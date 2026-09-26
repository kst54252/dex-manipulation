"""Training/evaluation orchestration, invoked after Isaac startup."""

from ..configuration import read_config
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..fk import HandModel
from ..data import legacy_demo_paths
from dex_manipulation.policy.trajectory import ReferenceMotion, digest
from .floating_env import PhysxResidualEnv
from .ppo import PPO
from .trainer import train_policy
from .evaluation import evaluate, evaluate_contacts, plot_comparison
from dex_manipulation.tasks.registry import attach_grasp_reward


def run(args, root, on_ready=None, is_running=None):
    speed = getattr(args, "speed", 1.0)
    if speed != 1.0 and args.mode != "play":
        raise ValueError("--speed applies only to play; training timing is unchanged")
    print("[startup] load configuration and reference", flush=True)
    config = read_config(Path(args.config))
    config["source_profile"] = args.source_profile or config.get("source_profile", "leap")
    if config["source_profile"] == "wuji":
        config["reward"]["object_std_m"] = 0.015
        config["reward"]["action_l2_weight"] = 0.5
        config["domain_randomization"]["hand_friction"] = [0.5, 1.5]
        config["domain_randomization"]["joint_default_offset_rad"] = 0.0
        config["domain_randomization"]["object_com_range_m"] = [0.005, 0.02, 0.005]
        config["domain_randomization"]["com_basis"] = "Source Wuji rigid screwdriver XYZ COM ranges"
    if args.device is not None:
        config["physics_device"] = args.device
    if args.policy_device is None:
        args.policy_device = config["physics_device"]
    if args.num_envs is None:
        args.num_envs = config["training"]["num_envs"] if args.mode == "train" else 2
    if args.iterations is None:
        args.iterations = config["training"]["iterations"]
    if args.save_every is None:
        args.save_every = config["training"]["save_every"]
    if args.rsi_sampling is not None:
        config["rsi"]["sampling"] = args.rsi_sampling
    if args.enable_rsi:
        config["rsi"]["enabled"] = True
    for name, key in (
        ("disable_rsi", "rsi"),
        ("disable_augmentation", "augmentation"),
        ("disable_gravity_curriculum", "gravity_curriculum"),
    ):
        if getattr(args, name):
            config[key]["enabled"] = False
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = HandModel.load(root / config["model"])
    reference = ReferenceMotion(
        root / config["reference"], model, root / config["object_geometry"], config["world_frame"]
    )
    if (
        args.mode == "train"
        and on_ready is None
        and "base_diameter_m" in reference.object_geometry.get("specification", {})
    ):
        # Keep historical checkpoints inspectable, but prevent new learning
        # from silently repeating an inverted asymmetric-can experiment.
        reference.validate_can_base_down()
    # Preserve the exact metadata contract of pre-v7 checkpoints.
    if "reference_timing" in config:
        reference.configure_timing(config)
    else:
        reference.retime_for_control_horizon(
            config["episode_length_s"], config["physics_dt"] * config["control_decimation"]
        )
    last_frame = reference.metadata["control_reference_frames"] - 1
    dt = config["physics_dt"] * config["control_decimation"]
    if config["augmentation"].get("mode", "fade") == "fade":
        for key, ratio in (
            ("blend_start_s", "phase_start_ratio"),
            ("blend_end_s", "phase_end_ratio"),
        ):
            config["augmentation"][key] = int(last_frame * config["augmentation"][ratio]) * dt
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        config=config,
        reference=reference.metadata,
        model_sha256=digest(root / config["model"]),
        observation_schema="regrind_revo2_v4_actor67_critic94"
        if config["observation"].get("include_object_velocity", False)
        else "regrind_revo2_v3_actor67_critic88",
        implementation="independent task and simulator adapter; general RSL-RL PPO; no REGRIND imports",
    )
    if config.get("reference_phase") == "endpoint_normalized":
        metadata["observation_schema"] = "regrind_revo2_v6_actor67_critic94"
    robot_mode = getattr(args, "robot", "floating")
    native_arm = config.get("arm_training", {}).get("enabled", False)
    arm_config = None
    if native_arm and robot_mode != "arm":
        raise ValueError("An arm-trained policy requires --robot arm and real arm observations")
    if robot_mode == "arm":
        arm_path = getattr(args, "arm_config", None) or (
            root / config["arm_training"]["arm_config"]
            if native_arm
            else root / "config/tasks/can_pick/ik_demo2.json"
        )
        arm_config = read_config(Path(arm_path))
        if native_arm:
            metadata["observation_schema"] = "revo2_arm_actor87_critic114_v1"
            metadata["arm_training"] = dict(
                config=arm_config,
                assets={
                    name: digest(root / arm_config[name])
                    for name in ("usd", "arm_model", "workcell", "alignment")
                },
            )
        elif args.mode != "play":
            raise ValueError("Arm training requires config.arm_training.enabled")
    from ..scene import FloatingSurface

    metadata["surface"] = FloatingSurface(config["surface"], reference.object[0, :3, 3]).metadata()
    if native_arm:
        from ..scene import Workcell

        metadata["surface"] = Workcell.load(root / arm_config["workcell"]).metadata()
    metadata["contract_hash"] = hashlib.sha256(
        json.dumps(legacy_demo_paths(metadata), sort_keys=True).encode()
    ).hexdigest()
    # Preserve strict historical checkpoint contracts. Deployment material
    # changes belong to an explicit execution profile, never forged metadata.
    from dex_manipulation.policy.trajectory import prepare_playback

    contact_mode = getattr(args, "contact_materials", None) or (
        "rubber" if args.mode == "play" else "checkpoint"
    )
    training_reference = reference
    reference, env_config, playback_timing = prepare_playback(reference, config, speed)
    placement = None
    if getattr(args, "random_can", False):
        if args.mode != "play" or robot_mode != "arm" or native_arm:
            raise ValueError("--random-can supports demo 2 floating-trained arm playback only")
        from .placement import load_random_placement

        placement = load_random_placement(
            root, config, arm_config, speed, getattr(args, "placement_seed", None), reference
        )
    if contact_mode == "rubber":
        env_config["contact_materials"] = read_config(
            root / "config/tasks/can_pick/policy_demo2.json"
        )["contact_materials"]
    control_mode = getattr(args, "motion_control", None) or "checkpoint"
    stable = (
        read_config(root / "config/tasks/can_pick/policy_demo2.json")["motion_control"]
        if control_mode == "stable"
        else None
    )
    if stable is not None:
        stable = dict(stable, enabled=True)
    if native_arm:
        from .arm_train_env import ArmTrainingEnv

        env = ArmTrainingEnv(
            root, model, reference, env_config, arm_config, args.num_envs, render=not args.headless
        )
    elif robot_mode == "arm":
        from .arm_play_env import ArmPolicyEnv

        env = ArmPolicyEnv(root, model, reference, env_config, arm_config, render=not args.headless)
        # Constructor reset uses the nominal placement; consume the random
        # sequence only when the first actual playback episode begins.
        if placement is not None:
            env.placement_sampler = placement
    else:
        env = PhysxResidualEnv(
            root,
            model,
            reference,
            env_config,
            args.num_envs,
            render=not args.headless,
            motion_control_override=stable,
        )
    table_mode = getattr(args, "table_safety", None) or (
        "protect" if args.mode == "play" and not native_arm else "checkpoint"
    )
    if table_mode == "protect":
        table_settings = read_config(root / "config/tasks/can_pick/policy_demo2.json")[
            "table_safety"
        ]
        env.configure_table_safety(dict(table_settings, enabled=True, guard_enabled=True))
    attach_grasp_reward(env, metadata, training_reference)
    metadata["physics"] = env.metadata
    # Execution/output choices do not change the policy/checkpoint contract.
    metadata["execution"] = dict(
        headless=args.headless,
        render=not args.headless,
        viewport_updates=not args.headless,
        skip_evaluation=args.skip_evaluation,
        export=args.export,
        automatic_replay_output=not args.skip_evaluation and not native_arm,
        mode=args.mode,
    )
    metadata["execution"]["robot"] = robot_mode
    metadata["execution"]["playback_timing"] = playback_timing
    if placement is not None:
        metadata["execution"]["random_can"] = placement.metadata
    if speed != 1.0:
        print(
            f"[playback] {speed:g}x | reference {playback_timing['input_duration_s']:g}s -> {reference.duration:g}s | physics/control dt unchanged",
            flush=True,
        )
    metadata["execution"]["contact_materials"] = dict(
        mode=contact_mode,
        trained=config.get("contact_materials"),
        applied=env_config.get("contact_materials"),
        differs_from_training=config.get("contact_materials")
        != env_config.get("contact_materials"),
    )
    if metadata["execution"]["contact_materials"]["differs_from_training"]:
        print(
            "[contact] Five rubber pads enabled for playback; contact physics differs from this checkpoint training. Use --contact-materials checkpoint to reproduce it.",
            flush=True,
        )
    metadata["execution"]["table_safety"] = dict(
        mode=table_mode,
        trained=config.get("table_safety"),
        applied=env.table_safety.config if env.table_safety is not None else None,
        interpretation="collision-geometry command guard; dynamic contacts still require validation",
    )
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "config.resolved.json").write_text(json.dumps(config, indent=2) + "\n")
    try:
        print(
            "[recipe]",
            json.dumps(
                dict(
                    rsi=config["rsi"]["enabled"],
                    gravity_curriculum=config["gravity_curriculum"]["enabled"],
                    augmentation=config["augmentation"]["enabled"],
                    device=str(env.device),
                    num_envs=env.num_envs,
                    gpu_dynamics=env.metadata.get("gpu_dynamics_enabled"),
                    broadphase=env.metadata.get("broadphase"),
                    learning=on_ready is None and args.mode == "train",
                )
            ),
            flush=True,
        )
        if on_ready is not None:
            return on_ready(env, metadata)
        print("[startup] allocate PPO model and rollout storage", flush=True)
        learner = PPO(env.task, config["ppo"], device=args.policy_device, num_envs=env.num_envs)
        if args.checkpoint:
            loaded = learner.load(args.checkpoint, metadata, resume=args.mode == "train")
            if robot_mode == "arm":
                env.checkpoint_physics_names = loaded["metadata"]["physics"]
            if loaded.get("training_state") is not None:
                state = loaded["training_state"]
                if speed != 1.0 and hasattr(env, "sampler"):
                    # RSI is disabled in play. Its saved time-bin histogram
                    # belongs to the original horizon and cannot be restored
                    # into the shorter one; keep all saved physical properties.
                    # The floating-to-arm adapter has no RSI sampler and only
                    # restores named physical properties from this state.
                    state = dict(state, sampler=env.sampler.state_dict())
                env.load_training_state_dict(state)
            elif robot_mode == "arm":
                raise ValueError(
                    "Arm deployment requires saved named hand/object physical properties"
                )
            elif args.mode == "train":
                raise ValueError("Resume requires complete curriculum state")
        initialize = getattr(args, "initialize_actor", None)
        if initialize:
            metadata["execution"]["actor_initialization"] = learner.initialize_actor(
                initialize, metadata
            )
            metadata["execution"]["actor_initialization"].update(
                path=str(initialize), sha256=digest(initialize)
            )
        metadata["execution"]["motion_control"] = dict(
            mode=control_mode,
            trained=config.get("motion_control"),
            applied=env.motion_controller.config,
            differs_from_training=config.get("motion_control") != env.motion_controller.config,
        )
        from ..materials import pad_contact_report

        metadata["execution"]["contact_materials"]["runtime"] = pad_contact_report(env)
        (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if args.mode != "train":
            print(
                "[motion control]", json.dumps(metadata["execution"]["motion_control"]), flush=True
            )
            print("[table safety]", json.dumps(metadata["execution"]["table_safety"]), flush=True)
        if args.mode == "play":
            from .play import play

            if getattr(args, "record_tactile", False):
                from ..sensors.physx_tactile import attach_tactile_recording

                attach_tactile_recording(env, args.output)
            return play(
                env,
                learner,
                args.output,
                args.episodes,
                is_running,
                protocol="source" if args.evaluation_protocol == "source" else "strict",
            )
        baseline = None
        primary_protocol = "source" if args.evaluation_protocol == "source" else "strict"
        contact_only = bool(config.get("five_finger_grasp"))
        if not args.skip_evaluation and not contact_only:
            baseline = evaluate(
                env,
                learner,
                args.output,
                "zero_residual",
                config["seed"] + 1,
                zero=True,
                protocol=primary_protocol,
            )
        if args.mode == "train":
            train_policy(env, learner, args, metadata)
        if args.export:
            learner.export(args.output / "exported")
        if args.skip_evaluation:
            return
        if contact_only:
            return evaluate_contacts(env, learner, args.output)
        trained = evaluate(
            env,
            learner,
            args.output,
            "trained_residual",
            config["seed"] + 1,
            protocol=primary_protocol,
        )
        if args.evaluation_protocol == "both":
            evaluate(
                env,
                learner,
                args.output,
                "zero_residual_source_play",
                config["seed"] + 1,
                zero=True,
                protocol="source",
            )
            evaluate(
                env,
                learner,
                args.output,
                "trained_residual_source_play",
                config["seed"] + 1,
                protocol="source",
            )
        comparison = dict(
            training_steps=learner.steps,
            zero_residual_success=baseline["success_count"],
            trained_success=trained["success_count"],
            episodes=env.num_envs,
            return_difference_mean=float(
                np.mean(trained["returns"]) - np.mean(baseline["returns"])
            ),
            simulated_full_sequence_tracking_success_observed=bool(trained["success_count"] > 0),
            policy_status="candidate_only; no automatic replacement of geometric reference",
            limitations=[
                "single sequence",
                "assumed table/world frame",
                "physical arm with sampled IK transition checks; not a general collision certificate"
                if native_arm
                else "idealized floating wrist",
                "original capture fps unknown",
            ],
            training_rsi=config["rsi"],
            training_gravity_curriculum=config["gravity_curriculum"],
            training_augmentation=config["augmentation"],
        )
        (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        plot_comparison(args.output)
        if not native_arm:
            from .replay import write_replay

            write_replay(args.output, model, reference)
    finally:
        env.close()
