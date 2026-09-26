"""Record, replay and execute a fixed arm/hand motion without object perception."""

import json
from pathlib import Path
import numpy as np
from dex_manipulation.robot.trajectory import RecordedCommands, SCHEMA, sha256
from dex_manipulation.ik import pose_error
import argparse
import asyncio
from datetime import datetime
import math
import sys
from dex_manipulation.configuration import read_config
from dex_manipulation.data import resolve_demo_path


def run(env, metadata, checkpoint, output, *, replay_path=None, hold_s=1.0, record_tactile=False):
    completed = False
    try:
        result = _run(
            env,
            metadata,
            checkpoint,
            output,
            replay_path=replay_path,
            hold_s=hold_s,
            record_tactile=record_tactile,
        )
        completed = True
        return result
    finally:
        recorder = getattr(env, "tactile_recorder", None)
        if record_tactile and recorder is not None and recorder.active:
            recorder.finish({"end": "completed" if completed else "aborted"})


def _run(env, metadata, checkpoint, output, *, replay_path=None, hold_s=1.0, record_tactile=False):
    import torch
    from dex_manipulation.policy.ppo import PPO

    output, checkpoint = Path(output), Path(checkpoint)
    if env.num_envs != 1 or not hasattr(env, "arm") or not hasattr(env.task, "contacts"):
        raise ValueError("Recording requires one assembled arm and pad/can contacts")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["metadata"]["contract_hash"] != metadata["contract_hash"]:
        raise ValueError("Checkpoint contract differs from recording physics/reference")
    env.checkpoint_physics_names = saved["metadata"]["physics"]
    env.load_training_state_dict(saved["training_state"])
    learner = None
    if replay_path is None:
        learner = PPO(env.task, env.cfg["ppo"], device=str(env.device), num_envs=1)
        learner.load(checkpoint, metadata, resume=False)
        learner.actor.eval()
        learner.critic.eval()
    env.set_training(False)
    env.evaluation_protocol = "strict"
    env.reset(randomize=False)
    initial_state = env.state()
    initial = np.r_[initial_state["q_arm"][0].cpu().numpy(), initial_state["q"][0].cpu().numpy()]
    initial_object = initial_state["object_position"][0].cpu().numpy().copy()
    names = env.arm.active_names + env.model.active_names
    frozen = RecordedCommands(replay_path) if replay_path else None
    if frozen:
        if frozen.names != tuple(names) or frozen.metadata["checkpoint_sha256"] != sha256(
            checkpoint
        ):
            raise ValueError("Recorded names/checkpoint changed")
        if (
            not np.allclose(frozen.metadata["world_from_source"], env.world_from_source, atol=1e-10)
            or frozen.metadata["workcell"] != env.workcell.metadata()
            or frozen.metadata["arm_asset_sha256"] != env.metadata["assembled_usd_sha256"]
            or frozen.metadata["arm_model_sha256"] != env.metadata["arm_model_sha256"]
            or not np.isclose(frozen.dt, env.task.dt)
        ):
            raise ValueError("Recorded placement or timing changed")
        # Initialization only. Motion below uses drive targets, never state writes.
        qa = torch.tensor(frozen.initial[:6][None], device=env.device, dtype=torch.float32)
        qf = frozen.initial[6:] @ env.model.coupling.T + env.model.offset
        for ids, q in (
            (env.arm_ids, qa),
            (env.full_ids, torch.tensor(qf[None], device=env.device, dtype=torch.float32)),
        ):
            env.robot.set_joint_positions(q, joint_indices=ids)
            env.robot.set_joint_velocities(torch.zeros_like(q), joint_indices=ids)
            env.robot.set_joint_position_targets(q, joint_indices=ids)
        env.world.physics_sim_view.update_articulations_kinematic()
        initial = frozen.initial.copy()
    writes = env.object_reset_count
    count = len(frozen.times) if frozen else env.reference_frame_count
    commands, velocities, rows, ik_ok = [], [], [], []
    hold_count = round(hold_s / env.task.dt)
    if hold_count < 1:
        raise ValueError("At least one final hold interval is required")
    recorder = None
    if record_tactile:
        from dex_manipulation.sensors.physx_tactile import attach_tactile_recording

        recorder = attach_tactile_recording(env, output)
        recorder.begin(1)
    target_q = None
    with torch.no_grad():
        for i in range(count + hold_count):
            hold = i >= count
            if recorder is not None:
                recorder.reference_time_override = min(i, count - 1) * env.task.dt
            if not frozen and not hold:
                action = learner.act(env.observation(), deterministic=True)
                _, _, term, trunc, info = env.step(action, auto_reset=False)
                target_q = np.r_[
                    info["applied_targets"]["q_arm"][0].cpu().numpy(),
                    info["applied_targets"]["active_q"][0].cpu().numpy(),
                ]
                velocity = info["applied_targets"]["q_arm_velocity"][0].cpu().numpy().copy()
                metrics = {k: float(v[0]) for k, v in info["metrics"].items()}
                good_ik = metrics["arm_ik_success"] > 0.5
                if (
                    not good_ik
                    or metrics["early_failure"]
                    or (bool(term[0] or trunc[0]) and i != count - 1)
                ):
                    raise RuntimeError(f"Incomplete policy episode at command {i}: {metrics}")
                forces = None  # Already consumed by the task; metrics retain contact duty.
            else:
                if not hold:
                    target_q = frozen.q[i]
                    velocity = frozen.data["q_arm_velocity_rad_s"][i]
                else:
                    velocity = np.zeros(6)
                full = target_q[6:] @ env.model.coupling.T + env.model.offset

                def tensor(value):
                    return torch.tensor(
                        np.asarray(value)[None], device=env.device, dtype=torch.float32
                    )

                env.robot.set_joint_position_targets(
                    tensor(target_q[:6]), joint_indices=env.arm_ids
                )
                env.robot.set_joint_velocity_targets(tensor(velocity), joint_indices=env.arm_ids)
                env.robot.set_joint_position_targets(tensor(full), joint_indices=env.full_ids)
                for _ in range(env.cfg["control_decimation"]):
                    env.world.step(render=False)
                forces = env.task.contacts.consume()[:, 0].cpu().numpy()
                contacts = forces >= env.task.settings["minimum_force_n"]
                metrics = dict(
                    grasp_opposition_duty=float((contacts[:, 0] & contacts[:, 1:].any(1)).mean()),
                    grasp_all_five_duty=float(contacts.all(1).mean()),
                    grasp_contact_count=float(contacts.sum(1).mean()),
                )
                good_ik = True
                if env.render:
                    env.world.render()
            state = {k: v[0].detach().cpu().numpy().copy() for k, v in env.state().items()}
            t = min(i, count - 1) * env.task.dt
            reference = env.reference.sample(min(t, env.reference.duration))
            expected = env.reference.points(
                reference["object_position"], reference["object_quaternion"]
            )[0]
            actual = env.reference.points(
                state["object_position"][None], state["object_quaternion"][None]
            )[0]
            position = env.bridge.target(state["wrist_position"], state["wrist_quaternion"])
            target_pose = env.arm.pose(target_q[:6])
            error = pose_error(target_pose, position)
            sigma, condition = env.bridge.solver.singularity(state["q_arm"])
            row = dict(
                timestamp_s=(i + 1) * env.task.dt,
                hold=hold,
                q_arm=state["q_arm"],
                q_finger=state["q"],
                q_finger_command=target_q[6:].copy(),
                finger_tracking_error_rad=target_q[6:] - state["q"],
                finger_guard_conflict=metrics.get("finger_guard_conflict", 0.0),
                q_full=state["full_q"],
                object_position_source=state["object_position"],
                object_quaternion_xyzw=state["object_quaternion"],
                object_error_m=np.linalg.norm(actual - expected, axis=1).mean(),
                wrist_actual_base=position,
                wrist_command_base=target_pose,
                wrist_position_error_m=np.linalg.norm(error[:3]),
                wrist_rotation_error_rad=np.linalg.norm(error[3:]),
                coupling_error_rad=np.abs(
                    state["full_q"] - state["q"] @ env.model.coupling.T - env.model.offset
                ).max(),
                sigma_min=sigma,
                condition=condition,
                opposition_duty=metrics["grasp_opposition_duty"],
                all_five_duty=metrics["grasp_all_five_duty"],
                contact_count=metrics["grasp_contact_count"],
            )
            if not all(np.isfinite(v).all() for v in row.values()):
                raise RuntimeError("Nonfinite recorded physics")
            rows.append(row)
            if not hold:
                commands.append(target_q.copy())
                velocities.append(velocity.copy())
                ik_ok.append(good_ik)
    if writes != env.object_reset_count:
        raise RuntimeError("Object reset inside the recorded episode")
    arrays = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    np.savez_compressed(output / "rollout.npz", **arrays)
    motion = ~arrays["hold"]
    held = arrays["hold"]
    lift = arrays["object_position_source"][:, 2] - initial_object[2]
    # A lift and a held opposing-finger contact, not completion alone.
    grasp = bool(
        lift[held].min() > 0.05
        and arrays["opposition_duty"][held].min() >= 0.75
        and arrays["object_error_m"].max() < env.cfg["success_object_error_m"]
    )
    report = dict(
        schema="recorded_grasp_report_v1",
        simulated_grasp=grasp,
        mode="frozen_joint_commands" if frozen else "policy_capture",
        command_count=count,
        control_dt_s=env.task.dt,
        reference_duration_s=env.reference.duration,
        execution_duration_s=count * env.task.dt,
        hold_s=hold_count * env.task.dt,
        checkpoint_sha256=sha256(checkpoint),
        checkpoint_iteration=saved["iteration"],
        object_estimator_required_for_replay=False,
        policy_calls=0 if frozen else count,
        online_ik_calls=0 if frozen else count,
        object_writes_during_motion=0,
        can_initial_world_pose=env.initial_can_pose.tolist(),
        mean_object_error_mm=float(arrays["object_error_m"][motion].mean() * 1000),
        max_object_error_mm=float(arrays["object_error_m"].max() * 1000),
        final_lift_mm=float(lift[-1] * 1000),
        minimum_hold_lift_mm=float(lift[held].min() * 1000),
        minimum_hold_opposition_duty=float(arrays["opposition_duty"][held].min()),
        mean_hold_all_five_duty=float(arrays["all_five_duty"][held].mean()),
        mean_wrist_tracking_error_mm=float(arrays["wrist_position_error_m"][motion].mean() * 1000),
        max_wrist_tracking_error_mm=float(arrays["wrist_position_error_m"].max() * 1000),
        minimum_sigma=float(arrays["sigma_min"].min()),
        maximum_coupling_error_rad=float(arrays["coupling_error_rad"].max()),
        strict_coupling_0_01rad_pass=bool(arrays["coupling_error_rad"].max() < 0.01),
        maximum_finger_tracking_error_deg=float(
            np.rad2deg(np.abs(arrays["finger_tracking_error_rad"])).max()
        ),
        per_joint_mean_tracking_error_deg=np.rad2deg(
            np.abs(arrays["finger_tracking_error_rad"]).mean(0)
        ).tolist(),
        per_joint_max_tracking_error_deg=np.rad2deg(
            np.abs(arrays["finger_tracking_error_rad"]).max(0)
        ).tolist(),
        per_joint_hold_tracking_error_deg=np.rad2deg(
            np.abs(arrays["finger_tracking_error_rad"])[held].mean(0)
        ).tolist(),
        finger_guard_conflict_frames=np.flatnonzero(arrays["finger_guard_conflict"] > 0).tolist(),
        hardware_validated=False,
    )
    settings = env.cfg.get("finger_tracking", {})
    if settings.get("enabled", False):
        limits = np.array([settings["max_error_rad"][n] for n in env.model.active_names])
        excessive = np.abs(arrays["finger_tracking_error_rad"]) > limits + 1e-4
        report["finger_tracking"] = dict(
            max_error_deg=np.rad2deg(limits).tolist(),
            exceeds_bound_frames=np.flatnonzero(excessive.any(1)).tolist(),
            hold_within_bound=bool(not excessive[held].any()),
            all_frames_within_bound=bool(not excessive.any()),
            interpretation="Measured end-of-control-interval error including fixed final hold; issued-target bounds do not guarantee post-step tracking",
        )
    if not frozen:
        m = dict(
            schema=SCHEMA,
            angle_unit="rad",
            length_unit="m",
            quaternion_order="xyzw",
            command_semantics="start_of_interval_zero_order_hold",
            control_dt_s=env.task.dt,
            reference_duration_s=env.reference.duration,
            execution_duration_s=count * env.task.dt,
            checkpoint=str(checkpoint.resolve()),
            checkpoint_sha256=sha256(checkpoint),
            arm_asset_sha256=env.metadata["assembled_usd_sha256"],
            arm_model_sha256=env.metadata["arm_model_sha256"],
            policy_contract_hash=metadata["contract_hash"],
            config=metadata["config"],
            world_from_source=env.world_from_source.tolist(),
            workcell=env.workcell.metadata(),
            initial_can_world_pose=env.initial_can_pose.tolist(),
            timing_note="One unchanged reset-to-first-command interval is included; no speed scaling",
            capture_report=report,
            hardware_calibrated=False,
        )
        path = output / "commands.npz"
        np.savez_compressed(
            path,
            timestamp_s=np.arange(count) * env.task.dt,
            q_command_rad=np.asarray(commands),
            q_arm_velocity_rad_s=np.asarray(velocities),
            initial_q_rad=initial,
            joint_names=np.asarray(names),
            ik_success=np.asarray(ik_ok),
            lower_rad=np.r_[env.arm.lower, env.model.lower],
            upper_rad=np.r_[env.arm.upper, env.model.upper],
            velocity_limit_rad_s=np.r_[env.arm.velocity, env.model.velocity],
            metadata_json=np.asarray(json.dumps(m)),
        )
        recording = RecordedCommands(path)
        recording.export_csv(output / "commands.csv")
        report["commands_sha256"] = sha256(path)
    else:
        report["commands_sha256"] = sha256(frozen.path)
    if recorder is not None:
        recorder.metadata.update(
            source_commands_sha256=report["commands_sha256"],
            time_zero="first recorded command",
            command_hz=1 / env.task.dt,
            hold_s=hold_count * env.task.dt,
        )
        (recorder.output / "metadata.json").write_text(
            json.dumps(recorder.metadata, indent=2) + "\n"
        )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("[recorded grasp]", json.dumps(report), flush=True)
    if not grasp:
        raise RuntimeError(
            "Saved diagnostics, but this run did not pass the simulated lift-and-hold criterion"
        )
    return report


ROOT = Path(__file__).resolve().parents[3]


def _main(argv=None):
    defaults = read_config(ROOT / "config/execution.json")
    parser = argparse.ArgumentParser(prog="./run.sh execute")
    parser.add_argument(
        "mode", choices=("record", "replay", "inspect", "dry-run", "probe", "hardware")
    )
    parser.add_argument("--recording", type=Path, default=ROOT / defaults["recording"])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="record: latest completed demo policy by default; replay: recorded checkpoint",
    )
    parser.add_argument("--arm-config", type=Path, default=ROOT / defaults["arm_config"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hold", type=float, default=defaults["hold_s"])
    parser.add_argument(
        "--gui", action="store_true", help="Show the requested simulation (default headless)"
    )
    parser.add_argument(
        "--hardware-config", type=Path, default=ROOT / "config/hardware.example.json"
    )
    parser.add_argument(
        "--send", action="store_true", help="Actually connect and stream to commissioned hardware"
    )
    parser.add_argument(
        "--record-tactile",
        action="store_true",
        help="Record tactile on the same I/O owner as motion",
    )
    parser.add_argument(
        "--tactile-hz",
        type=float,
        default=100.0,
        help="Requested hardware tactile rate; limited by available command time",
    )
    parser.add_argument("--seconds", type=float, default=5.0, help="Read-only probe duration")
    args = parser.parse_args(argv)
    for name in ("recording", "checkpoint", "arm_config", "hardware_config", "output"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, (ROOT / value).resolve())
    if args.send and args.mode != "hardware":
        parser.error("--send requires hardware mode")
    if args.record_tactile and args.mode not in ("hardware", "probe", "record", "replay"):
        parser.error("--record-tactile requires hardware, probe, record or replay")
    if not math.isfinite(args.hold) or args.hold <= 0:
        parser.error("--hold must be positive and finite")
    if not math.isfinite(args.tactile_hz) or args.tactile_hz <= 0:
        parser.error("--tactile-hz must be positive and finite")
    from dex_manipulation.robot.trajectory import RecordedCommands, MockBackend, stream

    frozen = None if args.mode in ("record", "probe") else RecordedCommands(args.recording)
    if args.checkpoint is None and args.mode in ("record", "replay"):
        if frozen:
            args.checkpoint = resolve_demo_path(frozen.metadata["checkpoint"], ROOT).resolve()
        else:
            from dex_manipulation.policies import latest_policy

            args.checkpoint = Path(latest_policy(ROOT, defaults["demo_id"], "arm")["checkpoint"])
        print(f"정책: {args.checkpoint}", flush=True)
    if args.mode == "inspect":
        print(json.dumps(frozen.summary(), indent=2))
        return 0
    output = (
        args.output
        or ROOT
        / (
            "local/results/execution/"
            + args.mode
            + "_"
            + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
    ).resolve()
    if not output.is_relative_to(ROOT / "local") or output.exists():
        parser.error("Use a new output directory under local/")
    if args.mode == "probe":
        from dex_manipulation.robot.hardware import probe

        result = asyncio.run(
            probe(read_config(args.hardware_config), output, args.seconds, args.record_tactile)
        )
        print(json.dumps(result, indent=2))
        print(f"Log: {output}")
        return 0
    if args.mode == "dry-run":
        result = asyncio.run(
            stream(
                frozen,
                MockBackend(frozen.initial),
                output,
                start_tolerance_rad=[0.01] * 12,
                tracking_tolerance_rad=[0.01] * 12,
                maximum_feedback_age_s=0.1,
                maximum_lateness_s=0.02,
                hold_s=args.hold,
            )
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.mode == "hardware":
        from dex_manipulation.robot.hardware import hardware_plan, execute

        config = read_config(args.hardware_config)
        config["simulation_validation"] = str((ROOT / config["simulation_validation"]).resolve())
        plan = hardware_plan(frozen, config)
        if not args.send:
            print(json.dumps(plan, indent=2))
            return 0
        if args.record_tactile:
            from dex_manipulation.sensors.session import MotionTelemetry

            MotionTelemetry(frozen, output, args.tactile_hz)  # Validate before connecting.
        print(
            json.dumps(
                asyncio.run(
                    execute(
                        frozen,
                        config,
                        output,
                        args.hold,
                        record_tactile=args.record_tactile,
                        tactile_hz=args.tactile_hz,
                    )
                ),
                indent=2,
            )
        )
        print(f"Log: {output}")
        return 0
    output.mkdir(parents=True)
    config = (
        frozen.metadata["config"]
        if frozen
        else read_config(args.checkpoint.parent / "config.resolved.json")
    )
    if (
        frozen
        and args.checkpoint.resolve()
        != resolve_demo_path(frozen.metadata["checkpoint"], ROOT).resolve()
    ):
        parser.error("Replay must use the checkpoint recorded in the trajectory")
    config_path = output / "physics_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    from scripts.policy import parse_args, launch

    sim_args = parse_args(
        [
            "--mode",
            "play",
            "--robot",
            "arm",
            "--episodes",
            "1",
            "--speed",
            "1",
            "--evaluation-protocol",
            "strict",
            "--contact-materials",
            "checkpoint",
            "--table-safety",
            "checkpoint",
            "--checkpoint",
            str(args.checkpoint),
            "--config",
            str(config_path),
            "--arm-config",
            str(args.arm_config),
            "--output",
            str(output),
        ]
        + ([] if args.gui else ["--headless"])
    )

    def ready(env, metadata):
        pass

        return run(
            env,
            metadata,
            args.checkpoint,
            output,
            replay_path=args.recording if args.mode == "replay" else None,
            hold_s=args.hold,
            record_tactile=args.record_tactile,
        )

    return launch(sim_args, on_ready=ready)


def main(argv=None):
    try:
        return _main(argv)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"실행 준비/제어 실패: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
