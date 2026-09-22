"""Freeze one policy rollout, solve once, and measure open-loop arm tracking.

Planning is simulator-independent. Replay uses saved named joint targets only;
the actor, online IK and virtual wrist response are never called in the loop.
"""

from ..configuration import read_config
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..fk import ArmModel, HandModel
from ..ik import IKOptions, model_fingerprint, pose_error, solve_trajectory
from ..joint_trajectory import JointReference
from ..scene import Workcell
from ..transforms import transform
from .trajectory import ReferenceMotion, digest


def frozen_source(rollout, reference, control_dt, kind="command", time_scale=1.0):
    """Retain initial rest state and all post-interval samples, including tick 0.

    Raw policy targets are desired PD poses, not measured floating poses. The
    latter can be selected explicitly as a separate motion-reproduction test.
    One reset-to-first-command interval is included, so a 1.3 s / 40-command
    reference occupies 40/30 s of physics, not 39/30 s.
    """
    if kind not in ("command", "measured") or not np.isfinite(time_scale) or time_scale <= 0:
        raise ValueError("Choose command/measured and a positive finite time scale")
    count = reference.metadata["control_reference_frames"]
    expected = np.arange(1, count + 1) * control_dt
    if (
        len(rollout["physics_time_s"]) != count
        or not np.allclose(rollout["physics_time_s"], expected, atol=1e-6, rtol=0)
        or not np.allclose(rollout["time_s"], np.arange(count) * control_dt, atol=2e-6, rtol=0)
    ):
        raise ValueError("Capture must contain every command from frame zero through demo end")
    p, r, q = (
        ("raw_target_position", "raw_target_quaternion", "raw_target_active_q")
        if kind == "command"
        else ("wrist_position", "wrist_quaternion", "q")
    )
    positions, quaternions, joints = (np.asarray(rollout[k]) for k in (p, r, q))
    if (
        positions.shape != (count, 3)
        or quaternions.shape != (count, 4)
        or joints.shape != (count, 6)
        or not all(np.isfinite(v).all() for v in (positions, quaternions, joints))
        or not np.allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError("Malformed policy wrist/finger samples")
    wrist = np.tile(np.eye(4), (count + 1, 1, 1))
    wrist[0] = reference.wrist[0]
    wrist[1:, :3, 3] = positions
    wrist[1:, :3, :3] = Rotation.from_quat(quaternions).as_matrix()
    obj = np.tile(np.eye(4), (count + 1, 1, 1))
    obj[0] = reference.object[0]
    obj[1:, :3, 3] = rollout["reference_object_position"]
    obj[1:, :3, :3] = Rotation.from_quat(rollout["reference_object_quaternion"]).as_matrix()
    return dict(
        timestamps_s=np.r_[0.0, expected] * time_scale,
        wrist_transform=wrist,
        object_transform=obj,
        active_q_rad=np.vstack([reference.q[0], joints]),
        active_joint_names=np.array(reference.model.active_names),
        frame_ids=np.arange(count + 1),
        valid=np.ones(count + 1, bool),
        transition_valid=np.ones(count + 1, bool),
        object_geometry_fingerprint=np.array(reference.metadata["object_geometry_fingerprint"]),
    )


def plan(root, capture, arm_config_path, output, kind="command", time_scale=1.0):
    root, capture, output = map(Path, (root, capture, output))
    if (output / "trajectory.npz").exists():
        raise ValueError("Choose a new output directory; existing trajectories are immutable")
    metadata = read_config(capture / "run_metadata.json")
    playback = read_config(capture / "playback.json")
    if playback["robot"] != "floating Revo2" or playback["protocol"] != "strict":
        raise ValueError("Use one strict floating policy capture without augmentation")
    config = metadata["config"]
    hand = HandModel.load(root / config["model"])
    reference = ReferenceMotion(
        root / config["reference"], hand, root / config["object_geometry"], config["world_frame"]
    )
    reference.configure_timing(config)
    if reference.metadata != metadata["reference"]:
        raise ValueError("Capture reference changed")
    with np.load(capture / "first_episode.npz", allow_pickle=False) as data:
        source = frozen_source(
            data, reference, config["physics_dt"] * config["control_decimation"], kind, time_scale
        )
    arm_config = read_config(Path(arm_config_path))
    arm = ArmModel.load(root / arm_config["arm_model"])
    workcell = Workcell.load(root / arm_config["workcell"])
    alignment = workcell.resolve_alignment(read_config(root / arm_config["alignment"]))
    if digest(root / arm_config["input"]) != reference.metadata["reference_sha256"]:
        raise ValueError("Arm placement config must refer to the captured reference")
    # Use the control grid even after slowing the saved motion; never reduce
    # controller/physics frequencies to implement slow playback.
    substeps = round(time_scale)
    if substeps < 1 or not np.isclose(time_scale, substeps):
        raise ValueError("Tracking plans currently require an integer time scale >= 1")
    output.mkdir(parents=True, exist_ok=True)
    source_path = output / "frozen_source.npz"
    np.savez_compressed(source_path, **source)
    report = solve_trajectory(
        arm,
        hand,
        source_path,
        output,
        alignment["base_from_source"],
        arm_config["seed_q_rad"],
        IKOptions(**arm_config["solver"]),
        alignment_status="frozen_policy_simulation_diagnostic",
        trajectory_substeps=substeps,
    )
    manifest = dict(
        schema="frozen_policy_tracking_v1",
        capture=str(capture.resolve()),
        capture_sha256=digest(capture / "first_episode.npz"),
        policy_contract_hash=metadata["contract_hash"],
        policy_iteration=playback["checkpoint_iteration"],
        config=config,
        arm_config=arm_config,
        arm_config_sha256=digest(arm_config_path),
        kind=kind,
        time_scale=time_scale,
        source_reference_duration_s=reference.duration,
        execution_duration_s=float(source["timestamps_s"][-1]),
        timestamps="Post-interval times; includes the initial reset-to-first-target control interval",
        command_semantics=(
            "Frozen actor PD pose and finger position targets"
            if kind == "command"
            else "Frozen measured floating wrist AND measured finger motion, not actor targets"
        ),
        world_from_source=alignment["world_from_source"],
        trajectory_sha256=digest(output / "trajectory.npz"),
        physics_condition="Preserves the captured checkpoint can orientation and physics; slowing is an open-loop experiment, not retraining",
        collision_validation="No geometric collision certificate; dynamic contact is measured during replay",
    )
    (output / "frozen.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if report["continuous_reference_valid"]:
        JointReference(output / "trajectory.npz").export_csv(output / "targets.csv")
    return report


def stats(values):
    values = np.asarray(values)
    return dict(
        mean=float(values.mean()), maximum=float(values.max()), p95=float(np.percentile(values, 95))
    )


def replay(
    env,
    metadata,
    root,
    plan_path,
    checkpoint_path,
    output,
    repeats=3,
    hold_s=1.0,
    velocity_feedforward=True,
):
    """Use PhysX drives, never articulation-state writes during motion.

    The checkpoint is read only to restore materials/masses/gains. No network
    is constructed. All IK and joint velocities are resolved before playback.
    """
    import torch

    root, plan_path, output = map(Path, (root, plan_path, output))
    manifest = read_config(plan_path.parent / "frozen.json")
    reference = JointReference(plan_path)
    if digest(plan_path) != manifest["trajectory_sha256"]:
        raise ValueError("Saved trajectory changed after planning")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not (
        saved["metadata"]["contract_hash"]
        == manifest["policy_contract_hash"]
        == metadata["contract_hash"]
    ):
        raise ValueError("Capture/checkpoint/physics configuration mismatch")
    if (
        model_fingerprint(env.arm) != reference.metadata["arm_fingerprint"]
        or model_fingerprint(env.model) != reference.metadata["hand_fingerprint"]
        or not np.allclose(env.world_from_source, manifest["world_from_source"], atol=1e-10, rtol=0)
    ):
        raise ValueError("Model or placement changed after planning")
    dt = env.task.dt
    if not np.allclose(np.diff(reference.times), dt, atol=1e-7, rtol=0):
        raise ValueError("Plan must lie on the fixed control grid")
    if repeats < 1 or not np.isfinite(hold_s) or hold_s < 0:
        raise ValueError("Invalid repeat/hold duration")
    if (output / "summary.json").exists():
        raise ValueError("Choose a new replay output directory")
    env.checkpoint_physics_names = saved["metadata"]["physics"]
    env.load_training_state_dict(saved["training_state"])
    env.set_training(False)
    env.evaluation_protocol = "strict"
    base_from_source = np.asarray(reference.metadata["base_from_source"])
    d = reference.data
    qa, qf = d["q_arm"], d["q_finger"]
    qv = np.vstack([np.zeros(6), np.diff(qa, axis=0) / dt])
    full = qf @ env.model.coupling.T + env.model.offset
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for repeat in range(repeats):
        env.reset(randomize=False)
        # Initialization only. This is not a trajectory from an arbitrary robot posture.
        for ids, values in ((env.arm_ids, qa[:1]), (env.full_ids, full[:1])):
            values = torch.as_tensor(values, device=env.device, dtype=torch.float32)
            env.robot.set_joint_positions(values, joint_indices=ids)
            env.robot.set_joint_velocities(torch.zeros_like(values), joint_indices=ids)
            env.robot.set_joint_position_targets(values, joint_indices=ids)
        env.world.physics_sim_view.update_articulations_kinematic()
        initial = env.state()
        if np.linalg.norm(initial["q_arm"][0].cpu().numpy() - qa[0]) > 1e-5:
            raise RuntimeError("Initial articulation state does not match the planned branch")
        writes = env.object_reset_count
        rows = []
        count = len(qa) - 1
        hold_count = round(hold_s / dt)
        for step in range(count + hold_count):
            i = min(step + 1, len(qa) - 1)
            hold = step >= count
            arm_velocity = np.zeros(6) if hold or not velocity_feedforward else qv[i]
            env.robot.set_joint_position_targets(
                torch.tensor(qa[i : i + 1], device=env.device, dtype=torch.float32),
                joint_indices=env.arm_ids,
            )
            env.robot.set_joint_velocity_targets(
                torch.tensor(arm_velocity[None], device=env.device, dtype=torch.float32),
                joint_indices=env.arm_ids,
            )
            env.robot.set_joint_position_targets(
                torch.tensor(full[i : i + 1], device=env.device, dtype=torch.float32),
                joint_indices=env.full_ids,
            )
            # Deliberately no env.step/actor/IK/response call. Do not early-reset
            # after object loss; record the entire motion and requested hold.
            for substep in range(env.cfg["control_decimation"]):
                env.world.step(render=False)
                s = {k: v[0].detach().cpu().numpy() for k, v in env.state().items()}
                actual = base_from_source @ transform(
                    Rotation.from_quat(s["wrist_quaternion"]).as_matrix(), s["wrist_position"]
                )
                fk = env.arm.pose(s["q_arm"])
                error = pose_error(d["wrist_target_pose"][i], actual)
                consistency = pose_error(fk, actual)
                sigma, condition = env.bridge.solver.singularity(s["q_arm"])
                obj = base_from_source @ transform(
                    Rotation.from_quat(s["object_quaternion"]).as_matrix(), s["object_position"]
                )
                points = env.reference.object_local
                expected_points = (
                    points @ d["object_transform"][i, :3, :3].T + d["object_transform"][i, :3, 3]
                )
                actual_points = points @ obj[:3, :3].T + obj[:3, 3]
                rows.append(
                    dict(
                        timestamp_s=(step * env.cfg["control_decimation"] + substep + 1)
                        * env.cfg["physics_dt"],
                        target_timestamp_s=reference.times[i],
                        command_index=i,
                        control_endpoint=substep == env.cfg["control_decimation"] - 1,
                        hold=hold,
                        q_arm=s["q_arm"],
                        q_arm_velocity=s["q_arm_velocity"],
                        q_finger=s["q"],
                        q_full=s["full_q"],
                        target_q_arm=qa[i],
                        target_q_finger=qf[i],
                        target_q_arm_velocity=arm_velocity,
                        wrist_target=d["wrist_target_pose"][i],
                        wrist_actual=actual,
                        fk_actual=fk,
                        object_actual=obj,
                        position_error_mm=np.linalg.norm(error[:3]) * 1000,
                        orientation_error_deg=np.linalg.norm(error[3:]) * 180 / np.pi,
                        fk_consistency_mm=np.linalg.norm(consistency[:3]) * 1000,
                        object_error_mm=np.linalg.norm(
                            actual_points - expected_points, axis=1
                        ).mean()
                        * 1000,
                        sigma_min=sigma,
                        condition=condition,
                        arm_limit_violation_rad=max(
                            0.0,
                            np.max(env.arm.lower - s["q_arm"]),
                            np.max(s["q_arm"] - env.arm.upper),
                        ),
                        arm_velocity_violation_rad_s=np.maximum(
                            np.abs(s["q_arm_velocity"]) - env.arm.velocity, 0
                        ).max(),
                        finger_velocity_violation_rad_s=np.maximum(
                            np.abs(s["full_q_velocity"]) - env.model.full_velocity, 0
                        ).max(),
                        coupling_error_rad=np.abs(
                            s["full_q"] - s["q"] @ env.model.coupling.T - env.model.offset
                        ).max(),
                    )
                )
                if not all(np.isfinite(v).all() for v in rows[-1].values()):
                    raise RuntimeError("Nonfinite physical rollout; no reset or masking applied")
            if env.render:
                env.world.render()
        if env.object_reset_count != writes:
            raise RuntimeError("Unexpected object reset during open-loop replay")
        arrays = {k: np.asarray([row[k] for row in rows]) for k in rows[0]}
        np.savez_compressed(output / f"run_{repeat + 1:02d}.npz", **arrays)
        motion = ~arrays["hold"]
        end = motion & arrays["control_endpoint"]
        summary = dict(
            repeat=repeat + 1,
            command_count=count,
            physics_samples=int(motion.sum()),
            policy_inferences=0,
            online_ik_calls=0,
            object_writes_during_motion=0,
            endpoint={
                k: stats(arrays[k][end])
                for k in ("position_error_mm", "orientation_error_deg", "object_error_mm")
            },
            all_physics_ticks={
                k: stats(arrays[k][motion]) for k in ("position_error_mm", "orientation_error_deg")
            },
            per_arm_joint_error_deg={
                n: stats(
                    np.rad2deg(np.abs(arrays["q_arm"][end, j] - arrays["target_q_arm"][end, j]))
                )
                for j, n in enumerate(env.arm.active_names)
            },
            per_finger_joint_error_deg={
                n: stats(
                    np.rad2deg(
                        np.abs(arrays["q_finger"][end, j] - arrays["target_q_finger"][end, j])
                    )
                )
                for j, n in enumerate(env.model.active_names)
            },
            maximum_fk_consistency_mm=float(arrays["fk_consistency_mm"].max()),
            min_sigma=float(arrays["sigma_min"].min()),
            max_condition=float(arrays["condition"].max()),
            near_singular_sample_indices=np.flatnonzero(
                (arrays["sigma_min"] < env.bridge.solver.options.near_sigma_min)
                | (arrays["condition"] > env.bridge.solver.options.near_condition)
            ).tolist(),
            maximum_arm_limit_violation_rad=float(arrays["arm_limit_violation_rad"].max()),
            maximum_arm_velocity_violation_rad_s=float(
                arrays["arm_velocity_violation_rad_s"].max()
            ),
            maximum_finger_velocity_violation_rad_s=float(
                arrays["finger_velocity_violation_rad_s"].max()
            ),
            maximum_coupling_error_rad=float(arrays["coupling_error_rad"].max()),
        )
        if hold_count:
            summary["hold_final"] = {
                k: float(arrays[k][-1])
                for k in ("position_error_mm", "orientation_error_deg", "object_error_mm")
            }
        summaries.append(summary)
        (output / f"run_{repeat + 1:02d}.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("[frozen tracking]", json.dumps(summary["endpoint"]), flush=True)
    report = dict(
        schema="frozen_tracking_report_v1",
        plan=str(plan_path.resolve()),
        plan_sha256=digest(plan_path),
        checkpoint_sha256=digest(checkpoint_path),
        policy_contract_hash=manifest["policy_contract_hash"],
        kind=manifest["kind"],
        time_scale=manifest["time_scale"],
        duration_s=float(reference.times[-1]),
        hold_s=hold_count * dt,
        velocity_feedforward=velocity_feedforward,
        physics=env.metadata,
        policy_inferences_during_replay=0,
        online_ik_during_replay=0,
        initialization="Set first saved joint state at reset only; start from rest, no motion-time state writes",
        timing="Targets held per 30Hz command; errors recorded each 120Hz physics tick and separately at control endpoints",
        interpretation="Physical simulated tracking of frozen commands; not learned closed-loop grasp or hardware validation",
        runs=summaries,
    )
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
