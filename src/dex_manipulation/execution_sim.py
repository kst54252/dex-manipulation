"""Isaac adapter for recording and replaying a single physical arm grasp."""

import json
from pathlib import Path

import numpy as np

from .execution import RecordedCommands, SCHEMA, sha256
from .ik import pose_error


def run(env, metadata, checkpoint, output, *, replay_path=None, hold_s=1.0, record_tactile=False):
    completed = False
    try:
        result = _run(env, metadata, checkpoint, output, replay_path=replay_path,
                      hold_s=hold_s, record_tactile=record_tactile)
        completed = True
        return result
    finally:
        recorder = getattr(env, "tactile_recorder", None)
        if record_tactile and recorder is not None and recorder.active:
            recorder.finish({"end": "completed" if completed else "aborted"})


def _run(env, metadata, checkpoint, output, *, replay_path=None, hold_s=1.0, record_tactile=False):
    import torch
    from .policy.ppo import PPO

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
        from .sensors.physx_tactile import attach_tactile_recording

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
        hardware_validated=False,
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
        recorder.metadata.update(source_commands_sha256=report["commands_sha256"],
                                 time_zero="first recorded command", command_hz=1 / env.task.dt,
                                 hold_s=hold_count * env.task.dt)
        (recorder.output / "metadata.json").write_text(json.dumps(recorder.metadata, indent=2) + "\n")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("[recorded grasp]", json.dumps(report), flush=True)
    if not grasp:
        raise RuntimeError(
            "Saved diagnostics, but this run did not pass the simulated lift-and-hold criterion"
        )
    return report
