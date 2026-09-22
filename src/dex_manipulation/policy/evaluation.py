"""Trajectory-level tracking metrics, independent of the simulator and RSI."""

import numpy as np
import torch
import json
from pathlib import Path
from .math3d import numpy_tree
from .contact_reward import SCHEMA, FINGERS


def full_horizon_metrics(
    keypoint_error,
    failure,
    height,
    target_height,
    *,
    mean_threshold_m=0.005,
    peak_threshold_m=0.025,
):
    """Failures remain in the denominator; short RSI episodes cannot pass."""
    error, failure, height, target = map(
        np.asarray, (keypoint_error, failure, height, target_height)
    )
    if (
        error.ndim != 2
        or not error.size
        or any(a.shape != error.shape for a in (failure, height, target))
    ):
        raise ValueError("Expected nonempty [time, environment] evaluation arrays")
    if not all(np.isfinite(a).all() for a in (error, height, target)) or (error < 0).any():
        raise ValueError("Nonfinite or negative evaluation errors")
    complete = ~failure.any(axis=0)
    means, peaks = error.mean(axis=0), error.max(axis=0)
    good = complete & (means <= mean_threshold_m) & (peaks <= peak_threshold_m)
    return dict(
        frames=error.shape[0],
        episodes=error.shape[1],
        includes_failed_frames=True,
        mean_error_mm=float(error.mean() * 1000),
        max_error_mm=float(error.max() * 1000),
        per_episode_mean_error_mm=(means * 1000).tolist(),
        per_episode_max_error_mm=(peaks * 1000).tolist(),
        completed_count=int(complete.sum()),
        within_mean_and_peak_tolerances_count=int(good.sum()),
        mean_threshold_mm=mean_threshold_m * 1000,
        peak_threshold_mm=peak_threshold_m * 1000,
        mean_final_height_error_mm=float(np.abs(height[-1] - target[-1]).mean() * 1000),
        mean_final_lift_mm=float((height[-1] - height[0]).mean() * 1000),
        failed_environment_indices=np.flatnonzero(~complete).tolist(),
        interpretation="Dynamic object tracking only; does not certify joint-speed limits, nonpenetration or real-world grasp robustness.",
    )


def evaluate(env, learner, output, label, seed, zero=False, protocol="strict"):
    # Evaluation must not advance curriculum, train the RSI sampler, or consume training RNG.
    saved = env.training_state_dict()
    threshold = env.cfg["failure_object_error_m"]
    if protocol not in ("source", "strict"):
        raise ValueError("Unknown evaluation protocol")
    try:
        env.evaluation_protocol = protocol
        if protocol == "source":
            env.cfg["failure_object_error_m"] = 1.0
        env.set_training(False)
        env.rng = np.random.default_rng(seed)
        env.generator.manual_seed(seed)
        return _evaluate_episode_batch(env, learner, output, label, seed, zero)
    finally:
        env.cfg["failure_object_error_m"] = threshold
        env.load_training_state_dict(saved)


def _evaluate_episode_batch(env, learner, output, label, seed, zero=False):
    """Full sequence from frame 15; never counts random mid-sequence resets as success."""
    env.reset(randomize=True)
    start_reset_count = env.object_reset_count
    obs = env.observation()
    n = env.num_envs
    alive, success = np.ones(n, bool), np.zeros(n, bool)
    demo_completed, tracking_only = np.zeros(n, bool), np.zeros(n, bool)
    initial_height = numpy_tree(env.state()["object_position"])[:, 2].copy()
    maximum_lift = np.zeros(n)
    rows, metric_rows, returns, lengths = [], [], np.zeros(n), np.zeros(n, int)
    max_error = np.zeros(n)
    constraints_ok = np.ones(n, bool)
    failure_reasons = [None] * n
    first_done = None
    for step in range(int(round(env.reference.duration / env.task.dt)) + 1):
        if zero:
            action = np.zeros((n, 12), np.float32)
        else:
            with torch.no_grad():
                action = learner.act(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action, auto_reset=False)
        if env.object_reset_count != start_reset_count:
            raise AssertionError("Object was reset during evaluation")
        info = {k: numpy_tree(v) for k, v in info.items() if k != "final_observation"}
        terminated, truncated = numpy_tree(terminated), numpy_tree(truncated)
        reward, action = numpy_tree(reward), numpy_tree(action)
        metrics = info["metrics"]
        metric_rows.append({k: v.copy() for k, v in metrics.items()} | {"alive": alive.copy()})
        returns += reward * alive
        lengths += alive
        max_error = np.maximum(max_error, metrics["object_keypoint_error_m"] * alive)
        maximum_lift = np.maximum(
            maximum_lift, (metrics["object_height_m"] - initial_height) * alive
        )
        step_constraints = (
            (metrics["joint_limit_violation_rad"] < 1e-3)
            & (metrics["joint_velocity_violation_rad_s"] < 1e-2)
            & (metrics["coupling_error_rad"] < 1e-2)
        )
        if "table_clearance_m" in metrics:
            step_constraints &= metrics["table_clearance_m"] >= 0.0
        if "arm_ik_success" in metrics:
            options = env.ik.options if hasattr(env, "ik") else env.bridge.solver.options
            step_constraints &= (
                (metrics["arm_ik_success"] > 0.5)
                & (metrics["arm_joint_limit_violation_rad"] < 1e-3)
                & (metrics["arm_joint_velocity_violation_rad_s"] < 1e-2)
                & (metrics["arm_actual_sigma_min"] >= options.singular_sigma_min)
                & (metrics["arm_actual_condition"] <= options.max_condition)
            )
        constraints_ok &= ~alive | step_constraints
        done = alive & (terminated | truncated)
        for i in np.flatnonzero(done):
            demo_completed[i] = bool(not metrics["early_failure"][i] and metrics["demo_end"][i])
            tracking_only[i] = bool(
                demo_completed[i]
                and max_error[i] < env.cfg["success_object_error_m"]
                and abs(metrics["object_height_m"][i] - metrics["target_object_height_m"][i])
                < env.cfg["success_final_height_error_m"]
            )
            success[i] = tracking_only[i] and constraints_ok[i]
            failure_reasons[i] = (
                "tracking_threshold_or_coupling_or_drop"
                if metrics["early_failure"][i]
                else (None if success[i] else "completed_but_tracking_tolerance_failed")
            )
        # Save the entire horizon, including the aftermath of a failure.
        state, ref = info["state"], info["reference"]
        rows.append(
            dict(
                time_s=info["reference_time"][0],
                action=action[0].copy(),
                physics_time_s=(step + 1) * env.task.dt,
                terminated=bool(terminated[0]),
                truncated=bool(truncated[0]),
                **{k: v[0].copy() for k, v in state.items()},
                **{"raw_target_" + k: v[0].copy() for k, v in info["raw_targets"].items()},
                **{"applied_target_" + k: v[0].copy() for k, v in info["applied_targets"].items()},
                **{"reference_" + k: v[0].copy() for k, v in ref.items() if k != "time"},
            )
        )
        if done[0] and first_done is None:
            first_done = dict(terminated=bool(terminated[0]), truncated=bool(truncated[0]))
        alive[done] = False
    output = Path(output)
    arrays = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
    arrays["body_names"] = np.array(env.robot.body_names)
    arrays["active_joint_names"] = np.array(env.model.active_names)
    arrays["full_joint_names"] = np.array(env.model.full_names)
    arrays["camera_to_world"] = env.reference.camera_to_world
    for key in info["metrics"]:
        arrays["metric_" + key] = np.array([r[key][0] for r in metric_rows[: len(rows)]])
    arrays["joint_constraints_valid"] = (
        (arrays["metric_joint_limit_violation_rad"] < 1e-3)
        & (arrays["metric_joint_velocity_violation_rad_s"] < 1e-2)
        & (arrays["metric_coupling_error_rad"] < 1e-2)
    )
    arrays["episode_alive"] = np.logical_not(np.maximum.accumulate(arrays["metric_early_failure"]))
    arrays["valid"] = arrays["joint_constraints_valid"] & arrays["episode_alive"]
    if "metric_table_clearance_m" in arrays:
        arrays["table_clearance_valid"] = arrays["metric_table_clearance_m"] >= 0.0
        arrays["valid"] &= arrays["table_clearance_valid"]
    if "metric_arm_ik_success" in arrays:
        arrays["arm_joint_names"] = np.asarray(env.arm.active_names)
        arrays["arm_constraints_valid"] = (
            (arrays["metric_arm_ik_success"] > 0.5)
            & (arrays["metric_arm_joint_limit_violation_rad"] < 1e-3)
            & (arrays["metric_arm_joint_velocity_violation_rad_s"] < 1e-2)
            & (arrays["metric_arm_actual_sigma_min"] >= options.singular_sigma_min)
            & (arrays["metric_arm_actual_condition"] <= options.max_condition)
        )
        arrays["valid"] &= arrays["arm_constraints_valid"]
    arrays["tracking_within_tolerance"] = (
        arrays["metric_object_keypoint_error_m"] < env.cfg["success_object_error_m"]
    )
    # Exact link-local semantics from measured rigid bodies, without imposing ideal coupling again.
    from scipy.spatial.transform import Rotation

    kp = []
    for point in env.model.keypoints:
        index = env.robot.body_names.index(point["link"])
        poses = arrays["link_transforms"][:, index]
        kp.append(Rotation.from_quat(poses[:, 3:7]).apply(point["xyz"]) + poses[:, :3])
    arrays["robot_keypoints"] = np.stack(kp, axis=1)
    arrays["object_keypoints"] = env.reference.points(
        arrays["object_position"], arrays["object_quaternion"]
    )
    arrays["semantic_names"] = np.array(env.model.semantic_names)
    np.savez_compressed(output / f"{label}_rollout.npz", **arrays)
    mask = np.stack([r["alive"] for r in metric_rows])
    report = dict(
        label=label,
        seed=seed,
        episodes=n,
        success_count=int(success.sum()),
        demo_completed_count=int(demo_completed.sum()),
        object_tracking_only_count=int(tracking_only.sum()),
        object_tracking_only_definition="full sequence and object tracking/height tolerances; excludes joint constraints, not a grasp certificate",
        maximum_lift_m=maximum_lift.tolist(),
        mean_maximum_lift_m=float(maximum_lift.mean()),
        gravity_m_s2=env.gravity.value,
        rsi_enabled=False,
        protocol=env.evaluation_protocol,
        motion_control=env.motion_controller.config,
        table_safety=env.table_safety.config
        if getattr(env, "table_safety", None) is not None
        else None,
        motion_control_differs_from_training=env.motion_controller.config
        != env.cfg.get("motion_control"),
        augmentation_enabled=env.evaluation_protocol == "source"
        and env.cfg["augmentation"]["enabled"],
        observation_noise_and_delay_enabled=env.evaluation_protocol == "source",
        failure_object_error_m=env.cfg["failure_object_error_m"],
        success_definition="full sequence, no early failure, every-step object 50-keypoint mean error <25mm, final height error <25mm, joint/coupling tolerances and nonnegative recorded table clearance when enabled",
        joint_constraints_passed=constraints_ok.tolist(),
        returns=returns.tolist(),
        episode_steps=lengths.tolist(),
        failure_reasons=failure_reasons,
        maximum_object_error_m=max_error.tolist(),
        first_rollout_end=first_done,
        object_writes_during_rollout=env.object_reset_count - start_reset_count,
        interpretation="dynamic simulated tracking; not a real-world grasp certification",
        valid_mask_definition="finite simulated step, no early termination, joint/coupling numerical tolerances and nonnegative table clearance when enabled; not a general nonpenetration or grasp certificate",
    )
    if "table_clearance_m" in metric_rows[0]:
        report["table_clearance"] = dict(
            minimum_m=float(min(r["table_clearance_m"].min() for r in metric_rows)),
            maximum_target_lift_m=float(max(r["table_target_lift_m"].max() for r in metric_rows)),
            violating_step_count=int(sum((r["table_clearance_m"] < 0).sum() for r in metric_rows)),
            geometry="conservative collision-mesh enclosing boxes against tabletop half-space",
        )
    if "arm_ik_success" in metric_rows[0]:
        report["arm"] = dict(
            ik_success_fraction=float(np.stack([r["arm_ik_success"] for r in metric_rows]).mean()),
            ik_failure_step_indices_first_environment=np.flatnonzero(
                ~arrays["metric_arm_ik_success"].astype(bool)
            ).tolist(),
            constraints_passed_first_environment=bool(arrays["arm_constraints_valid"].all()),
            minimum_actual_sigma=float(min(r["arm_actual_sigma_min"].min() for r in metric_rows)),
            maximum_actual_condition=float(
                max(r["arm_actual_condition"].max() for r in metric_rows)
            ),
        )
        for name in (
            "arm_ik_position_error_m",
            "arm_ik_orientation_error_rad",
            "arm_tracking_position_error_m",
            "arm_tracking_orientation_error_rad",
            "arm_fk_consistency_position_error_m",
        ):
            values = np.stack([r[name] for r in metric_rows])
            report["arm"][name] = dict(mean=float(values.mean()), maximum=float(values.max()))
        report["success_definition"] += (
            "; arm IK, arm velocity/limits and actual Jacobian singularity checks at every step"
        )
        report["valid_mask_definition"] += (
            "; also includes arm IK, velocity/limits and singularity checks"
        )
    for key in (
        "object_keypoint_error_m",
        "wrist_error_m",
        "joint_rmse_rad",
        "coupling_error_rad",
        "joint_limit_violation_rad",
        "joint_velocity_violation_rad_s",
    ):
        values = np.stack([r[key] for r in metric_rows])[mask]
        report[key] = dict(mean=float(values.mean()), maximum=float(values.max()))
    report["full_horizon"] = full_horizon_metrics(
        np.stack([r["object_keypoint_error_m"] for r in metric_rows]),
        np.stack([r["early_failure"] for r in metric_rows]),
        np.stack([r["object_height_m"] for r in metric_rows]),
        np.stack([r["target_object_height_m"] for r in metric_rows]),
    )
    report["legacy_error_metrics"] = (
        "Metrics outside full_horizon stop at first failure; never use them alone to claim tracking performance."
    )
    (output / f"{label}_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("[evaluation]", json.dumps(report), flush=True)
    return report


def plot_comparison(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for label, color in (("zero_residual", "#dd8050"), ("trained_residual", "#3d92d1")):
        path = Path(output) / f"{label}_rollout.npz"
        if not path.exists():
            continue
        with np.load(path) as d:
            t = d["time_s"]
            axes[0].plot(t, d["object_position"][:, 2], label=label, color=color)
            axes[0].plot(
                t,
                d["reference_object_position"][:, 2],
                "k--",
                alpha=0.6,
                label="reference" if label == "zero_residual" else None,
            )
            axes[1].plot(
                t,
                np.linalg.norm(d["object_position"] - d["reference_object_position"], axis=-1)
                * 1000,
                color=color,
            )
            axes[2].plot(
                t,
                np.linalg.norm(d["wrist_position"] - d["reference_wrist_position"], axis=-1) * 1000,
                color=color,
            )
    axes[0].set_ylabel("Can height (m)")
    axes[1].set_ylabel("Can position error (mm)")
    axes[2].set_ylabel("Wrist error (mm)")
    axes[2].set_xlabel("Retimed sequence time (s)")
    axes[0].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("Dynamic PhysX evaluation: full horizon, including failed steps")
    fig.tight_layout()
    fig.savefig(Path(output) / "evaluation.png", dpi=150)
    plt.close(fig)


def evaluate_contacts(env, learner, output):
    """Full-gravity, frame-15 starts; no RSI, no reset within an episode.

    Failed/unobserved suffixes count as missing contacts. Geometric tracking
    alone is never relabeled five-finger success.
    """
    import numpy as np

    env.evaluation_protocol = "strict"
    env.set_training(False)
    env.reset(randomize=False)
    obs = env.observation()
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    completed = torch.zeros_like(alive)
    all_five = torch.zeros(env.num_envs, device=env.device)
    per_finger = torch.zeros(env.num_envs, 5, device=env.device)
    max_error = torch.zeros_like(all_five)
    max_force = torch.zeros_like(all_five)
    table_ok = torch.ones_like(alive)
    joints_ok = torch.ones_like(alive)
    window_steps = 0
    rows = []
    with torch.no_grad():
        for step in range(round(env.reference.duration / env.task.dt) + 1):
            obs, _, term, trunc, info = env.step(learner.act(obs), auto_reset=False)
            m = info["metrics"]
            on = m["grasp_active"] > 0.5
            window_steps += int(on[0])
            mask = (on & alive).float()
            all_five += mask * m["grasp_all_five_duty"]
            duty = torch.stack([m[f"grasp_{name}_duty"] for name in FINGERS], dim=-1)
            per_finger += mask[:, None] * duty
            max_error = torch.maximum(max_error, m["object_keypoint_error_m"])
            max_force = torch.maximum(max_force, m["grasp_max_force_n"])
            table_ok &= m.get("table_clearance_m", torch.zeros_like(max_error)) >= 0
            joints_ok &= (
                (m["joint_limit_violation_rad"] < 1e-3)
                & (m["coupling_error_rad"] < 1e-2)
                & (m["joint_velocity_violation_rad_s"] < 1e-2)
            )
            done = term | trunc
            completed |= alive & done & m["demo_end"].bool() & ~m["early_failure"].bool()
            rows.append(
                {
                    "time_s": info["reference_time"].cpu().numpy(),
                    "alive": alive.cpu().numpy(),
                    **{name: value.cpu().numpy() for name, value in m.items()},
                }
            )
            alive &= ~done
    if not window_steps:
        raise RuntimeError("Evaluation did not cover the requested grasp interval")
    fraction = all_five / window_steps
    success = (
        completed
        & table_ok
        & joints_ok
        & (max_error <= env.cfg["success_object_error_m"])
        & (fraction >= 0.95)
    )
    report = dict(
        schema=SCHEMA,
        gravity_m_s2=env.gravity.value,
        rsi=False,
        start_time_s=env.task.schedule["start_time_s"],
        window_control_steps=window_steps,
        full_sequence_completed=completed.cpu().tolist(),
        all_five_contact_fraction=fraction.cpu().tolist(),
        per_finger_contact_fraction=(per_finger / window_steps).cpu().tolist(),
        finger_order=list(FINGERS),
        maximum_object_error_mm=(max_error * 1000).cpu().tolist(),
        maximum_pad_force_n=max_force.cpu().tolist(),
        table_clearance_passed=table_ok.cpu().tolist(),
        joint_constraints_passed=joints_ok.cpu().tolist(),
        five_finger_success=success.cpu().tolist(),
        success_count=int(success.sum()),
        definition="Full sequence, object mean-keypoint error <= configured success tolerance at every step, "
        "joint/coupling constraints, nonnegative tabletop clearance, and >=95% simultaneous five-pad/can contact duty in the whole grasp window. "
        "Failed suffixes count as noncontact; simulation diagnostic, not physical grasp certification.",
    )
    output = Path(output)
    (output / "five_finger_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        output / "five_finger_evaluation.npz",
        **{name: np.stack([row[name] for row in rows]) for name in rows[0]},
    )
    print("[five-finger evaluation]", json.dumps(report), flush=True)
    return report
