"""Named, rate-bounded 12-axis jogging; distinct from frozen grasp recordings."""

from pathlib import Path
import json
import time
from types import SimpleNamespace

import numpy as np

from ..execution import check_state


def plan_jog(request, model, measured, settings, dt, output, *, hardware=False):
    numeric = [
        dt,
        *[
            settings[k]
            for k in (
                "arm_velocity_deg_s",
                "finger_velocity_deg_s",
                "acceleration_deg_s2",
                "maximum_hardware_step_deg",
                "maximum_duration_s",
            )
        ],
    ]
    if not np.isfinite(numeric).all() or np.any(np.asarray(numeric) <= 0):
        raise ValueError("Jog timestep and limits must be positive and finite")
    trajectory = request.trajectory
    names = list(trajectory.joint_names)
    if len(names) != 12 or len(set(names)) != 12 or set(names) != set(model.names):
        raise ValueError("Jog requires all twelve named arm/finger joints")
    if (
        len(trajectory.points) != 2
        or trajectory.header.stamp.sec
        or trajectory.header.stamp.nanosec
        or request.multi_dof_trajectory.points
        or request.multi_dof_trajectory.joint_names
        or request.path_tolerance
        or request.goal_tolerance
        or request.component_path_tolerance
        or request.component_goal_tolerance
        or request.goal_time_tolerance.sec
        or request.goal_time_tolerance.nanosec
    ):
        raise ValueError("Jog requires two immediate position-only endpoints")
    order = [names.index(n) for n in model.names]
    positions = []
    times = []
    for point in trajectory.points:
        q = np.asarray(point.positions, float)
        if (
            q.shape != (12,)
            or not np.isfinite(q).all()
            or point.velocities
            or point.accelerations
            or point.effort
        ):
            raise ValueError("Jog positions must contain twelve finite radians")
        positions.append(q[order])
        times.append(point.time_from_start.sec + point.time_from_start.nanosec * 1e-9)
    initial, target = positions
    if (
        times[0] != 0
        or not np.isfinite(times).all()
        or times[1] <= 0
        or times[1] > settings["maximum_duration_s"]
    ):
        raise ValueError("Invalid jog duration")
    lower = np.r_[model.arm.lower, model.hand.lower]
    upper = np.r_[model.arm.upper, model.hand.upper]
    if np.any(np.array(positions) < lower - 1e-7) or np.any(np.array(positions) > upper + 1e-7):
        raise ValueError("Jog exceeds USD/coupled position limits")
    check_state(measured, initial, np.full(12, np.deg2rad(0.5)), 0.2, time.monotonic)
    delta = target - initial
    if hardware and np.abs(delta).max() > np.deg2rad(settings["maximum_hardware_step_deg"]) + 1e-9:
        raise ValueError("Hardware jog exceeds the configured per-action step limit")
    velocity = np.minimum(
        np.r_[model.arm.velocity, model.hand.velocity],
        np.deg2rad([settings["arm_velocity_deg_s"]] * 6 + [settings["finger_velocity_deg_s"]] * 6),
    )
    acceleration = np.full(12, np.deg2rad(settings["acceleration_deg_s2"]))
    if np.any(velocity <= 0) or np.any(acceleration <= 0):
        raise ValueError("Invalid jog rate limits")
    duration = max(
        times[1],
        float(np.max(1.875 * np.abs(delta) / velocity)),
        float(np.max(np.sqrt(5.774 * np.abs(delta) / acceleration))),
    )
    count = int(np.ceil(duration / dt)) + 1
    if (count - 1) * dt > settings["maximum_duration_s"]:
        raise ValueError("Target needs more than the allowed jog duration")
    u = np.linspace(0, 1, count)
    blend = 10 * u**3 - 15 * u**4 + 6 * u**5
    q = initial[None] + blend[:, None] * delta
    dq = np.gradient(q, dt, axis=0)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        kind="bounded_joint_jog",
        names=model.names,
        dt_s=dt,
        times_s=(np.arange(count) * dt).tolist(),
        q_rad=q.tolist(),
        initial_rad=initial.tolist(),
        joint_space_only=True,
        collision_free_certified=False,
        hardware=hardware,
    )
    with output.open("x") as f:
        json.dump(payload, f, indent=2)
    return SimpleNamespace(
        path=output,
        names=model.names,
        dt=dt,
        times=np.arange(count) * dt,
        q=q,
        initial=initial,
        duration=count * dt,
        data={
            "q_arm_velocity_rad_s": dq[:, :6],
            "lower_rad": lower,
            "upper_rad": upper,
            "velocity_limit_rad_s": velocity,
        },
        metadata=payload,
    )


def validate_hardware_jog(plan, config, model, root, arm_config):
    """Use measured motor mapping/physical limits; never borrow USD as calibration."""
    from ..hardware import connection_plan, vector
    from ..scene import Workcell
    from ..execution import sha256

    mapping = connection_plan(plan.names, config)
    calibration = config["calibration"]
    if (
        calibration["workcell_fingerprint"]
        != Workcell.load(root / arm_config["workcell"]).fingerprint
    ):
        raise ValueError("Commissioned workcell does not match the model")
    if calibration["mount_asset_sha256"] != sha256(root / arm_config["usd"]):
        raise ValueError("Commissioned mount does not match the assembly")
    arm, hand = zip(*(mapping.encode(q) for q in plan.q))
    arm, hand = np.array(arm), np.array(hand)
    lower = vector(config["rb3"]["lower_deg"], 6, "physical lower limits")
    upper = vector(config["rb3"]["upper_deg"], 6, "physical upper limits")
    velocity = vector(config["rb3"]["velocity_deg_s"], 6, "physical velocity")
    acceleration = vector(config["rb3"]["acceleration_deg_s2"], 6, "physical acceleration")
    hand_velocity = vector(config["revo2"]["velocity_units_s"], 6, "hand commissioned velocity")
    v = np.diff(np.vstack([arm[0], arm]), axis=0) / plan.dt
    a = np.diff(np.vstack([np.zeros(6), v, np.zeros(6)]), axis=0) / plan.dt
    if (
        np.any(lower >= upper)
        or np.any(velocity <= 0)
        or np.any(acceleration <= 0)
        or np.any(hand_velocity <= 0)
        or np.any(arm < lower)
        or np.any(arm > upper)
        or np.any(np.abs(arm) > 360)
        or np.any(np.abs(v) > velocity)
        or np.any(np.abs(a) > acceleration)
        or np.any(np.abs(np.diff(hand, axis=0) / plan.dt) > hand_velocity)
    ):
        raise ValueError("Jog exceeds commissioned physical joint/rate limits")
