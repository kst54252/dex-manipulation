"""Recorded arm/hand commands, guarded streaming and bounded virtual jogging."""

import asyncio
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from dex_manipulation.data import resolve_demo_path
from types import SimpleNamespace


SCHEMA = "dex_recorded_commands_v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RecordedCommands:
    def __init__(self, path):
        self.path = resolve_demo_path(path).resolve()
        with np.load(path, allow_pickle=False) as data:
            self.data = {k: data[k].copy() for k in data.files}
        d = self.data
        self.metadata = json.loads(str(d["metadata_json"]))
        m = self.metadata
        if (
            m["schema"] != SCHEMA
            or m["angle_unit"] != "rad"
            or m["command_semantics"] != "start_of_interval_zero_order_hold"
        ):
            raise ValueError("Unsupported command trajectory or units")
        self.names = tuple(map(str, d["joint_names"]))
        self.dt = float(m["control_dt_s"])
        self.times = d["timestamp_s"]
        self.q = d["q_command_rad"]
        self.initial = d["initial_q_rad"]
        self.duration = float(m["execution_duration_s"])
        n = len(self.times)
        if n < 2 or len(self.names) != 12 or len(set(self.names)) != 12:
            raise ValueError("Expected six named arm and six named finger joints")
        for key, shape in [
            ("q_command_rad", (n, 12)),
            ("q_arm_velocity_rad_s", (n, 6)),
            ("initial_q_rad", (12,)),
            ("lower_rad", (12,)),
            ("upper_rad", (12,)),
            ("velocity_limit_rad_s", (12,)),
        ]:
            if d[key].shape != shape or not np.isfinite(d[key]).all():
                raise ValueError(f"Invalid {key}")
        if (
            not np.isfinite([self.dt, self.duration]).all()
            or self.dt <= 0
            or not np.allclose(self.times, np.arange(n) * self.dt, atol=1e-8, rtol=0)
            or not np.isclose(self.duration, n * self.dt, atol=1e-8, rtol=0)
        ):
            raise ValueError("Commands must cover every control interval without retiming")
        if np.any(d["velocity_limit_rad_s"] <= 0) or np.any(d["lower_rad"] >= d["upper_rad"]):
            raise ValueError("Invalid model limits")
        all_q = np.vstack([self.initial, self.q])
        if np.any(all_q < d["lower_rad"] - 1e-6) or np.any(all_q > d["upper_rad"] + 1e-6):
            raise ValueError("Recorded command exceeds model position limits")
        self.segment_velocity = np.diff(all_q, axis=0) / self.dt
        if np.any(np.abs(self.segment_velocity) > d["velocity_limit_rad_s"] + 1e-5):
            raise ValueError("Recorded command exceeds model interval velocity limits")
        if np.any(np.abs(d["q_arm_velocity_rad_s"]) > d["velocity_limit_rad_s"][:6] + 1e-5):
            raise ValueError("Arm feedforward exceeds model velocity limits")
        if (
            d["ik_success"].shape != (n,)
            or d["ik_success"].dtype.kind != "b"
            or not d["ik_success"].all()
        ):
            raise ValueError("Recorded trajectory contains an IK failure")

    def export_csv(self, path):
        header = [
            "timestamp_s",
            *[n + "_rad" for n in self.names],
            *[n + "_velocity_rad_s" for n in self.names[:6]],
        ]
        np.savetxt(
            path,
            np.c_[self.times, self.q, self.data["q_arm_velocity_rad_s"]],
            delimiter=",",
            header=",".join(header),
            comments="",
        )

    def summary(self):
        return dict(
            command_count=len(self.times),
            reference_duration_s=self.metadata["reference_duration_s"],
            execution_duration_s=self.duration,
            control_hz=1 / self.dt,
            maximum_interval_velocity_rad_s=np.abs(self.segment_velocity).max(0).tolist(),
            initial_q_rad=self.initial.tolist(),
            joint_names=list(self.names),
            calibration_reference=dict(
                workcell_fingerprint=self.metadata["workcell"]["fingerprint"],
                mount_asset_sha256=self.metadata.get("arm_asset_sha256"),
                world_from_base=self.metadata["workcell"]["world_from_base"],
                can_initial_world_pose=self.metadata["initial_can_world_pose"],
            ),
            interpretation="Recorded joint commands; no object pose estimator or online policy/IK needed",
        )


def check_state(state, expected, tolerance, maximum_age, clock):
    q = np.asarray(state["q_rad"], float)
    if q.shape != (12,) or not np.isfinite(q).all() or not state["ready"]:
        raise RuntimeError("Robot feedback is invalid or robot is not ready")
    age = clock() - float(state["sample_time_s"])
    if not 0 <= age <= maximum_age:
        raise RuntimeError("Robot feedback is stale")
    # Do not wrap angles: a different 360-degree branch is a different start.
    error = np.abs(q - expected)
    if np.any(error > tolerance):
        raise RuntimeError(f"Joint tracking/start mismatch: {np.rad2deg(error).tolist()} degrees")
    return q


async def stream(
    trajectory,
    backend,
    output,
    *,
    start_tolerance_rad,
    tracking_tolerance_rad,
    maximum_feedback_age_s,
    maximum_lateness_s,
    hold_s=1.0,
    telemetry=None,
    clock=time.monotonic,
    sleep=asyncio.sleep,
):
    """Send once, use feedback only for tracking/stop, never adjust the object.

    The backend owns unit conversion, I/O timeouts and a finite controller-side
    stream timeout. Host Python scheduling is not a hard real-time guarantee.
    No automatic move to the starting pose, retries, repeats or clock catch-up.
    """
    limits = [np.asarray(x, float) for x in (start_tolerance_rad, tracking_tolerance_rad)]
    if any(x.shape != (12,) or not np.isfinite(x).all() or np.any(x <= 0) for x in limits):
        raise ValueError("Twelve explicit positive feedback tolerances required")
    if (
        not np.isfinite([maximum_feedback_age_s, maximum_lateness_s, hold_s]).all()
        or not 0 < maximum_lateness_s < trajectory.dt
        or maximum_feedback_age_s <= 0
        or hold_s < 0
    ):
        raise ValueError("Invalid timing/feedback guard")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    events = []
    report = dict(
        completed=False,
        source_sha256=sha256(trajectory.path),
        hardware=bool(getattr(backend, "hardware", False)),
        speed=1.0,
        object_feedback=False,
        feedback_source=getattr(backend, "feedback_source", None),
    )
    try:
        await backend.connect()
        if telemetry is not None:
            await telemetry.prepare(backend)
        state = await backend.read()
        check_state(state, trajectory.initial, limits[0], maximum_feedback_age_s, clock)
        start = clock()
        if telemetry is not None:
            telemetry.start(start)
            telemetry.feedback(state, trajectory.initial, -1)
        report["start_monotonic_s"] = start
        hold_count = int(np.ceil(hold_s / trajectory.dt))
        previous = trajectory.initial
        with (output / "commands.jsonl").open("w") as log:
            for i in range(len(trajectory.times) + hold_count):
                deadline = start + i * trajectory.dt
                await sleep(max(0.0, deadline - clock()))
                late = clock() - deadline
                if late > maximum_lateness_s:
                    raise TimeoutError("Command deadline missed; refusing catch-up jump")
                if i:
                    state = await backend.read()
                    if telemetry is not None:
                        telemetry.feedback(state, previous, i - 1)
                    check_state(state, previous, limits[1], maximum_feedback_age_s, clock)
                    if clock() - deadline > maximum_lateness_s:
                        raise TimeoutError("Feedback exceeded command deadline")
                index = min(i, len(trajectory.times) - 1)
                velocity = (
                    trajectory.data["q_arm_velocity_rad_s"][index]
                    if i < len(trajectory.times)
                    else np.zeros(6)
                )
                command = trajectory.q[index]
                report["attempted_commands"] = i + 1
                sending = clock() - start
                await backend.send(command, velocity, trajectory.dt)
                event = dict(
                    index=i,
                    scheduled_s=i * trajectory.dt,
                    sent_s=clock() - start,
                    send_started_s=sending,
                    feedback_sample_s=float(state["sample_time_s"]) - start,
                    feedback_read_window_s=float(state.get("read_window_s", 0.0)),
                    hold=i >= len(trajectory.times),
                    q_command_rad=command.tolist(),
                    q_measured_rad=np.asarray(state["q_rad"]).tolist(),
                    feedback_source=report["feedback_source"],
                )
                log.write(json.dumps(event) + "\n")
                log.flush()
                events.append(event)
                previous = command
                if clock() - deadline >= trajectory.dt:
                    raise TimeoutError("Command I/O exceeded its interval")
                if telemetry is not None:
                    await telemetry.poll_until(backend, deadline + trajectory.dt, clock, sleep)
            await sleep(
                max(0.0, start + (len(trajectory.times) + hold_count) * trajectory.dt - clock())
            )
            state = await backend.read()
            if telemetry is not None:
                telemetry.feedback(state, previous, len(trajectory.times) + hold_count - 1)
            check_state(state, previous, limits[1], maximum_feedback_age_s, clock)
        report.update(completed=True, command_count=len(trajectory.times), hold_count=hold_count)
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        # A stopped arm must not be paired with an automatic hand-open command.
        cleanup_errors = []
        for operation in (backend.stop, backend.close):
            try:
                await operation()
            except BaseException as error:
                cleanup_errors.append(f"{operation.__name__}: {type(error).__name__}: {error}")
        if cleanup_errors:
            report.update(completed=False, cleanup_errors=cleanup_errors)
        report["sent_commands"] = len(events)
        if telemetry is not None:
            try:
                telemetry.finish(report)
            except BaseException as error:
                cleanup_errors.append(f"measurement: {type(error).__name__}: {error}")
                report.update(completed=False, cleanup_errors=cleanup_errors)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if cleanup_errors and "error" not in report:
            raise RuntimeError("; ".join(cleanup_errors))
    return report


class MockBackend:
    """Named-joint transport exercise. It models perfect tracking, not physics."""

    hardware = False
    feedback_source = dict(arm="mock_command_hold", hand="mock_command_hold")

    def __init__(self, initial, clock=time.monotonic):
        self.q = np.asarray(initial).copy()
        self.clock = clock

    async def connect(self):
        pass

    async def read(self):
        return dict(q_rad=self.q.copy(), sample_time_s=self.clock(), ready=True)

    async def send(self, q, arm_velocity, dt):
        self.q = np.asarray(q).copy()

    async def stop(self):
        pass

    async def close(self):
        pass


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
    from dex_manipulation.robot.hardware import connection_plan, vector
    from dex_manipulation.scene import Workcell

    pass

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


class VirtualJointBackend:
    hardware = False
    feedback_source = dict(arm="virtual_joint_servo", hand="virtual_joint_servo")

    def __init__(self, initial, velocity, time_constant=0.025, clock=time.monotonic):
        self.q = np.asarray(initial, float).copy()
        self.target = self.q.copy()
        self.velocity = np.asarray(velocity, float)
        self.tau = float(time_constant)
        self.clock = clock
        if (
            self.q.shape != (12,)
            or self.velocity.shape != (12,)
            or not np.isfinite(np.r_[self.q, self.velocity, self.tau]).all()
            or self.tau <= 0
            or np.any(self.velocity <= 0)
        ):
            raise ValueError("Invalid virtual joint-servo configuration")
        self.last = clock()
        self.sent = False

    def advance(self):
        now = self.clock()
        dt = max(0.0, now - self.last)
        self.last = now
        step = (self.target - self.q) * (1 - np.exp(-dt / self.tau))
        self.q += np.clip(step, -self.velocity * dt, self.velocity * dt)

    async def connect(self):
        self.last = self.clock()

    async def read(self):
        self.advance()
        return dict(
            q_rad=self.q.copy(),
            sample_time_s=self.clock(),
            ready=True,
            feedback_source=self.feedback_source,
            physical_simulation=False,
        )

    async def send(self, q, arm_velocity, dt):
        self.advance()
        q = np.asarray(q, float)
        if q.shape != (12,) or not np.isfinite(q).all():
            raise ValueError("Invalid virtual command")
        self.target = q.copy()
        self.sent = True

    async def stop(self):
        self.advance()
        self.target = self.q.copy()
        self.sent = False

    async def close(self):
        pass
