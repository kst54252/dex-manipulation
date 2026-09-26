"""Recorded 12-axis commands and simulator-independent, bounded execution.

The saved commands use start-of-interval timestamps and zero-order hold. Actual
measured positions are separate: a policy PD target is not an encoder sample.
"""

import asyncio
import hashlib
import json
from pathlib import Path
import time

import numpy as np

SCHEMA = "dex_recorded_commands_v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RecordedCommands:
    def __init__(self, path):
        self.path = Path(path).resolve()
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
                    feedback_read_window_s=float(state.get("read_window_s", 0.)),
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
