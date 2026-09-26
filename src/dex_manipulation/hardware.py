"""Explicitly calibrated RBpodo + Revo2 RS485 adapters; no Isaac dependency.

Official SDK APIs only. Importing/planning does not open ports. Commissioning
data are mandatory; USD limits are not a motor-unit calibration.
"""

from .configuration import read_config
import asyncio
import importlib.metadata
import inspect
import json
from pathlib import Path
import time

import numpy as np

from .execution import sha256, stream

VERSIONS = {"rbpodo": "0.16.14", "bc-stark-sdk": "2.0.3"}
ARM_NAMES = ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3")


def vector(value, count, name):
    value = np.asarray(value, float)
    if value.shape != (count,) or not np.isfinite(value).all():
        raise ValueError(f"{name}: {count} finite calibrated values required")
    return value


class ArmCalibration:
    def __init__(self, names, config):
        if tuple(names[:6]) != ARM_NAMES:
            raise ValueError("RBpodo joint order differs from the recorded model")
        self.sign = vector(config["arm_sign"], 6, "arm_sign")
        self.offset = vector(config["arm_offset_deg"], 6, "arm_offset_deg")
        if not np.isin(self.sign, [-1, 1]).all():
            raise ValueError("arm_sign must contain measured +1/-1 axis directions")

    def encode_arm(self, q):
        return np.rad2deg(vector(q, 6, "arm target")) * self.sign + self.offset

    def decode_arm(self, degrees):
        return np.deg2rad((vector(degrees, 6, "RB3 feedback") - self.offset) / self.sign)


def rb_status(s):
    """Decode current faults, excluding historical collision/timezone bits."""
    masks = {
        "init_error": 0xFFF,
        "op_stat_collision_occur": 0x3,
        "op_stat_ems_flag": 0x3F,
        "op_stat_self_collision": 0x3,
        "op_stat_sos_flag": 0x3F,
        "is_freedrive_mode": 0x3,
    }
    faults = {key: int(getattr(s, key)) & mask for key, mask in masks.items()}
    faults["op_stat_soft_estop_occur"] = int(s.op_stat_soft_estop_occur)
    return dict(
        mode=int(s.real_vs_simulation_mode) & 0xF,
        init_stage=int(s.init_state_info) & 0x3F,
        robot_state=int(s.robot_state),
        task_state=int(s.task_state),
        faults=faults,
        raw={key: int(getattr(s, key)) for key in faults},
    )


class JointCalibration(ArmCalibration):
    def __init__(self, names, config):
        super().__init__(names, config)
        rows = config["fingers"]
        if len(rows) != 6 or {r["joint"] for r in rows} != set(names[6:]):
            raise ValueError("Six explicit Revo2 model-to-SDK joint mappings required")
        self.fingers = []
        for name in names[6:]:
            row = next(r for r in rows if r["joint"] == name)
            index = row["sdk_index"]
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 6:
                raise ValueError("Revo2 sdk_index must be explicitly calibrated")
            q, u = np.asarray(row["rad"], float), np.asarray(row["units"], float)
            if (
                q.ndim != 1
                or len(q) < 2
                or u.shape != q.shape
                or not np.isfinite(q).all()
                or not np.isfinite(u).all()
                or not (np.diff(q) > 0).all()
                or not ((np.diff(u) > 0).all() or (np.diff(u) < 0).all())
                or u.min() < 0
                or u.max() > 1000
            ):
                raise ValueError(
                    "Revo2 requires measured monotonic rad <-> 0..1000 calibration knots"
                )
            self.fingers.append((index, q, u))
        if len({i for i, _, _ in self.fingers}) != 6:
            raise ValueError("Duplicate Revo2 motor index")

    def encode(self, q):
        q = vector(q, 12, "joint targets")
        arm = self.encode_arm(q[:6])
        fingers = np.empty(6, int)
        for value, (index, knots, units) in zip(q[6:], self.fingers):
            if not knots[0] - 1e-7 <= value <= knots[-1] + 1e-7:
                raise ValueError(
                    "Finger target outside calibrated range; no clipping/extrapolation"
                )
            fingers[index] = int(round(np.interp(value, knots, units)))
        return arm, fingers

    def decode(self, arm, fingers):
        arm, fingers = vector(arm, 6, "RB3 feedback"), vector(fingers, 6, "Revo2 feedback")
        q = list(self.decode_arm(arm))
        for index, knots, units in self.fingers:
            order = np.argsort(units)
            if not units.min() - 1 <= fingers[index] <= units.max() + 1:
                raise ValueError("Finger feedback outside calibrated range")
            q.append(float(np.interp(fingers[index], units[order], knots[order])))
        return np.asarray(q)


def connection_plan(names, config):
    """Validate read-only connection and encoder mapping without motion settings."""
    if config.get("schema") != "dex_rb3_revo2_hardware_v1":
        raise ValueError("Unsupported hardware configuration")
    calibration = config["calibration"]
    if not calibration.get("id"):
        raise ValueError(
            "Supply measured arm/hand/mount/table calibration; example config is not commissioned"
        )
    mapping = JointCalibration(names, calibration)
    transport_plan(config)
    return mapping


def transport_plan(config):
    """Validate only explicitly supplied connection settings for a read-only probe."""
    if config.get("schema") != "dex_rb3_revo2_hardware_v1":
        raise ValueError("Unsupported hardware configuration")
    rb, revo = config["rb3"], config["revo2"]
    if not rb["address"] or not revo["port"] or revo["transport"] != "rs485":
        raise ValueError("Provide explicit RB3 IP and Revo2 RS485 port")
    if not isinstance(revo["baudrate_enum"], str) or not revo["baudrate_enum"].startswith("Baud"):
        raise ValueError("Provide the official SDK baudrate enum for the connected hand")
    if not isinstance(revo["slave_id"], int) or not 1 <= revo["slave_id"] <= 247:
        raise ValueError("Provide the actual Revo2 slave ID")
    if not np.isfinite(config["io_timeout_s"]) or config["io_timeout_s"] <= 0:
        raise ValueError("I/O timeout must be positive and finite")
    for key in ("command_port", "data_port"):
        if not isinstance(rb[key], int) or not 1 <= rb[key] <= 65535:
            raise ValueError("Invalid RB3 port")


def hardware_plan(trajectory, config):
    """No connection; reject missing commissioning information before any I/O."""
    mapping = connection_plan(trajectory.names, config)
    calibration = config["calibration"]
    if calibration["workcell_fingerprint"] != trajectory.metadata["workcell"]["fingerprint"]:
        raise ValueError("Real tabletop/base placement has not been matched to this recording")
    if calibration["mount_asset_sha256"] != trajectory.metadata["arm_asset_sha256"]:
        raise ValueError("Physical straight mount is not confirmed against the recorded assembly")
    arm, hand = zip(*(mapping.encode(q) for q in np.vstack([trajectory.initial, trajectory.q])))
    arm, hand = np.asarray(arm), np.asarray(hand)
    lower = vector(config["rb3"]["lower_deg"], 6, "RB3 physical lower limits")
    upper = vector(config["rb3"]["upper_deg"], 6, "RB3 physical upper limits")
    vmax = vector(config["rb3"]["velocity_deg_s"], 6, "RB3 physical velocity limits")
    amax = vector(config["rb3"]["acceleration_deg_s2"], 6, "RB3 physical acceleration limits")
    if np.any(lower >= upper) or np.any(vmax <= 0) or np.any(amax <= 0):
        raise ValueError("Invalid physical arm limits")
    velocity = np.diff(arm, axis=0) / trajectory.dt
    acceleration = np.diff(np.vstack([np.zeros(6), velocity, np.zeros(6)]), axis=0) / trajectory.dt
    if (
        np.any(arm < lower)
        or np.any(arm > upper)
        or np.any(np.abs(arm) > 360)
        or np.any(np.abs(velocity) > vmax)
        or np.any(np.abs(acceleration) > amax)
    ):
        raise ValueError(
            "Recorded speed/acceleration exceeds physical RB3 limits; speed will not be changed silently"
        )
    hv = vector(config["revo2"]["velocity_units_s"], 6, "Revo2 commissioned velocity limits")
    if np.any(hv <= 0) or np.any(np.abs(np.diff(hand, axis=0) / trajectory.dt) > hv):
        raise ValueError("Recorded finger speed exceeds commissioned limits")
    rb = config["rb3"]
    if not np.isfinite(config["io_timeout_s"]) or not 0 < config["io_timeout_s"] < trajectory.dt:
        raise ValueError("I/O timeout must be finite and shorter than one command interval")
    servo = rb["servo"]
    values = np.asarray([servo[k] for k in ("t1_s", "t2_s", "gain", "alpha")], float)
    if (
        not np.isfinite(values).all()
        or values[0] < 0.002
        or not 0.02 < values[1] < 0.2
        or values[2] <= 0
        or not 0 < values[3] < 1
        or not np.isclose(values[0], trajectory.dt)
    ):
        raise ValueError(
            "Commission RB3 Servo J parameters; t1 must match the recorded 30Hz interval"
        )
    validation = read_config(Path(config["simulation_validation"]))
    if (
        validation["commands_sha256"] != sha256(trajectory.path)
        or not validation["simulated_grasp"]
        or not validation["command_collision_samples_passed"]
    ):
        raise ValueError("Require matching grasp replay and sampled command collision validation")
    return dict(
        **trajectory.summary(),
        backend="rbpodo Servo J + bc-stark-sdk RS485",
        calibration_id=calibration["id"],
        rb3_initial_deg=arm[0].tolist(),
        revo2_initial_units=hand[0].tolist(),
        rb3_peak_velocity_deg_s=np.abs(velocity).max(0).tolist(),
        rb3_peak_interval_acceleration_deg_s2=np.abs(acceleration).max(0).tolist(),
        revo2_peak_units_s=np.abs(np.diff(hand, axis=0) / trajectory.dt).max(0).tolist(),
        rb3_feedforward="SDK Servo J takes position; saved Isaac velocity feedforward is not sent",
        hardware_validated=False,
    )


class RBPodoStark:
    hardware = True
    feedback_source = dict(arm="physical_encoder_jnt_ang", hand="physical_motor_positions")

    def __init__(self, names, config):
        self.config = config
        self.mapping = JointCalibration(names, config["calibration"])
        self.robot = self.data = self.hand = None
        self.previous_device_time = None
        self.last_finger = None
        self.sent = False
        self.hand_lock = asyncio.Lock()

    async def connect(self):
        for name, version in VERSIONS.items():
            if importlib.metadata.version(name) != version:
                raise RuntimeError(f"Use {name}=={version}; revalidate other API versions")
        import rbpodo
        import bc_stark_sdk.main_mod as stark

        self.rb_sdk, self.hand_sdk = rbpodo, stark
        rb, hand = self.config["rb3"], self.config["revo2"]
        # Do not activate motors, change operation mode, gains, collision flags,
        # current limits or move to a start pose as a side effect of connection.
        self.robot = rbpodo.Cobot(rb["address"], rb["command_port"])
        self.data = rbpodo.CobotData(rb["address"], rb["data_port"])
        self.hand = await asyncio.wait_for(
            stark.modbus_open(hand["port"], getattr(stark.Baudrate, hand["baudrate_enum"])), 2.0
        )
        hand_type = await self.call_hand("get_hand_type")
        mode = await self.call_hand("get_finger_unit_mode")
        if hand_type != stark.HandType.Right or mode != stark.FingerUnitMode.Normalized:
            raise RuntimeError("Expected commissioned right hand with normalized position units")
        if not await self.call_hand("uses_revo2_motor_api"):
            raise RuntimeError("Connected hand does not support the Revo2 motor API")

    async def call_hand(self, method, *args):
        async with self.hand_lock:
            value = getattr(self.hand, method)(self.config["revo2"]["slave_id"], *args)
            return (
                await asyncio.wait_for(value, self.config["io_timeout_s"])
                if inspect.isawaitable(value)
                else value
            )

    async def prepare_tactile(self):
        if not await self.call_hand("is_touch_hand"):
            raise ValueError("The connected Revo2 has no tactile sensors")
        enabled = int(await self.call_hand("get_touch_sensor_enabled"))
        if enabled & 31 != 31:
            raise ValueError("All five tactile sensors must already be enabled")
        return dict(
            type="Revo2 capacitive", enabled_mask=enabled,
            firmware=str(await self.call_hand("get_touch_sensor_fw_versions")),
            io_timeout_s=self.config["io_timeout_s"],
            connection="shared controller RS485 handle; no sensor/calibration setters",
        )

    async def read_tactile(self):
        from .sensors.hardware_tactile import decode_sdk_sample

        started = time.monotonic()
        value = await self.call_hand("get_touch_sensor_status")
        completed = time.monotonic()
        return dict(sample=decode_sdk_sample(value), started_s=started, completed_s=completed)

    async def read(self):
        started = time.monotonic()
        arm, hand = await asyncio.gather(
            asyncio.to_thread(self.data.request_data, self.config["io_timeout_s"]),
            self.call_hand("get_motor_status"),
        )
        if arm is None:
            raise TimeoutError("RB3 feedback timeout")
        s = arm.sdata
        device_time = float(s.time)
        if not np.isfinite(device_time) or (
            self.previous_device_time is not None and device_time <= self.previous_device_time
        ):
            raise RuntimeError("RB3 feedback clock is stale or moved backwards")
        self.previous_device_time = device_time
        status = rb_status(s)
        ready = (
            status["mode"] == 0
            and status["init_stage"] == 6
            and not any(status["faults"].values())
            and s.robot_state in (1, 3)
        )
        if not self.sent:
            ready &= s.robot_state == 1 and s.task_state == 1
        allowed = (self.hand_sdk.MotorState.Idle, self.hand_sdk.MotorState.Running)
        ready &= len(hand.states) == 6 and all(state in allowed for state in hand.states)
        self.last_finger = list(hand.positions)
        return dict(
            q_rad=self.mapping.decode(s.jnt_ang, hand.positions),
            ready=bool(ready),
            sample_time_s=started,
            rb3_device_time_s=device_time,
            read_window_s=time.monotonic() - started,
            rb3_status=status,
            feedback_source=dict(arm="physical_encoder_jnt_ang", hand="physical_motor_positions"),
            revo2_status=dict(motor_states=[str(value) for value in hand.states]),
        )

    async def send(self, q, arm_velocity, dt):
        arm, fingers = self.mapping.encode(q)
        p = self.config["rb3"]["servo"]

        def arm_command():
            rc = self.rb_sdk.ResponseCollector()
            result = self.robot.move_servo_j(
                rc,
                arm,
                p["t1_s"],
                p["t2_s"],
                p["gain"],
                p["alpha"],
                self.config["io_timeout_s"],
                True,
            )
            if not result.is_success():
                raise RuntimeError("RB3 rejected Servo J or did not acknowledge")

        # Mark before sending: a partial two-device failure still needs stopping.
        self.sent = True
        results = await asyncio.gather(
            asyncio.to_thread(arm_command),
            self.call_hand("set_finger_positions", fingers.tolist()),
            return_exceptions=True,
        )
        if not isinstance(results[1], BaseException) and results[1] is not False:
            self.last_finger = fingers.tolist()
        else:
            # Do not replay an older, possibly more open, hand target on failure.
            self.last_finger = None
        for result in results:
            if isinstance(result, BaseException):
                raise result
            if result is False:
                raise RuntimeError("Revo2 rejected finger command")

    async def stop(self):
        if not self.sent:
            return
        # Finite t2 is the controller-side stream expiry. task_stop is an
        # additional best-effort stop, not a substitute for the physical E-stop.
        calls = []
        if self.robot is not None:
            calls.append(
                asyncio.to_thread(
                    self.robot.task_stop,
                    self.rb_sdk.ResponseCollector(),
                    self.config["io_timeout_s"],
                    True,
                )
            )
        if self.hand is not None:

            async def hold_hand():
                position = self.last_finger
                if position is None:
                    position = list((await self.call_hand("get_motor_status")).positions)
                return await self.call_hand("set_finger_positions", position)

            calls.append(hold_hand())
        results = await asyncio.gather(*calls, return_exceptions=True)
        if any(
            isinstance(r, BaseException)
            or r is False
            or (hasattr(r, "is_success") and not r.is_success())
            for r in results
        ):
            raise RuntimeError("Stop/hold acknowledgment failed; use the physical stop")

    async def close(self):
        try:
            if self.hand is not None:
                await asyncio.wait_for(
                    self.hand_sdk.modbus_close(self.hand), self.config["io_timeout_s"]
                )
        finally:
            self.hand = self.robot = self.data = None


async def execute(trajectory, config, output, hold_s, *, record_tactile=False, tactile_hz=100.):
    plan = hardware_plan(trajectory, config)
    guard = config["guard"]
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new hardware output directory")
    from .sensors.session import MotionTelemetry

    telemetry = MotionTelemetry(trajectory, output, tactile_hz) if record_tactile else None
    try:
        return await stream(
            trajectory,
            RBPodoStark(trajectory.names, config),
            output,
            start_tolerance_rad=np.deg2rad(guard["start_tolerance_deg"]),
            tracking_tolerance_rad=np.deg2rad(guard["tracking_tolerance_deg"]),
            maximum_feedback_age_s=guard["maximum_feedback_age_s"],
            maximum_lateness_s=guard["maximum_lateness_s"],
            hold_s=hold_s,
            telemetry=telemetry,
        )
    finally:
        if output.is_dir():
            (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
            (output / "hardware_config.json").write_text(json.dumps(config, indent=2) + "\n")
