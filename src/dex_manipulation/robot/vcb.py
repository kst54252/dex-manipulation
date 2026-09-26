"""Rainbow Virtual Control Box adapter and connection commands."""

import asyncio
import importlib.metadata
import time
import numpy as np
from dex_manipulation.robot.hardware import ARM_NAMES, ArmCalibration, rb_status, vector
from dex_manipulation.configuration import read_config
import argparse
from datetime import datetime
import json
from pathlib import Path


SDK_VERSION = "0.16.14"
FEEDBACK_SOURCE = dict(arm="vcb_controller_reference_jnt_ref", hand="virtual_command_hold")


def connection_plan(config):
    if (
        config.get("schema") != "dex_rb3_vcb_v1"
        or config.get("endpoint_kind") != "virtual_control_box"
    ):
        raise ValueError("Expected a dedicated Virtual Control Box configuration")
    rb = config["rb3"]
    if not isinstance(rb["address"], str) or not rb["address"].strip():
        raise ValueError("Set the actual VM address with ./run.sh vcb init --address VM_IP")
    for key in ("command_port", "data_port"):
        if type(rb[key]) is not int or not 1 <= rb[key] <= 65535:
            raise ValueError(f"Invalid {key}")
    if not np.isfinite(config["io_timeout_s"]) or config["io_timeout_s"] <= 0:
        raise ValueError("I/O timeout must be positive and finite")


def motion_plan(recording, config):
    connection_plan(config)
    mapping = ArmCalibration(recording.names, config["mapping"])
    arm = np.asarray(
        [mapping.encode_arm(q[:6]) for q in np.vstack([recording.initial, recording.q])]
    )
    if np.any(np.abs(arm) > 360):
        raise ValueError("Mapped commands exceed Servo J +/-360 degrees; no angle wrapping")
    p = config["rb3"]["servo"]
    t1, t2, gain, alpha = np.asarray([p[k] for k in ("t1_s", "t2_s", "gain", "alpha")], float)
    if (
        not np.isfinite([t1, t2, gain, alpha]).all()
        or t1 < 0.002
        or not 0.02 < t2 < 0.2
        or gain <= 0
        or not 0 < alpha < 1
        or not np.isclose(t1, recording.dt, atol=1e-9, rtol=0)
    ):
        raise ValueError("Invalid VCB Servo J settings; t1 must equal recording dt")
    guard = config["guard"]
    for key in ("start_tolerance_deg", "tracking_tolerance_deg"):
        if np.any(vector(guard[key], 12, key) <= 0):
            raise ValueError(f"Invalid {key}")
    age, late = guard["maximum_feedback_age_s"], guard["maximum_lateness_s"]
    if (
        not np.isfinite([age, late]).all()
        or age <= 0
        or not 0 < late < recording.dt
        or config["io_timeout_s"] >= recording.dt
    ):
        raise ValueError("VCB timing guards must fit the recorded command interval")
    return dict(
        **recording.summary(),
        backend="vcb",
        rb3_initial_deg=arm[0].tolist(),
        rb3_joint_names=list(ARM_NAMES),
        feedback_source=FEEDBACK_SOURCE,
        rb3_peak_velocity_deg_s=np.abs(np.diff(arm, axis=0) / recording.dt).max(0).tolist(),
        hardware=False,
        physical_grasp_evaluation=False,
        feedforward="Servo J position only; Isaac velocity feedforward is not sent",
    )


class VCBConnection:
    """Read-only status first; opening a connection sends no motion commands."""

    def __init__(self, config):
        connection_plan(config)
        self.config = config
        self.data = self.robot = self.sdk = None
        self.previous_device_time = None
        self.last_raw = None
        self.sent = False

    async def connect(self):
        if importlib.metadata.version("rbpodo") != SDK_VERSION:
            raise RuntimeError(f"Use rbpodo=={SDK_VERSION}; revalidate other API versions")
        import rbpodo

        self.sdk = rbpodo
        rb = self.config["rb3"]
        self.data = rbpodo.CobotData(rb["address"], rb["data_port"])
        # Probe mode on data-only socket before even opening a command socket.
        await self.read_raw()

    async def read_raw(self):
        started = time.monotonic()
        self.last_raw = None
        while True:
            remaining = self.config["io_timeout_s"] - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("VCB clock is stale or status timed out")
            packet = await asyncio.to_thread(self.data.request_data, remaining)
            if packet is None:
                raise TimeoutError("VCB status timeout on data port")
            s = packet.sdata
            status = rb_status(s)
            if status["mode"] != 1:
                raise RuntimeError("VCB requires Simulation mode (1); refusing Real/unknown mode")
            device_time = float(s.time)
            if not np.isfinite(device_time) or (
                self.previous_device_time is not None and device_time < self.previous_device_time
            ):
                raise RuntimeError("VCB clock moved backwards or is invalid")
            if device_time != self.previous_device_time:
                break
            # Two host reads can fall inside one controller tick; retry only
            # inside the original I/O budget, never publish a duplicate sample.
            await asyncio.sleep(min(0.002, max(0.0, remaining)))
        self.previous_device_time = device_time
        ready = (
            not any(status["faults"].values())
            and status["robot_state"] in (1, 3)
            and status["task_state"] in (1, 3)
        )
        if not self.sent:
            ready &= status["robot_state"] == 1 and status["task_state"] == 1
        # VCB has no real motors: do not require hardware activation stage 6.
        self.last_raw = dict(
            jnt_ref_deg=vector(s.jnt_ref, 6, "VCB reference").tolist(),
            jnt_ang_deg=vector(s.jnt_ang, 6, "encoder field").tolist(),
            sample_time_s=started,
            rb3_device_time_s=device_time,
            read_window_s=time.monotonic() - started,
            rb3_status=status,
            ready=bool(ready),
            feedback_source=FEEDBACK_SOURCE,
        )
        return self.last_raw

    def assert_ready(self):
        state = self.last_raw
        if (
            state is None
            or state["rb3_status"]["mode"] != 1
            or not state["ready"]
            or not 0
            <= time.monotonic() - state["sample_time_s"]
            <= self.config["guard"]["maximum_feedback_age_s"]
        ):
            raise RuntimeError(
                "Fresh, ready Simulation-mode feedback required before a VCB command"
            )

    async def command(self, name, *args):
        self.assert_ready()
        if self.robot is None:
            rb = self.config["rb3"]
            self.robot = self.sdk.Cobot(rb["address"], rb["command_port"])
        self.sent = True  # A timed-out acknowledgment can still mean it was applied.
        result = await asyncio.to_thread(
            getattr(self.robot, name),
            self.sdk.ResponseCollector(),
            *args,
            self.config["io_timeout_s"],
            True,
        )
        if not result.is_success():
            raise RuntimeError(f"VCB rejected {name} or acknowledgment timed out")

    async def stop(self):
        if not self.sent or self.robot is None:
            return
        result = await asyncio.to_thread(
            self.robot.task_stop, self.sdk.ResponseCollector(), self.config["io_timeout_s"], True
        )
        if not result.is_success():
            raise RuntimeError("VCB stop acknowledgment failed")
        self.sent = False

    async def close(self):
        self.robot = self.data = None


class VCBBackend(VCBConnection):
    hardware = False
    feedback_source = FEEDBACK_SOURCE

    def __init__(self, recording, config):
        motion_plan(recording, config)
        super().__init__(config)
        self.recording = recording
        self.mapping = ArmCalibration(recording.names, config["mapping"])
        self.finger = recording.initial[6:].copy()

    async def read(self):
        state = await self.read_raw()
        return dict(
            state,
            q_rad=np.r_[self.mapping.decode_arm(state["jnt_ref_deg"]), self.finger],
            revo2_status=dict(kind="virtual", motor_connected=False),
        )

    async def send(self, q, arm_velocity, dt):
        q = vector(q, 12, "VCB joint target")
        if not np.isclose(dt, self.recording.dt, atol=1e-9, rtol=0):
            raise ValueError("VCB command dt differs from recorded timing")
        if np.any(q < self.recording.data["lower_rad"] - 1e-6) or np.any(
            q > self.recording.data["upper_rad"] + 1e-6
        ):
            raise ValueError("VCB target exceeds recorded model limits")
        arm = self.mapping.encode_arm(q[:6])
        if np.any(np.abs(arm) > 360):
            raise ValueError("VCB target exceeds Servo J angular range")
        p = self.config["rb3"]["servo"]
        await self.command("move_servo_j", arm, p["t1_s"], p["t2_s"], p["gain"], p["alpha"])
        self.finger = q[6:].copy()


async def probe(config, samples=30, rate=30.0):
    """No command socket: mode, clock, raw joints and round-trip read timing."""
    if type(samples) is not int or samples < 1 or not np.isfinite(rate) or rate <= 0:
        raise ValueError("Probe requires positive sample count and rate")
    device = VCBConnection(config)
    rows = []
    try:
        await device.connect()
        rows.append(device.last_raw)
        for _ in range(samples - 1):
            await asyncio.sleep(1 / rate)
            rows.append(await device.read_raw())
    finally:
        await device.close()
    return dict(
        backend="vcb",
        samples=len(rows),
        latest=rows[-1],
        mean_read_ms=1000 * float(np.mean([s["read_window_s"] for s in rows])),
        max_read_ms=1000 * max(s["read_window_s"] for s in rows),
        motion_commands_sent=0,
        physical_hardware_tested=False,
    )


async def prepare(recording, config, timeout_s=90.0):
    """Explicit VCB-only Move J to the recording start, at 10 deg/s."""
    if not np.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Invalid preparation timeout")
    device = VCBBackend(recording, config)
    started = time.monotonic()
    try:
        await device.connect()
        target = device.mapping.encode_arm(recording.initial[:6])
        await device.command("move_j", target, 10.0, 20.0)
        while time.monotonic() - started < timeout_s:
            await asyncio.sleep(1 / 30)
            state = await device.read()
            if not state["ready"]:
                raise RuntimeError("VCB fault while moving to initial pose")
            error = np.abs(state["q_rad"][:6] - recording.initial[:6])
            if state["rb3_status"]["robot_state"] == 1 and np.all(error <= np.deg2rad(0.1)):
                return dict(
                    backend="vcb",
                    prepared=True,
                    elapsed_s=time.monotonic() - started,
                    initial_deg=target.tolist(),
                    error_deg=np.rad2deg(error).tolist(),
                    feedback_source=FEEDBACK_SOURCE,
                )
        raise TimeoutError("VCB did not reach the recording initial pose")
    finally:
        try:
            await device.stop()
        finally:
            await device.close()


ROOT = Path(__file__).resolve().parents[3]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="./run.sh vcb")
    parser.add_argument(
        "mode", choices=("init", "probe", "plan", "prepare", "bridge", "send", "mirror", "status")
    )
    parser.add_argument("--config", type=Path, default=ROOT / "local/vcb.json")
    parser.add_argument("--address", help="Actual VCB VM IP/hostname; only for init")
    parser.add_argument(
        "--arm-sign", type=float, nargs=6, help="Verified USD-to-VCB axis signs, for init"
    )
    parser.add_argument(
        "--arm-offset-deg", type=float, nargs=6, help="Verified USD-to-VCB zero offsets, for init"
    )
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--ros-config", type=Path, default=ROOT / "config/ros_vcb.json")
    parser.add_argument(
        "--read-only", action="store_true", help="Bridge without accepting motion goals"
    )
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true", help="Mirror without a window")
    args = parser.parse_args(argv)
    try:
        pass

        path = (ROOT / args.config).resolve()
        if args.mode == "init":
            if not path.is_relative_to(ROOT / "local"):
                raise ValueError("Keep VM settings under local/")
            config = read_config(ROOT / "config/vcb.example.json")
            config["rb3"]["address"] = args.address
            connection_plan(config)
            if (args.arm_sign is None) != (args.arm_offset_deg is None):
                raise ValueError(
                    "Provide both axis signs and zero offsets, or leave both for later"
                )
            if args.arm_sign is not None:
                from dex_manipulation.robot.hardware import ARM_NAMES, ArmCalibration

                config["mapping"] = dict(arm_sign=args.arm_sign, arm_offset_deg=args.arm_offset_deg)
                ArmCalibration(ARM_NAMES, config["mapping"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as file:
                file.write(json.dumps(config, indent=2) + "\n")
            print(f"VCB config: {path}\nNext: ./run.sh vcb probe")
            return 0
        if args.address or args.arm_sign is not None or args.arm_offset_deg is not None:
            raise ValueError(
                "Address/mapping options belong to init; otherwise edit the selected config"
            )
        if args.read_only and args.mode != "bridge":
            raise ValueError("--read-only belongs to bridge")
        if args.mode in ("bridge", "send", "mirror", "status"):
            from dex_manipulation.robot.ros import main as ros_main

            forwarded = [
                args.mode,
                "--config",
                str(args.ros_config),
                "--seconds",
                str(args.seconds),
            ]
            if args.mode == "bridge":
                forwarded += ["--backend", "vcb", "--vcb-config", str(path)]
                if not args.read_only:
                    forwarded += ["--enable-motion"]
            if args.recording:
                forwarded += ["--recording", str(args.recording)]
            if args.headless:
                forwarded += ["--headless"]
            return ros_main(forwarded)
        config = read_config(path)
        if args.mode == "probe":
            report = asyncio.run(probe(config))
        else:
            from dex_manipulation.robot.trajectory import RecordedCommands

            defaults = read_config(ROOT / "config/execution.json")
            recording = RecordedCommands(ROOT / (args.recording or defaults["recording"]))
            plan = motion_plan(recording, config)
            report = plan if args.mode == "plan" else asyncio.run(prepare(recording, config))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.mode in ("probe", "prepare"):
            output = ROOT / "local/results/vcb" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output.mkdir(parents=True, exist_ok=False)
            (output / f"{args.mode}.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"Log: {output}")
        return 0
    except KeyboardInterrupt:
        return 130
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        ImportError,
        TimeoutError,
    ) as error:
        print(f"VCB 실행 실패: {error}", flush=True)
        return 2
