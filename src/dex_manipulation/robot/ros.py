"""ROS 2 arm/hand actions, feedback bridge and command-line clients."""

from dex_manipulation.configuration import read_config
import asyncio
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import datetime
import json
from pathlib import Path
import threading
import time
import numpy as np
from dex_manipulation.robot.trajectory import MockBackend, check_state, stream
from dex_manipulation.robot.hardware import RBPodoStark, connection_plan, hardware_plan
from dex_manipulation.robot.model import RobotState
import argparse
import math


class MotionCancelled(RuntimeError):
    pass


def goal_arrays(recording, hold_s):
    return (
        np.r_[recording.times, recording.duration + hold_s],
        np.vstack([recording.q, recording.q[-1]]),
    )


def make_goal(recording, hold):
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(recording.names)
    times, positions = goal_arrays(recording, hold)
    for stamp, q in zip(times, positions):
        point = JointTrajectoryPoint()
        point.positions = q.tolist()
        ns = round(stamp * 1e9)
        point.time_from_start.sec = ns // 10**9
        point.time_from_start.nanosec = ns % 10**9
        goal.trajectory.points.append(point)
    return goal


def validate_goal(request, recording, hold_s, guard):
    """Accept a named permutation of the recording, never silently retime it."""
    trajectory = request.trajectory
    names = list(trajectory.joint_names)
    if len(names) != 12 or set(names) != set(recording.names):
        raise ValueError("Expected all twelve arm and finger joint names")
    if (
        trajectory.header.stamp.sec
        or trajectory.header.stamp.nanosec
        or request.multi_dof_trajectory.points
        or request.multi_dof_trajectory.joint_names
        or request.component_path_tolerance
        or request.component_goal_tolerance
        or request.goal_time_tolerance.sec
        or request.goal_time_tolerance.nanosec
    ):
        raise ValueError(
            "Only immediate single-DoF goals with fixed recording timing are supported"
        )
    times, positions = goal_arrays(recording, hold_s)
    if len(trajectory.points) != len(times):
        raise ValueError("Goal must contain the selected recording plus its final hold endpoint")
    indices = [names.index(n) for n in recording.names]
    for point, stamp, q in zip(trajectory.points, times, positions):
        values = np.asarray(point.positions, float)
        if (
            values.shape != (12,)
            or not np.isfinite(values).all()
            or point.velocities
            or point.accelerations
            or point.effort
            or abs(point.time_from_start.sec + point.time_from_start.nanosec * 1e-9 - stamp) > 2e-9
            or not np.allclose(values[indices], q, atol=1e-8, rtol=0)
        ):
            raise ValueError(
                "Goal differs from the configured recording; select/validate that recording first"
            )
    tolerances = []
    for rows in (request.path_tolerance, request.goal_tolerance):
        limit = np.deg2rad(guard["tracking_tolerance_deg"]).copy()
        seen = set()
        for row in rows:
            if (
                row.name not in recording.names
                or row.name in seen
                or not np.isfinite([row.position, row.velocity, row.acceleration]).all()
                or row.position < 0
                or row.velocity
                or row.acceleration
            ):
                raise ValueError("Only named position tolerances may tighten the configured guard")
            seen.add(row.name)
            if row.position:
                i = recording.names.index(row.name)
                limit[i] = min(limit[i], row.position)
        tolerances.append(limit)
    return tolerances


class ObservedBackend:
    def __init__(self, worker, cancel, feedback):
        self.worker, self.cancel, self.feedback = worker, cancel, feedback
        self.hardware = worker.backend.hardware
        self.feedback_source = getattr(worker.backend, "feedback_source", None)
        self.desired = worker.recording.initial.copy()
        self.started = time.monotonic()

    async def connect(self):
        pass  # Connection belongs to the worker lifetime.

    async def close(self):
        pass

    def check_cancel(self):
        if self.cancel.is_set() or self.worker.shutdown.is_set():
            raise MotionCancelled("ROS goal canceled/stopped")

    async def read(self):
        self.check_cancel()
        state = await self.worker.read()
        self.feedback(state, self.desired, time.monotonic() - self.started)
        return state

    async def send(self, q, velocity, dt):
        self.check_cancel()
        await self.worker.backend.send(q, velocity, dt)
        self.desired = q.copy()

    async def prepare_tactile(self):
        self.check_cancel()
        return await self.worker.backend.prepare_tactile()

    async def read_tactile(self):
        self.check_cancel()
        return await self.worker.backend.read_tactile()

    async def stop(self):
        await self.worker.backend.stop()
        if hasattr(self.worker.backend, "sent"):
            self.worker.backend.sent = False


class DeviceWorker:
    def __init__(
        self, recording, backend, guard, rate, publish, *, record_tactile=False, tactile_hz=100.0
    ):
        self.recording, self.backend, self.guard = recording, backend, guard
        self.rate, self.publish = rate, publish
        self.record_tactile, self.tactile_hz = record_tactile, tactile_hz
        self.shutdown = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.last = None
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._thread, name="robot_io", daemon=True)

    def _thread(self):
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._session())
        finally:
            self.loop.close()

    async def _session(self):
        self.lock = asyncio.Lock()
        try:
            await self.backend.connect()
            await self.read()
            self.ready.set()
            while not self.shutdown.is_set():
                started = time.monotonic()
                async with self.lock:
                    if not self.shutdown.is_set():
                        await self.read()
                await asyncio.sleep(max(0.0, 1 / self.rate - (time.monotonic() - started)))
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            self.ready.set()
        finally:
            for operation in (self.backend.stop, self.backend.close):
                try:
                    await operation()
                except Exception as error:
                    self.error = f"{operation.__name__}: {error}"

    async def read(self):
        state = await self.backend.read()
        check_state(
            dict(state, ready=True),
            np.asarray(state["q_rad"]),
            np.full(12, np.inf),
            self.guard["maximum_feedback_age_s"],
            time.monotonic,
        )
        if "full_q_rad" in state:
            full = np.asarray(state["full_q_rad"], float)
            if full.shape != (17,) or not np.isfinite(full).all():
                raise RuntimeError("Invalid full joint feedback")
        self.last = state
        self.publish(state)
        return state

    def start(self):
        self.thread.start()
        if not self.ready.wait(5):
            raise TimeoutError("Robot connection/initial feedback timed out")
        if self.error:
            raise RuntimeError(self.error)

    async def _motion(self, output, hold, cancel, feedback, tolerances, trajectory=None):
        async with self.lock:
            trajectory = self.recording if trajectory is None else trajectory
            observed = ObservedBackend(self, cancel, feedback)
            observed.desired = trajectory.initial.copy()
            from dex_manipulation.sensors.session import MotionTelemetry

            telemetry = (
                MotionTelemetry(trajectory, output, self.tactile_hz)
                if self.record_tactile
                else None
            )
            try:
                result = await stream(
                    trajectory,
                    observed,
                    output,
                    start_tolerance_rad=np.deg2rad(self.guard["start_tolerance_deg"]),
                    tracking_tolerance_rad=tolerances[0],
                    maximum_feedback_age_s=self.guard["maximum_feedback_age_s"],
                    maximum_lateness_s=self.guard["maximum_lateness_s"],
                    hold_s=hold,
                    telemetry=telemetry,
                )
            except Exception:
                report = Path(output) / "report.json"
                if report.exists() and read_config(report).get("cleanup_errors"):
                    self.error = (
                        "Robot stop acknowledgment failed; restart after checking physical state"
                    )
                    raise RuntimeError(self.error)
                raise
            check_state(
                await self.read(),
                trajectory.q[-1],
                tolerances[1],
                self.guard["maximum_feedback_age_s"],
                time.monotonic,
            )
            return result

    def submit(self, *args):
        if self.error or not self.thread.is_alive():
            raise RuntimeError(self.error or "Device worker stopped")
        return asyncio.run_coroutine_threadsafe(self._motion(*args), self.loop)

    def close(self):
        self.shutdown.set()
        self.thread.join(5)
        if self.thread.is_alive():
            raise RuntimeError("Robot I/O did not shut down within five seconds")


def make_node(
    root,
    settings,
    recording,
    hardware_config,
    *,
    backend="mock",
    enable_motion=False,
    hold_s=1.0,
    record_tactile=False,
    tactile_hz=100.0,
    allow_jog=False,
):
    import rclpy
    from rclpy.action import ActionServer, CancelResponse, GoalResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.time import Time
    from control_msgs.action import FollowJointTrajectory
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import JointState
    from std_srvs.srv import Trigger
    from scipy.spatial.transform import Rotation

    if backend == "hardware":
        connection_plan(recording.names, hardware_config)
        if enable_motion:
            hardware_plan(recording, hardware_config)
    elif backend == "vcb":
        from dex_manipulation.robot.vcb import VCBBackend
    elif backend not in ("mock", "virtual", "isaac"):
        raise ValueError("Unknown ROS device backend")
    guard = hardware_config["guard"]
    arm_config = read_config(root / settings["arm_config"])
    model = RobotState(root, arm_config)
    if model.names != recording.names:
        raise ValueError("Recording and current model joint names differ")

    class RobotNode(Node):
        def __init__(self):
            super().__init__("rb3_revo2", namespace=settings["namespace"])
            if self.get_parameter("use_sim_time").value:
                raise ValueError("Hardware feedback requires wall clock, not /clock")
            self.group = ReentrantCallbackGroup()
            qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.states = self.create_publisher(JointState, "joint_states", qos)
            self.full_states = self.create_publisher(JointState, "model_joint_states", qos)
            self.wrist = self.create_publisher(PoseStamped, "wrist_pose", qos)
            self.diagnostics = self.create_publisher(DiagnosticArray, "diagnostics", 10)
            self.motion_enabled = backend in ("mock", "virtual", "isaac") or enable_motion
            self.jog_enabled = allow_jog and self.motion_enabled
            self.pending_jog = None
            self.busy = False
            self.goal_lock = threading.Lock()
            self.cancel = threading.Event()
            self.last_error = ""
            self.samples = 0
            if backend == "mock":
                device = MockBackend(recording.initial)
            elif backend == "virtual":
                from dex_manipulation.robot.trajectory import VirtualJointBackend

                device = VirtualJointBackend(
                    recording.initial,
                    np.r_[model.arm.velocity, model.hand.velocity],
                    settings.get("virtual_time_constant_s", 0.025),
                )
            elif backend == "isaac":
                from dex_manipulation.robot.sim import IsaacBackend

                device = IsaacBackend(settings, model)
            elif backend == "vcb":
                device = VCBBackend(recording, hardware_config)
            else:
                device = RBPodoStark(recording.names, hardware_config)
            self.worker = DeviceWorker(
                recording,
                device,
                guard,
                settings["state_rate_hz"],
                self.publish_state,
                record_tactile=record_tactile,
                tactile_hz=tactile_hz,
            )
            self.action = ActionServer(
                self,
                FollowJointTrajectory,
                "follow_joint_trajectory",
                goal_callback=self.accept_goal,
                cancel_callback=self.cancel_goal,
                execute_callback=self.execute_goal,
                callback_group=self.group,
            )
            self.jog_action = ActionServer(
                self,
                FollowJointTrajectory,
                "jog",
                goal_callback=self.accept_jog,
                cancel_callback=self.cancel_goal,
                execute_callback=self.execute_jog,
                callback_group=self.group,
            )
            self.stop_service = self.create_service(
                Trigger, "stop", self.stop_motion, callback_group=self.group
            )
            self.timer = self.create_timer(0.2, self.publish_diagnostics, callback_group=self.group)

        def publish_state(self, state):
            age = time.monotonic() - state["sample_time_s"]
            stamp = Time(nanoseconds=self.get_clock().now().nanoseconds - int(age * 1e9)).to_msg()
            message = JointState()
            message.header.stamp = stamp
            message.header.frame_id = settings["frame_id"]
            message.name = list(model.names)
            message.position = np.asarray(state["q_rad"]).tolist()
            self.states.publish(message)
            full = JointState()
            full.header = message.header
            full.name = list(model.model_names)
            full.position = np.asarray(
                state.get("full_q_rad", model.expand(np.asarray(state["q_rad"])))
            ).tolist()
            self.full_states.publish(full)
            pose = model.arm.pose(np.asarray(state["q_rad"])[:6])
            quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
            wrist = PoseStamped()
            wrist.header = message.header
            wrist.pose.position.x, wrist.pose.position.y, wrist.pose.position.z = map(
                float, pose[:3, 3]
            )
            (
                wrist.pose.orientation.x,
                wrist.pose.orientation.y,
                wrist.pose.orientation.z,
                wrist.pose.orientation.w,
            ) = map(float, quat)
            self.wrist.publish(wrist)
            self.samples += 1

        def publish_diagnostics(self):
            state = self.worker.last
            age = time.monotonic() - state["sample_time_s"] if state else float("inf")
            status = DiagnosticStatus()
            status.name = "rb3_revo2"
            status.hardware_id = backend
            error = self.worker.error or self.last_error
            ready = bool(state and state["ready"])
            status.level = (
                DiagnosticStatus.ERROR
                if error or not ready or age > settings["state_timeout_s"]
                else DiagnosticStatus.OK
            )
            status.message = error or (
                "feedback stale"
                if age > settings["state_timeout_s"]
                else "robot not ready"
                if not ready
                else "moving"
                if self.busy
                else "ready"
            )
            values = dict(
                backend=backend,
                motion_enabled=self.motion_enabled,
                jog_enabled=self.jog_enabled,
                busy=self.busy,
                ready=ready,
                io_healthy=self.worker.error is None,
                feedback_age_s=age,
                state_samples=self.samples,
                command_hz=1 / recording.dt,
                feedback_rate_requested_hz=settings["state_rate_hz"],
                dependent_fingers="PhysX joint state"
                if backend == "isaac"
                else "derived from coupling; not measured",
                effort="unavailable; JointState.effort empty",
            )
            values["feedback_source"] = getattr(self.worker.backend, "feedback_source", None)
            if state:
                for key in ("rb3_device_time_s", "rb3_status", "revo2_status", "read_window_s"):
                    if key in state:
                        values[key] = state[key]
            status.values = [KeyValue(key=k, value=json.dumps(v)) for k, v in values.items()]
            msg = DiagnosticArray()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.status = [status]
            self.diagnostics.publish(msg)

        def accept_goal(self, request):
            try:
                validate_goal(request, recording, hold_s, guard)
                if not self.motion_enabled:
                    raise ValueError("Read-only bridge; --enable-motion was not set")
                if self.worker.error or self.worker.last is None:
                    raise ValueError("Feedback is unavailable")
                check_state(
                    self.worker.last,
                    recording.initial,
                    np.deg2rad(guard["start_tolerance_deg"]),
                    guard["maximum_feedback_age_s"],
                    time.monotonic,
                )
                with self.goal_lock:
                    if self.busy:
                        raise ValueError("Another motion is active; cancel/finish it first")
                    self.busy = True
                    self.cancel.clear()
                    self.last_error = ""
                return GoalResponse.ACCEPT
            except (ValueError, RuntimeError, KeyError) as error:
                self.get_logger().warning(str(error))
                return GoalResponse.REJECT

        def accept_jog(self, request):
            try:
                if record_tactile:
                    raise ValueError(
                        "Tactile session recording requires a saved-motion goal, not jog"
                    )
                if not self.jog_enabled or self.worker.error or self.worker.last is None:
                    raise ValueError("Jog is disabled or robot feedback unavailable")
                with self.goal_lock:
                    if self.busy:
                        raise ValueError("Another motion is active")
                    from dex_manipulation.robot.trajectory import plan_jog, validate_hardware_jog

                    output = (
                        root
                        / "local/results/robot/jog_plans"
                        / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
                    )
                    plan = plan_jog(
                        request,
                        model,
                        self.worker.last,
                        settings["jog"],
                        recording.dt,
                        output,
                        hardware=backend == "hardware",
                    )
                    if backend == "hardware":
                        validate_hardware_jog(plan, hardware_config, model, root, arm_config)
                    elif backend == "vcb":
                        from dex_manipulation.robot.hardware import ArmCalibration

                        mapping = ArmCalibration(plan.names, hardware_config["mapping"])
                        mapped = np.asarray([mapping.encode_arm(q[:6]) for q in plan.q])
                        if np.any(np.abs(mapped) > 360):
                            raise ValueError("Jog exceeds VCB Servo J +/-360 degree range")
                    self.pending_jog = plan
                    self.busy = True
                    self.cancel.clear()
                    self.last_error = ""
                return GoalResponse.ACCEPT
            except (ValueError, RuntimeError, KeyError) as error:
                self.get_logger().warning(str(error))
                return GoalResponse.REJECT

        def execute_jog(self, handle):
            plan, self.pending_jog = self.pending_jog, None
            return self.execute_goal(handle, plan=plan)

        def cancel_goal(self, handle):
            self.cancel.set()
            return CancelResponse.ACCEPT

        def stop_motion(self, request, response):
            self.cancel.set()
            response.success = True
            response.message = "Cancellation requested; completion/stop acknowledgment appears in action result and diagnostics"
            return response

        def execute_goal(self, handle, plan=None):
            result = FollowJointTrajectory.Result()

            def feedback(state, desired, elapsed):
                if not handle.is_active:
                    return
                msg = FollowJointTrajectory.Feedback()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.joint_names = list(recording.names)
                msg.desired.positions = desired.tolist()
                msg.actual.positions = np.asarray(state["q_rad"]).tolist()
                msg.error.positions = (desired - np.asarray(state["q_rad"])).tolist()
                for point in (msg.desired, msg.actual, msg.error):
                    ns = int(elapsed * 1e9)
                    point.time_from_start.sec = ns // 10**9
                    point.time_from_start.nanosec = ns % 10**9
                handle.publish_feedback(msg)

            try:
                output = root / "local/results/ros" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                limits = (
                    validate_goal(handle.request, recording, hold_s, guard)
                    if plan is None
                    else [np.deg2rad(guard["tracking_tolerance_deg"])] * 2
                )
                future = self.worker.submit(
                    output, hold_s if plan is None else 0.3, self.cancel, feedback, limits, plan
                )
                while True:
                    try:
                        future.result(timeout=0.05)
                        break
                    except FutureTimeout:
                        if future.done():
                            raise
                        if handle.is_cancel_requested or not rclpy.ok():
                            self.cancel.set()
                handle.succeed()
                result.error_code = result.SUCCESSFUL
                result.error_string = f"Completed; log: {output}"
            except MotionCancelled as error:
                if handle.is_cancel_requested:
                    handle.canceled()
                else:
                    handle.abort()
                result.error_code = result.PATH_TOLERANCE_VIOLATED
                result.error_string = str(error)
            except Exception as error:
                handle.abort()
                self.last_error = str(error)
                result.error_code = result.PATH_TOLERANCE_VIOLATED
                result.error_string = str(error)
            finally:
                with self.goal_lock:
                    self.busy = False
            return result

        def close(self):
            self.cancel.set()
            self.worker.close()
            self.action.destroy()
            self.jog_action.destroy()
            self.destroy_node()

    return RobotNode()


ROOT = Path(__file__).resolve().parents[3]


def send(recording, settings, hold, wait):
    import rclpy
    from rclpy.action import ActionClient
    from control_msgs.action import FollowJointTrajectory
    from action_msgs.msg import GoalStatus

    node = rclpy.create_node("recorded_motion_client", namespace=settings["namespace"])
    client = ActionClient(node, FollowJointTrajectory, "follow_joint_trajectory")
    handle = None
    try:
        if not client.wait_for_server(timeout_sec=wait):
            raise TimeoutError("ROS action server not found")
        future = client.send_goal_async(make_goal(recording, hold))
        rclpy.spin_until_future_complete(node, future, timeout_sec=wait)
        if not future.done():
            raise TimeoutError("Goal acknowledgment timed out")
        handle = future.result()
        if not handle.accepted:
            raise RuntimeError("Goal rejected; check bridge diagnostics and initial pose")
        result = handle.get_result_async()
        try:
            rclpy.spin_until_future_complete(
                node, result, timeout_sec=recording.duration + hold + wait
            )
            if not result.done():
                raise TimeoutError("Motion result timed out")
        except (KeyboardInterrupt, TimeoutError):
            cancellation = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(node, cancellation, timeout_sec=2.0)
            rclpy.spin_until_future_complete(node, result, timeout_sec=3.0)
            raise
        response = result.result()
        print(
            json.dumps(
                dict(
                    status=response.status,
                    error_code=response.result.error_code,
                    message=response.result.error_string,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return (
            0
            if response.status == GoalStatus.STATUS_SUCCEEDED and response.result.error_code == 0
            else 1
        )
    finally:
        client.destroy()
        node.destroy_node()


def monitor(root, settings, seconds, output):
    import rclpy
    from dex_manipulation.robot.model import RobotState, subscribe_state

    model = RobotState(root, read_config(root / settings["arm_config"]))
    node = subscribe_state(settings, model, "robot_state_monitor")
    start = time.monotonic()
    last_print = 0.0
    ages = []
    try:
        while rclpy.ok() and (seconds == 0 or time.monotonic() - start < seconds):
            rclpy.spin_once(node, timeout_sec=0.02)
            if model.stamp_ns:
                ages.append((node.get_clock().now().nanoseconds - model.stamp_ns) * 1e-9)
            if time.monotonic() - last_print >= 1:
                last_print = time.monotonic()
                print(
                    f"states={model.accepted} fresh={model.fresh(settings['state_timeout_s'])} joints=12",
                    flush=True,
                )
        report = dict(
            messages=model.accepted,
            rejected=model.rejected,
            elapsed_s=time.monotonic() - start,
            fresh=model.fresh(settings["state_timeout_s"]),
            latest_q_rad=model.q.tolist() if model.q is not None else None,
            observed_age_max_s=max(ages) if ages else None,
        )
        print(json.dumps(report, indent=2))
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
        return 0 if report["messages"] and report["fresh"] else 1
    finally:
        node.destroy_node()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="./run.sh ros")
    parser.add_argument("mode", choices=("bridge", "send", "status", "mirror"))
    parser.add_argument(
        "--backend", choices=("mock", "virtual", "isaac", "hardware", "vcb"), default="mock"
    )
    parser.add_argument(
        "--enable-motion", action="store_true", help="Allow the selected recorded ROS goal"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/ros.json")
    parser.add_argument(
        "--hardware-config", type=Path, default=ROOT / "config/hardware.example.json"
    )
    parser.add_argument("--vcb-config", type=Path, default=ROOT / "local/vcb.json")
    parser.add_argument("--recording", type=Path)
    parser.add_argument(
        "--allow-jog",
        action="store_true",
        help="Enable bounded named joint jogging through the same I/O owner",
    )
    parser.add_argument("--hold", type=float)
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="Duration for bridge/status/mirror; 0 keeps running",
    )
    parser.add_argument(
        "--wait", type=float, default=10.0, help="Action server/result timeout margin"
    )
    parser.add_argument("--headless", action="store_true", help="Isaac mirror without a window")
    parser.add_argument("--output", type=Path, help="Local JSON report for status/mirror")
    parser.add_argument(
        "--record-tactile",
        action="store_true",
        help="Record tactile/feedback during hardware actions through the shared RS485 connection",
    )
    parser.add_argument("--tactile-hz", type=float, default=100.0)
    args = parser.parse_args(argv)
    try:
        if args.allow_jog and (
            args.mode != "bridge"
            or (args.backend in ("hardware", "vcb") and not args.enable_motion)
        ):
            raise ValueError("Jog requires bridge and motion-enabled hardware/VCB")
        if args.record_tactile and (
            args.mode != "bridge" or args.backend != "hardware" or not args.enable_motion
        ):
            raise ValueError("--record-tactile requires bridge --backend hardware --enable-motion")
        if not math.isfinite(args.tactile_hz) or args.tactile_hz <= 0:
            raise ValueError("--tactile-hz must be positive")
        if args.enable_motion and (
            args.mode != "bridge" or args.backend not in ("hardware", "vcb")
        ):
            raise ValueError("--enable-motion requires bridge --backend hardware or vcb")
        for key in ("seconds", "wait"):
            if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
                raise ValueError("Invalid duration")
        settings = read_config(ROOT / args.config)
        for key in ("state_rate_hz", "state_timeout_s", "mirror_rate_hz"):
            if not math.isfinite(settings[key]) or settings[key] <= 0:
                raise ValueError(f"Invalid {key}")
        output = (ROOT / args.output).resolve() if args.output else None
        if output and (not output.is_relative_to(ROOT / "local") or output.exists()):
            raise ValueError("Use a new report path under local/")
        if args.mode == "mirror":
            from dex_manipulation.robot.model import run_mirror as run

            return run(ROOT, settings, args.seconds, args.headless, output)
        import rclpy
        from rclpy.signals import SignalHandlerOptions

        rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
        try:
            if args.mode == "status":
                return monitor(ROOT, settings, args.seconds, output)
            from dex_manipulation.robot.trajectory import RecordedCommands

            defaults = read_config(ROOT / "config/execution.json")
            recording = RecordedCommands(ROOT / (args.recording or defaults["recording"]))
            hold = args.hold if args.hold is not None else defaults["hold_s"]
            if not math.isfinite(hold) or hold < 0:
                raise ValueError("Hold must be finite and nonnegative")
            if args.mode == "send":
                return send(recording, settings, hold, args.wait)
            from rclpy.executors import MultiThreadedExecutor

            pass

            config = read_config(
                ROOT / (args.vcb_config if args.backend == "vcb" else args.hardware_config)
            )
            if args.backend != "vcb":
                config["simulation_validation"] = str(ROOT / config["simulation_validation"])
            node = make_node(
                ROOT,
                settings,
                recording,
                config,
                backend=args.backend,
                enable_motion=args.enable_motion,
                hold_s=hold,
                record_tactile=args.record_tactile,
                tactile_hz=args.tactile_hz,
                allow_jog=args.allow_jog,
            )
            executor = MultiThreadedExecutor(num_threads=3)
            executor.add_node(node)
            try:
                node.worker.start()
                print(
                    f"ROS ready: backend={args.backend}, motion={node.motion_enabled}, "
                    f"action={settings['namespace']}/follow_joint_trajectory",
                    flush=True,
                )
                started = time.monotonic()
                while rclpy.ok() and (
                    args.seconds == 0 or time.monotonic() - started < args.seconds
                ):
                    executor.spin_once(timeout_sec=0.05)
            finally:
                node.cancel.set()
                node.worker.close()
                executor.shutdown(timeout_sec=5.0)
                node.action.destroy()
                node.jog_action.destroy()
                node.destroy_node()
            return 0
        finally:
            if rclpy.ok():
                rclpy.shutdown()
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, ImportError, TimeoutError) as error:
        print(f"ROS 실행 실패: {error}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
