"""Loopback-only control panel; ROS clients never open vendor connections."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import queue
import secrets
import threading
import time

import numpy as np


def run_panel(root, settings, recording, hold, port, page):
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from rclpy.action import ActionClient
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from control_msgs.action import FollowJointTrajectory
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
    from sensor_msgs.msg import JointState
    from std_srvs.srv import Trigger
    from trajectory_msgs.msg import JointTrajectoryPoint
    from ..configuration import read_config
    from ..ros_state import RobotState
    from ..ros_bridge import make_goal

    model = RobotState(root, read_config(root / settings["arm_config"]))
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("robot_panel", namespace=settings["namespace"])
    client = ActionClient(node, FollowJointTrajectory, "jog")
    playback = ActionClient(node, FollowJointTrajectory, "follow_joint_trajectory")
    stop = node.create_client(Trigger, "stop")
    lock = threading.Lock()
    state = dict(
        q_deg=None,
        command_deg=None,
        error_deg=None,
        status="Waiting for feedback",
        result="",
        busy=False,
        received=0.0,
        backend=settings["backend"],
        namespace=settings["namespace"],
        names=model.names,
        lower_deg=np.rad2deg(np.r_[model.arm.lower, model.hand.lower]).tolist(),
        upper_deg=np.rad2deg(np.r_[model.arm.upper, model.hand.upper]).tolist(),
        initial_deg=np.rad2deg(recording.initial).tolist(),
        enabled=False,
        jog_enabled=False,
        diagnostic_ok=False,
    )
    commands = queue.Queue(maxsize=8)
    token = secrets.token_urlsafe(32)

    def on_state(msg):
        try:
            now = node.get_clock().now().nanoseconds
            model.receive(
                msg.name,
                msg.position,
                msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec,
                now,
                settings["state_timeout_s"],
            )
            with lock:
                state.update(q_deg=np.rad2deg(model.q).tolist(), received=time.monotonic())
        except ValueError:
            pass

    def on_diagnostic(msg):
        for row in msg.status:
            values = {x.key: json.loads(x.value) for x in row.values}
            with lock:
                state.update(
                    status=row.message,
                    enabled=values.get("motion_enabled", False),
                    jog_enabled=values.get("jog_enabled", False),
                    diagnostic_ok=values.get("ready", False)
                    and values.get("io_healthy", row.level == DiagnosticStatus.OK),
                    feedback_source=values.get("feedback_source"),
                    dependent_fingers=values.get("dependent_fingers"),
                )

    node.create_subscription(JointState, "joint_states", on_state, qos_profile_sensor_data)
    node.create_subscription(DiagnosticArray, "diagnostics", on_diagnostic, 10)

    def finished(future):
        try:
            result = future.result()
            message = f"{result.result.error_code}: {result.result.error_string}"
        except Exception as error:
            message = str(error)
        with lock:
            state.update(result=message, busy=False, command_deg=None, error_deg=None)

    def feedback(message):
        sample = message.feedback
        order = [sample.joint_names.index(name) for name in model.names]
        with lock:
            state.update(
                command_deg=np.rad2deg(np.asarray(sample.desired.positions)[order]).tolist(),
                error_deg=np.rad2deg(np.asarray(sample.error.positions)[order]).tolist(),
            )

    def acknowledged(future):
        try:
            handle = future.result()
            if not handle.accepted:
                raise ValueError("Rejected by robot bridge; check pose, limits and readiness")
            handle.get_result_async().add_done_callback(finished)
        except Exception as error:
            with lock:
                state.update(result=str(error), busy=False)

    def dispatch():
        try:
            kind, payload = commands.get_nowait()
        except queue.Empty:
            return
        try:
            if kind == "stop":
                if not stop.service_is_ready():
                    raise ValueError("Stop service unavailable")
                stop.call_async(Trigger.Request())
                return
            with lock:
                if (
                    state["busy"]
                    or not state["diagnostic_ok"]
                    or time.monotonic() - state["received"] > settings["state_timeout_s"]
                ):
                    raise ValueError("Robot busy, faulted, or feedback stale")
                if not state["enabled"] or (kind == "jog" and not state["jog_enabled"]):
                    raise ValueError("Motion is disabled")
                q = np.deg2rad(state["q_deg"])
            if kind == "recording":
                goal = make_goal(recording, hold)
                action = playback
            else:
                target = np.deg2rad(np.asarray(payload["q_deg"], float))
                duration = float(payload.get("duration_s", 2.0))
                if (
                    target.shape != (12,)
                    or not np.isfinite(target).all()
                    or not np.isfinite(duration)
                    or duration <= 0
                    or duration > settings["jog"]["maximum_duration_s"]
                ):
                    raise ValueError("Expected twelve finite targets and duration in (0,30] s")
                goal = FollowJointTrajectory.Goal()
                goal.trajectory.joint_names = list(model.names)
                for t, values in ((0.0, q), (duration, target)):
                    point = JointTrajectoryPoint()
                    point.positions = values.tolist()
                    ns = round(t * 1e9)
                    point.time_from_start.sec = ns // 10**9
                    point.time_from_start.nanosec = ns % 10**9
                    goal.trajectory.points.append(point)
                action = client
            if not action.server_is_ready():
                raise ValueError("Action server unavailable")
            with lock:
                state.update(busy=True, result="Goal submitted")
            action.send_goal_async(goal, feedback_callback=feedback).add_done_callback(acknowledged)
        except Exception as error:
            with lock:
                state["result"] = str(error)

    node.create_timer(0.02, dispatch)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, code, body, content_type="application/json"):
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def valid_host(self):
            return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def do_GET(self):
            if not self.valid_host():
                return self.respond(403, '{"error":"Loopback host required"}')
            if self.path == "/":
                return self.respond(
                    200, page.read_text().replace("__TOKEN__", token), "text/html; charset=utf-8"
                )
            if self.path == "/state":
                with lock:
                    snapshot = dict(state)
                snapshot["age_s"] = time.monotonic() - snapshot.pop("received")
                return self.respond(200, json.dumps(snapshot, allow_nan=False))
            self.respond(404, "{}")

        def do_POST(self):
            if not self.valid_host() or not secrets.compare_digest(
                self.headers.get("X-Robot-Token", ""), token
            ):
                return self.respond(403, '{"error":"Invalid local session"}')
            if self.path not in ("/jog", "/recording", "/stop"):
                return self.respond(404, "{}")
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request length")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Expected an object")
                commands.put_nowait((self.path[1:], payload))
                self.respond(202, '{"queued":true}')
            except (ValueError, queue.Full) as error:
                self.respond(400, json.dumps(dict(error=str(error))))

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"[robot] Control panel: http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        executor.shutdown(timeout_sec=2)
        thread.join(2)
        client.destroy()
        playback.destroy()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
