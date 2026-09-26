"""Physical Isaac device and ROS proxy, separated from the joint controller."""

import asyncio
import json
import threading
import time

import numpy as np


class IsaacBackend:
    hardware = False
    feedback_source = dict(arm="PhysX joint position", hand="PhysX joint position")

    def __init__(self, settings, model):
        self.settings, self.model = settings, model
        self.last = None
        self.lock = threading.Lock()
        self.sent = False
        self.node = self.executor = self.thread = None

    async def connect(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        from std_srvs.srv import Trigger

        self.node = rclpy.create_node("isaac_device_proxy", namespace=self.settings["namespace"])
        self.command = self.node.create_publisher(JointState, "device/command", 1)
        self.stop_client = self.node.create_client(Trigger, "device/stop")

        def receive(msg):
            names = list(msg.name)
            q = np.asarray(msg.position, float)
            age = (
                self.node.get_clock().now().nanoseconds
                - msg.header.stamp.sec * 10**9
                - msg.header.stamp.nanosec
            ) * 1e-9
            if (
                len(names) != len(set(names))
                or set(names) != set(self.model.model_names)
                or q.shape != (17,)
                or not np.isfinite(q).all()
                or not 0 <= age <= self.settings["state_timeout_s"]
            ):
                return
            with self.lock:
                self.last = dict(
                    q_rad=q[[names.index(n) for n in self.model.names]],
                    full_q_rad=q[[names.index(n) for n in self.model.model_names]],
                    sample_time_s=time.monotonic() - age,
                    ready=True,
                )

        self.node.create_subscription(
            JointState,
            "device/state",
            receive,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            with self.lock:
                ready = self.last is not None
            if ready and self.stop_client.service_is_ready():
                return
            await asyncio.sleep(0.02)
        raise TimeoutError("Isaac device state/stop service unavailable")

    async def read(self):
        deadline = time.monotonic() + self.settings["state_timeout_s"]
        while time.monotonic() < deadline:
            with self.lock:
                state = dict(self.last) if self.last is not None else None
            if (
                state is not None
                and time.monotonic() - state["sample_time_s"] < self.settings["state_timeout_s"]
            ):
                return state
            await asyncio.sleep(0.005)
        raise TimeoutError("Isaac device feedback timed out")

    async def send(self, q, arm_velocity, dt):
        from sensor_msgs.msg import JointState

        message = JointState()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.name = list(self.model.names)
        message.position = np.asarray(q, float).tolist()
        message.velocity = np.r_[arm_velocity, np.zeros(6)].tolist()
        self.sent = True
        self.command.publish(message)

    async def stop(self):
        if self.node is None or not self.sent:
            return
        from std_srvs.srv import Trigger

        if not self.stop_client.service_is_ready():
            raise RuntimeError("Isaac stop service unavailable")
        future = self.stop_client.call_async(Trigger.Request())
        deadline = time.monotonic() + 1
        while not future.done() and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        if not future.done() or not future.result().success:
            raise RuntimeError("Isaac stop not acknowledged")
        self.sent = False

    async def close(self):
        if self.executor:
            self.executor.shutdown(timeout_sec=2)
        if self.thread:
            self.thread.join(2)
        if self.node:
            self.node.destroy_node()
            self.node = None


def run_device(root, settings, recording, output):
    # SimulationApp must precede all Isaac/PhysX imports.
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    import torch
    from sensor_msgs.msg import JointState
    from std_srvs.srv import Trigger
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from ..configuration import read_config
    from .model import RobotState
    from dex_manipulation.policy.trajectory import ReferenceMotion
    from ..policy.arm_play_env import ArmPolicyEnv

    node = None
    try:
        cfg = read_config(root / settings["physics_config"])
        arm_cfg = read_config(root / settings["arm_config"])
        # Disable the optional policy wrist response: commands are already arm joints.
        arm_cfg.setdefault("policy_control", {}).update(
            wrist_response="direct", contact_feedback_gain=0.0
        )
        model = RobotState(root, arm_cfg)
        reference = ReferenceMotion(
            root / cfg["reference"], model.hand, root / cfg["object_geometry"], cfg["world_frame"]
        )
        env = ArmPolicyEnv(root, model.hand, reference, cfg, arm_cfg, render=False)
        full_ids = torch.cat([env.arm_ids, env.full_ids])

        def tensor(values):
            return torch.as_tensor(values[None], dtype=torch.float32, device=env.device)

        target = recording.initial.copy()
        velocity = np.zeros(6)
        last_command = 0.0
        holding = True
        env.robot.set_joint_positions(tensor(model.expand(target)), joint_indices=full_ids)
        env.robot.set_joint_velocities(tensor(np.zeros(17)), joint_indices=full_ids)
        rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
        node = rclpy.create_node("isaac_joint_device", namespace=settings["namespace"])
        pub = node.create_publisher(
            JointState,
            "device/state",
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        lower = np.r_[model.arm.lower, model.hand.lower]
        upper = np.r_[model.arm.upper, model.hand.upper]
        vmax = np.r_[model.arm.velocity, model.hand.velocity]

        def current():
            return (
                env.robot.get_joint_positions(joint_indices=full_ids)[0].cpu().numpy().astype(float)
            )

        def hold():
            nonlocal target, velocity, holding, last_stamp
            q = current()
            target = q[[model.model_names.index(n) for n in model.names]]
            velocity = np.zeros(6)
            holding = True
            last_stamp = node.get_clock().now().nanoseconds

        last_stamp = 0

        def command(msg):
            nonlocal target, velocity, last_command, holding, last_stamp
            q = np.asarray(msg.position, float)
            v = np.asarray(msg.velocity, float)
            names = list(msg.name)
            stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            age = (node.get_clock().now().nanoseconds - stamp) * 1e-9
            if (
                len(names) != 12
                or len(set(names)) != 12
                or set(names) != set(model.names)
                or q.shape != (12,)
                or v.shape != (12,)
                or not np.isfinite(np.r_[q, v]).all()
                or stamp <= last_stamp
                or not 0 <= age <= settings["state_timeout_s"]
            ):
                return
            order = [names.index(n) for n in model.names]
            q, v = q[order], v[order]
            if (
                np.any(q < lower - 1e-6)
                or np.any(q > upper + 1e-6)
                or np.any(np.abs(v) > vmax + 1e-6)
            ):
                return
            target = q
            velocity = v[:6]
            last_command = time.monotonic()
            holding = False
            last_stamp = stamp

        def stop(request, response):
            hold()
            response.success = True
            response.message = "PhysX targets held at measured position"
            return response

        node.create_subscription(JointState, "device/command", command, 1)
        node.create_service(Trigger, "device/stop", stop)
        output.mkdir(parents=True, exist_ok=True)
        metadata = dict(
            env.metadata,
            backend="isaac_joint_device",
            controller="named joint PD targets; no online IK or policy",
            initial_q_rad=target.tolist(),
            watchdog_s=settings["sim_command_timeout_s"],
        )
        (output / "physics.json").write_text(json.dumps(metadata, indent=2) + "\n")
        dt = cfg["physics_dt"]
        next_step = time.monotonic()
        steps = 0
        while app.is_running() and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            now = time.monotonic()
            if not holding and now - last_command > settings["sim_command_timeout_s"]:
                hold()
            env.robot.set_joint_position_targets(
                tensor(model.expand(target)), joint_indices=full_ids
            )
            env.robot.set_joint_velocity_targets(tensor(velocity), joint_indices=env.arm_ids)
            env.world.step(render=False)
            steps += 1
            if steps % max(1, round(1 / (dt * settings["sim_feedback_rate_hz"]))) == 0:
                q = current()
                if not np.isfinite(q).all():
                    raise RuntimeError("Non-finite PhysX joint state")
                msg = JointState()
                msg.header.stamp = node.get_clock().now().to_msg()
                msg.name = list(model.model_names)
                msg.position = q.tolist()
                pub.publish(msg)
                if steps > 120 and not (output / "ready.json").exists():
                    (output / "ready.json").write_text(json.dumps(dict(ready=True, physics_dt=dt)))
            next_step += dt
            delay = next_step - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -dt:
                next_step = time.monotonic()  # Do not fast-forward after a stall.
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        app.close()
