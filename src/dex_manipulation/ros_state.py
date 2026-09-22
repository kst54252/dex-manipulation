"""Named joint-state transport and USD display; independent of ROS/Isaac."""
import time

import numpy as np

from .fk import ArmModel, HandModel


def subscribe_state(settings, model, name):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    node = rclpy.create_node(name, namespace=settings['namespace'])

    def receive(message):
        try:
            stamp = message.header.stamp.sec*10**9+message.header.stamp.nanosec
            model.receive(message.name, message.position, stamp, node.get_clock().now().nanoseconds,
                          settings['state_timeout_s'])
        except ValueError as error:
            node.get_logger().warning(str(error), throttle_duration_sec=2.)
    node.create_subscription(JointState, 'joint_states', receive, qos_profile_sensor_data)
    return node


class RobotState:
    def __init__(self, root, arm_config):
        self.arm = ArmModel.load(root/arm_config['arm_model'])
        self.hand = HandModel.load(root/arm_config['hand_model'])
        self.names = tuple(self.arm.active_names+self.hand.active_names)
        self.model_names = tuple(self.arm.active_names+self.hand.full_names)
        self.q = None
        self.stamp_ns = None
        self.received = None
        self.accepted = self.rejected = 0

    def receive(self, names, positions, stamp_ns, now_ns, maximum_age_s, *, clock=time.monotonic):
        """No guessing joint order, replaying old packets, extrapolation or wrap."""
        try:
            names = tuple(names)
            values = np.asarray(positions, float)
            if len(names) != len(set(names)) or set(names) != set(self.names):
                raise ValueError('Expected exactly the twelve named joints')
            if values.shape != (12,) or not np.isfinite(values).all():
                raise ValueError('Invalid joint positions')
            age = (now_ns-stamp_ns)*1e-9
            if stamp_ns <= 0 or not 0 <= age <= maximum_age_s:
                raise ValueError('Stale/future state timestamp; synchronize host clocks')
            if self.stamp_ns is not None and stamp_ns <= self.stamp_ns:
                raise ValueError('Out-of-order state timestamp')
            self.q = values[[names.index(n) for n in self.names]]
            self.stamp_ns, self.received = stamp_ns, clock()
            self.accepted += 1
        except (ValueError, TypeError):
            self.rejected += 1
            raise

    def fresh(self, timeout, *, clock=time.monotonic):
        return self.received is not None and 0 <= clock()-self.received <= timeout

    def expand(self, q):
        return np.r_[q[:6], self.hand.expand(q[6:])]

    def link_poses(self, q):
        arm = self.arm.link_transforms(q[:6])
        return dict(arm, **self.hand.link_transforms(q[6:], arm[self.arm.wrist]))


class UsdStateMirror:
    """Display received joint FK, with physics disabled in an in-memory USD layer.

    Followers are model-derived, not extra measured encoders. No robot commands
    or object pose are synthesized by this visualization.
    """
    def __init__(self, stage, state, assembly_path, world_from_base):
        from pxr import Usd, UsdGeom, UsdPhysics
        self.stage, self.state = stage, state
        self.world_from_base = np.asarray(world_from_base)
        required = set(state.link_poses(np.zeros(12)))
        self.prims = {}
        assembly = stage.GetPrimAtPath(assembly_path)
        for prim in Usd.PrimRange(assembly):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
                if prim.GetName() in required:
                    if prim.GetName() in self.prims:
                        raise ValueError(f'Duplicate link name: {prim.GetName()}')
                    self.prims[prim.GetName()] = prim
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
            if prim.IsA(UsdGeom.Sphere) and '/kp_' in str(prim.GetPath()):
                UsdGeom.Imageable(prim).MakeInvisible()
        if set(self.prims) != required:
            raise ValueError(f'USD missing measured model links: {required-set(self.prims)}')
        self.ops = {}
        for name, prim in self.prims.items():
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            # Reset stack makes world transforms unambiguous in nested link USDs.
            xform.SetResetXformStack(True)
            self.ops[name] = xform.AddTransformOp(opSuffix='measured')

    def apply(self, q):
        from pxr import Gf
        poses = self.state.link_poses(q)
        for name, pose in poses.items():
            self.ops[name].Set(Gf.Matrix4d((self.world_from_base@pose).T.tolist()))
        return poses
