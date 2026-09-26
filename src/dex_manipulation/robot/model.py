"""Combined robot feedback model, USD mirror and RViz description export."""

import time
import numpy as np
from ..fk import ArmModel, HandModel
from pathlib import Path
import hashlib
import json
import struct
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
from ..configuration import read_config
from ..scene import Workcell
from ..transforms import inverse, apply


def subscribe_state(settings, model, name):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState

    node = rclpy.create_node(name, namespace=settings["namespace"])

    def receive(message):
        try:
            stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
            model.receive(
                message.name,
                message.position,
                stamp,
                node.get_clock().now().nanoseconds,
                settings["state_timeout_s"],
            )
        except ValueError as error:
            node.get_logger().warning(str(error), throttle_duration_sec=2.0)

    node.create_subscription(JointState, "joint_states", receive, qos_profile_sensor_data)
    return node


class RobotState:
    def __init__(self, root, arm_config):
        self.arm = ArmModel.load(root / arm_config["arm_model"])
        self.hand = HandModel.load(root / arm_config["hand_model"])
        self.names = tuple(self.arm.active_names + self.hand.active_names)
        self.model_names = tuple(self.arm.active_names + self.hand.full_names)
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
                raise ValueError("Expected exactly the twelve named joints")
            if values.shape != (12,) or not np.isfinite(values).all():
                raise ValueError("Invalid joint positions")
            age = (now_ns - stamp_ns) * 1e-9
            if stamp_ns <= 0 or not 0 <= age <= maximum_age_s:
                raise ValueError("Stale/future state timestamp; synchronize host clocks")
            if self.stamp_ns is not None and stamp_ns <= self.stamp_ns:
                raise ValueError("Out-of-order state timestamp")
            self.q = values[[names.index(n) for n in self.names]]
            self.stamp_ns, self.received = stamp_ns, clock()
            self.accepted += 1
        except (ValueError, TypeError):
            self.rejected += 1
            raise

    def fresh(self, timeout, *, clock=time.monotonic):
        return self.received is not None and 0 <= clock() - self.received <= timeout

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
                        raise ValueError(f"Duplicate link name: {prim.GetName()}")
                    self.prims[prim.GetName()] = prim
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
            if prim.IsA(UsdGeom.Sphere) and "/kp_" in str(prim.GetPath()):
                UsdGeom.Imageable(prim).MakeInvisible()
        if set(self.prims) != required:
            raise ValueError(f"USD missing measured model links: {required - set(self.prims)}")
        self.ops = {}
        for name, prim in self.prims.items():
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            # Reset stack makes world transforms unambiguous in nested link USDs.
            xform.SetResetXformStack(True)
            self.ops[name] = xform.AddTransformOp(opSuffix="measured")

    def apply(self, q):
        from pxr import Gf

        poses = self.state.link_poses(q)
        for name, pose in poses.items():
            self.ops[name].Set(Gf.Matrix4d((self.world_from_base @ pose).T.tolist()))
        return poses


def numbers(values):
    return " ".join(format(float(v), ".12g") for v in values)


def origin(parent, matrix):
    ET.SubElement(
        parent,
        "origin",
        xyz=numbers(matrix[:3, 3]),
        rpy=numbers(Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz")),
    )


def write_stl(path, vertices, triangles):
    p = np.asarray(vertices)[np.asarray(triangles)]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-15)
    rows = np.empty(
        len(p),
        dtype=np.dtype([("normal", "<f4", (3,)), ("points", "<f4", (3, 3)), ("attribute", "<u2")]),
    )
    rows["normal"], rows["points"], rows["attribute"] = normal, p, 0
    Path(path).write_bytes(
        b"dex-manipulation USD display mesh".ljust(80, b"\0")
        + struct.pack("<I", len(rows))
        + rows.tobytes()
    )


def export_meshes(asset, state, output):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(asset))
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    cache = UsdGeom.XformCache()
    names = set(state.link_poses(np.zeros(12)))
    bodies = {
        str(p.GetPath()): p.GetName()
        for p in stage.Traverse()
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetName() in names
    }
    meshes = {n: [] for n in names}
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(prim).ComputePurpose() not in (
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
        ):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        parent = prim
        while parent and str(parent.GetPath()) not in bodies:
            parent = parent.GetParent()
        if not parent:
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.array(mesh.GetPointsAttr().Get(), float)
        transform = (
            inverse(np.asarray(cache.GetLocalToWorldTransform(parent)).T)
            @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
        )
        points = apply(transform, points) * scale
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), int)
        triangles = []
        start = 0
        for count in counts:
            face = indices[start : start + count]
            triangles.extend([face[[0, i, i + 1]] for i in range(1, count - 1)])
            start += count
        triangles = np.asarray(triangles, int)
        if mesh.GetOrientationAttr().Get() == "leftHanded" or np.linalg.det(transform[:3, :3]) < 0:
            triangles = triangles[:, [0, 2, 1]]
        name = bodies[str(parent.GetPath())]
        path = output / f"{name}_{len(meshes[name])}.stl"
        write_stl(path, points, triangles)
        color = mesh.GetDisplayColorAttr().Get()
        color = (
            list(color[0])
            if color and len(color)
            else ([0.2, 0.24, 0.29] if name.startswith("right_") else [0.8, 0.81, 0.83])
        )
        meshes[name].append(
            dict(
                path=str(path.resolve()),
                color=[*color, 1.0],
                vertices=len(points),
                triangles=len(triangles),
            )
        )
    if not any(meshes.values()):
        raise ValueError("No visible USD meshes mapped to named robot links")
    return meshes


def build_urdf(state, workcell, meshes, *, mimic=False):
    robot = ET.Element("robot", name="rb3_730_revo2")
    links = set(state.link_poses(np.zeros(12)))
    elements = {name: ET.SubElement(robot, "link", name=name) for name in sorted(links)}
    ET.SubElement(robot, "link", name="world")
    mount = ET.SubElement(robot, "joint", name="workcell_to_robot", type="fixed")
    ET.SubElement(mount, "parent", link="world")
    ET.SubElement(mount, "child", link=state.arm.root)
    origin(mount, workcell.world_from_base)
    for name, parts in meshes.items():
        for i, part in enumerate(parts):
            visual = ET.SubElement(elements[name], "visual", name=f"{name}_visual_{i}")
            geom = ET.SubElement(visual, "geometry")
            ET.SubElement(geom, "mesh", filename=Path(part["path"]).as_uri())
            material = ET.SubElement(visual, "material", name=f"{name}_material_{i}")
            ET.SubElement(material, "color", rgba=numbers(part["color"]))
    for model in (state.arm, state.hand):
        for j in model.joints:
            f0, f1 = np.asarray(j["frame0"]), np.asarray(j["frame1"])
            reverse = j.get("reversed", False)
            pre, post = (f1, inverse(f0)) if reverse else (f0, inverse(f1))
            a, b = j["name"] + "_frame", j["name"] + "_motion"
            ET.SubElement(robot, "link", name=a)
            ET.SubElement(robot, "link", name=b)
            before = ET.SubElement(robot, "joint", name=j["name"] + "_before", type="fixed")
            ET.SubElement(before, "parent", link=j["parent"])
            ET.SubElement(before, "child", link=a)
            origin(before, pre)
            joint = ET.SubElement(robot, "joint", name=j["name"], type=j["kind"])
            ET.SubElement(joint, "parent", link=a)
            ET.SubElement(joint, "child", link=b)
            if j["kind"] != "fixed":
                ET.SubElement(
                    joint, "axis", xyz=numbers(np.asarray(j["axis"]) * (-1 if reverse else 1))
                )
                ET.SubElement(
                    joint,
                    "limit",
                    lower=str(j["lower"]),
                    upper=str(j["upper"]),
                    velocity=str(j["velocity"]),
                    effort=str(j.get("effort", 0.0) or 0.0),
                )
                if mimic and j.get("mimic"):
                    m = j["mimic"]
                    ET.SubElement(
                        joint,
                        "mimic",
                        joint=m["leader"],
                        multiplier=str(m["multiplier"]),
                        offset=str(m["offset"]),
                    )
            after = ET.SubElement(robot, "joint", name=j["name"] + "_after", type="fixed")
            ET.SubElement(after, "parent", link=b)
            ET.SubElement(after, "child", link=j["child"])
            origin(after, post)
    for name, box in workcell.boxes().items():
        name = "workcell_" + name
        link = ET.SubElement(robot, "link", name=name)
        joint = ET.SubElement(robot, "joint", name=name + "_fixed", type="fixed")
        ET.SubElement(joint, "parent", link="world")
        ET.SubElement(joint, "child", link=name)
        t = np.eye(4)
        t[:3, 3] = box["center"]
        origin(joint, t)
        v = ET.SubElement(link, "visual")
        g = ET.SubElement(v, "geometry")
        ET.SubElement(g, "box", size=numbers(box["size"]))
        material = ET.SubElement(v, "material", name=name + "_material")
        ET.SubElement(material, "color", rgba="0.38 0.42 0.46 1")
    ET.indent(robot)
    return ET.tostring(robot, encoding="unicode") + "\n"


def export_description(root, settings, output):
    root, output = Path(root), Path(output)
    if not output.resolve().is_relative_to((root / "local").resolve()):
        raise ValueError("Generated robot description belongs under local/")
    cfg = read_config(root / settings["arm_config"])
    state = RobotState(root, cfg)
    cell = Workcell.load(root / cfg["workcell"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "meshes").mkdir(exist_ok=True)
    sources = [root / cfg[k] for k in ("usd", "arm_model", "hand_model", "workcell")]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    meshes = export_meshes(root / cfg["usd"], state, output / "meshes")
    for name, mimic in [("measured.urdf", False), ("coupled.urdf", True)]:
        (output / name).write_text(build_urdf(state, cell, meshes, mimic=mimic))
    metadata = dict(
        schema="dex_robot_description_v1",
        sources=hashes,
        joint_names=state.names,
        model_joint_names=state.model_names,
        meshes=meshes,
        measured="all 17 supplied positions; no mimic overwrite of simulated follower feedback",
        coupled="nominal USD affine mimic relationships",
        physics_model=False,
    )
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output / "measured.urdf"


def run_mirror(root, settings, seconds, headless, output):
    from isaacsim import SimulationApp

    app = SimulationApp(
        dict(
            headless=headless,
            multi_gpu=False,
            enable_crashreporter=False,
            hide_ui=headless,
            disable_viewport_updates=headless,
        )
    )
    code = 0
    try:
        import rclpy
        import omni.usd
        from pxr import UsdGeom, UsdLux
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.viewports import set_camera_view

        rclpy.init(args=[])
        config = read_config(root / settings["arm_config"])
        state = RobotState(root, config)
        workcell = Workcell.load(root / config["workcell"])
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        add_reference_to_stage(str(root / config["usd"]), "/Robot")
        workcell.create_usd(stage, collision=False)
        UsdLux.DomeLight.Define(stage, "/Light").CreateIntensityAttr(900)
        mirror = UsdStateMirror(stage, state, "/Robot", workcell.world_from_base)
        UsdGeom.Imageable(stage.GetPrimAtPath("/Robot")).MakeInvisible()
        set_camera_view(eye=np.array([1.65, -1.9, 1.25]), target=np.array([0.35, 0, 0.05]))
        node = subscribe_state(settings, state, "robot_state_mirror")
        print(
            f"Mirror: joint-state display on {settings['namespace']}; "
            "feedback sources are listed in diagnostics",
            flush=True,
        )
        started = time.monotonic()
        last_stamp = None
        updates = 0
        fresh_before = None
        last_error = 0.0
        previous_q = None
        maximum_step = 0.0
        try:
            while app.is_running() and (seconds == 0 or time.monotonic() - started < seconds):
                tick = time.monotonic()
                rclpy.spin_once(node, timeout_sec=0.0)
                fresh = state.fresh(settings["state_timeout_s"])
                if fresh and state.stamp_ns != last_stamp:
                    if previous_q is not None:
                        maximum_step = max(
                            maximum_step, float(np.max(np.abs(state.q - previous_q)))
                        )
                    previous_q = state.q.copy()
                    expected = mirror.apply(state.q)
                    UsdGeom.Imageable(stage.GetPrimAtPath("/Robot")).MakeVisible()
                    cache = UsdGeom.XformCache()
                    for name, pose in expected.items():
                        actual = np.asarray(cache.GetLocalToWorldTransform(mirror.prims[name])).T
                        last_error = max(
                            last_error,
                            float(np.max(np.abs(actual - workcell.world_from_base @ pose))),
                        )
                    updates += 1
                    last_stamp = state.stamp_ns
                if fresh != fresh_before:
                    print(
                        "Mirror: joint feedback live"
                        if fresh
                        else "Mirror: feedback stale; holding last received pose",
                        flush=True,
                    )
                    fresh_before = fresh
                app.update()
                time.sleep(max(0.0, 1 / settings["mirror_rate_hz"] - (time.monotonic() - tick)))
        finally:
            report = dict(
                messages=state.accepted,
                rejected=state.rejected,
                usd_updates=updates,
                maximum_link_matrix_error=last_error,
                fresh=state.fresh(settings["state_timeout_s"]),
                maximum_measured_joint_step_rad=maximum_step,
                latest_q_rad=state.q.tolist() if state.q is not None else None,
                physical_simulation=False,
                robot_command_publishing=False,
                object_pose_received=False,
            )
            if output:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
            node.destroy_node()
            rclpy.shutdown()
        if not updates:
            raise RuntimeError("No fresh ROS joint feedback was received")
    except Exception:
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        app.close(exit_code=code)
    return code
