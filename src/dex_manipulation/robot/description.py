"""Generate RViz URDF/STL from the current USD and extracted joint frames.

Display models are derived artifacts under local/. Source assets are read-only.
Two fixed helper frames around each moving joint preserve both USD joint frames.
"""

from pathlib import Path
import hashlib
import json
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from ..configuration import read_config
from ..ros_state import RobotState
from ..scene import Workcell
from ..transforms import inverse, apply


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
