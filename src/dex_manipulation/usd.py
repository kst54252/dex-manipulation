"""One-time USD extraction. No simulator, source-code dependencies or asset writes."""

from .configuration import read_config
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from .transforms import inverse, transform, apply


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convex_surface(points):
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    hull = ConvexHull(points)
    faces = hull.simplices.copy()
    triangles = points[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    flip = np.einsum("ij,ij->i", normals, hull.equations[:, :3]) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    used, indices = np.unique(faces, return_inverse=True)
    return points[used], indices.reshape(-1, 3)


def extract_hand(stage_path, keypoint_path, output_path):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise ValueError(f"Cannot open USD: {stage_path}")
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    cache = UsdGeom.XformCache()
    raw_keypoints = read_config(Path(keypoint_path))
    raw_keypoints = [raw_keypoints[k] for k in sorted(raw_keypoints, key=int)]
    root_name = next(k["parent_link"] for k in raw_keypoints if k["name"].endswith("_wrist"))
    prims = list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))
    roots = [p for p in prims if p.GetName() == root_name and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(roots) != 1:
        raise ValueError(f"Expected one rigid palm {root_name}; found {len(roots)}")
    root = roots[0]
    root_path = root.GetPath()

    def world(prim):
        matrix = np.array(cache.GetLocalToWorldTransform(prim)).T
        matrix[:3, 3] *= scale
        return matrix

    def frame(joint, side):
        pos = np.asarray(getattr(joint, f"GetLocalPos{side}Attr")().Get(), dtype=float) * scale
        q = getattr(joint, f"GetLocalRot{side}Attr")().Get()
        quat = [*q.GetImaginary(), q.GetReal()]  # USD Gf quat is real + imaginary.
        return transform(Rotation.from_quat(quat).as_matrix(), pos).tolist()

    all_joints = []
    for prim in prims:
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        if joint.GetJointEnabledAttr().Get() is False:
            continue
        a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        # Scope selection only; graph traversal below determines kinematics.
        if (
            len(a) != 1
            or len(b) != 1
            or not a[0].HasPrefix(root_path)
            or not b[0].HasPrefix(root_path)
        ):
            continue
        kind = {
            "PhysicsRevoluteJoint": "revolute",
            "PhysicsPrismaticJoint": "prismatic",
            "PhysicsFixedJoint": "fixed",
        }.get(prim.GetTypeName())
        if kind is None:
            raise ValueError(f"Unsupported joint: {prim.GetPath()} {prim.GetTypeName()}")
        record = dict(
            name=prim.GetName(),
            usd_path=str(prim.GetPath()),
            body0=str(a[0]),
            body1=str(b[0]),
            kind=kind,
            frame0=frame(joint, 0),
            frame1=frame(joint, 1),
            mimic=None,
        )
        if kind != "fixed":
            factor = np.pi / 180 if kind == "revolute" else scale
            record["axis"] = np.eye(3)[
                "XYZ".index(prim.GetAttribute("physics:axis").Get())
            ].tolist()
            for bound in ("lower", "upper"):
                value = prim.GetAttribute(f"physics:{bound}Limit").Get()
                if value is None or not np.isfinite(value):
                    raise ValueError(f"Missing finite {bound} limit: {prim.GetName()}")
                record[bound] = float(value * factor)
            physx_v = prim.GetAttribute("physxJoint:maxJointVelocity").Get()
            urdf_v = prim.GetAttribute("urdf:limit:velocity").Get()
            if physx_v is None and urdf_v is None:
                raise ValueError(f"Missing velocity limit: {prim.GetName()}")
            velocity = float(physx_v * factor) if physx_v is not None else float(urdf_v)
            record.update(
                velocity=velocity,
                velocity_source="physxJoint:maxJointVelocity"
                if physx_v is not None
                else "urdf:limit:velocity",
            )
            record["legacy_urdf_velocity"] = None if urdf_v is None else float(urdf_v)
            record["velocity_overrides_urdf"] = bool(
                urdf_v is not None and not np.isclose(velocity, urdf_v, rtol=1e-4)
            )
            leaders = prim.GetRelationship("newton:mimicJoint").GetTargets()
            if leaders and prim.GetAttribute("newton:mimicEnabled").Get() is not False:
                if len(leaders) != 1:
                    raise ValueError("Multiple mimic leaders")
                coef0 = prim.GetAttribute("newton:mimicCoef0").Get()
                coef1 = prim.GetAttribute("newton:mimicCoef1").Get()
                record["mimic"] = dict(
                    leader=leaders[0].name,
                    multiplier=1.0 if coef1 is None else float(coef1),
                    offset=0.0 if coef0 is None else float(coef0 * factor),
                    schema="NewtonMimicAPI",
                )
            unsupported = [
                a.GetName()
                for a in prim.GetAttributes()
                if "mimic" in a.GetName().lower() and not a.GetName().startswith("newton:")
            ]
            if unsupported:
                raise ValueError(f"Unimplemented coupling schema: {unsupported}")
        all_joints.append(record)

    # Breadth-first traversal, also supports reversed body0/body1 definitions.
    reached = {str(root_path)}
    ordered = []
    remaining = list(all_joints)
    while remaining:
        progress = False
        for joint in remaining[:]:
            a, b = joint["body0"], joint["body1"]
            if a in reached and b in reached:
                raise ValueError("Closed kinematic loop: unsupported")
            if (a in reached) != (b in reached):
                reverse = b in reached
                parent, child = (b, a) if reverse else (a, b)
                joint.update(
                    parent=parent.split("/")[-1], child=child.split("/")[-1], reversed=reverse
                )
                reached.add(child)
                ordered.append(joint)
                remaining.remove(joint)
                progress = True
        if not progress:
            raise ValueError("Disconnected joint graph")
    link_paths = {p.split("/")[-1]: p for p in reached}
    if len(link_paths) != len(reached):
        raise ValueError("Ambiguous link names")
    root_inverse = inverse(world(root))
    zero_links = {
        name: (root_inverse @ world(stage.GetPrimAtPath(path))).tolist()
        for name, path in link_paths.items()
    }
    for name, matrix in zero_links.items():
        r = np.asarray(matrix)[:3, :3]
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or np.linalg.det(r) < 0:
            raise ValueError(f"Scaled/reflected rigid link is unsupported: {name}")
    keypoints = []
    for k in raw_keypoints:
        if k["parent_link"] not in link_paths:
            raise ValueError(f"Unknown keypoint parent: {k}")
        matches = [
            p
            for p in prims
            if p.GetName() == k["name"] and p.GetParent().GetPath() == link_paths[k["parent_link"]]
        ]
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous USD keypoint: {k['name']}")
        local = inverse(world(stage.GetPrimAtPath(link_paths[k["parent_link"]]))) @ world(
            matches[0]
        )
        if not np.allclose(local[:3, 3], k["xyz"], atol=1e-7):
            raise ValueError(f"JSON local coordinates disagree with USD: {k['name']}")
        keypoints.append(
            dict(
                name=k["name"],
                semantic=re.sub(r"^kp_\d+_", "", k["name"]),
                link=k["parent_link"],
                xyz=k["xyz"],
            )
        )

    colliders = []
    for prim in prims:
        if not prim.GetPath().HasPrefix(root_path) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        parent = prim
        while parent and str(parent.GetPath()) not in reached:
            parent = parent.GetParent()
        if not parent:
            raise ValueError(f"Unattached collider: {prim.GetPath()}")
        approximation = prim.GetAttribute("physics:approximation").Get()
        if not prim.IsA(UsdGeom.Mesh) or approximation != "convexHull":
            raise ValueError(f"Unsupported collider: {prim.GetPath()} {approximation}")
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float) * scale
        local = inverse(world(parent)) @ world(prim)
        points = apply(local, points)
        vertices, faces = convex_surface(points)
        colliders.append(
            dict(
                link=parent.GetName(),
                usd_path=str(prim.GetPath()),
                approximation=approximation,
                vertices=vertices.tolist(),
                faces=faces.tolist(),
                original_vertex_count=len(points),
            )
        )
    if not colliders:
        raise ValueError("No collision geometry extracted")
    description = dict(
        schema_version=1,
        source=str(Path(stage_path).resolve()),
        source_sha256=file_hash(stage_path),
        keypoint_sha256=file_hash(keypoint_path),
        length_unit="m",
        angle_unit="rad",
        quaternion_order="xyzw",
        usd_meters_per_unit=scale,
        usd_up_axis=str(UsdGeom.GetStageUpAxis(stage)),
        root_link=root_name,
        root_usd_path=str(root_path),
        link_paths=link_paths,
        joints=ordered,
        keypoints=keypoints,
        colliders=colliders,
        authored_zero_link_transforms=zero_links,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(description, indent=2) + "\n")
    return description


def extract_object(mesh_stage_path, collision_stage_path, output_path):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(mesh_stage_path))
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    cache = UsdGeom.XformCache()
    root_inverse = np.linalg.inv(np.array(cache.GetLocalToWorldTransform(stage.GetDefaultPrim())).T)
    vertices, faces, offset = [], [], 0
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        matrix = root_inverse @ np.array(cache.GetLocalToWorldTransform(prim)).T
        points = apply(matrix, np.asarray(mesh.GetPointsAttr().Get())) * scale
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        cursor = 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            polygon = indices[cursor : cursor + count]
            for k in range(1, count - 1):
                faces.append([offset + polygon[0], offset + polygon[k], offset + polygon[k + 1]])
            cursor += count
        vertices.extend(points)
        offset += len(points)
    if not vertices:
        raise ValueError("No can mesh found; dimensions and alignment must be supplied")
    collision_stage = Usd.Stage.Open(str(collision_stage_path))
    collision_scale = UsdGeom.GetStageMetersPerUnit(collision_stage)
    cache = UsdGeom.XformCache()
    root_inverse = np.linalg.inv(
        np.array(cache.GetLocalToWorldTransform(collision_stage.GetDefaultPrim())).T
    )
    shapes = []
    for prim in collision_stage.Traverse():
        if (
            not prim.HasAPI(UsdPhysics.CollisionAPI)
            or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False
        ):
            continue
        if not prim.IsA(UsdGeom.Cylinder):
            raise ValueError(f"Unsupported can collision geometry: {prim.GetPath()}")
        cylinder = UsdGeom.Cylinder(prim)
        local = root_inverse @ np.array(cache.GetLocalToWorldTransform(prim)).T
        local[:3, 3] *= collision_scale
        if not np.allclose(local[:3, :3].T @ local[:3, :3], np.eye(3), atol=1e-7):
            raise ValueError("Scaled cylinder unsupported")
        axis = cylinder.GetAxisAttr().Get()
        axis_rotation = {
            "X": Rotation.from_rotvec([0, np.pi / 2, 0]).as_matrix(),
            "Y": Rotation.from_rotvec([-np.pi / 2, 0, 0]).as_matrix(),
            "Z": np.eye(3),
        }[axis]
        local = local @ transform(axis_rotation)
        shapes.append(
            dict(
                type="cylinder",
                radius=cylinder.GetRadiusAttr().Get() * collision_scale,
                height=cylinder.GetHeightAttr().Get() * collision_scale,
                transform=local.tolist(),
                usd_path=str(prim.GetPath()),
            )
        )
    if not shapes:
        raise ValueError("No object collision geometry")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, vertices=np.asarray(vertices), faces=np.asarray(faces, dtype=np.int32)
    )
    metadata = dict(
        mesh_source=str(mesh_stage_path),
        collision_source=str(collision_stage_path),
        units="m",
        frame="mesh stage default prim; alignment with dataset must be confirmed",
        stage_up_axis=str(UsdGeom.GetStageUpAxis(stage)),
        collision_shapes=shapes,
        bounds_m=[np.min(vertices, axis=0).tolist(), np.max(vertices, axis=0).tolist()],
    )
    import hashlib
    from .geometry import geometry_fingerprint

    metadata.update(
        mesh_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest(),
        collision_source_sha256=hashlib.sha256(Path(collision_stage_path).read_bytes()).hexdigest(),
    )
    metadata["fingerprint"] = geometry_fingerprint(metadata)
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def extract_arm(stage_path, base_path, flange_path, wrist_path, output_path):
    """Extract the actual composed arm/mount chain, including Xform joint bodies.

    Isaac's assembler can connect named Xform attachments rather than rigid-body
    prims. Resolve each attachment to its rigid ancestor and compose both frames.
    Frame paths are explicit inputs; no inferred physical flange-face offset.
    """
    from collections import deque
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise ValueError(f"Cannot open {stage_path}")
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    cache = UsdGeom.XformCache()

    def world(prim):
        matrix = np.array(cache.GetLocalToWorldTransform(prim)).T
        matrix[:3, 3] *= scale
        if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6):
            raise ValueError(f"Scaled rigid frame: {prim.GetPath()}")
        return matrix

    def rigid(path):
        prim = stage.GetPrimAtPath(path)
        while prim and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim = prim.GetParent()
        return prim

    for path in (base_path, flange_path, wrist_path):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"Select an actual rigid link, not a mesh: {path}")
    adjacency = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        if joint.GetJointEnabledAttr().Get() is False:
            continue
        targets = [getattr(joint, f"GetBody{i}Rel")().GetTargets() for i in (0, 1)]
        if any(len(t) != 1 for t in targets):
            continue
        bodies = [rigid(t[0]) for t in targets]
        if not all(bodies):  # World anchoring is outside this base-relative chain.
            continue
        paths = [str(p.GetPath()) for p in bodies]
        if paths[0] == paths[1]:
            continue
        record = dict(
            name=prim.GetName(),
            usd_path=str(prim.GetPath()),
            mimic=None,
            attachment_paths=[str(t[0]) for t in targets],
            body0=paths[0],
            body1=paths[1],
        )
        for i in (0, 1):
            quat = getattr(joint, f"GetLocalRot{i}Attr")().Get()
            local = transform(
                Rotation.from_quat([*quat.GetImaginary(), quat.GetReal()]).as_matrix(),
                np.asarray(getattr(joint, f"GetLocalPos{i}Attr")().Get()) * scale,
            )
            attachment = stage.GetPrimAtPath(targets[i][0])
            record[f"frame{i}"] = (inverse(world(bodies[i])) @ world(attachment) @ local).tolist()
        if prim.IsA(UsdPhysics.FixedJoint):
            record["kind"] = "fixed"
        elif prim.IsA(UsdPhysics.RevoluteJoint):
            record["kind"] = "revolute"
            record["axis"] = np.eye(3)[
                "XYZ".index(prim.GetAttribute("physics:axis").Get())
            ].tolist()
            for name in ("lower", "upper"):
                value = prim.GetAttribute(f"physics:{name}Limit").Get()
                if value is None or not np.isfinite(value):
                    raise ValueError(f"Missing finite {name} limit: {prim.GetPath()}")
                record[name] = float(np.deg2rad(value))
            value = prim.GetAttribute("physxJoint:maxJointVelocity").Get()
            source = "physxJoint:maxJointVelocity"
            if value is not None:
                value = np.deg2rad(value)
            else:
                value = prim.GetAttribute("urdf:limit:velocity").Get()
                source = "urdf:limit:velocity"
            if value is None or not np.isfinite(value) or value <= 0:
                raise ValueError(f"Missing velocity limit: {prim.GetPath()}")
            record.update(velocity=float(value), velocity_source=source)
            record["has_mimic"] = any("mimic" in name.lower() for name in prim.GetAppliedSchemas())
        else:
            record["kind"] = "unsupported"
        for a, b, reverse in ((paths[0], paths[1], False), (paths[1], paths[0], True)):
            adjacency.setdefault(a, []).append((b, record, reverse))
    queue = deque([(base_path, [])])
    seen = {base_path}
    chain = None
    while queue:
        parent, prefix = queue.popleft()
        if parent == wrist_path:
            chain = prefix
            break
        for child, record, reverse in adjacency.get(parent, []):
            if child in seen:
                continue
            seen.add(child)
            queue.append(
                (
                    child,
                    prefix
                    + [
                        dict(
                            record,
                            parent=parent.split("/")[-1],
                            child=child.split("/")[-1],
                            reversed=reverse,
                        )
                    ],
                )
            )
    if not chain:
        raise ValueError("No connected joint chain between selected base and wrist")
    if any(j["kind"] not in ("revolute", "fixed") or j.get("has_mimic") for j in chain):
        raise ValueError("Unsupported joint/coupling in selected arm chain")
    full_paths = {base_path}
    for joint in chain:
        full_paths.update([joint["body0"], joint["body1"]])
    if flange_path not in full_paths:
        raise ValueError("Selected flange is not on the wrist chain")
    # Flange-to-palm must consist only of fixed joints.
    flange_name = flange_path.split("/")[-1]
    after_flange = False
    mount = np.eye(4)
    for j in chain:
        if j["parent"] == flange_name:
            after_flange = True
        if after_flange:
            if j["kind"] != "fixed":
                raise ValueError("A moving joint separates flange and wrist")
            relative = np.array(j["frame0"]) @ inverse(np.array(j["frame1"]))
            mount = mount @ (inverse(relative) if j["reversed"] else relative)
    if not after_flange:
        raise ValueError("Missing explicit fixed flange-to-wrist mount")
    observed_mount = inverse(world(stage.GetPrimAtPath(flange_path))) @ world(
        stage.GetPrimAtPath(wrist_path)
    )
    if not np.allclose(mount, observed_mount, atol=2e-6):
        raise ValueError("Authored wrist transform disagrees with assembler joint frames")
    link_paths = {p.split("/")[-1]: p for p in full_paths}
    if len(link_paths) != len(full_paths):
        raise ValueError("Ambiguous rigid-link names")
    root_inv = inverse(world(stage.GetPrimAtPath(base_path)))
    description = dict(
        schema_version=1,
        source=str(Path(stage_path).resolve()),
        source_sha256=file_hash(stage_path),
        layer_sha256={
            str(Path(layer.realPath).resolve()): file_hash(layer.realPath)
            for layer in stage.GetUsedLayers()
            if layer.realPath
        },
        length_unit="m",
        angle_unit="rad",
        quaternion_order="xyzw",
        usd_meters_per_unit=scale,
        usd_up_axis=str(UsdGeom.GetStageUpAxis(stage)),
        root_link=base_path.split("/")[-1],
        root_usd_path=base_path,
        flange_link=flange_name,
        flange_usd_path=flange_path,
        wrist_link=wrist_path.split("/")[-1],
        wrist_usd_path=wrist_path,
        flange_frame_definition="selected USD last arm rigid-link frame; no inferred flange face/TCP",
        flange_to_wrist=mount.tolist(),
        joints=chain,
        keypoints=[],
        colliders=[],
        link_paths=link_paths,
        base_in_stage=world(stage.GetPrimAtPath(base_path)).tolist(),
        authored_zero_link_transforms={
            name: (root_inv @ world(stage.GetPrimAtPath(path))).tolist()
            for name, path in link_paths.items()
        },
    )
    from .fk import ArmModel

    model = ArmModel(description)
    fk = model.link_transforms(np.zeros(6))
    zero_error = max(
        np.max(np.abs(fk[name] - pose))
        for name, pose in description["authored_zero_link_transforms"].items()
    )
    if zero_error > 2e-6:
        raise ValueError(f"Zero FK disagrees with USD by {zero_error}")
    description["zero_fk_usd_max_matrix_error"] = float(zero_error)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(description, indent=2) + "\n")
    return description
