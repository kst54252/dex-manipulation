"""Dimensioned can asset generation; object-local SI geometry, without Isaac Sim."""

from .configuration import read_config
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .geometry import generate_object_points
from .usd import extract_object


def stepped_can(spec):
    """Closed exterior mesh and two exact cylinders, including the exposed annulus."""
    r, R = spec["body_diameter_m"] / 2, spec["base_diameter_m"] / 2
    h, b = spec["total_height_m"], spec["base_height_m"]
    n = spec["radial_segments"]
    center = np.asarray(spec["center_in_object_m"], dtype=float)
    if (
        spec["length_unit"] != "m"
        or spec["axis"] != "+Z"
        or center.shape != (3,)
        or not np.isfinite([r, R, h, b, spec["mass_kg"], *center]).all()
        or not 0 < r <= R
        or not 0 < b < h
        or spec["mass_kg"] <= 0
        or not isinstance(n, int)
        or isinstance(n, bool)
        or n < 16
    ):
        raise ValueError("Invalid stepped can dimensions/frame")
    z0 = -h / 2
    angles = np.arange(n) * 2 * np.pi / n
    rings = [(R, z0), (R, z0 + b), (r, z0 + b), (r, h / 2)]
    vertices = (
        np.vstack(
            [
                np.c_[radius * np.cos(angles), radius * np.sin(angles), np.full(n, z)]
                for radius, z in rings
            ]
            + [np.array([[0, 0, z0], [0, 0, h / 2]])]
        )
        + center
    )
    faces = []
    for ring in range(3):
        for j in range(n):
            k = (j + 1) % n
            a, c, d, e = ring * n + j, ring * n + k, (ring + 1) * n + k, (ring + 1) * n + j
            faces.extend([[a, c, d], [a, d, e]])
    for j in range(n):
        k = (j + 1) % n
        faces.extend([[4 * n, k, j], [4 * n + 1, 3 * n + j, 3 * n + k]])
    shapes = []
    for name, radius, height, z in [("Base", R, b, z0 + b / 2), ("Body", r, h - b, b / 2)]:
        local = np.eye(4)
        local[:3, 3] = center + [0, 0, z]
        shapes.append(
            dict(name=name, type="cylinder", radius=radius, height=height, transform=local.tolist())
        )
    volumes = np.array([np.pi * s["radius"] ** 2 * s["height"] for s in shapes])
    masses = spec["mass_kg"] * volumes / volumes.sum()
    centers = np.array([np.array(s["transform"])[:3, 3] for s in shapes])
    com = np.sum(masses[:, None] * centers, axis=0) / masses.sum()
    inertia = np.zeros(3)
    for mass, shape, position in zip(masses, shapes, centers):
        transverse = mass * (3 * shape["radius"] ** 2 + shape["height"] ** 2) / 12
        local = np.array([transverse, transverse, mass * shape["radius"] ** 2 / 2])
        delta = position - com
        inertia += local + mass * (np.dot(delta, delta) - delta**2)
    return vertices, np.array(faces, dtype=np.int32), shapes, com, inertia


def build_can(spec_path, asset_path, model_dir):
    """Author a standalone USD; extract the actual USD before surface sampling."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

    spec_path, asset_path, model_dir = map(Path, (spec_path, asset_path, model_dir))
    spec = read_config(spec_path)
    vertices, faces, shapes, com, inertia = stepped_can(spec)
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(asset_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = UsdGeom.Xform.Define(stage, "/Can").GetPrim()
    stage.SetDefaultPrim(root)
    root.SetCustomDataByKey("shape_spec_json", json.dumps(spec, sort_keys=True))
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(spec["mass_kg"])
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*com))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0))
    mesh = UsdGeom.Mesh.Define(stage, "/Can/Visual")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(False)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.55, 0.59)])
    for shape in shapes:
        cylinder = UsdGeom.Cylinder.Define(stage, "/Can/Collision" + shape["name"])
        cylinder.CreateAxisAttr("Z")
        cylinder.CreateRadiusAttr(shape["radius"])
        cylinder.CreateHeightAttr(shape["height"])
        cylinder.AddTranslateOp().Set(Gf.Vec3d(*np.array(shape["transform"])[:3, 3]))
        cylinder.CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    stage.GetRootLayer().Save()
    model_dir.mkdir(parents=True, exist_ok=True)
    metadata = extract_object(asset_path, asset_path, model_dir / "can_mesh.npz")
    metadata.update(
        name=spec["name"],
        specification=spec,
        frame="dataset object pose frame; original geometric center and +Z axis preserved",
        center_of_mass_m=com.tolist(),
        diagonal_inertia_kg_m2=inertia.tolist(),
        mesh_sha256=hashlib.sha256((model_dir / "can_mesh.npz").read_bytes()).hexdigest(),
        collision_source_sha256=hashlib.sha256(asset_path.read_bytes()).hexdigest(),
    )
    from .geometry import geometry_fingerprint

    metadata["fingerprint"] = geometry_fingerprint(metadata)
    (model_dir / "can_mesh.json").write_text(json.dumps(metadata, indent=2) + "\n")
    points = generate_object_points(
        model_dir / "can_mesh.npz",
        model_dir / "can_points.npz",
        count=spec["surface_point_count"],
        seed=spec["surface_sampling_seed"],
    )
    overlay = Usd.Stage.CreateNew(str(asset_path.with_name("keypoints.usda")))
    UsdGeom.SetStageMetersPerUnit(overlay, 1.0)
    UsdGeom.SetStageUpAxis(overlay, "Z")
    overlay_root = UsdGeom.Xform.Define(overlay, "/Can").GetPrim()
    overlay_root.GetReferences().AddReference(os.path.relpath(asset_path, asset_path.parent))
    overlay.SetDefaultPrim(overlay_root)
    markers = UsdGeom.Points.Define(overlay, "/Can/Keypoints")
    markers.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    markers.CreateWidthsAttr([0.0015] * len(points))
    markers.CreateIdsAttr(list(range(len(points))))
    markers.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.75, 0.05)])
    overlay.GetRootLayer().Save()
    np.savetxt(
        asset_path.with_name("keypoints.csv"),
        np.c_[np.arange(len(points)), points],
        delimiter=",",
        header="index,x_m,y_m,z_m",
        comments="",
        fmt=["%d", "%.12g", "%.12g", "%.12g"],
    )
    return metadata
