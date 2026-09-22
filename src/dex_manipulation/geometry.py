"""Persistent mesh surface samples and complete convex collider queries."""

from .configuration import read_config
import json
import hashlib
from pathlib import Path
import numpy as np


def can_orientation_report(poses, geometry):
    """Inspect the actual stepped-can collider centers in a Z-up frame."""
    parts = {
        s.get("name", s.get("usd_path", "").rsplit("/", 1)[-1]).removeprefix("Collision"): s
        for s in geometry["collision_shapes"]
    }
    if not {"Base", "Body"} <= parts.keys():
        raise ValueError("Base/body collider identities are required to verify can orientation")
    base, body = parts["Base"], parts["Body"]
    if base["radius"] <= body["radius"] or base["height"] >= body["height"]:
        raise ValueError("Expected the explicitly dimensioned wider, shorter can base")
    poses = np.asarray(poses, dtype=float).reshape(-1, 4, 4)
    if not np.isfinite(poses).all():
        raise ValueError("Nonfinite can poses")
    centers = np.array([np.asarray(s["transform"])[:3, 3] for s in (base, body)])
    z = np.einsum("tj,kj->tk", poses[:, 2, :3], centers) + poses[:, 2, 3, None]
    direction = centers[1] - centers[0]
    up = poses[:, 2, :3] @ (direction / np.linalg.norm(direction))
    return dict(
        initial_base_center_z_m=float(z[0, 0]),
        initial_body_center_z_m=float(z[0, 1]),
        initial_base_below_body=bool(up[0] > 1e-8),
        base_below_body_count=int(np.sum(up > 1e-8)),
        frame_count=len(poses),
        base_not_below_body_indices=np.flatnonzero(up <= 1e-8).tolist(),
        base_to_body_world_up_cosine=up.tolist(),
    )


def can_base_down_alignment(initial_world_pose, geometry):
    """Turn the replacement can around its confirmed center, not the world/hand.

    This is one fixed right-multiplied asset-to-dataset-pose transform. It changes
    the asymmetric object's physical orientation, requiring new retargeting and
    training; it is not a visual-only correction.
    """
    spec = geometry["specification"]
    if spec["axis"] != "+Z":
        raise ValueError("Expected a confirmed object-local +Z can axis")
    center = np.asarray(spec["center_in_object_m"], dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Confirmed can geometric center is required")
    before = can_orientation_report(initial_world_pose, geometry)
    if abs(before["base_to_body_world_up_cosine"][0]) < 0.1:
        raise ValueError(
            "Initial can axis is nearly horizontal; specify the physical alignment explicitly"
        )
    alignment = np.eye(4)
    if not before["initial_base_below_body"]:
        alignment[:3, :3] = np.diag([1.0, -1.0, -1.0])
        alignment[:3, 3] = center - alignment[:3, :3] @ center
    after = can_orientation_report(np.asarray(initial_world_pose) @ alignment, geometry)
    if not after["initial_base_below_body"]:
        raise ValueError("Can alignment did not put the wider base below the body")
    return alignment, dict(
        before=before,
        after=after,
        flipped_about_geometric_center=not before["initial_base_below_body"],
        geometric_center_object_m=center.tolist(),
        method="one fixed asset-to-dataset-pose SE(3); no world/hand flip",
    )


def sample_surface(vertices, faces, count=50, seed=20260918, candidates=20000):
    """Area-weighted triangle sampling, then deterministic farthest-point thinning.

    Points remain on the actual triangles; no pose-dependent resampling occurs.
    """
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    triangles = vertices[faces]
    areas = (
        np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
        )
        / 2
    )
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError("Degenerate object mesh")
    rng = np.random.default_rng(seed)
    face_ids = rng.choice(len(faces), size=max(candidates, count), p=areas / areas.sum())
    uv = rng.random((len(face_ids), 2))
    s = np.sqrt(uv[:, 0])
    barycentric = np.column_stack((1 - s, s * (1 - uv[:, 1]), s * uv[:, 1]))
    cloud = np.einsum("ni,nij->nj", barycentric, triangles[face_ids])
    selected = [int(np.argmax(np.linalg.norm(cloud - cloud.mean(0), axis=1)))]
    distances = np.full(len(cloud), np.inf)
    for _ in range(count - 1):
        distances = np.minimum(distances, np.sum((cloud - cloud[selected[-1]]) ** 2, axis=1))
        selected.append(int(np.argmax(distances)))
    return cloud[selected], face_ids[selected], barycentric[selected]


def geometry_fingerprint(metadata):
    """Content contract for collision geometry and the sampled visual surface."""
    fields = {k: metadata[k] for k in ("collision_shapes", "mesh_sha256") if k in metadata}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def generate_object_points(mesh_path, output_path, count=50, seed=20260918):
    with np.load(mesh_path, allow_pickle=False) as mesh:
        points, face_ids, barycentric = sample_surface(
            mesh["vertices"], mesh["faces"], count, seed=seed
        )
    np.savez_compressed(
        output_path,
        points_local=points,
        point_ids=np.arange(count),
        triangle_ids=face_ids,
        barycentric=barycentric,
    )
    return points


class CollisionScene:
    """All extracted hand convex collision meshes against each can collider.

    FCL GJK/EPA checks geometry, not the semantic points. Checks are at the
    evaluated configuration, not a continuous-motion collision certificate.
    The solver queries the full shapes at endpoints and interpolation substeps.
    """

    def __init__(self, model, object_metadata):
        import fcl

        self.fcl = fcl
        self.model = model
        if isinstance(object_metadata, (str, Path)):
            object_metadata = read_config(Path(object_metadata))
        self.fingerprint = geometry_fingerprint(object_metadata)
        self.hand = []
        for collider in model.colliders:
            vertices = np.array(collider["vertices"], dtype=np.float64)
            faces = np.array(collider["faces"], dtype=np.int32)
            polygons = np.column_stack([np.full(len(faces), 3), faces]).ravel()
            geometry = fcl.Convex(vertices, len(faces), polygons)
            self.hand.append((collider["link"], fcl.CollisionObject(geometry)))
        self.objects = []
        for shape in object_metadata["collision_shapes"]:
            if shape["type"] != "cylinder":
                raise ValueError("Unsupported object collision primitive")
            self.objects.append(
                (
                    np.asarray(shape["transform"]),
                    fcl.CollisionObject(fcl.Cylinder(shape["radius"], shape["height"])),
                )
            )
        if not self.hand or not self.objects:
            raise ValueError("Collision verification requires complete collision geometry")
        self.pair_names = [
            f"{link}:can_{i}" for link, _ in self.hand for i in range(len(self.objects))
        ]
        self.request = fcl.DistanceRequest(enable_signed_distance=True, enable_nearest_points=True)

    def distances(self, links, object_transform, *, with_witnesses=False):
        fcl = self.fcl
        for local, obj in self.objects:
            matrix = object_transform @ local
            obj.setTransform(fcl.Transform(matrix[:3, :3], matrix[:3, 3]))
        values, witnesses = [], []
        for link, hand in self.hand:
            matrix = links[link]
            hand.setTransform(fcl.Transform(matrix[:3, :3], matrix[:3, 3]))
            for _, obj in self.objects:
                result = fcl.DistanceResult()
                distance = fcl.distance(hand, obj, self.request, result)
                if not np.isfinite(distance):
                    raise RuntimeError("Non-finite collision distance")
                values.append(distance)
                if with_witnesses:
                    witnesses.append(result.nearest_points)
        if with_witnesses:
            return np.array(values), np.asarray(witnesses)
        return np.array(values)

    def collision_flags(self, links, object_transform):
        """Separate collide query used for post-solve verification."""
        self.distances(links, object_transform)
        return np.array(
            [
                bool(
                    self.fcl.collide(
                        hand, obj, self.fcl.CollisionRequest(), self.fcl.CollisionResult()
                    )
                )
                for _, hand in self.hand
                for _, obj in self.objects
            ]
        )
