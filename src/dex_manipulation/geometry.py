"""Persistent mesh surface samples and complete convex collider queries."""
import json
import hashlib
from pathlib import Path
import numpy as np
from .transforms import apply


def sample_surface(vertices, faces, count=50, seed=20260918, candidates=20000):
    """Area-weighted triangle sampling, then deterministic farthest-point thinning.

    Points remain on the actual triangles; no pose-dependent resampling occurs.
    """
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
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
    fields = {k: metadata[k] for k in ('collision_shapes', 'mesh_sha256') if k in metadata}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def generate_object_points(mesh_path, output_path, count=50, seed=20260918):
    with np.load(mesh_path, allow_pickle=False) as mesh:
        points, face_ids, barycentric = sample_surface(mesh["vertices"], mesh["faces"], count, seed=seed)
    np.savez_compressed(output_path, points_local=points, point_ids=np.arange(count),
                        triangle_ids=face_ids, barycentric=barycentric)
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
            object_metadata = json.loads(Path(object_metadata).read_text())
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
            self.objects.append((np.asarray(shape["transform"]), fcl.CollisionObject(fcl.Cylinder(shape["radius"], shape["height"]))))
        if not self.hand or not self.objects:
            raise ValueError("Collision verification requires complete collision geometry")
        self.pair_names = [f"{link}:can_{i}" for link, _ in self.hand for i in range(len(self.objects))]
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
        return np.array([bool(self.fcl.collide(hand, obj, self.fcl.CollisionRequest(), self.fcl.CollisionResult()))
                         for _, hand in self.hand for _, obj in self.objects])
