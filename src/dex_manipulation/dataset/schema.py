"""Small, explicit interchange contracts. Transforms map local points to parent."""

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from ..data import DEXYCB_SEMANTICS

SCHEMA = 1
HAND_NAMES = DEXYCB_SEMANTICS
CAMERA_AXES = "x_right_y_down_z_forward"
WORLD_AXES = "x_right_y_forward_z_up"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    atomic_write(
        path,
        lambda p: p.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        ),
    )


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as f:
        temporary = Path(f.name)
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_npz(path, **arrays):
    atomic_write(path, lambda p: np.savez_compressed(p, **arrays))


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata(archive):
    value = json.loads(str(archive["metadata_json"]))
    if value.get("schema") != SCHEMA or value.get("length_unit") != "m":
        raise ValueError("Annotations require schema=1 and length_unit=m")
    return value


def timeline(frame_ids, timestamps_s):
    ids, times = np.asarray(frame_ids), np.asarray(timestamps_s, dtype=float)
    if (
        ids.ndim != 1
        or not len(ids)
        or not np.issubdtype(ids.dtype, np.integer)
        or np.any(np.diff(ids) <= 0)
    ):
        raise ValueError("frame_ids must be nonempty, strictly increasing integers")
    if times.shape != ids.shape or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps_s must be finite, strictly increasing and match frame_ids")
    return ids, times


def same_frames(archive, ids):
    if not np.array_equal(archive["frame_ids"], ids):
        raise ValueError("Frame IDs differ; implicit index alignment is forbidden")


def transforms(value, name="transform", valid=None):
    array = np.asarray(value, dtype=float)
    if array.shape[-2:] != (4, 4):
        raise ValueError(f"{name}: expected (...,4,4)")
    selected = array if valid is None else array[np.asarray(valid, dtype=bool)]
    if (
        not np.isfinite(selected).all()
        or not np.allclose(selected[..., 3, :], [0, 0, 0, 1], atol=1e-7)
        or not np.allclose(
            np.swapaxes(selected[..., :3, :3], -1, -2) @ selected[..., :3, :3], np.eye(3), atol=1e-5
        )
        or not np.allclose(np.linalg.det(selected[..., :3, :3]), 1, atol=1e-5)
    ):
        raise ValueError(f"{name}: expected finite SE(3), no scaling/reflection")
    return array


def transform_points(pose, points):
    return np.einsum("...ij,...vj->...vi", pose[..., :3, :3], points) + pose[..., None, :3, 3]


def bool_mask(value, shape, name):
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.shape != shape:
        raise ValueError(f"{name} must be a boolean array with shape {shape}")
    return array


def faces(value, vertex_count):
    array = np.asarray(value)
    if (
        array.ndim != 2
        or array.shape[1] != 3
        or not len(array)
        or not np.issubdtype(array.dtype, np.integer)
        or array.min() < 0
        or array.max() >= vertex_count
    ):
        raise ValueError("Invalid triangular mesh topology")
    return array


def edges(triangles, bidirectional=True):
    pairs = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    if bidirectional:
        pairs = np.concatenate([pairs, pairs[:, ::-1]])
    return np.unique(pairs, axis=0)


def new_directory(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing output: {path}")
    path.mkdir(parents=True)
    return path
