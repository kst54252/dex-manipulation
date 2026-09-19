"""Input validation and explicit named correspondences; no hidden mirroring."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

DEXYCB_SEMANTICS = ("wrist",) + tuple(f"{finger}_{joint}" for finger in ("thumb", "index", "middle", "ring", "little")
                                      for joint in ("mcp", "pip", "dip", "tip"))

# Explicit fields prevent static arrays (e.g. 50 object samples) from being
# accidentally cropped when their size happens to equal a sequence length.
FRAME_FIELDS = frozenset(('frame_ids', 'pose_y', 'joint_3d', 'timestamps_s',
    'wrist_transform', 'wrist_translation_m', 'wrist_quaternion_xyzw',
    'active_q_rad', 'full_q_rad', 'human_keypoints', 'robot_keypoints',
    'object_keypoints', 'object_transform', 'valid', 'transition_valid'))


def frame_mask(frame_ids, frame_range=None):
    ids = np.asarray(frame_ids)
    if ids.ndim != 1 or not len(ids) or np.any(np.diff(ids) <= 0):
        raise ValueError('Frame IDs must increase strictly')
    if frame_range is None:
        return np.ones(len(ids), dtype=bool)
    if (len(frame_range) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in frame_range)
            or frame_range[0] > frame_range[1]):
        raise ValueError('frame_range must be [first_id, last_id], inclusive')
    selected = (ids >= frame_range[0]) & (ids <= frame_range[1])
    if not selected.any():
        raise ValueError('No frames remain inside frame_range')
    return selected


def select_frame_arrays(data, frame_range):
    mask = frame_mask(data['frame_ids'], frame_range)
    out = {key: value.copy() for key, value in data.items()}
    for key in FRAME_FIELDS & data.keys():
        if data[key].shape[0] != len(mask):
            raise ValueError(f'Mismatched per-frame array: {key}')
        out[key] = data[key][mask].copy()
    if 'transition_valid' in out:
        out['transition_valid'][0] = False  # No preceding frame inside this segment.
    return out


@dataclass
class Sequence:
    frame_ids: np.ndarray
    hand: np.ndarray
    object_poses: np.ndarray
    times: np.ndarray | None
    valid: np.ndarray
    metadata: dict


def correspondence(model):
    names = model.semantic_names
    if set(names) != set(DEXYCB_SEMANTICS):
        raise ValueError(f"Semantic mismatch: {set(names) ^ set(DEXYCB_SEMANTICS)}")
    return np.array([DEXYCB_SEMANTICS.index(name) for name in names], dtype=int)


def load_sequence(path, model, config, *, geometric_debug=False):
    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text())
    else:
        config = dict(config)
    required = ("object_index", "object_name", "mesh_to_object", "hand_side", "length_unit", "camera_axes")
    missing = [name for name in required if config.get(name) is None]
    if missing:
        raise ValueError(f"Missing confirmed sequence metadata: {missing}")
    if config["length_unit"] != "m" or config["hand_side"] not in ("left", "right"):
        raise ValueError("Expected metres and explicit left/right hand metadata")
    if not isinstance(config['object_name'], str) or not config['object_name'].strip():
        raise ValueError('An explicit object model name is required')
    alignment = np.asarray(config["mesh_to_object"], dtype=float)
    if alignment.shape != (4, 4) or not np.allclose(alignment[3], [0, 0, 0, 1]):
        raise ValueError("mesh_to_object must be a 4x4 SE(3) matrix")
    if not np.allclose(alignment[:3, :3].T @ alignment[:3, :3], np.eye(3), atol=1e-7) or np.linalg.det(alignment[:3, :3]) < 0:
        raise ValueError("mesh_to_object cannot scale or reflect the object")
    with np.load(path, allow_pickle=False) as archive:
        frame_ids = archive["frame_ids"].copy()
        all_poses = archive["pose_y"].astype(float)
        hand = archive["joint_3d"].astype(float)
        if 'frame_metadata_json' in archive:
            frame = json.loads(str(archive['frame_metadata_json']))
            config['frame_metadata'] = frame
            config['source_camera_axes'] = config['camera_axes']
            config['camera_axes'] = frame['world_axes']
            config['coordinate_frame'] = frame['coordinate_frame']
    if hand.shape != (len(frame_ids), 1, 21, 3) or all_poses.shape[0] != len(frame_ids) or all_poses.shape[2:] != (3, 4):
        raise ValueError("Unexpected sequence dimensions")
    if np.any(np.diff(frame_ids) <= 0):
        raise ValueError("Frame IDs must increase strictly")
    index = config["object_index"]
    if not isinstance(index, int) or not 0 <= index < all_poses.shape[1]:
        raise ValueError("Object index outside pose_y")
    selected = all_poses[:, index]
    poses = np.tile(np.eye(4), (len(frame_ids), 1, 1))
    poses[:, :3] = selected
    poses = poses @ alignment
    hand = hand[:, 0, correspondence(model)]
    valid = np.isfinite(hand).all((1, 2)) & np.isfinite(poses).all((1, 2)) & ~np.all(hand == -1, axis=(1, 2))
    for i, pose in enumerate(poses):
        if valid[i]:
            r = pose[:3, :3]
            valid[i] &= np.allclose(r.T @ r, np.eye(3), atol=1e-4) and abs(np.linalg.det(r) - 1) < 1e-4
    if config.get("timestamps_s") is not None:
        times = np.asarray(config["timestamps_s"], dtype=float)
        if times.shape != frame_ids.shape or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Timestamp count/order invalid")
    elif config.get("retimed_fps") is not None:
        if config.get("time_basis") != "retimed_user_authorized":
            raise ValueError("Retiming requires an explicitly authorized time_basis")
        fps = float(config["retimed_fps"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("retimed_fps must be positive")
        times = (frame_ids - frame_ids[0]) / fps
    elif config.get("fps") is not None:
        fps = float(config["fps"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive")
        times = (frame_ids - frame_ids[0]) / fps
    elif geometric_debug:
        times = None
    else:
        raise ValueError("Actual fps/timestamps or explicitly authorized retiming required for velocity constraints; no default fps is assumed")
    if config["hand_side"] == "left" and config.get("left_hand_policy") != "preserve_source":
        raise ValueError("Left-to-right hand handling must be explicit: choose preserve_source (no mirroring)")
    mask = frame_mask(frame_ids, config.get('frame_range'))
    return Sequence(frame_ids[mask], hand[mask], poses[mask], None if times is None else times[mask], valid[mask], config)
