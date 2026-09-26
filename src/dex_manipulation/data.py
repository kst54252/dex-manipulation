"""Input validation and explicit named correspondences; no hidden mirroring."""

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

CONFIG_PATH_ALIASES = {
    "config/policy.json": "config/policy_demo2.json",
    "config/ik.json": "config/ik_demo2.json",
    "config/retargeting.json": "config/retargeting_demo2.json",
    "config/arm_ground_frame.json": "config/arm_placement_demo2.json",
    "config/policy_demo2_grasp.json": "config/policy_demo2_contact_only.json",
}

# Task-specific settings moved together; historical snapshots keep their strings.
TASK_CONFIG_PATH_ALIASES = {
    f"config/{name}.json": f"config/tasks/can_pick/{name}.json"
    for name in (
        "play",
        "policy_demo2",
        "policy_demo2_contact",
        "policy_demo2_contact_only",
        "policy_arm",
        "ik_demo2",
        "retargeting_demo2",
        "arm_placement_demo2",
    )
}


def resolve_demo_path(path, root=None):
    """Resolve historical demo/config paths without rewriting saved provenance.

    New can-pick callers use data/can_grasping/demo2 (27 poses).
    Older checkpoint configurations
    and immutable NPZ provenance retain their original path strings.
    """
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    from .portable import relocate_path

    path = relocate_path(path, root)
    candidate = path if path.is_absolute() else root / path
    if candidate.exists():
        return candidate
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return candidate
    aliases = {
        **{old: TASK_CONFIG_PATH_ALIASES.get(new, new) for old, new in CONFIG_PATH_ALIASES.items()},
        **TASK_CONFIG_PATH_ALIASES,
        "data/demo2": "data/can_grasping/demo2",
        "data/current": "data/can_grasping/demo2",
        "data/raw/839512060362": "data/can_grasping/demo2/raw",
        "data/poses/839512060362.npz": "data/can_grasping/demo2/poses.npz",
        "data/reference/839512060362.npz": "data/can_grasping/demo2/retargeted.npz",
        "data/grounded": "data/can_grasping/demo2/grounded",
        "local/policy/reference.npz": "data/can_grasping/demo2/policy_reference.npz",
    }
    for old, new in aliases.items():
        if relative.is_relative_to(old):
            return root / new / relative.relative_to(old)
    return candidate


def legacy_demo_paths(value):
    """Stable checkpoint-hash spelling for renamed data and configuration files.

    The retained can-pick demo2 is the historical "current" demo.
    Only these exact path prefixes change. Reference bytes/hashes, model,
    controller and every other contract field remain checked as before.
    Do not use this representation for opening files or user-facing paths.
    """
    if isinstance(value, dict):
        return {key: legacy_demo_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [legacy_demo_paths(item) for item in value]
    if isinstance(value, str):
        root = str(Path(__file__).resolve().parents[2]) + "/"
        for prefix in ("", root):
            for old, new in TASK_CONFIG_PATH_ALIASES.items():
                if value == prefix + new:
                    value = prefix + old
                    break
            for old, new in CONFIG_PATH_ALIASES.items():
                if value == prefix + new:
                    return prefix + old
            for new, old in (
                ("data/can_grasping/demo2", "data/current"),
                ("data/demo2", "data/current"),
            ):
                if value == prefix + new or value.startswith(prefix + new + "/"):
                    return prefix + old + value[len(prefix + new) :]
    return value


DEXYCB_SEMANTICS = ("wrist",) + tuple(
    f"{finger}_{joint}"
    for finger in ("thumb", "index", "middle", "ring", "little")
    for joint in ("mcp", "pip", "dip", "tip")
)

# Explicit fields prevent static arrays (e.g. 50 object samples) from being
# accidentally cropped when their size happens to equal a sequence length.
FRAME_FIELDS = frozenset(
    (
        "frame_ids",
        "pose_y",
        "joint_3d",
        "timestamps_s",
        "wrist_transform",
        "wrist_translation_m",
        "wrist_quaternion_xyzw",
        "active_q_rad",
        "full_q_rad",
        "human_keypoints",
        "robot_keypoints",
        "object_keypoints",
        "object_transform",
        "valid",
        "transition_valid",
    )
)


def frame_mask(frame_ids, frame_range=None):
    ids = np.asarray(frame_ids)
    if ids.ndim != 1 or not len(ids) or np.any(np.diff(ids) <= 0):
        raise ValueError("Frame IDs must increase strictly")
    if frame_range is None:
        return np.ones(len(ids), dtype=bool)
    if (
        len(frame_range) != 2
        or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in frame_range)
        or frame_range[0] > frame_range[1]
    ):
        raise ValueError("frame_range must be [first_id, last_id], inclusive")
    selected = (ids >= frame_range[0]) & (ids <= frame_range[1])
    if not selected.any():
        raise ValueError("No frames remain inside frame_range")
    return selected


def select_frame_arrays(data, frame_range):
    mask = frame_mask(data["frame_ids"], frame_range)
    out = {key: value.copy() for key, value in data.items()}
    for key in FRAME_FIELDS & data.keys():
        if data[key].shape[0] != len(mask):
            raise ValueError(f"Mismatched per-frame array: {key}")
        out[key] = data[key][mask].copy()
    if "transition_valid" in out:
        out["transition_valid"][0] = False  # No preceding frame inside this segment.
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


def import_human_demo(source, output, geometry_path):
    """Import explicit MANO21/object annotations, never a robot reference.

    Supports the source project's preprocessed DexYCB interchange metadata.
    Already converted right-hand landmarks are validated against the declared
    object-local reflection; no second reflection or robot-index remap occurs.
    The source fps is a declared playback clock, not a measured camera rate.
    """
    import hashlib
    from scipy.spatial.transform import Rotation

    source, output, geometry_path = map(Path, (source, output, geometry_path))
    with np.load(source, allow_pickle=False) as data:
        values = {
            k: data[k].copy()
            for k in (
                "source_frame_indices",
                "mano_joint_coords_right_mano21",
                "mano_joint_coords_original",
                "object_pos",
                "object_quat",
                "quaternion_order",
                "fps",
                "sequence_name",
                "source_camera_serial",
                "source_hand_side",
                "target_hand_side",
                "hand_conversion",
                "mano_source_joint_order",
                "source_grasped_ycb_id",
                "grasped_object_name",
            )
        }
    ids = values["source_frame_indices"]
    frame_mask(ids)
    hand = np.asarray(values["mano_joint_coords_right_mano21"], dtype=float)
    original = np.asarray(values["mano_joint_coords_original"], dtype=float)
    position = np.asarray(values["object_pos"], dtype=float)
    quaternion = np.asarray(values["object_quat"], dtype=float)
    if (
        hand.shape != (len(ids), 21, 3)
        or original.shape != hand.shape
        or position.shape != (len(ids), 3)
        or quaternion.shape != (len(ids), 4)
        or not all(np.isfinite(v).all() for v in (hand, original, position, quaternion))
    ):
        raise ValueError("Invalid demo landmark/object shapes or nonfinite annotations")
    if (
        str(values["mano_source_joint_order"]) != "mano21_sequential_thumb_index_middle_ring_little"
        or str(values["target_hand_side"]) != "right"
    ):
        raise ValueError("Expected explicit sequential MANO21 and right-hand conversion metadata")
    order = str(values["quaternion_order"])
    if order not in ("wxyz", "xyzw") or not np.allclose(
        np.linalg.norm(quaternion, axis=-1), 1, atol=1e-5
    ):
        raise ValueError("Invalid declared quaternion order or norm")
    rotation = Rotation.from_quat(
        quaternion[:, [1, 2, 3, 0]] if order == "wxyz" else quaternion
    ).as_matrix()
    conversion = str(values["hand_conversion"])
    if str(values["source_hand_side"]) == "left" and conversion == "object_local_x_reflection":
        local = np.einsum("tji,tkj->tki", rotation, original - position[:, None])
        local[:, :, 0] *= -1
        expected = np.einsum("tij,tkj->tki", rotation, local) + position[:, None]
    elif str(values["source_hand_side"]) == "right" and conversion in ("none", "identity"):
        expected = original
    else:
        raise ValueError("Unsupported or inconsistent hand-conversion metadata")
    reflection_error = float(np.max(np.abs(expected - hand)))
    if reflection_error > 2e-6:
        raise ValueError("Right-hand annotations disagree with the declared conversion")
    fps = float(values["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Demo must declare positive finite playback fps")
    poses = np.concatenate((rotation, position[:, :, None]), axis=2)[:, None]
    geometry = json.loads(geometry_path.read_text())
    # Camera gravity and the replacement asset's bottom-to-top direction are
    # different conventions; the original left-demo object +Z points down.
    from .geometry import can_base_down_alignment

    world_from_camera = np.eye(4)
    world_from_camera[:3, :3] = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
    first = np.eye(4)
    first[:3] = poses[0, 0]
    alignment, orientation = can_base_down_alignment(world_from_camera @ first, geometry)
    metadata = dict(
        object_index=0,
        object_name=geometry.get("name", str(values["grasped_object_name"])),
        mesh_to_object=alignment.tolist(),
        hand_side="right",
        left_hand_policy="preserve_source",
        length_unit="m",
        camera_axes="x_right_y_down_z_forward",
        fps=None,
        grounding_mode="camera_negative_y_up",
        object_orientation_requirement="initial_base_below_body",
        object_alignment_check=orientation,
        timestamps_s=((ids - ids[0]) / fps).tolist(),
        time_basis="source_declared_playback_fps",
        source_playback_fps=fps,
        frame_range=[int(ids[0]), int(ids[-1])],
        dataset_object_name=str(values["grasped_object_name"]),
        source_grasped_ycb_id=int(values["source_grasped_ycb_id"]),
        object_geometry=str(geometry_path),
        source_demo=dict(
            path=str(source.resolve()),
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            sequence=str(values["sequence_name"]),
            camera=str(values["source_camera_serial"]),
            source_hand_side=str(values["source_hand_side"]),
            hand_conversion=conversion,
            conversion_max_error_m=reflection_error,
            source_quaternion_order=order,
        ),
        semantic_order=list(DEXYCB_SEMANTICS),
        shape_replacement="Use explicit stepped-can geometry with one center-preserving base-down alignment to the original object pose frame; no shape inference from poses",
        timing_note="Source-declared playback fps retained; capture FPS not independently measured",
        robot_reference_used=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / "poses.npz", output / "retargeting.json"]
    if any(p.exists() or p.resolve() in (source.resolve(), geometry_path.resolve()) for p in paths):
        raise ValueError("Demo output already exists or would overwrite an input")
    np.savez_compressed(
        paths[0],
        frame_ids=ids,
        pose_y=poses,
        joint_3d=hand[:, None],
        source_metadata_json=np.array(json.dumps(metadata["source_demo"])),
    )
    paths[1].write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_sequence(path, model, config, *, geometric_debug=False):
    if isinstance(config, (str, Path)):
        config = json.loads(resolve_demo_path(config).read_text())
    else:
        config = dict(config)
    required = (
        "object_index",
        "object_name",
        "mesh_to_object",
        "hand_side",
        "length_unit",
        "camera_axes",
    )
    missing = [name for name in required if config.get(name) is None]
    if missing:
        raise ValueError(f"Missing confirmed sequence metadata: {missing}")
    if config["length_unit"] != "m" or config["hand_side"] not in ("left", "right"):
        raise ValueError("Expected metres and explicit left/right hand metadata")
    if not isinstance(config["object_name"], str) or not config["object_name"].strip():
        raise ValueError("An explicit object model name is required")
    alignment = np.asarray(config["mesh_to_object"], dtype=float)
    if alignment.shape != (4, 4) or not np.allclose(alignment[3], [0, 0, 0, 1]):
        raise ValueError("mesh_to_object must be a 4x4 SE(3) matrix")
    if (
        not np.allclose(alignment[:3, :3].T @ alignment[:3, :3], np.eye(3), atol=1e-7)
        or np.linalg.det(alignment[:3, :3]) < 0
    ):
        raise ValueError("mesh_to_object cannot scale or reflect the object")
    with np.load(path, allow_pickle=False) as archive:
        frame_ids = archive["frame_ids"].copy()
        all_poses = archive["pose_y"].astype(float)
        hand = archive["joint_3d"].astype(float)
        if "frame_metadata_json" in archive:
            frame = json.loads(str(archive["frame_metadata_json"]))
            config["frame_metadata"] = frame
            config["source_camera_axes"] = config["camera_axes"]
            config["camera_axes"] = frame["world_axes"]
            config["coordinate_frame"] = frame["coordinate_frame"]
    if (
        hand.shape != (len(frame_ids), 1, 21, 3)
        or all_poses.shape[0] != len(frame_ids)
        or all_poses.shape[2:] != (3, 4)
    ):
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
    valid = (
        np.isfinite(hand).all((1, 2))
        & np.isfinite(poses).all((1, 2))
        & ~np.all(hand == -1, axis=(1, 2))
    )
    for i, pose in enumerate(poses):
        if valid[i]:
            r = pose[:3, :3]
            valid[i] &= (
                np.allclose(r.T @ r, np.eye(3), atol=1e-4) and abs(np.linalg.det(r) - 1) < 1e-4
            )
    if config.get("timestamps_s") is not None:
        times = np.asarray(config["timestamps_s"], dtype=float)
        if (
            times.shape != frame_ids.shape
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
        ):
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
        raise ValueError(
            "Actual fps/timestamps or explicitly authorized retiming required for velocity constraints; no default fps is assumed"
        )
    if config["hand_side"] == "left" and config.get("left_hand_policy") != "preserve_source":
        raise ValueError(
            "Left-to-right hand handling must be explicit: choose preserve_source (no mirroring)"
        )
    mask = frame_mask(frame_ids, config.get("frame_range"))
    return Sequence(
        frame_ids[mask],
        hand[mask],
        poses[mask],
        None if times is None else times[mask],
        valid[mask],
        config,
    )
