"""Named hand landmarks and independently measured metric alignment."""

import json

import numpy as np

from .schema import CAMERA_AXES, HAND_NAMES, bool_mask, faces, metadata, same_frames


def validate_hands(archive, ids, require_metric=True):
    same_frames(archive, ids)
    meta = metadata(archive)
    if meta.get("camera_axes") != CAMERA_AXES or meta.get("coordinate_frame") != "camera":
        raise ValueError("Hands must be in the calibrated OpenCV camera frame")
    names = list(map(str, archive["joint_names"]))
    sides = list(map(str, archive["hand_sides"]))
    if len(names) != 21 or set(names) != set(HAND_NAMES) or sides != ["left", "right"]:
        raise ValueError("Declare 21 unique semantic names and hand_sides=[left,right]")
    joints = np.asarray(archive["joints_camera_m"], float)
    if joints.shape != (len(ids), 2, 21, 3):
        raise ValueError("joints_camera_m must be (T,2,21,3)")
    valid = bool_mask(archive["valid"], (len(ids), 2), "hand valid")
    if not np.isfinite(joints[valid]).all() or np.any(np.all(joints[valid] == 0, axis=(1, 2))):
        raise ValueError("Valid hands cannot contain missing or zero-filled poses")
    if not meta.get("method") or not meta.get("metric_source"):
        raise ValueError("Hand estimates must declare method and metric_source")
    if require_metric and meta.get("metric_calibrated") is not True:
        raise ValueError(
            "Metric hand alignment is missing; use measured anchors before build/export"
        )
    for side in sides:
        key = f"vertices_{side}_camera_m"
        if key in archive:
            vertices = archive[key]
            if vertices.ndim != 3 or vertices.shape[0] != len(ids) or vertices.shape[2] != 3:
                raise ValueError(f"Invalid {key}")
            if not np.isfinite(vertices[valid[:, sides.index(side)]]).all():
                raise ValueError(f"Invalid mesh for visible {side} hand")
            faces(archive[f"faces_{side}"], vertices.shape[1])
    return joints[:, :, [names.index(name) for name in HAND_NAMES]], valid, meta


def fit_metric_alignment(source, target, estimate_scale=True):
    """Similarity fit to >=3 non-collinear corresponding 3D joint centres."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.ndim != 2 or source.shape[1] != 3 or target.shape != source.shape:
        raise ValueError("Corresponding hand anchors must have matching (N,3) shapes")
    valid = np.isfinite(source).all(-1) & np.isfinite(target).all(-1)
    source, target = source[valid], target[valid]
    if len(source) < 3:
        raise ValueError("At least three measured 3D hand anchors are needed")
    a, b = source - source.mean(0), target - target.mean(0)
    if min(np.linalg.matrix_rank(a, tol=1e-7), np.linalg.matrix_rank(b, tol=1e-7)) < 2:
        raise ValueError("Hand anchors are collinear")
    u, singular, vt = np.linalg.svd(a.T @ b)
    signs = np.ones(3)
    signs[-1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ np.diag(signs) @ u.T
    scale = float(np.dot(singular, signs) / np.sum(a * a)) if estimate_scale else 1.0
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid measured hand scale")
    translation = target.mean(0) - scale * rotation @ source.mean(0)
    residual = source @ rotation.T * scale + translation - target
    return scale, rotation, translation, float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))


def align_hands(archive, anchors, max_error_m=0.01):
    joints, valid, meta = validate_hands(archive, archive["frame_ids"], require_metric=False)
    target, measured, anchor_meta = validate_hands(
        anchors, archive["frame_ids"], require_metric=True
    )
    if "timestamps_s" in archive and "timestamps_s" in anchors:
        if np.asarray(archive["timestamps_s"]).shape != np.asarray(
            anchors["timestamps_s"]
        ).shape or not np.allclose(
            archive["timestamps_s"], anchors["timestamps_s"], rtol=0, atol=1e-7
        ):
            raise ValueError("Hand anchor timestamps differ from the estimates")
    # Sparse anchors use a separate mask, not NaNs in a declared valid full hand.
    mask = anchors.get("anchor_mask", np.ones(target.shape[:-1], bool))
    bool_mask(mask, target.shape[:-1], "anchor_mask")
    anchor_names = list(map(str, anchors["joint_names"]))
    mask = mask[:, :, [anchor_names.index(name) for name in HAND_NAMES]]
    result = {k: v.copy() for k, v in archive.items()}
    aligned = np.full_like(joints, np.nan)
    errors, scales = np.full(valid.shape, np.nan), np.full(valid.shape, np.nan)
    accepted = np.zeros_like(valid)
    if not np.isfinite(max_error_m) or max_error_m <= 0:
        raise ValueError("max_error_m must be positive")
    for frame, side in np.argwhere(valid & measured):
        keep = mask[frame, side]
        try:
            scale, rotation, translation, error = fit_metric_alignment(
                joints[frame, side, keep], target[frame, side, keep]
            )
        except ValueError:
            continue
        errors[frame, side], scales[frame, side] = error, scale
        if error > max_error_m:
            continue
        accepted[frame, side] = True
        aligned[frame, side] = joints[frame, side] @ rotation.T * scale + translation
        key = f"vertices_{('left', 'right')[side]}_camera_m"
        if key in result:
            result[key][frame] = archive[key][frame] @ rotation.T * scale + translation
    for side, name in enumerate(("left", "right")):
        if f"vertices_{name}_camera_m" in result:
            result[f"vertices_{name}_camera_m"][~accepted[:, side]] = np.nan
    meta.update(
        metric_calibrated=True,
        metric_source=anchor_meta["metric_source"],
        alignment="per_frame_measured_joint_similarity",
        max_alignment_error_m=max_error_m,
    )
    result.update(
        joints_camera_m=aligned,
        joint_names=np.array(HAND_NAMES),
        valid=accepted,
        alignment_error_m=errors,
        measured_scale=scales,
        metadata_json=np.array(json.dumps(meta)),
    )
    return result
