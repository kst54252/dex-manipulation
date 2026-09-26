"""Measured marker/CAD correspondences -> object or table-frame pose."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .schema import (
    CAMERA_AXES,
    WORLD_AXES,
    faces,
    fingerprint,
    transforms,
    transform_points,
    save_npz,
)


def load_mesh(path, unit_to_m, mesh_to_object):
    import trimesh

    if unit_to_m is None or not np.isfinite(unit_to_m) or unit_to_m <= 0:
        raise ValueError(
            "Measured mesh unit_to_m is required; shape/scale is never inferred from pose"
        )
    alignment = transforms(mesh_to_object, "mesh_to_object")
    mesh = trimesh.load(Path(path), process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(
            "Use a single triangulated OBJ/PLY/STL mesh, with no implicit scene transforms"
        )
    vertices = transform_points(alignment, np.asarray(mesh.vertices) * unit_to_m)
    triangles = faces(mesh.faces, len(vertices))
    if not np.isfinite(vertices).all() or np.linalg.matrix_rank(vertices - vertices.mean(0)) < 2:
        raise ValueError("Invalid object geometry")
    return vertices, triangles, fingerprint(path)


def solve_pose(points_m, pixels, calibration, max_error_px=3.0, ambiguity_px=0.15):
    """Reject depth flips and ambiguous planar solutions; error is actual reprojection RMS."""
    import cv2

    points, uv = np.asarray(points_m, float), np.asarray(pixels, float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or uv.shape != (len(points), 2)
        or len(points) < 4
        or not np.isfinite(points).all()
        or not np.isfinite(uv).all()
        or not np.isfinite([max_error_px, ambiguity_px]).all()
        or max_error_px <= 0
        or ambiguity_px < 0
    ):
        raise ValueError(
            "PnP requires >=4 finite, measured 3D/2D correspondences and valid thresholds"
        )
    centered = points - points.mean(0)
    rank = np.linalg.matrix_rank(centered, tol=1e-8)
    if rank < 2:
        raise ValueError("Collinear PnP landmarks")
    matrix, distortion = (
        np.asarray(calibration["K"], float),
        np.asarray(calibration["distortion"], float),
    )
    flag = cv2.SOLVEPNP_IPPE if rank == 2 else cv2.SOLVEPNP_SQPNP
    result = cv2.solvePnPGeneric(points, uv, matrix, distortion, flags=flag)
    candidates = []
    for rvec, tvec in zip(result[1], result[2]):
        pose = np.eye(4)
        pose[:3, :3] = cv2.Rodrigues(rvec)[0]
        pose[:3, 3] = tvec.ravel()
        if np.min(transform_points(pose, points)[:, 2]) <= 0:
            continue
        projected = cv2.projectPoints(points, rvec, tvec, matrix, distortion)[0].reshape(-1, 2)
        error = float(np.sqrt(np.mean(np.sum((projected - uv) ** 2, axis=1))))
        if np.isfinite(error):
            candidates.append((error, pose))
    candidates.sort(key=lambda c: c[0])
    if not candidates:
        return np.full((4, 4), np.nan), np.nan, False
    error, pose = candidates[0]
    ambiguous = False
    if len(candidates) > 1 and candidates[1][0] - error < ambiguity_px:
        other = candidates[1][1]
        rotation_gap = Rotation.from_matrix(pose[:3, :3].T @ other[:3, :3]).magnitude()
        ambiguous = (
            rotation_gap > np.deg2rad(2) or np.linalg.norm(pose[:3, 3] - other[:3, 3]) > 0.002
        )
    return pose, error, bool(error <= max_error_px and not ambiguous)


def track_markers(capture, calibration, marker_config, output):
    """ArUco corner coordinates are supplied in the desired object/table frame."""
    import cv2
    from .capture import read_capture

    info, ids, times, images = read_capture(capture)
    if info["image_size"] != calibration["image_size"]:
        raise ValueError("Capture/calibration resolutions differ")
    if (
        marker_config.get("length_unit") != "m"
        or not marker_config.get("measurement_source")
        or marker_config.get("frame") not in ("object", "world")
    ):
        raise ValueError("Markers need measured corners in metres and frame=object/world")
    if marker_config["frame"] == "world" and (
        marker_config.get("world_axes") != WORLD_AXES
        or marker_config.get("world_origin") != "table_surface"
    ):
        raise ValueError("World markers must define a measured Z-up table_surface frame")
    dictionary_name = marker_config["dictionary"]
    if not dictionary_name.startswith("DICT_") or not hasattr(cv2.aruco, dictionary_name):
        raise ValueError("Unknown ArUco dictionary")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    detector = cv2.aruco.ArucoDetector(dictionary)
    markers = {int(m["id"]): np.asarray(m["corners_m"], float) for m in marker_config["markers"]}
    if len(markers) != len(marker_config["markers"]) or not markers:
        raise ValueError("Marker IDs must be unique and nonempty")
    if any(p.shape != (4, 3) or not np.isfinite(p).all() for p in markers.values()):
        raise ValueError("Each marker needs four measured corners in detection order")
    poses = np.full((len(ids), 4, 4), np.nan)
    errors, valid = np.full(len(ids), np.nan), np.zeros(len(ids), bool)
    for i, path in enumerate(images):
        image = cv2.imread(str(path))
        corners, detected, _ = detector.detectMarkers(image)
        pairs = (
            []
            if detected is None
            else [
                (markers[int(k)], c.reshape(4, 2))
                for k, c in zip(detected.ravel(), corners)
                if int(k) in markers
            ]
        )
        if pairs:
            poses[i], errors[i], valid[i] = solve_pose(
                np.concatenate([p[0] for p in pairs]),
                np.concatenate([p[1] for p in pairs]),
                calibration,
                marker_config.get("max_reprojection_px", 3.0),
                marker_config.get("ambiguity_px", 0.15),
            )
    key = "T_camera_object" if marker_config["frame"] == "object" else "T_world_camera"
    if key == "T_world_camera":
        poses[valid] = np.linalg.inv(poses[valid])
        poses[~valid] = np.nan
    if Path(output).exists():
        raise FileExistsError(output)
    save_npz(
        output,
        frame_ids=ids,
        timestamps_s=times,
        **{key: poses},
        valid=valid,
        reprojection_error_px=errors,
        metadata_json=np.array(
            json.dumps(
                dict(
                    schema=1,
                    length_unit="m",
                    camera_axes=CAMERA_AXES,
                    method="measured_aruco_pnp",
                    capture_sha256=fingerprint(capture),
                    coordinate_frame=marker_config["frame"],
                    world_axes=marker_config.get("world_axes"),
                    world_origin=marker_config.get("world_origin"),
                    measurement_source=marker_config["measurement_source"],
                    marker_config=marker_config,
                )
            )
        ),
    )
    return {"frames": len(ids), "valid": int(valid.sum()), "output": str(output)}
