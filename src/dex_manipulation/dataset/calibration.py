"""Camera calibration and measured metric frame transforms."""

import numpy as np

from .schema import CAMERA_AXES, WORLD_AXES, transforms, read_json


def load_calibration(path):
    config = read_json(path)
    if config.get("schema") != 1 or config.get("camera_axes") != CAMERA_AXES:
        raise ValueError("Calibration must declare schema=1 and OpenCV camera axes")
    matrix = np.asarray(config["K"], dtype=float)
    distortion = np.asarray(config["distortion"], dtype=float)
    size = np.asarray(config["image_size"])
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or matrix[0, 0] <= 0
        or matrix[1, 1] <= 0
        or not np.allclose(matrix[2], [0, 0, 1])
        or not np.isclose(matrix[0, 1], 0)
        or not np.isclose(matrix[1, 0], 0)
        or distortion.ndim != 1
        or len(distortion) not in (4, 5, 8, 12, 14)
        or not np.isfinite(distortion).all()
        or size.shape != (2,)
        or not np.issubdtype(size.dtype, np.integer)
        or np.any(size <= 0)
    ):
        raise ValueError("Invalid camera intrinsics/distortion/image_size")
    if config.get("T_world_camera") is not None:
        if config.get("camera_motion") != "fixed" or not config.get("extrinsics_source"):
            raise ValueError("Static extrinsics require camera_motion=fixed and extrinsics_source")
        if config.get("world_axes") != WORLD_AXES or config.get("world_origin") != "table_surface":
            raise ValueError("Declare measured table_surface origin and Z-up world_axes")
        transforms(config["T_world_camera"], "T_world_camera")
    return config


def calibrate_checkerboard(images, board, square_m):
    import cv2

    if len(board) != 2 or min(board) < 3 or not np.isfinite(square_m) or square_m <= 0:
        raise ValueError("Declare inner corner counts and measured square_m")
    grid = np.zeros((board[0] * board[1], 3), np.float32)
    grid[:, :2] = np.mgrid[: board[0], : board[1]].T.reshape(-1, 2) * square_m
    objects, pixels, used, size = [], [], [], None
    for path in images:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot read calibration image {path}")
        current = (image.shape[1], image.shape[0])
        if size is not None and current != size:
            raise ValueError("Calibration images must have the same resolution")
        size = current
        found, corners = cv2.findChessboardCornersSB(image, tuple(board))
        if found:
            objects.append(grid.copy())
            pixels.append(corners)
            used.append(str(path))
    if len(used) < 8:
        raise ValueError("At least 8 checkerboard detections at varied poses are required")
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(objects, pixels, size, None, None)
    return dict(
        schema=1,
        camera_axes=CAMERA_AXES,
        K=matrix.tolist(),
        distortion=distortion.ravel().tolist(),
        image_size=list(size),
        camera_motion="unknown",
        T_world_camera=None,
        extrinsics_source=None,
        calibration_rms_px=float(rms),
        square_m=square_m,
        images=used,
    )


def backproject(pixels, depth_m, calibration):
    """Depth must be optical-axis Z, aligned with the RGB pixels (not ray range)."""
    import cv2

    pixels, depth_m = np.asarray(pixels, float), np.asarray(depth_m, float)
    if pixels.shape[:-1] != depth_m.shape or pixels.shape[-1] != 2:
        raise ValueError("Pixel/depth shapes differ")
    rays = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2),
        np.asarray(calibration["K"]),
        np.asarray(calibration["distortion"]),
    ).reshape(pixels.shape)
    result = np.concatenate([rays, np.ones((*rays.shape[:-1], 1))], axis=-1) * depth_m[..., None]
    result[~np.isfinite(depth_m) | (depth_m <= 0)] = np.nan
    return result
