"""Compact offline playback: static meshes once, joint/SE(3) poses per frame."""

from .configuration import read_config
import json
import base64
from pathlib import Path
import numpy as np


def viewer_payload(
    trajectory_path, report_path=None, model=None, object_mesh_path=None, reference_dir=None
):
    if model is None:
        raise ValueError("The FK model is required for joint-consistent interpolated playback")
    with np.load(trajectory_path, allow_pickle=False) as archive:
        d = {key: archive[key] for key in archive.files}
    # Original RGB remains in the original camera frame. Only this display copy
    # is mapped back; the grounded dataset and simulator reference stay untouched.
    display_frame = "input"
    if reference_dir is not None and "frame_metadata_json" in d:
        from .coordinates import reference_in_source_frame

        d = reference_in_source_frame(d)
        display_frame = "original_camera_for_rgb_comparison"
    report = {} if report_path is None else read_config(Path(report_path))

    def numbers(values):
        values = np.asarray(values)
        if np.issubdtype(values.dtype, np.floating):
            values = np.round(values, 8)
            return np.where(np.isfinite(values), values, None).tolist()
        return values.tolist()

    times = d["timestamps_s"]
    timed = bool(len(times) and np.isfinite(times).all())
    if not timed:
        times = np.arange(len(d["frame_ids"])) * 0.1
    times = times - times[0]
    raw = model.description
    joints = []
    for j in raw["joints"]:
        joints.append(
            {
                k: j[k]
                for k in ("name", "parent", "child", "kind", "frame0", "frame1", "reversed")
                if k in j
            }
            | {"axis": j.get("axis", [0, 0, 0])}
        )
    all_points = np.concatenate(
        [d[k].reshape(-1, 3) for k in ("human_keypoints", "robot_keypoints", "object_keypoints")]
    )
    all_points = all_points[np.isfinite(all_points).all(1)]
    geometry = []
    for collider in model.colliders:
        geometry.append(
            dict(
                link=collider["link"],
                vertices=numbers(collider["vertices"]),
                faces=collider["faces"],
            )
        )
    object_mesh = None
    if object_mesh_path is not None:
        metadata_path = Path(object_mesh_path).with_suffix(".json")
        if metadata_path.exists():
            object_geometry = read_config(metadata_path)
            if (
                object_geometry.get("fingerprint")
                and str(d.get("object_geometry_fingerprint", "")) != object_geometry["fingerprint"]
            ):
                raise ValueError("Viewer mesh differs from reference geometry; rerun retargeting")
        with np.load(object_mesh_path, allow_pickle=False) as mesh:
            object_mesh = dict(vertices=numbers(mesh["vertices"]), faces=mesh["faces"].tolist())
    metadata = report.get("sequence_metadata", {})
    from .metrics import finger_edges, direction_error

    edges = finger_edges(model.semantic_names)
    quality = []
    for human, robot in zip(d["human_keypoints"], d["robot_keypoints"]):
        if not np.isfinite(human).all() or not np.isfinite(robot).all():
            quality.append(None)
        else:
            quality.append(
                dict(
                    keypoint_rmse_mm=float(
                        np.sqrt(np.mean(np.sum((human - robot) ** 2, axis=1))) * 1000
                    ),
                    finger_direction_mean_deg=direction_error(human, robot, edges)["mean_deg"],
                )
            )
    reference = None
    if reference_dir is not None:
        from .data import correspondence

        reference_dir = Path(reference_dir)
        photos, raw_points, raw_uv = [], [], []
        image_shape = None
        for frame in d["frame_ids"]:
            photos.append(
                "data:image/jpeg;base64,"
                + base64.b64encode((reference_dir / f"color_{frame:06d}.jpg").read_bytes()).decode(
                    "ascii"
                )
            )
            with np.load(reference_dir / f"labels_{frame:06d}.npz") as label:
                if image_shape is not None and label["seg"].shape != image_shape:
                    raise ValueError("Reference images use inconsistent dimensions")
                image_shape = label["seg"].shape
                raw_points.append(label["joint_3d"][0, correspondence(model)])
                raw_uv.append(label["joint_2d"][0, correspondence(model)])
        raw_points, raw_uv = np.asarray(raw_points), np.asarray(raw_uv)
        if not np.allclose(raw_points, d["human_keypoints"], atol=1e-7, equal_nan=True):
            raise ValueError("RGB labels and trajectory hand annotations differ")
        xyz, uv = raw_points.reshape(-1, 3), raw_uv.reshape(-1, 2)
        fx, cx = np.linalg.lstsq(
            np.c_[xyz[:, 0] / xyz[:, 2], np.ones(len(xyz))], uv[:, 0], rcond=None
        )[0]
        fy, cy = np.linalg.lstsq(
            np.c_[xyz[:, 1] / xyz[:, 2], np.ones(len(xyz))], uv[:, 1], rcond=None
        )[0]
        projection = np.c_[fx * xyz[:, 0] / xyz[:, 2] + cx, fy * xyz[:, 1] / xyz[:, 2] + cy]
        error = float(np.linalg.norm(projection - uv, axis=1).max())
        if error > 0.1:
            raise ValueError(f"Cannot reconstruct a matching pinhole camera: {error}px")
        reference = dict(
            images=photos,
            points_2d=numbers(raw_uv),
            width=image_shape[1],
            height=image_shape[0],
            intrinsics=numbers([fx, fy, cx, cy]),
            reprojection_max_px=error,
        )
    return dict(
        frame_ids=d["frame_ids"].tolist(),
        times=numbers(times),
        timed=timed,
        display_frame=display_frame,
        time_basis=metadata.get("time_basis", "source_timestamps" if timed else "preview_only"),
        synthetic=metadata.get("kind", "").startswith("synthetic"),
        classification=report.get("classification", "geometric_debug"),
        valid=d["valid"].tolist(),
        transition_valid=d["transition_valid"].tolist(),
        human=numbers(d["human_keypoints"]),
        robot=numbers(d["robot_keypoints"]),
        wrist=numbers(d["wrist_transform"]),
        q=numbers(d["active_q_rad"]),
        object_poses=numbers(d["object_transform"]),
        object_points=numbers(d["object_points_local"]),
        object_mesh=object_mesh,
        geometry=geometry,
        reference=reference,
        quality=quality,
        model=dict(
            root=raw["root_link"],
            joints=joints,
            full_names=model.full_names,
            coupling=numbers(model.coupling),
            offset=numbers(model.offset),
            keypoints=model.keypoints,
        ),
        semantic_names=model.semantic_names,
        bounds=numbers([all_points.min(0) - 0.025, all_points.max(0) + 0.025]),
        failures=report.get("failed_frame_ids", []),
    )


def comparison_viewer(
    trajectory_path,
    output_path,
    report_path=None,
    model=None,
    object_mesh_path=None,
    reference_dir=None,
):
    payload = viewer_payload(trajectory_path, report_path, model, object_mesh_path, reference_dir)
    static = Path(__file__).parent / "static"
    template = (static / "viewer.html").read_text()
    bundle = (static / "viewer.bundle.js").read_text()
    document = template.replace(
        "__PAYLOAD__",
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/"),
    )
    document = document.replace("__BUNDLE__", bundle.replace("</script", "<\\/script"))
    Path(output_path).write_text(document, encoding="utf-8")
