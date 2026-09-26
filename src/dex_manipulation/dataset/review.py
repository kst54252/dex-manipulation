"""Self-contained browser review and RGB bounding-box annotation; no simulator."""

import json
from pathlib import Path

import numpy as np

from .capture import read_capture
from .egophi import validate_interaction
from .schema import fingerprint, load_npz, new_directory, same_frames, transform_points, write_json
from .sequence import DemoSequence


def json_finite(value):
    if isinstance(value, np.ndarray):
        return json_finite(value.tolist())
    if isinstance(value, dict):
        return {k: json_finite(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_finite(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def make_review(capture_path, output, sequence_path=None, mesh_path=None, interaction=None):
    import cv2

    capture, ids, times, images = read_capture(capture_path)
    payload = dict(
        frame_ids=ids,
        times=times,
        image_size=capture["image_size"],
        capture_sha256=fingerprint(capture_path),
        hand=None,
        hand_meshes=None,
        joints_are_proxies=False,
        assumptions=[],
        object=None,
        contact=None,
        prediction=None,
        source_camera=None,
        summary=None,
    )
    if sequence_path:
        sequence = DemoSequence.load(sequence_path)
        same_frames(sequence.arrays, ids)
        source_hash = sequence.meta.get("sources", {}).get("capture", {}).get("sha256")
        if source_hash and source_hash != fingerprint(capture_path):
            raise ValueError("Review RGB capture differs from the sequence source")
        a = sequence.arrays
        mesh = load_npz(mesh_path)
        if sequence.meta.get("processed_mesh_sha256") and sequence.meta[
            "processed_mesh_sha256"
        ] != fingerprint(mesh_path):
            raise ValueError("Review mesh differs from the sequence geometry")
        # Subsample only the viewer, retaining original mesh indices for contact colours.
        selected = np.arange(0, len(mesh["vertices_m"]), max(1, len(mesh["vertices_m"]) // 1500))
        object_world = transform_points(
            a["T_world_object"], np.repeat(mesh["vertices_m"][None, selected], len(ids), axis=0)
        )
        hand = a["hand_world_m"].copy()
        hand[~a["hand_valid"]] = np.nan
        object_world[~a["object_valid"]] = np.nan
        payload.update(
            hand=hand,
            object=object_world,
            summary=sequence.summary(),
            geometry_status=sequence.meta["geometry_status"],
            hand_valid=a["hand_valid"],
            object_valid=a["object_valid"],
            camera_valid=a["camera_valid"],
            assumptions=sequence.meta.get("assumptions", []),
            joints_are_proxies=bool(
                sequence.meta.get("joints_are_proxies", False)
                or sequence.meta.get("hand_estimator", {}).get("joints_are_proxies", False)
                or sequence.meta.get("hand_estimator", {}).get(
                    "landmarks_are_format_proxies", False
                )
            ),
        )
        hand_meshes = {}
        for side, name in enumerate(("left", "right")):
            key = f"vertices_{name}_camera_m"
            if key not in a:
                continue
            vertices = transform_points(a["T_world_camera"], a[key])
            valid = a["hand_valid"][:, side] & a["camera_valid"]
            vertices[~valid] = np.nan
            # DemoSequence validates face indices against each hand's own vertex count.
            hand_meshes[name] = dict(vertices=vertices, faces=a[f"faces_{name}"], valid=valid)
        payload["hand_meshes"] = hand_meshes or None
        k = np.asarray(sequence.meta["calibration"]["K"], float)
        d = np.asarray(sequence.meta["calibration"]["distortion"], float)
        payload["source_camera"] = dict(
            T_world_camera=a["T_world_camera"],
            valid=a["camera_valid"],
            K=k,
            distortion=d,
            image_size=sequence.meta["calibration"]["image_size"],
            intrinsics_source=sequence.meta["calibration"].get("intrinsics_source", "unspecified"),
            assumed=sequence.meta["geometry_status"] == "test_assumptions",
        )
        overlay = []
        object_overlay = []
        for i in range(len(ids)):
            if not a["camera_valid"][i]:
                overlay.append(None)
                object_overlay.append(None)
                continue
            camera_hand = transform_points(
                np.linalg.inv(a["T_world_camera"][i]), hand[i].reshape(-1, 3)
            )
            uv = cv2.projectPoints(camera_hand, np.zeros(3), np.zeros(3), k, d)[0].reshape(2, 21, 2)
            uv[camera_hand.reshape(2, 21, 3)[..., 2] <= 0] = np.nan
            overlay.append(uv)
            camera_object = transform_points(np.linalg.inv(a["T_world_camera"][i]), object_world[i])
            if a["object_valid"][i] and np.all(camera_object[:, 2] > 0):
                uv_object = cv2.projectPoints(
                    camera_object, np.zeros(3), np.zeros(3), k, d
                )[0].reshape(-1, 2)
                # Convex silhouette is a calibration diagnostic, not segmentation.
                object_overlay.append(cv2.convexHull(uv_object.astype(np.float32)).reshape(-1, 2))
            else:
                object_overlay.append(None)
        payload["overlay"] = overlay
        payload["object_overlay"] = object_overlay
        if interaction:
            interaction_data, _ = validate_interaction(
                interaction, sequence, sequence_path, mesh_path
            )
            points = transform_points(
                a["T_world_camera"], interaction_data["object_vertices_camera_m"][:, selected]
            )
            points[~interaction_data["prediction_available"]] = np.nan
            payload["contact"] = dict(
                points=points,
                probability=interaction_data["contact_object"][:, selected],
                valid=interaction_data["valid"],
                consistency_pass=interaction_data.get(
                    "consistency_pass", interaction_data["valid"]
                ),
                available=interaction_data["prediction_available"],
            )
            # The force/contact-stage vertices and final pose are distinct upstream
            # outputs. Keep both instead of silently treating them as the same mesh.
            predicted_world = transform_points(
                a["T_world_camera"] @ interaction_data["T_camera_object_predicted"],
                np.repeat(mesh["vertices_m"][None, selected], len(ids), axis=0),
            )
            predicted_world[~interaction_data["prediction_available"]] = np.nan
            payload["prediction"] = dict(
                points=predicted_world,
                available=interaction_data["prediction_available"],
                valid=interaction_data["valid"],
                pose_difference_m=interaction_data.get("pose_difference_m"),
                pose_difference_deg=interaction_data.get("pose_difference_deg"),
                mesh_pose_error_m=interaction_data.get("mesh_pose_error_m"),
            )
    output = new_directory(output)
    (output / "rgb").mkdir()
    names = []
    for i, source in enumerate(images):
        image = cv2.imread(str(source))
        ratio = min(1, 960 / image.shape[1])
        small = cv2.resize(image, (round(image.shape[1] * ratio), round(image.shape[0] * ratio)))
        name = f"rgb/{i:06d}.jpg"
        if not cv2.imwrite(str(output / name), small, [cv2.IMWRITE_JPEG_QUALITY, 85]):
            raise OSError(name)
        names.append(name)
    payload["images"] = names
    static = Path(__file__).resolve().parents[1] / "static"
    serialized = json.dumps(json_finite(payload), ensure_ascii=False, allow_nan=False).replace(
        "<", "\\u003c"
    )
    html = (static / "dataset.html").read_text().replace("__PAYLOAD__", serialized)
    html = html.replace("__BUNDLE__", (static / "dataset.bundle.js").read_text())
    (output / "review.html").write_text(html)
    if payload["summary"]:
        write_json(output / "summary.json", payload["summary"])
    return output / "review.html"
