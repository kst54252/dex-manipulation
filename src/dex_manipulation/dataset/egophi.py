"""EgoPHI adapter; neural inference lives in a separate Python process."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from .schema import (
    edges,
    faces,
    fingerprint,
    load_npz,
    metadata,
    same_frames,
    save_npz,
    transform_points,
    transforms,
    write_json,
)


def run_worker(root, backend, python, request, output, request_path):
    if Path(output).exists():
        raise FileExistsError(output)
    request = {**request, "output": str(Path(output).resolve())}
    write_json(request_path, request)
    # Do not inherit Isaac-specific Python packages into the estimator environment.
    environment = dict(
        os.environ, PYTHONPATH=str(Path(root).resolve() / "src"), PYTHONDONTWRITEBYTECODE="1"
    )
    subprocess.run(
        [
            str(python),
            str(Path(root) / "scripts/dataset_worker.py"),
            backend,
            "--request",
            str(Path(request_path).resolve()),
        ],
        check=True,
        env=environment,
    )
    if not Path(output).is_file():
        raise RuntimeError(f"{backend} worker did not save its output")


def normalized_object(vertices):
    # Match official _load_articulated_mesh: mean-centred / max abs ORIGINAL coordinate.
    vertices = np.asarray(vertices, float)
    center, scale = vertices.mean(0), float(np.abs(vertices).max())
    if not np.isfinite(vertices).all() or scale <= 0:
        raise ValueError("Invalid metric object template")
    return (vertices - center) / scale, center, scale


def object_from_prediction(rotation, translation, center):
    """Network translation refers to the centred template, not the dataset object origin."""
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation - rotation @ center
    return transforms(pose, "EgoPHI object pose")


def prediction_consistency(
    final_pose,
    tracked_pose,
    predicted_vertices,
    local_vertices,
    max_position_m,
    max_angle_deg,
    max_mesh_error_m,
):
    position = float(np.linalg.norm(final_pose[:3, 3] - tracked_pose[:3, 3]))
    angle = float(
        np.rad2deg(Rotation.from_matrix(tracked_pose[:3, :3].T @ final_pose[:3, :3]).magnitude())
    )
    mesh_error = float(
        np.max(
            np.linalg.norm(
                predicted_vertices - transform_points(final_pose, local_vertices), axis=-1
            )
        )
    )
    return (
        position <= max_position_m and angle <= max_angle_deg and mesh_error <= max_mesh_error_m,
        position,
        angle,
        mesh_error,
    )


def _load_official_model(factory, options, checkpoint, device):
    """Load the complete checkpoint without downloading an unused ViT initializer."""
    from unittest.mock import patch

    import timm
    import torch

    create_model = timm.create_model

    def without_pretrained(*args, **kwargs):
        return create_model(*args, **{**kwargs, "pretrained": False})

    # Upstream hardcodes pretrained=True. The complete strict state includes the
    # backbone, so ImageNet initialization is redundant. Restore timm even if the
    # upstream constructor fails; this override only lives in the isolated worker.
    with patch.object(timm, "create_model", side_effect=without_pretrained):
        net = factory(**options)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    net.load_state_dict(state["model_state"], strict=True)
    return net.to(device).eval()


def infer(request):
    """Call official InteractionGNN directly, without GT-aligned dataset loaders."""
    import cv2
    import torch
    from .capture import read_capture
    from .sequence import DemoSequence

    repo, checkpoint = Path(request["repo"]), Path(request["checkpoint"])
    if not (repo / "model.py").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("EgoPHI model.py and checkpoint are required")
    # Worker is isolated; upstream's unqualified `angular` import cannot pollute the simulator.
    sys.path.insert(0, str(repo.resolve()))
    from model import InteractionGNN

    config = request["options"]
    sequence = DemoSequence.load(request["sequence"])
    a = sequence.arrays
    capture, ids, _, images = read_capture(request["capture"])
    same_frames(a, ids)
    capture_hash = sequence.meta.get("sources", {}).get("capture", {}).get("sha256")
    if capture_hash and capture_hash != fingerprint(request["capture"]):
        raise ValueError("EgoPHI RGB capture differs from the sequence source")
    mesh = load_npz(request["mesh"])
    if sequence.meta.get("processed_mesh_sha256") and sequence.meta[
        "processed_mesh_sha256"
    ] != fingerprint(request["mesh"]):
        raise ValueError("EgoPHI mesh differs from the sequence geometry")
    vertices, triangles = mesh["vertices_m"], faces(mesh["faces"], len(mesh["vertices_m"]))
    template, center, scale = normalized_object(vertices)
    calibration = sequence.meta["calibration"]
    if capture["image_size"] != calibration["image_size"]:
        raise ValueError("Capture/calibration resolutions differ")
    for side in ("left", "right"):
        if f"vertices_{side}_camera_m" not in a:
            raise ValueError("EgoPHI requires both MANO mesh arrays, with explicit validity masks")
        if a[f"vertices_{side}_camera_m"].shape[1] != 778:
            raise ValueError("Official EgoPHI expects MANO's 778-vertex hand topology")
        faces(a[f"faces_{side}"], 778)
    missing_policy = config["missing_hand_policy"]
    if missing_policy not in ("skip", "zero_context_debug"):
        raise ValueError("missing_hand_policy must be skip or zero_context_debug")
    limits = [
        float(config[k])
        for k in ("max_pose_difference_m", "max_pose_difference_deg", "max_mesh_pose_error_m")
    ]
    if not np.isfinite(limits).all() or min(limits) <= 0:
        raise ValueError("EgoPHI consistency limits must be positive")
    device = torch.device(config["device"])
    net = _load_official_model(InteractionGNN, config["model"], checkpoint, device)

    def tensor(x, dtype=torch.float32):
        return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)

    def numpy(x):
        return x.detach().cpu().numpy()[0]

    graph = [
        tensor(edges(a[f"faces_{side}"], bidirectional=False), torch.long)
        for side in ("left", "right")
    ]
    object_graph = tensor(edges(triangles).T, torch.long)
    n = len(ids)
    result = dict(
        frame_ids=ids,
        timestamps_s=a["timestamps_s"],
        valid=np.zeros(n, bool),
        consistency_pass=np.zeros(n, bool),
        prediction_available=np.zeros(n, bool),
        context_complete=a["hand_valid"].all(axis=1),
        T_camera_object_predicted=np.full((n, 4, 4), np.nan),
        object_vertices_camera_m=np.full((n, len(vertices), 3), np.nan),
        pose_difference_m=np.full(n, np.nan),
        pose_difference_deg=np.full(n, np.nan),
        mesh_pose_error_m=np.full(n, np.nan),
    )
    for side, count in (("left", 778), ("right", 778), ("object", len(vertices))):
        result[f"contact_{side}"] = np.full((n, count), np.nan, np.float32)
        result[f"force_{side}_normalized"] = np.full((n, count), np.nan, np.float32)
        result[f"force_{side}_direction_camera"] = np.full((n, count, 3), np.nan, np.float32)
    k = np.asarray(calibration["K"], float)
    distortion = np.asarray(calibration["distortion"], float)
    width, height = capture["image_size"]
    for i, path in enumerate(images):
        if not a["camera_valid"][i] or not a["object_valid"][i] or not a["hand_valid"][i].any():
            continue
        if not result["context_complete"][i] and missing_policy == "skip":
            continue
        image = cv2.imread(str(path))
        image = cv2.undistort(image, k, distortion)
        # Match upstream RGB / 255 preprocessing, with the actual source resolution and K.
        rgb = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2RGB) / 255.0
        tracked_pose = np.linalg.inv(a["T_world_camera"][i]) @ a["T_world_object"][i]
        camera_vertices = transform_points(tracked_pose, vertices)
        if np.min(camera_vertices[:, 2]) <= 0:
            continue
        uv = camera_vertices @ k.T
        uv = uv[:, :2] / uv[:, 2:]
        low = np.maximum(uv.min(0), [0, 0])
        high = np.minimum(uv.max(0), [width, height])
        if np.any(high - low < 2):
            continue
        bbox = np.r_[low, high] * np.array([224 / width, 224 / height] * 2)
        hand = [
            tensor(
                a[f"vertices_{side}_camera_m"][i] if a["hand_valid"][i, j] else np.zeros((778, 3))
            )[None]
            for j, side in enumerate(("left", "right"))
        ]
        with torch.inference_mode():
            pred = net(
                tensor(rgb.transpose(2, 0, 1))[None],
                *hand,
                hand[0].mean(1),
                hand[1].mean(1),
                tensor(template)[None],
                tensor([scale]),
                tensor(bbox)[None],
                tensor(k)[None],
                *graph,
                object_graph,
                height,
                width,
            )
        if len(pred) != 14 or any(not torch.isfinite(value).all() for value in pred):
            raise ValueError(f"Invalid official EgoPHI output at frame {ids[i]}")
        pose = object_from_prediction(numpy(pred[12]), numpy(pred[11]), center)
        predicted_vertices = numpy(pred[13])
        consistent, pos, angle, mesh_error = prediction_consistency(
            pose, tracked_pose, predicted_vertices, vertices, *limits
        )
        result["prediction_available"][i] = True
        result["consistency_pass"][i] = consistent
        result["valid"][i] = (
            consistent
            and result["context_complete"][i]
            and sequence.meta.get("geometry_status") != "test_assumptions"
        )
        result["T_camera_object_predicted"][i] = pose
        result["object_vertices_camera_m"][i] = predicted_vertices
        result["pose_difference_m"][i], result["pose_difference_deg"][i] = pos, angle
        result["mesh_pose_error_m"][i] = mesh_error
        for j, side in enumerate(("left", "right", "object")):
            if j < 2 and not a["hand_valid"][i, j]:
                continue  # Missing branch context never becomes a labelled hand.
            result[f"contact_{side}"][i] = numpy(torch.sigmoid(pred[j])).reshape(-1)
            result[f"force_{side}_normalized"][i] = numpy(torch.sigmoid(pred[j + 3])).reshape(-1)
            direction = numpy(pred[j + 6])
            result[f"force_{side}_direction_camera"][i] = direction / (
                np.linalg.norm(direction, axis=-1, keepdims=True) + 1e-8
            )
    meta = dict(
        schema=1,
        length_unit="m",
        label_source="egophi_prediction",
        input_geometry_status=sequence.meta.get("geometry_status", "unspecified"),
        assumptions=sequence.meta.get("assumptions", []),
        measured_force=False,
        force_unit="normalized_model_output",
        camera_axes=sequence.meta["camera_axes"],
        force_conversion="none; official compute_metrics.py uses *100 N, not a calibration for this drill",
        checkpoint_sha256=fingerprint(checkpoint),
        model_sha256=fingerprint(repo / "model.py"),
        angular_sha256=fingerprint(repo / "angular.py"),
        sequence_sha256=fingerprint(request["sequence"]),
        mesh_sha256=fingerprint(request["mesh"]),
        capture_sha256=fingerprint(request["capture"]),
        missing_hand_policy=missing_policy,
        options=config,
        object_bbox_source="projection_of_independently_tracked_mesh",
        object_pose_source="independent_tracker_preserved",
        mesh_pose_check="network force mesh and final pose checked separately",
    )
    result["metadata_json"] = np.array(json.dumps(meta))
    save_npz(request["output"], **result)


def validate_interaction(path, sequence, sequence_path, mesh_path=None):
    a = load_npz(path)
    same_frames(a, sequence.arrays["frame_ids"])
    m = metadata(a)
    if m.get("sequence_sha256") != fingerprint(sequence_path):
        raise ValueError("Interaction labels belong to a different sequence revision")
    if not np.array_equal(a["timestamps_s"], sequence.arrays["timestamps_s"]):
        raise ValueError("Interaction timestamps differ from the sequence")
    if (
        m.get("label_source") != "egophi_prediction"
        or m.get("force_unit") != "normalized_model_output"
        or m.get("measured_force") is not False
    ):
        raise ValueError("Declare EgoPHI prediction source and normalized force units")
    object_count = None
    if mesh_path is not None:
        if m.get("mesh_sha256") != fingerprint(mesh_path):
            raise ValueError("Interaction mesh differs from exported geometry")
        object_count = len(load_npz(mesh_path)["vertices_m"])
    n = len(sequence.arrays["frame_ids"])
    for name in ("valid", "prediction_available", "context_complete"):
        from .schema import bool_mask

        bool_mask(a[name], (n,), name)
    if np.any(a["valid"] & ~(a["prediction_available"] & a["context_complete"])):
        raise ValueError("Incomplete interaction predictions cannot be marked valid")
    if "consistency_pass" in a:
        bool_mask(a["consistency_pass"], (n,), "consistency_pass")
        if np.any(a["consistency_pass"] & ~a["prediction_available"]):
            raise ValueError("Consistency cannot pass without a prediction")
        if np.any(a["valid"] & ~a["consistency_pass"]):
            raise ValueError("Inconsistent interaction predictions cannot be marked valid")
    if (
        sequence.meta.get("geometry_status") == "test_assumptions"
        or m.get("input_geometry_status") == "test_assumptions"
    ) and a["valid"].any():
        raise ValueError("Interaction predictions based on test assumptions cannot be marked valid")
    if np.any(a["context_complete"] & ~sequence.arrays["hand_valid"].all(axis=1)):
        raise ValueError("Interaction context cannot include missing hands")
    available = a["prediction_available"]
    poses = a["T_camera_object_predicted"]
    if poses.shape != (n, 4, 4):
        raise ValueError("Invalid predicted object pose shape")
    transforms(poses, "predicted object pose", available)
    points = a["object_vertices_camera_m"]
    if (
        points.ndim != 3
        or points.shape[0] != n
        or points.shape[2] != 3
        or not np.isfinite(points[available]).all()
    ):
        raise ValueError("Invalid predicted object vertices")
    if object_count is not None and points.shape[1] != object_count:
        raise ValueError("Predicted object vertex count differs from mesh")
    for side in ("left", "right", "object"):
        contact = a[f"contact_{side}"]
        force = a[f"force_{side}_normalized"]
        if contact.ndim != 2 or contact.shape[0] != n or force.shape != contact.shape:
            raise ValueError("Invalid contact/force shape")
        expected = points.shape[1] if side == "object" else 778
        direction = a[f"force_{side}_direction_camera"]
        if contact.shape[1] != expected or direction.shape != (*contact.shape, 3):
            raise ValueError("Contact/force topology differs from the labelled mesh")
        if not np.isfinite(direction[a["valid"]]).all():
            raise ValueError("Valid force directions must be finite")
        values = np.concatenate([contact[a["valid"]].ravel(), force[a["valid"]].ravel()])
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Valid normalized contact/force values must be finite and in [0,1]")
    return a, m
