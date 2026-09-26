"""Thin official HaMeR API adapter. Its absolute position is not a metric measurement."""

import json
import os
from pathlib import Path
import sys

import numpy as np

from .capture import read_capture
from .schema import CAMERA_AXES, HAND_NAMES, fingerprint, read_json, save_npz


def infer(request):
    import cv2
    import torch

    repo, checkpoint = Path(request["repo"]).resolve(), Path(request["checkpoint"]).resolve()
    if not repo.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("HaMeR repository/checkpoint and licensed MANO assets are required")
    sys.path.insert(0, str(repo))
    # Upstream model configuration uses repo-relative MANO asset paths; worker is isolated.
    os.chdir(repo)
    from hamer.models import load_hamer
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.utils import recursive_to
    from hamer.utils.renderer import cam_crop_to_full

    capture, ids, times, images = read_capture(request["capture"])
    annotations = read_json(request["boxes"])
    if (
        annotations.get("schema") != 1
        or annotations.get("image_size") != capture["image_size"]
        or annotations.get("frame_ids") != ids.tolist()
        or annotations.get("capture_sha256") != fingerprint(request["capture"])
    ):
        raise ValueError("Bounding box annotations do not match the capture")
    boxes = annotations["boxes"]
    if len(boxes) != len(ids):
        raise ValueError("Expected boxes for every frame, with null for missing hands")
    device = torch.device(request.get("device", "cuda:0"))
    model, config = load_hamer(str(checkpoint))
    model.to(device).eval()
    joints = np.full((len(ids), 2, 21, 3), np.nan)
    valid = np.zeros((len(ids), 2), bool)
    meshes = [np.full((len(ids), 778, 3), np.nan) for _ in range(2)]
    width, height = capture["image_size"]
    for i, image_path in enumerate(images):
        image = cv2.imread(str(image_path))
        for side, name in enumerate(("left", "right")):
            box = boxes[i].get(name)
            if box is None:
                continue
            box = np.asarray(box, float)
            if (
                box.shape != (4,)
                or not np.isfinite(box).all()
                or np.any(box[:2] < 0)
                or np.any(box[2:] > [width, height])
                or np.any(box[2:] <= box[:2])
            ):
                raise ValueError(f"Invalid {name} box at frame {ids[i]}")
            dataset = ViTDetDataset(config, image, box[None], np.array([side]), rescale_factor=2.0)
            batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)))
            batch = recursive_to(batch, device)
            with torch.inference_mode():
                output = model(batch)
                camera = output["pred_cam"].clone()
                camera[:, 1] *= 2 * side - 1
                focal = config.EXTRA.FOCAL_LENGTH / config.MODEL.IMAGE_SIZE * max(width, height)
                translation = cam_crop_to_full(
                    camera,
                    batch["box_center"].float(),
                    batch["box_size"].float(),
                    batch["img_size"].float(),
                    focal,
                )
            local_joints = output["pred_keypoints_3d"][0].detach().cpu().numpy().copy()
            local_mesh = output["pred_vertices"][0].detach().cpu().numpy().copy()
            if local_joints.shape != (21, 3) or local_mesh.shape != (778, 3):
                raise ValueError("Unexpected upstream HaMeR joint/vertex contract")
            local_joints[:, 0] *= 2 * side - 1
            local_mesh[:, 0] *= 2 * side - 1
            shift = translation[0].detach().cpu().numpy()
            joints[i, side] = local_joints + shift
            meshes[side][i] = local_mesh + shift
            valid[i, side] = (
                np.isfinite(joints[i, side]).all() and np.isfinite(meshes[side][i]).all()
            )
    faces_right = np.asarray(model.mano.faces, dtype=np.int64)
    meta = dict(
        schema=1,
        length_unit="m",
        coordinate_frame="camera",
        camera_axes=CAMERA_AXES,
        method="official_hamer",
        metric_source="monocular_mano_shape_prior",
        metric_calibrated=False,
        checkpoint_sha256=fingerprint(checkpoint),
        capture_sha256=fingerprint(request["capture"]),
        model_sha256=fingerprint(repo / "hamer/models/hamer.py"),
        joint_mapping="HaMeR MANO wrapper OpenPose21 -> explicit thumb/index/middle/ring/little semantics",
        camera_translation="upstream approximate focal length; requires measured alignment",
    )
    save_npz(
        request["output"],
        frame_ids=ids,
        timestamps_s=times,
        joint_names=np.array(HAND_NAMES),
        hand_sides=np.array(["left", "right"]),
        joints_camera_m=joints,
        valid=valid,
        vertices_left_camera_m=meshes[0],
        vertices_right_camera_m=meshes[1],
        faces_left=faces_right[:, [0, 2, 1]],
        faces_right=faces_right,
        metadata_json=np.array(json.dumps(meta)),
    )
