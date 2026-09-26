"""Export new demos without touching existing inputs or simulator/checkpoint contracts."""

import json
from pathlib import Path
import shutil

import numpy as np

from .egophi import validate_interaction
from .schema import CAMERA_AXES, WORLD_AXES, faces, fingerprint, load_npz, save_npz, write_json
from .sequence import DemoSequence


def export_demo(sequence_path, mesh_path, output, hand_side="right", interaction=None, events=None):
    sequence = DemoSequence.load(sequence_path)
    a, meta = sequence.arrays, sequence.meta
    if hand_side not in ("left", "right"):
        raise ValueError("Declare hand_side=left/right")
    side = ("left", "right").index(hand_side)
    valid = a["hand_valid"][:, side] & a["object_valid"] & a["camera_valid"]
    if not valid.any():
        raise ValueError("No valid hand/object pairs to export")
    mesh = load_npz(mesh_path)
    vertices = np.asarray(mesh["vertices_m"], float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Invalid metric object mesh")
    faces(mesh["faces"], len(vertices))
    if meta.get("processed_mesh_sha256") and meta["processed_mesh_sha256"] != fingerprint(
        mesh_path
    ):
        raise ValueError("Object mesh differs from the sequence geometry")
    interactions = (
        validate_interaction(interaction, sequence, sequence_path, mesh_path)
        if interaction
        else None
    )
    event_data = [] if events is None else events
    allowed_phases = {"approach", "grasp", "lift", "hold", "release"}
    for event in event_data:
        if (
            event.get("phase") not in allowed_phases
            or event.get("source") != "manual"
            or event.get("start_frame_id") not in a["frame_ids"]
            or event.get("end_frame_id") not in a["frame_ids"]
            or event["start_frame_id"] > event["end_frame_id"]
        ):
            raise ValueError("Events require valid original frame intervals and source=manual")
    output = Path(output)
    names = [
        "poses.npz",
        "object_mesh.npz",
        "retargeting_input.json",
        "manifest.json",
        "events.json",
    ]
    if interaction:
        names.append("interaction.npz")
    if any((output / name).exists() for name in names):
        raise FileExistsError("Dataset output already exists; use a new demo/output directory")
    output.mkdir(parents=True, exist_ok=True)
    # Pose is object->world; object_mesh vertices have already been converted to object-local metres.
    frame = dict(
        dataset_schema=1,
        coordinate_frame="table",
        world_axes=WORLD_AXES,
        transform_convention="T_parent_local",
        geometry_status=meta["geometry_status"],
        assumptions=meta.get("assumptions", []),
    )
    exported_hand = a["hand_world_m"][:, side : side + 1].copy()
    exported_pose = a["T_world_object"][:, None, :3, :].copy()
    exported_hand[~valid], exported_pose[~valid] = np.nan, np.nan
    save_npz(
        output / "poses.npz",
        frame_ids=a["frame_ids"],
        timestamps_s=a["timestamps_s"],
        pose_y=exported_pose,
        joint_3d=exported_hand,
        valid=valid,
        frame_metadata_json=np.array(json.dumps(frame)),
    )
    shutil.copyfile(mesh_path, output / "object_mesh.npz")
    config = dict(
        dataset_schema=1,
        object_index=0,
        object_name=meta["object_name"],
        mesh_to_object=np.eye(4).tolist(),
        hand_side=hand_side,
        left_hand_policy="preserve_source",
        length_unit="m",
        camera_axes=CAMERA_AXES,
        coordinate_frame="table",
        time_basis=meta["time_basis"],
        timestamps_s=a["timestamps_s"].tolist(),
        geometry_status=meta["geometry_status"],
        assumptions=meta.get("assumptions", []),
        task_id=meta["task_id"],
        demo_id=meta["demo_id"],
    )
    write_json(output / "retargeting_input.json", config)
    write_json(output / "events.json", event_data)
    if interaction:
        shutil.copyfile(interaction, output / "interaction.npz")
    manifest = dict(
        schema=1,
        **{k: meta[k] for k in ("task_id", "demo_id", "object_name", "geometry_status")},
        source=meta,
        sequence_sha256=fingerprint(sequence_path),
        files={name: fingerprint(output / name) for name in names if name != "manifest.json"},
        summary=sequence.summary(hand_side),
        interaction=None
        if not interactions
        else dict(
            label_source=interactions[1]["label_source"],
            valid_frames=int(interactions[0]["valid"].sum()),
            measured_force=False,
        ),
    )
    write_json(output / "manifest.json", manifest)
    return manifest
