"""Prepare source demos, scene coordinates and derived training references."""

import argparse
from pathlib import Path
import sys
import numpy as np
import json


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dex_manipulation.configuration import read_config
from dex_manipulation.fk import HandModel


def read_labels(source, frame_range=None):
    files = sorted(source.glob("labels_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        raise ValueError(f"No labels_*.npz in {source}")
    from dex_manipulation.data import frame_mask

    mask = frame_mask([int(path.stem.split("_")[-1]) for path in files], frame_range)
    files = [path for path, keep in zip(files, mask) if keep]
    poses, joints = [], []
    for path in files:
        with np.load(path, allow_pickle=False) as labels:
            pose, hand = labels["pose_y"], labels["joint_3d"]
        if pose.ndim != 3 or pose.shape[1:] != (3, 4) or not len(pose):
            raise ValueError(f"Invalid pose_y shape in {path}: {pose.shape}")
        if hand.shape != (1, 21, 3):
            raise ValueError(f"Invalid joint_3d shape in {path}: {hand.shape}")
        if not np.isfinite(pose).all() or not np.isfinite(hand).all() or np.all(hand == -1):
            raise ValueError(f"Missing or non-finite annotations in {path}")
        rotation = pose[..., :3].astype(np.float64)
        if not np.allclose(
            np.swapaxes(rotation, -1, -2) @ rotation, np.eye(3), atol=1e-4
        ) or not np.allclose(np.linalg.det(rotation), 1, atol=1e-4):
            raise ValueError(f"Invalid object rotation in {path}")
        poses.append(pose)
        joints.append(hand)
    return (
        np.array([int(p.stem.split("_")[-1]) for p in files], dtype=np.int32),
        np.stack(poses),
        np.stack(joints),
    )


def extract_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/can_grasping/demo2/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data/can_grasping/demo2/poses.npz")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/tasks/can_pick/retargeting_demo2.json"
    )
    parser.add_argument(
        "--frame-range", nargs=2, type=int, help="Inclusive frame IDs; defaults to dataset config"
    )
    args = parser.parse_args(argv)
    config = read_config(args.config)
    frame_ids, poses, joints = read_labels(
        args.source, args.frame_range or config.get("frame_range")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, frame_ids=frame_ids, pose_y=poses, joint_3d=joints)
    print(f"{len(frame_ids)} frames; pose_y {poses.shape}; joint_3d {joints.shape}")
    print(args.output.resolve())


def demo_main(argv=None):
    from dex_manipulation.data import import_human_demo

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--object-geometry", type=Path, default=ROOT / "assets/models/can_mesh.json"
    )
    args = parser.parse_args(argv)
    report = import_human_demo(args.source, args.output, args.object_geometry)
    print(
        json.dumps(
            dict(
                output=str(args.output),
                source=report["source_demo"],
                frame_range=report["frame_range"],
                playback_fps=report["source_playback_fps"],
            ),
            indent=2,
        )
    )


def ground_main(argv=None):
    from dex_manipulation.coordinates import ground_dataset, stabilize_grounded_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", type=Path, default=ROOT / "data/can_grasping/demo2/poses.npz")
    parser.add_argument(
        "--reference", type=Path, default=ROOT / "data/can_grasping/demo2/retargeted.npz"
    )
    parser.add_argument("--geometry", type=Path, default=ROOT / "assets/models/can_mesh.json")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/tasks/can_pick/retargeting_demo2.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/can_grasping/demo2/grounded")
    parser.add_argument(
        "--stabilize",
        action="store_true",
        help="Level an existing grounded sequence at the initial can base; write separate derived inputs",
    )
    args = parser.parse_args(argv)
    metadata = (
        stabilize_grounded_dataset(args.poses, args.reference, args.geometry, args.output)
        if args.stabilize
        else ground_dataset(args.poses, args.reference, args.geometry, args.config, args.output)
    )
    print(
        f"{args.output}: first frame {metadata['initial_frame_id']}, can bottom z={metadata['initial_collision_bottom_z_m']:.12g} m"
    )


def object_main(argv=None):
    from dex_manipulation.object import build_can

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=ROOT / "assets/can/spec.json")
    parser.add_argument("--asset", type=Path, default=ROOT / "assets/can/can.usda")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "assets/models")
    args = parser.parse_args(argv)
    metadata = build_can(args.spec, args.asset, args.model_dir)
    # Keep the installed project's asset metadata portable along with its USD.
    if args.asset.resolve().is_relative_to(ROOT):
        for key in ("mesh_source", "collision_source"):
            metadata[key] = str(args.asset.resolve().relative_to(ROOT))
        (args.model_dir / "can_mesh.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"{metadata['name']}: {len(metadata['collision_shapes'])} colliders, fingerprint={metadata['fingerprint']}"
    )


def rollout_main(argv=None):
    from dex_manipulation.policy.rollout_reference import prepare_contact_reference

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reference", type=Path, default=ROOT / "data/can_grasping/demo2/grounded/reference.npz"
    )
    p.add_argument("--rollout", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, default=ROOT / "assets/models/revo2.json")
    p.add_argument("--object-geometry", type=Path, default=ROOT / "assets/models/can_mesh.json")
    p.add_argument("--environment", type=int, default=0)
    a = p.parse_args(argv)
    report = prepare_contact_reference(
        a.reference,
        a.rollout,
        a.output,
        HandModel.load(a.model),
        a.object_geometry,
        environment=a.environment,
    )
    print(
        json.dumps(
            dict(
                output=str(a.output),
                frames=len(report["frames"]),
                object_targets_changed=False,
                solver_status_failures=[
                    r["frame_id"] for r in report["frames"] if not r["solver_success"]
                ],
            ),
            indent=2,
        )
    )


def contact_main(argv=None):
    from dex_manipulation.grasp_reference import prepare_grasp_reference

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start-frame", type=int, required=True)
    p.add_argument("--blend-start", type=float, required=True)
    p.add_argument("--table-margin", type=float, default=0.008)
    p.add_argument("--side-contact", action="store_true")
    a = p.parse_args(argv)
    r = prepare_grasp_reference(
        a.source,
        a.output,
        HandModel.load(ROOT / "assets/models/revo2.json"),
        ROOT / "assets/models/can_mesh.json",
        start_frame_id=a.start_frame,
        blend_start_s=a.blend_start,
        table_margin_m=a.table_margin,
        side_contact=a.side_contact,
    )
    print(json.dumps({k: v for k, v in r.items() if k not in ("frames", "transitions")}, indent=2))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    handlers = {
        "extract": extract_main,
        "contact-reference": contact_main,
        "import": demo_main,
        "ground": ground_main,
        "object": object_main,
        "rollout-reference": rollout_main,
    }
    parser = argparse.ArgumentParser(prog="./run.sh data", description=__doc__)
    parser.add_argument("command", choices=tuple(handlers))
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    if argv[0].startswith("--"):
        return extract_main(argv)  # Preserve scripts/data.py extraction options.
    command = parser.parse_args(argv[:1]).command
    return handlers[command](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
