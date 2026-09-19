#!/usr/bin/env python3
"""Extract source object 6D poses and MANO 21 points; preserve frame IDs and coordinates."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

def read_labels(source, frame_range=None):
    files = sorted(source.glob("labels_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        raise ValueError(f"No labels_*.npz in {source}")
    from dex_manipulation.data import frame_mask
    mask = frame_mask([int(path.stem.split('_')[-1]) for path in files], frame_range)
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
        if (not np.allclose(np.swapaxes(rotation, -1, -2) @ rotation, np.eye(3), atol=1e-4)
                or not np.allclose(np.linalg.det(rotation), 1, atol=1e-4)):
            raise ValueError(f"Invalid object rotation in {path}")
        poses.append(pose)
        joints.append(hand)
    return (np.array([int(p.stem.split("_")[-1]) for p in files], dtype=np.int32),
            np.stack(poses), np.stack(joints))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw/839512060362")
    parser.add_argument("--output", type=Path, default=ROOT / "data/poses/839512060362.npz")
    parser.add_argument('--config', type=Path, default=ROOT/'config/retargeting.json')
    parser.add_argument('--frame-range', nargs=2, type=int, help='Inclusive frame IDs; defaults to dataset config')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    frame_ids, poses, joints = read_labels(args.source, args.frame_range or config.get('frame_range'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, frame_ids=frame_ids, pose_y=poses, joint_3d=joints)
    print(f"{len(frame_ids)} frames; pose_y {poses.shape}; joint_3d {joints.shape}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
