#!/usr/bin/env python3
"""Create floor-aligned poses/reference together without changing the camera originals."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    from dex_manipulation.coordinates import ground_dataset, stabilize_grounded_dataset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poses', type=Path, default=ROOT/'data/demo2/poses.npz')
    parser.add_argument('--reference', type=Path, default=ROOT/'data/demo2/retargeted.npz')
    parser.add_argument('--geometry', type=Path, default=ROOT/'assets/models/can_mesh.json')
    parser.add_argument('--config', type=Path, default=ROOT/'config/retargeting.json')
    parser.add_argument('--output', type=Path, default=ROOT/'data/demo2/grounded')
    parser.add_argument('--stabilize', action='store_true',
                        help='Level an existing grounded sequence at the initial can base; write separate derived inputs')
    args = parser.parse_args()
    metadata = (stabilize_grounded_dataset(args.poses, args.reference, args.geometry, args.output)
                if args.stabilize else ground_dataset(args.poses, args.reference, args.geometry, args.config, args.output))
    print(f"{args.output}: first frame {metadata['initial_frame_id']}, can bottom z={metadata['initial_collision_bottom_z_m']:.12g} m")


if __name__ == '__main__':
    main()
