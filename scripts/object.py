#!/usr/bin/env python3
"""Build the configured stepped can USD, collision model and persistent keypoints."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    from dex_manipulation.object import build_can
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=ROOT/'assets/can/spec.json')
    parser.add_argument('--asset', type=Path, default=ROOT/'assets/can/can.usda')
    parser.add_argument('--model-dir', type=Path, default=ROOT/'assets/models')
    args = parser.parse_args()
    metadata = build_can(args.spec, args.asset, args.model_dir)
    # Keep the installed project's asset metadata portable along with its USD.
    if args.asset.resolve().is_relative_to(ROOT):
        for key in ('mesh_source', 'collision_source'):
            metadata[key] = str(args.asset.resolve().relative_to(ROOT))
        (args.model_dir/'can_mesh.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f"{metadata['name']}: {len(metadata['collision_shapes'])} colliders, fingerprint={metadata['fingerprint']}")


if __name__ == '__main__':main()
