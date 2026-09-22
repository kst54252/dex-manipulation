#!/usr/bin/env python3
"""Import a metadata-described human demo for independent retargeting."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from dex_manipulation.data import import_human_demo

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--object-geometry", type=Path, default=ROOT / "assets/models/can_mesh.json"
    )
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
