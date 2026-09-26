#!/usr/bin/env python3
"""Run optional neural estimators outside the simulator's Python environment."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("hamer", "egophi"))
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    if Path(request["output"]).exists():
        parser.error("Output already exists")
    if args.backend == "egophi":
        from dex_manipulation.dataset.egophi import infer
    else:
        from dex_manipulation.dataset.hamer import infer
    infer(request)


if __name__ == "__main__":
    main()
