#!/usr/bin/env python3
"""Offline RGB demonstration preparation; use ./run.sh dataset --help."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dex_manipulation.dataset.cli import main

if __name__ == "__main__":
    raise SystemExit(main(ROOT))
