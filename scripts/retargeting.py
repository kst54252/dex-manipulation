#!/usr/bin/env python3
"""Extract models, retarget hand motion, and create a comparison viewer."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dex_manipulation.cli import main

if __name__ == "__main__":
    main()
