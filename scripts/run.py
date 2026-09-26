#!/usr/bin/env python3
"""Short playback commands; `train` selects the separate training launcher."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    if sys.argv[1:2] == ["dataset"]:
        from dex_manipulation.dataset.cli import main

        sys.exit(main(ROOT, sys.argv[2:]))
    elif sys.argv[1:2] == ["tactile"]:
        from scripts.tactile import main

        sys.exit(main(ROOT, sys.argv[2:]))
    elif sys.argv[1:2] == ["tasks"]:
        from scripts.tasks import main

        sys.exit(main(sys.argv[2:]))
    elif sys.argv[1:2] == ["vcb"]:
        from scripts.vcb import main

        sys.exit(main(sys.argv[2:]))
    elif sys.argv[1:2] == ["ros"]:
        from scripts.ros import main

        sys.exit(main(sys.argv[2:]))
    elif sys.argv[1:2] == ["execute"]:
        from scripts.execute import main

        sys.exit(main(sys.argv[2:]))
    elif sys.argv[1:2] == ["train"]:
        from dex_manipulation.training import main

        sys.exit(main(ROOT, sys.argv[2:]))
    else:
        from dex_manipulation.launch import main

        sys.exit(main(ROOT))
