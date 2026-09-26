#!/usr/bin/env python3
"""Short playback commands; `train` selects the separate training launcher."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    from dex_manipulation.portable import prepare

    check = sys.argv[1:2] == ["check"]
    status = prepare(ROOT, verify=check)
    if status["installed"]:
        print(f"[runtime] Restored {status['installed']} execution inputs under local/", flush=True)
    if check:
        import json

        print(json.dumps(status, indent=2))
        sys.exit(1 if status["conflicts"] else 0)
    if sys.argv[1:2] == ["data"]:
        from scripts.data import main

        sys.exit(main(sys.argv[2:]))
    elif sys.argv[1:2] == ["retargeting"]:
        from dex_manipulation.cli import main

        sys.argv = ["./run.sh retargeting", *sys.argv[2:]]
        sys.exit(main())
    elif sys.argv[1:2] == ["ik"]:
        from scripts.ik import main

        sys.argv = ["./run.sh ik", *sys.argv[2:]]
        sys.exit(main())
    elif sys.argv[1:2] == ["tracking"]:
        from scripts.tracking import main

        sys.argv = ["./run.sh tracking", *sys.argv[2:]]
        sys.exit(main())
    elif sys.argv[1:2] == ["retarget-contact"]:
        from scripts.contact_retargeting import main

        sys.exit(main(sys.argv[2:]))
    elif sys.argv[1:2] == ["dataset"]:
        from dex_manipulation.dataset.cli import main

        sys.exit(main(ROOT, sys.argv[2:]))
    elif sys.argv[1:2] == ["tactile"]:
        from dex_manipulation.sensors.tactile import main

        sys.exit(main(ROOT, sys.argv[2:]))
    elif sys.argv[1:2] == ["tasks"]:
        from dex_manipulation.tasks.registry import main

        sys.exit(main(sys.argv[2:]))
    elif sys.argv[1:2] == ["vcb"]:
        from dex_manipulation.robot.vcb import main

        sys.exit(main(sys.argv[2:]))
    elif sys.argv[1:2] == ["robot"]:
        from scripts.robot import main

        sys.exit(main(sys.argv[2:]))
    elif sys.argv[1:2] == ["ros"]:
        from dex_manipulation.robot.ros import main

        sys.exit(main(sys.argv[2:]))
    elif sys.argv[1:2] == ["execute"]:
        from dex_manipulation.robot.recording import main

        sys.exit(main(sys.argv[2:]))
    elif sys.argv[1:2] == ["train"]:
        from dex_manipulation.training import main

        sys.exit(main(ROOT, sys.argv[2:]))
    else:
        from dex_manipulation.launch import main

        sys.exit(main(ROOT))
