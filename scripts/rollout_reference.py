"""Prepare a separate policy contact reference from an independently evaluated rollout."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dex_manipulation.fk import HandModel
from dex_manipulation.policy.rollout_reference import prepare_contact_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, default=ROOT / "data/demo2/grounded/reference.npz")
    p.add_argument("--rollout", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, default=ROOT / "assets/models/revo2.json")
    p.add_argument("--object-geometry", type=Path, default=ROOT / "assets/models/can_mesh.json")
    p.add_argument("--environment", type=int, default=0)
    a = p.parse_args()
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


if __name__ == "__main__":
    main()
