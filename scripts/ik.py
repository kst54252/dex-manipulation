#!/usr/bin/env python3
"""Extract the RB3 model or generate a simulator-independent 12-DoF reference."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dex_manipulation.configuration import read_config


def main():
    from dex_manipulation.fk import ArmModel, HandModel
    from dex_manipulation.ik import IKOptions, solve_trajectory, checked_pose
    from dex_manipulation.transforms import inverse
    from dex_manipulation.scene import Workcell
    from dex_manipulation.data import resolve_demo_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("extract", "solve", "place"))
    parser.add_argument("--config", type=Path, default=ROOT / "config/tasks/can_pick/ik_demo2.json")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "local/results/ik/full")
    parser.add_argument(
        "--alignment",
        type=Path,
        help="JSON with world_from_source or measured base_from_source and status",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--trajectory-substeps",
        type=int,
        help="IK samples per original interval; preserve original times/poses",
    )
    parser.add_argument(
        "--seed", nargs=6, type=float, metavar="RAD", help="Explicit initial q in USD joint order"
    )
    args = parser.parse_args()
    config = read_config(args.config)
    workcell = Workcell.load(ROOT / config["workcell"])
    if args.command == "extract":
        from dex_manipulation.usd import extract_arm

        extract_arm(
            ROOT / config["usd"],
            config["base_path"],
            config["flange_path"],
            config["wrist_path"],
            ROOT / config["arm_model"],
        )
        print(ROOT / config["arm_model"])
        return
    model = ArmModel.load(ROOT / config["arm_model"])
    hand = HandModel.load(ROOT / config["hand_model"])
    source = resolve_demo_path(args.input or config["input"], ROOT)
    seed = np.array(args.seed if args.seed is not None else config["seed_q_rad"])
    alignment_path = args.alignment or (ROOT / config["alignment"] if config["alignment"] else None)
    with np.load(source, allow_pickle=False) as data:
        grounded = (
            "frame_metadata_json" in data
            and json.loads(str(data["frame_metadata_json"]))["coordinate_frame"] == "ground"
        )
    if grounded and alignment_path is not None:
        specified = workcell.resolve_alignment(read_config(alignment_path))
        matrix = checked_pose(specified["world_from_source"])
        if specified["status"].startswith("simulation") and not np.allclose(
            matrix[2], [0, 0, 1, 0], atol=1e-9, rtol=0
        ):
            parser.error("Grounded simulation placement must preserve tabletop world Z=0.")
    if args.command == "place":
        if grounded:
            if alignment_path is None:
                parser.error(
                    "Grounded input requires an explicit planar alignment; wrist-based placement would lift the can."
                )
            args.output.mkdir(parents=True, exist_ok=True)
            path = args.output / "alignment.json"
            path.write_text(json.dumps(specified, indent=2) + "\n")
            print(path)
            return
        # One fixed scene transform, never a per-frame repair or target deformation.
        with np.load(source, allow_pickle=False) as data:
            first = checked_pose(data["wrist_transform"][0])
        alignment = model.pose(seed) @ inverse(first)
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / "alignment.json"
        path.write_text(
            json.dumps(
                dict(
                    base_from_source=alignment.tolist(),
                    status="simulation_placement_not_calibration",
                    method="single rigid transform mapping source first wrist to FK(seed); all relative poses preserved",
                    initial_seed_q_rad=seed.tolist(),
                    hardware_calibrated=False,
                ),
                indent=2,
            )
            + "\n"
        )
        print(path)
        return
    if alignment_path is None:
        parser.error(
            "Specify --alignment. Camera coordinates cannot silently be treated as RB3 base coordinates. Use place for an explicit simulation scenario."
        )
    alignment = workcell.resolve_alignment(read_config(alignment_path))
    report = solve_trajectory(
        model,
        hand,
        source,
        args.output,
        alignment["base_from_source"],
        seed,
        IKOptions(**config["solver"]),
        alignment["status"],
        args.max_frames,
        args.trajectory_substeps
        if args.trajectory_substeps is not None
        else config.get("trajectory_substeps", 1),
        scene_placement=alignment["scene_placement"],
    )
    if report["continuous_reference_valid"]:
        from dex_manipulation.joint_trajectory import JointReference

        reference = JointReference(args.output / "trajectory.npz")
        reference.export_csv(args.output / "reference_12dof.csv")
        reference.export_csv(args.output / "reference_120hz.csv", rate_hz=120)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "success_count",
                    "frame_count",
                    "continuous_reference_valid",
                    "failed_frame_ids",
                )
            }
        )
    )
    return 0 if report["continuous_reference_valid"] else 2


if __name__ == "__main__":
    sys.exit(main())
