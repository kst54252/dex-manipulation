import argparse
import json
from pathlib import Path
import numpy as np
from .fk import HandModel
from dex_manipulation.geometry import CollisionScene, generate_object_points
from .dataset.adapter import load_sequence
from .retargeting import solve_sequence, SolverOptions


def main():
    parser = argparse.ArgumentParser(
        description="Independent floating-hand interaction-mesh retargeting"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--usd", required=True, type=Path)
    extract.add_argument("--keypoints", required=True, type=Path)
    extract.add_argument("--can-mesh", required=True, type=Path)
    extract.add_argument("--can-collision", required=True, type=Path)
    extract.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("retarget")
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--norm", choices=("regrind", "omni"), default="regrind")
    p.add_argument("--max-iterations", type=int, default=400)
    p.add_argument(
        "--direction-weight",
        type=float,
        default=0.5,
        help="Additional semantic finger-direction fidelity weight",
    )
    p.add_argument(
        "--position-weight",
        type=float,
        default=200.0,
        help="Additional semantic point squared-error weight, 1/m",
    )
    p.add_argument(
        "--wrist-linear-limit",
        type=float,
        default=1.0,
        help="Algorithmic floating wrist bound, m/s",
    )
    p.add_argument(
        "--wrist-angular-limit",
        type=float,
        default=8.0,
        help="Algorithmic floating wrist bound, rad/s",
    )
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--metadata", required=True, type=Path)
    p.add_argument(
        "--geometric-debug",
        action="store_true",
        help="Allows missing timing; all valid flags remain false",
    )
    view = sub.add_parser("view")
    view.add_argument("--result", required=True, type=Path)
    view.add_argument("--model-dir", required=True, type=Path)
    view.add_argument(
        "--reference-dir",
        type=Path,
        help="Original RGB/label directory for synchronized camera comparison",
    )
    args = parser.parse_args()
    if args.command == "extract":
        from .usd import extract_hand, extract_object

        args.output.mkdir(parents=True, exist_ok=True)
        extract_hand(args.usd, args.keypoints, args.output / "revo2.json")
        extract_object(args.can_mesh, args.can_collision, args.output / "can_mesh.npz")
        generate_object_points(args.output / "can_mesh.npz", args.output / "can_points.npz")
        print(args.output)
        return
    model = HandModel.load(args.model_dir / "revo2.json")
    if args.command == "view":
        from .viewer import comparison_viewer

        comparison_viewer(
            args.result / "trajectory.npz",
            args.result / "comparison.html",
            args.result / "report.json",
            model,
            args.model_dir / "can_mesh.npz",
            args.reference_dir,
        )
        return
    with np.load(args.model_dir / "can_points.npz", allow_pickle=False) as archive:
        points = archive["points_local"]
    debug = args.geometric_debug
    sequence = load_sequence(args.input, model, args.metadata, geometric_debug=debug)
    scene = CollisionScene(model, args.model_dir / "can_mesh.json")
    options = SolverOptions(
        laplacian_norm=args.norm,
        max_iterations=args.max_iterations,
        finger_direction_weight=args.direction_weight,
        keypoint_position_weight=args.position_weight,
        wrist_linear_velocity_m_s=args.wrist_linear_limit,
        wrist_angular_velocity_rad_s=args.wrist_angular_limit,
    )
    report = solve_sequence(
        model, sequence, points, scene, args.output, options, debug, args.max_frames
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("classification", "frame_count", "valid_count", "failed_frame_ids")
            }
        )
    )


if __name__ == "__main__":
    main()
