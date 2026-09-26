"""Simple task/demo commands for offline RGB dataset preparation."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from .pipeline import DatasetProject
from .schema import load_npz, read_json, save_npz, write_json


def parser():
    p = argparse.ArgumentParser(prog="./run.sh dataset", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    descriptions = {
        "init": "Prepare demo directories and an input checklist",
        "status": "Show input/model availability without starting Isaac",
        "import": "Import RGB video using real presentation timestamps",
        "capture": "Record RGB camera frames and host receive times",
        "calibrate": "Estimate camera intrinsics from checkerboard RGB images",
        "track": "Track measured object/table markers in RGB",
        "annotate": "Create browser hand bounding-box editor",
        "hands": "Run official HaMeR in its own environment",
        "align": "Optionally align hand geometry to measured joint anchors",
        "build": "Assemble camera/hand/object annotations in the table frame",
        "infer": "Run official EgoPHI contact/force inference",
        "review": "Create RGB/3D comparison, without opening a window",
        "export": "Export immutable retargeting input to the demo directory",
    }
    for name, description in descriptions.items():
        q = sub.add_parser(name, help=description, description=description)
        q.add_argument("demo", nargs="?", default="1")
        q.add_argument("--task", default="drilling")
        q.add_argument("--config", type=Path, help="Task capture configuration")
        if name not in ("init", "status"):
            q.add_argument("--output", type=Path)
        if name in ("infer", "review", "export"):
            q.add_argument(
                "--build", type=Path, help="Build directory; defaults to latest successful build"
            )
        if name in ("hands", "infer"):
            q.add_argument("--python", type=Path, help="Separate estimator Python executable")
        if name == "import":
            q.add_argument("--video", required=True, type=Path)
            q.add_argument(
                "--timestamps", type=Path, help="Optional JSON array of actual frame times"
            )
        elif name == "capture":
            q.add_argument("--camera", type=int, required=True)
            q.add_argument("--seconds", type=float, required=True)
        elif name == "calibrate":
            q.add_argument("--images", type=Path, required=True)
            q.add_argument("--board", nargs=2, type=int, required=True, metavar=("COLUMNS", "ROWS"))
            q.add_argument("--square-m", type=float, required=True)
        elif name == "track":
            q.add_argument(
                "--markers", type=Path, required=True, help="Measured marker corner geometry JSON"
            )
        elif name == "align":
            q.add_argument("--anchors", type=Path, required=True)
            q.add_argument("--max-error-m", type=float, default=0.01)
        elif name == "build":
            q.add_argument("--hands", type=Path, help="Override hand annotation path")
    return p


def main(root, argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        project = DatasetProject(root, args.task, args.demo, args.config)
        command = args.command
        if command == "init":
            result = project.init()
        elif command == "status":
            result = project.status()
        elif command == "import":
            from .capture import import_video

            result = import_video(
                args.video,
                args.output or project.path("capture").parent,
                read_json(args.timestamps) if args.timestamps else None,
            )
            result = {k: result[k] for k in ("time_basis", "image_size")}
            result["output"] = str(args.output or project.path("capture").parent)
        elif command == "capture":
            from .capture import record_camera

            output = args.output or project.path("capture").parent
            recording = record_camera(args.camera, args.seconds, output)
            result = dict(
                frames=len(recording["frame_ids"]),
                output=str(output),
                time_basis=recording["time_basis"],
            )
        elif command == "calibrate":
            from .calibration import calibrate_checkerboard

            output = args.output or project.path("calibration")
            if output.exists():
                raise FileExistsError(output)
            images = sorted(
                p for p in args.images.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg")
            )
            result = calibrate_checkerboard(images, args.board, args.square_m)
            write_json(output, result)
            result = dict(
                output=str(output),
                rms_px=result["calibration_rms_px"],
                images=len(result["images"]),
            )
        elif command == "track":
            from .calibration import load_calibration
            from .object_pose import track_markers

            marker_config = read_json(args.markers)
            key = "camera_poses" if marker_config["frame"] == "world" else "object_poses"
            result = track_markers(
                project.path("capture"),
                load_calibration(project.path("calibration")),
                marker_config,
                args.output or project.path(key),
            )
        elif command == "hands":
            from .egophi import run_worker

            config = read_json(project.path("hamer_config"))
            python = args.python or config.get("python")
            if not python:
                raise ValueError(
                    "Set config/dataset/hamer.json python or --python to the HaMeR environment"
                )
            output = args.output or project.path("hands")
            run_worker(
                root,
                "hamer",
                python,
                dict(
                    repo=str((project.root / config["repo"]).resolve()),
                    checkpoint=str((project.root / config["checkpoint"]).resolve()),
                    device=config["device"],
                    capture=str(project.path("capture")),
                    boxes=str(project.path("boxes")),
                ),
                output,
                project.work / "hamer.request.json",
            )
            result = dict(output=str(output), geometry_status="monocular_estimate")
        elif command == "align":
            from .hand_pose import align_hands

            output = args.output or project.work / "hands_metric.npz"
            if output.exists():
                raise FileExistsError(output)
            aligned = align_hands(
                load_npz(project.path("hands")), load_npz(args.anchors), args.max_error_m
            )
            save_npz(output, **aligned)
            result = dict(output=str(output), valid=int(aligned["valid"].sum()))
        elif command == "build":
            output = project.build(args.output, args.hands)
            result = dict(directory=str(output), **read_json(output / "summary.json"))
        elif command in ("review", "annotate"):
            from .review import make_review

            build = (args.build or project.latest()) if command == "review" else None
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            output = args.output or project.work / (command + "_" + stamp)
            interaction = build / "interaction.npz" if build else None
            page = make_review(
                project.path("capture"),
                output,
                build / "sequence.npz" if build else None,
                build / "mesh.npz" if build else None,
                interaction if interaction and interaction.is_file() else None,
            )
            result = dict(html=str(page))
        elif command == "infer":
            from .egophi import run_worker

            config = read_json(project.path("egophi_config"))
            python = args.python or config.get("python")
            if not python:
                raise ValueError(
                    "Set config/dataset/egophi.json python or --python to the EgoPHI environment"
                )
            build = args.build or project.latest()
            output = args.output or build / "interaction.npz"
            run_worker(
                root,
                "egophi",
                python,
                dict(
                    repo=str((project.root / config["repo"]).resolve()),
                    checkpoint=str((project.root / config["checkpoint"]).resolve()),
                    options=config,
                    capture=str(project.path("capture")),
                    sequence=str((build / "sequence.npz").resolve()),
                    mesh=str((build / "mesh.npz").resolve()),
                ),
                output,
                project.work / "egophi.request.json",
            )
            saved = load_npz(output)
            result = dict(
                output=str(output),
                predictions=int(saved["prediction_available"].sum()),
                valid=int(saved["valid"].sum()),
                force_unit="normalized_model_output",
            )
        elif command == "export":
            from .export import export_demo

            build = args.build or project.latest()
            interaction = build / "interaction.npz"
            result = export_demo(
                build / "sequence.npz",
                build / "mesh.npz",
                args.output or project.demo_dir,
                project.config["hand_side"],
                interaction if interaction.is_file() else None,
                read_json(project.path("events")) if project.path("events").is_file() else None,
            )
            result = dict(
                output=str(args.output or project.demo_dir),
                geometry_status=result["geometry_status"],
                **result["summary"],
            )
        else:
            raise AssertionError(command)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as error:
        p.exit(2, f"dataset: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main(Path(__file__).resolve().parents[3]))
