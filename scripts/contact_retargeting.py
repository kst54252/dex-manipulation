#!/usr/bin/env python3
"""Independent contact retargeting experiments; existing outputs stay immutable."""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Contact retargeting, can_pick demo 2")
    parser.add_argument("method", choices=("hard", "virtual"))
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/tasks/can_pick/contact_retargeting.json"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New experiment directory under local/"
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--anchor-mode",
        choices=("human_surface_proxy", "robot_surface_seed"),
        help="Keep projected human contact points, or explicitly adapt placement to robot morphology",
    )
    parser.add_argument(
        "--controls",
        type=Path,
        help="Replay saved virtual-method controls with guidance OFF, without optimization",
    )
    args = parser.parse_args(argv)
    if not args.output.resolve().is_relative_to(ROOT / "local"):
        parser.error("Experiment outputs must be under local/")
    if args.output.exists():
        parser.error("Output exists; choose a new directory to preserve previous results")
    if args.controls is not None and args.method != "virtual":
        parser.error("--controls requires virtual")
    if args.method == "virtual" and args.max_frames is not None:
        parser.error("--max-frames is for geometric preflight only")
    cfg = json.loads(args.config.read_text())
    if args.anchor_mode is not None:
        cfg["anchor_mode"] = args.anchor_mode
    if cfg["task_id"] != "can_pick" or str(cfg["demo_id"]) != "2":
        parser.error("This contact annotation belongs to can_pick demo 2 only")
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    if args.method == "virtual":
        from isaacsim import SimulationApp

        app = SimulationApp(
            dict(
                headless=True,
                multi_gpu=False,
                enable_crashreporter=False,
                disable_viewport_updates=True,
                extra_args=["--enable", "isaacsim.core.api", "--enable", "isaacsim.core.prims"],
            )
        )
        code = 0
        try:
            from dex_manipulation.contact_physics import optimize_virtual_contacts

            optimize_virtual_contacts(ROOT, cfg, args.output, controls_path=args.controls)
        except Exception:
            import traceback

            traceback.print_exc()
            code = 1
        finally:
            app.close(exit_code=code)
        return code
    import numpy as np
    from dex_manipulation.contact_retargeting import (
        extract_contacts,
        adapt_contact_anchors,
        HardContacts,
    )
    from dex_manipulation.fk import HandModel
    from dex_manipulation.geometry import CollisionScene
    from dex_manipulation.retargeting import solve_sequence, SolverOptions
    from dex_manipulation.data import Sequence, resolve_demo_path

    with np.load(resolve_demo_path(cfg["reference"], ROOT), allow_pickle=False) as d:
        data = {k: d[k].copy() for k in d.files}
    model = HandModel.load(ROOT / cfg["model"])
    if list(data["semantic_names"]) != model.semantic_names:
        raise ValueError("Human keypoint semantics differ from model")
    plan = extract_contacts(data, ROOT / cfg["object_mesh"], cfg["contacts"])
    scene = CollisionScene(model, ROOT / cfg["object_geometry"])
    if cfg.get("anchor_mode", "human_surface_proxy") == "robot_surface_seed":
        plan = adapt_contact_anchors(plan, model, scene, data)
    elif cfg.get("anchor_mode", "human_surface_proxy") != "human_surface_proxy":
        raise ValueError("Unknown anchor mode")
    plan.save(args.output / "contacts.npz")
    (args.output / "contacts.json").write_text(json.dumps(plan.metadata, indent=2))
    hard = HardContacts(
        model, scene, plan, cfg["hard"]["tolerance_m"], cfg["hard"]["table_margin_m"]
    )
    options = SolverOptions(
        **{k: cfg["hard"][k] for k in ("max_iterations", "max_attempts", "collision_substeps")}
    )
    seq = Sequence(
        data["frame_ids"],
        data["human_keypoints"],
        data["object_transform"],
        data["timestamps_s"],
        np.ones(len(data["frame_ids"]), bool),
        dict(
            frame_metadata=json.loads(str(data["frame_metadata_json"])),
            contact_proxy=plan.metadata,
            initial_reference=cfg["reference"],
        ),
    )
    solve_sequence(
        model,
        seq,
        data["object_points_local"],
        scene,
        args.output,
        options,
        max_frames=args.max_frames,
        contact_constraints=hard,
        initial_trajectory=data,
    )
    from dex_manipulation.viewer import comparison_viewer

    comparison_viewer(
        args.output / "trajectory.npz",
        args.output / "comparison.html",
        args.output / "report.json",
        model,
        ROOT / cfg["object_mesh"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
