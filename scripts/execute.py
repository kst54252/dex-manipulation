#!/usr/bin/env python3
"""Record, replay or stream a fixed arm+hand motion without object perception."""

import argparse
import asyncio
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from dex_manipulation.configuration import read_config


def _main(argv=None):
    defaults = read_config(ROOT / "config/execution.json")
    parser = argparse.ArgumentParser(prog="./run.sh execute")
    parser.add_argument("mode", choices=("record", "replay", "inspect", "dry-run", "hardware"))
    parser.add_argument("--recording", type=Path, default=ROOT / defaults["recording"])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="record: latest completed demo policy by default; replay: recorded checkpoint",
    )
    parser.add_argument("--arm-config", type=Path, default=ROOT / defaults["arm_config"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hold", type=float, default=defaults["hold_s"])
    parser.add_argument(
        "--gui", action="store_true", help="Show the requested simulation (default headless)"
    )
    parser.add_argument(
        "--hardware-config", type=Path, default=ROOT / "config/hardware.example.json"
    )
    parser.add_argument(
        "--send", action="store_true", help="Actually connect and stream to commissioned hardware"
    )
    args = parser.parse_args(argv)
    for name in ("recording", "checkpoint", "arm_config", "hardware_config", "output"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, (ROOT / value).resolve())
    if args.send and args.mode != "hardware":
        parser.error("--send requires hardware mode")
    if args.hold <= 0:
        parser.error("--hold must be positive")
    from dex_manipulation.execution import RecordedCommands, MockBackend, stream

    frozen = None if args.mode == "record" else RecordedCommands(args.recording)
    if args.checkpoint is None and args.mode in ("record", "replay"):
        if frozen:
            args.checkpoint = (ROOT / frozen.metadata["checkpoint"]).resolve()
        else:
            from dex_manipulation.policies import latest_policy

            args.checkpoint = Path(latest_policy(ROOT, defaults["demo_id"], "arm")["checkpoint"])
        print(f"정책: {args.checkpoint}", flush=True)
    if args.mode == "inspect":
        print(json.dumps(frozen.summary(), indent=2))
        return 0
    output = (
        args.output
        or ROOT
        / (
            "local/results/execution/"
            + args.mode
            + "_"
            + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
    ).resolve()
    if not output.is_relative_to(ROOT / "local") or output.exists():
        parser.error("Use a new output directory under local/")
    if args.mode == "dry-run":
        result = asyncio.run(
            stream(
                frozen,
                MockBackend(frozen.initial),
                output,
                start_tolerance_rad=[0.01] * 12,
                tracking_tolerance_rad=[0.01] * 12,
                maximum_feedback_age_s=0.1,
                maximum_lateness_s=0.02,
                hold_s=args.hold,
            )
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.mode == "hardware":
        from dex_manipulation.hardware import hardware_plan, execute

        config = read_config(args.hardware_config)
        config["simulation_validation"] = str((ROOT / config["simulation_validation"]).resolve())
        plan = hardware_plan(frozen, config)
        if not args.send:
            print(json.dumps(plan, indent=2))
            return 0
        print(json.dumps(asyncio.run(execute(frozen, config, output, args.hold)), indent=2))
        return 0
    output.mkdir(parents=True)
    config = (
        frozen.metadata["config"]
        if frozen
        else read_config(args.checkpoint.parent / "config.resolved.json")
    )
    if frozen and args.checkpoint.resolve() != Path(frozen.metadata["checkpoint"]).resolve():
        parser.error("Replay must use the checkpoint recorded in the trajectory")
    config_path = output / "physics_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    from scripts.policy import parse_args, launch

    sim_args = parse_args(
        [
            "--mode",
            "play",
            "--robot",
            "arm",
            "--episodes",
            "1",
            "--speed",
            "1",
            "--evaluation-protocol",
            "strict",
            "--contact-materials",
            "checkpoint",
            "--table-safety",
            "checkpoint",
            "--checkpoint",
            str(args.checkpoint),
            "--config",
            str(config_path),
            "--arm-config",
            str(args.arm_config),
            "--output",
            str(output),
        ]
        + ([] if args.gui else ["--headless"])
    )

    def ready(env, metadata):
        from dex_manipulation.execution_sim import run

        return run(
            env,
            metadata,
            args.checkpoint,
            output,
            replay_path=args.recording if args.mode == "replay" else None,
            hold_s=args.hold,
        )

    return launch(sim_args, on_ready=ready)


def main(argv=None):
    try:
        return _main(argv)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"실행 준비/제어 실패: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
