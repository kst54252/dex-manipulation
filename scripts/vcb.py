#!/usr/bin/env python3
"""Short commands for an external Rainbow Virtual Control Box VM."""

from dex_manipulation.configuration import read_config
import argparse
import asyncio
from datetime import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="./run.sh vcb")
    parser.add_argument(
        "mode", choices=("init", "probe", "plan", "prepare", "bridge", "send", "mirror", "status")
    )
    parser.add_argument("--config", type=Path, default=ROOT / "local/vcb.json")
    parser.add_argument("--address", help="Actual VCB VM IP/hostname; only for init")
    parser.add_argument(
        "--arm-sign", type=float, nargs=6, help="Verified USD-to-VCB axis signs, for init"
    )
    parser.add_argument(
        "--arm-offset-deg", type=float, nargs=6, help="Verified USD-to-VCB zero offsets, for init"
    )
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--ros-config", type=Path, default=ROOT / "config/ros_vcb.json")
    parser.add_argument(
        "--read-only", action="store_true", help="Bridge without accepting motion goals"
    )
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true", help="Mirror without a window")
    args = parser.parse_args(argv)
    try:
        from dex_manipulation.vcb import connection_plan, motion_plan, prepare, probe

        path = (ROOT / args.config).resolve()
        if args.mode == "init":
            if not path.is_relative_to(ROOT / "local"):
                raise ValueError("Keep VM settings under local/")
            config = read_config(ROOT / "config/vcb.example.json")
            config["rb3"]["address"] = args.address
            connection_plan(config)
            if (args.arm_sign is None) != (args.arm_offset_deg is None):
                raise ValueError(
                    "Provide both axis signs and zero offsets, or leave both for later"
                )
            if args.arm_sign is not None:
                from dex_manipulation.hardware import ARM_NAMES, ArmCalibration

                config["mapping"] = dict(arm_sign=args.arm_sign, arm_offset_deg=args.arm_offset_deg)
                ArmCalibration(ARM_NAMES, config["mapping"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as file:
                file.write(json.dumps(config, indent=2) + "\n")
            print(f"VCB config: {path}\nNext: ./run.sh vcb probe")
            return 0
        if args.address or args.arm_sign is not None or args.arm_offset_deg is not None:
            raise ValueError(
                "Address/mapping options belong to init; otherwise edit the selected config"
            )
        if args.read_only and args.mode != "bridge":
            raise ValueError("--read-only belongs to bridge")
        if args.mode in ("bridge", "send", "mirror", "status"):
            from scripts.ros import main as ros_main

            forwarded = [
                args.mode,
                "--config",
                str(args.ros_config),
                "--seconds",
                str(args.seconds),
            ]
            if args.mode == "bridge":
                forwarded += ["--backend", "vcb", "--vcb-config", str(path)]
                if not args.read_only:
                    forwarded += ["--enable-motion"]
            if args.recording:
                forwarded += ["--recording", str(args.recording)]
            if args.headless:
                forwarded += ["--headless"]
            return ros_main(forwarded)
        config = read_config(path)
        if args.mode == "probe":
            report = asyncio.run(probe(config))
        else:
            from dex_manipulation.execution import RecordedCommands

            defaults = read_config(ROOT / "config/execution.json")
            recording = RecordedCommands(ROOT / (args.recording or defaults["recording"]))
            plan = motion_plan(recording, config)
            report = plan if args.mode == "plan" else asyncio.run(prepare(recording, config))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.mode in ("probe", "prepare"):
            output = ROOT / "local/results/vcb" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output.mkdir(parents=True, exist_ok=False)
            (output / f"{args.mode}.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"Log: {output}")
        return 0
    except KeyboardInterrupt:
        return 130
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        ImportError,
        TimeoutError,
    ) as error:
        print(f"VCB 실행 실패: {error}", flush=True)
        return 2
