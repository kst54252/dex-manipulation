#!/usr/bin/env python3
"""Run independent residual PPO with the actual floating Revo2/can USD assets."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args(argv=None, modes=("train", "evaluate"), default_mode="train"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/policy.json")
    parser.add_argument("--output", type=Path, default=ROOT / "local/results/policy")
    parser.add_argument("--mode", choices=modes, default=default_mode)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--policy-device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default=None, help="Physics and default policy device")
    parser.add_argument("--export", action="store_true", help="Export the checked checkpoint to JIT and ONNX after evaluation")
    parser.add_argument("--logger", choices=("tensorboard", "wandb", "none"), default="tensorboard")
    parser.add_argument("--console", choices=("pretty", "json"), default="pretty", help="Per-iteration terminal format; training.jsonl always contains full statistics")
    parser.add_argument("--source-profile", choices=("leap", "wuji"), default="leap", help="Source reward/material/default-joint-offset profile; hardware remains Revo2")
    parser.add_argument("--rsi-sampling", choices=("uniform", "adaptive"), default=None)
    rsi_flags = parser.add_mutually_exclusive_group()
    rsi_flags.add_argument("--enable-rsi", action="store_true", help="Enable random reference-state starts (off in the default config)")
    rsi_flags.add_argument("--disable-rsi", action="store_true", help="Always start from the first reference frame")
    parser.add_argument("--disable-augmentation", action="store_true")
    parser.add_argument("--disable-gravity-curriculum", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true", help="Train only; omit full-gravity baseline and final evaluation")
    parser.add_argument("--evaluation-protocol", choices=("source", "strict", "both"), default="both", help="Source PLAY noise/augmentation and 1 m deviation threshold; strict adds a fixed-reference diagnostic")
    args = parser.parse_args(argv)
    if any(v is not None and v < 1 for v in (args.num_envs, args.iterations, args.save_every)):
        parser.error("Counts must be positive")
    if args.mode == "evaluate" and not args.checkpoint:
        parser.error("Evaluation requires --checkpoint")
    if args.mode == "evaluate" and args.skip_evaluation:
        parser.error("--skip-evaluation cannot be used with --mode evaluate")

    return args


def launch(args, on_ready=None):
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless, "multi_gpu": False, "enable_crashreporter": False,
                         "extra_args": ["--enable", "isaacsim.core.api", "--enable", "isaacsim.core.prims"]})
    code = 0
    try:
        from dex_manipulation.policy.runner import run
        run(args, ROOT, on_ready=on_ready)
    except Exception:
        import traceback
        traceback.print_exc()
        code = 1
    finally:
        # Isaac Sim fast shutdown exits inside close(); preserve a failed check's status.
        app.close(exit_code=code)
    return code


if __name__ == "__main__":
    sys.exit(launch(parse_args()))
