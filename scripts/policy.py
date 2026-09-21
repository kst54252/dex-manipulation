#!/usr/bin/env python3
"""Run residual PPO on physical floating Revo2 or assembled RB3+Revo2 with online IK."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args(argv=None, modes=("train", "evaluate", "play"), default_mode="train"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/policy.json")
    parser.add_argument("--output", type=Path, default=ROOT / "local/results/policy")
    parser.add_argument("--mode", choices=modes, default=default_mode)
    parser.add_argument("--robot", choices=("floating", "arm"), default="floating", help="Floating hand or physical RB3+Revo2; arm training needs an arm_training config")
    parser.add_argument("--arm-config", type=Path, default=None)
    parser.add_argument('--initialize-actor',type=Path,help='Initialize an arm-training actor from a compatible floating checkpoint; not resume')
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=0, help="Play mode: repeat until stopped (0), or stop after this many episodes")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--policy-device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default=None, help="Physics and default policy device")
    parser.add_argument("--export", action="store_true", help="Export the checked checkpoint to JIT and ONNX after evaluation")
    parser.add_argument("--logger", choices=("tensorboard", "wandb", "none"), default="tensorboard")
    parser.add_argument("--console", choices=("pretty", "json"), default="pretty", help="Per-iteration terminal format; training.jsonl always contains full statistics")
    parser.add_argument("--source-profile", choices=("revo2", "leap", "wuji"), default=None, help="Recipe label/profile; default uses the config file's source_profile; hardware remains Revo2")
    parser.add_argument("--rsi-sampling", choices=("uniform_frames", "uniform", "adaptive"), default=None)
    rsi_flags = parser.add_mutually_exclusive_group()
    rsi_flags.add_argument("--enable-rsi", action="store_true", help="Enable random reference-state starts (on in the source-aligned default config)")
    rsi_flags.add_argument("--disable-rsi", action="store_true", help="Always start from the first reference frame")
    parser.add_argument("--disable-augmentation", action="store_true")
    parser.add_argument("--disable-gravity-curriculum", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true", help="Train only; omit full-gravity baseline and final evaluation")
    parser.add_argument("--evaluation-protocol", choices=("source", "strict", "both"), default="both", help="Source PLAY noise/augmentation and 1 m deviation threshold; strict adds a fixed-reference diagnostic")
    parser.add_argument("--motion-control", choices=("checkpoint", "stable"), default=None,
                        help="Play/evaluate default to the trained controller. Stable explicitly enables the optional command governor from config/policy.json. Training uses its config.")
    parser.add_argument('--table-safety', choices=('checkpoint','protect'), default=None,
                        help='Play defaults to protect: collision-geometry tabletop command guard. Checkpoint reproduces the trained setting. Training uses its config.')
    args = parser.parse_args(argv)
    if args.robot == 'arm':
        if args.mode != 'play' and args.config == ROOT/'config/policy.json':
            parser.error('Arm training/evaluation requires an explicit arm-training config')
        if args.evaluation_protocol == 'source' or args.motion_control == 'stable':
            parser.error('Arm playback uses strict observations and the checkpoint target mapping')
    if any(v is not None and v < 1 for v in (args.num_envs, args.iterations, args.save_every)):
        parser.error("Counts must be positive")
    if args.mode in ("evaluate", "play") and not args.checkpoint:
        parser.error("Evaluation/play requires --checkpoint")
    if args.mode in ("evaluate", "play") and args.skip_evaluation:
        parser.error("--skip-evaluation is only for training")
    if args.episodes < 0:
        parser.error("--episodes must be nonnegative")
    if args.mode == "train" and args.motion_control is not None:
        parser.error("Set motion_control in the training config; this flag is for play/evaluate")
    if args.mode == 'train' and args.table_safety is not None:
        parser.error('Set table_safety in the training config; this flag is for play/evaluate')
    if args.initialize_actor and (args.mode!='train' or args.robot!='arm' or args.checkpoint):
        parser.error('--initialize-actor is only for new arm training, without --checkpoint')
    if args.mode == "play":
        if args.num_envs not in (None, 1):
            parser.error("Play mode uses one robot (--num-envs 1)")
        args.num_envs = 1
        if args.output == ROOT / "local/results/policy":
            args.output = ROOT / ('local/results/policy/play_arm' if args.robot == 'arm' else 'local/results/policy/play')
        if args.output.resolve() == args.checkpoint.resolve().parent:
            parser.error("Use a separate play output directory to preserve training results")

    return args


def launch(args, on_ready=None):
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless, "multi_gpu": False, "enable_crashreporter": False,
                         "hide_ui": True if args.headless else None,
                         "disable_viewport_updates": args.headless,
                         "extra_args": ["--enable", "isaacsim.core.api", "--enable", "isaacsim.core.prims"]})
    code = 0
    try:
        from dex_manipulation.policy.runner import run
        run(args, ROOT, on_ready=on_ready, is_running=app.is_running)
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
