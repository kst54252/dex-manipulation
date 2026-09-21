#!/usr/bin/env python3
"""Separate demo2 five-finger experiment. Existing policy files stay untouched."""
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def prepare(argv=None):
    from scripts.policy import parse_args
    from dex_manipulation.training import scale_gravity
    argv = list(sys.argv[1:] if argv is None else argv)
    dry_run = '--dry-run' in argv
    if dry_run: argv.remove('--dry-run')
    args = parse_args(['--config', str(ROOT / 'config/policy_demo2_grasp.json'), *argv])
    if args.robot != 'floating' or args.initialize_actor or args.export or args.logger == 'wandb':
        raise ValueError('This separate experiment supports floating train/evaluate/play, tensorboard/none, and its own checkpoints')
    if args.mode == 'train' and not args.headless:
        args.headless = True
    if args.mode == 'train': args.skip_evaluation = True
    if args.mode != 'train':
        args.contact_materials = args.contact_materials or 'checkpoint'
        args.table_safety = args.table_safety or 'checkpoint'
    if args.checkpoint:
        source = args.checkpoint.resolve().parent / 'config.resolved.json'
        if not source.is_file(): raise ValueError('Checkpoint requires adjacent config.resolved.json')
    else:
        source = args.config
    config = json.loads(source.read_text())
    from dex_manipulation.policy.grasp import SCHEMA
    if config.get('five_finger_grasp', {}).get('schema') != SCHEMA:
        raise ValueError('Use the dedicated demo2 five-finger config/checkpoint; existing policies remain separate')
    from dex_manipulation.data import resolve_demo_path
    if resolve_demo_path(config['reference'], ROOT).resolve() != (ROOT / 'data/demo2/grounded/reference.npz').resolve():
        raise ValueError('This experiment requires the unchanged geometric demo2 reference')
    if config.get('reference_timing') != 'input_timestamps':
        raise ValueError('Preserve the source frame-to-time mapping')
    if not args.checkpoint and args.mode == 'train':
        iterations = args.iterations or config['training']['iterations']
        scale_gravity(config, iterations)
        config['training'].update(iterations=iterations,
            num_envs=args.num_envs or config['training']['num_envs'],
            save_every=args.save_every or config['training']['save_every'])
    if '--output' not in argv and not any(v.startswith('--output=') for v in argv):
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        args.output = ROOT / 'local/results/policy' / f'demo2_five_finger_{args.mode}_{stamp}'
    args.output = args.output.expanduser().resolve()
    if not args.output.is_relative_to(ROOT / 'local') or args.output.exists():
        raise ValueError('Choose a new output folder under local/; existing runs are never overwritten')
    # Validate the frame mapping without starting Isaac.
    import numpy as np
    with np.load(ROOT / config['reference'], allow_pickle=False) as data:
        ids, times = data['frame_ids'], data['timestamps_s']
        where = np.flatnonzero(ids == config['five_finger_grasp']['start_frame_id'])
        if len(where) != 1: raise ValueError('Requested contact start frame is missing')
        start = float(times[where[0]] - times[0])
    plan = dict(mode=args.mode, output=str(args.output), config=config,
                contact_start_frame=config['five_finger_grasp']['start_frame_id'], contact_start_time_s=start,
                existing_files_modified=False, long_training_started=False)
    if dry_run:
        print(json.dumps(plan, indent=2)); return None
    args.output.mkdir(parents=True)
    args.config = args.output / 'config.input.json'
    args.config.write_text(json.dumps(config, indent=2) + '\n')
    return args


def launch(args):
    from isaacsim import SimulationApp
    app = SimulationApp(dict(headless=args.headless, multi_gpu=False, enable_crashreporter=False,
        hide_ui=True if args.headless else None, disable_viewport_updates=args.headless,
        extra_args=['--enable', 'isaacsim.core.api', '--enable', 'isaacsim.core.prims']))
    code = 0
    try:
        from dex_manipulation.policy.demo2 import run
        run(args, ROOT, is_running=app.is_running)
    except Exception:
        import traceback
        traceback.print_exc(); code = 1
    finally:
        app.close(exit_code=code)
    return code


if __name__ == '__main__':
    arguments = prepare()
    sys.exit(launch(arguments) if arguments is not None else 0)
