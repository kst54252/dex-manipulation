#!/usr/bin/env python3
"""Record, plot and compare Revo2 tactile without issuing robot motion commands."""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path


def main(root, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('record', help='Read-only, exclusive RS485 connection to Revo2 capacitive tactile')
    p.add_argument('--port', required=True)
    p.add_argument('--slave-id', type=int, required=True)
    p.add_argument('--baudrate-enum', required=True, help='Actual configured BrainCo SDK Baudrate enum name')
    p.add_argument('--seconds', type=float, default=10.)
    p.add_argument('--hz', type=float, default=100., help='Host polling request rate, not sensor bandwidth')
    p.add_argument('--output', type=Path)
    c = sub.add_parser('compare', help='Compare real CSV to simulation NPZ after measured start alignment')
    c.add_argument('--sim', type=Path, required=True)
    c.add_argument('--real', type=Path, required=True)
    c.add_argument('--offset-s', type=float, help='real_host_time - simulation_time; automatic zero only for matching execute records')
    c.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('plot', help='Plot real and/or simulated normal force, tangential force and direction')
    p.add_argument('--sim', type=Path)
    p.add_argument('--real', type=Path)
    p.add_argument('--offset-s', type=float)
    p.add_argument('--grasp-start-s', type=float)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    for key in ('sim', 'real', 'output'):
        value = getattr(args, key, None)
        if value is not None:
            setattr(args, key, (root / value).resolve())
    if args.output is not None and (not args.output.is_relative_to(root / 'local') or args.output.exists()):
        parser.error('Use a new output directory under local/')
    if args.command == 'plot':
        from dex_manipulation.sensors.plotting import plot

        print(plot(args.output, sim_path=args.sim, real_path=args.real,
                   offset_s=args.offset_s, grasp_start_s=args.grasp_start_s))
        return 0
    if args.command == 'compare':
        from dex_manipulation.sensors.comparison import compare

        from dex_manipulation.sensors.plotting import aligned_offset

        offset, _ = aligned_offset(args.sim, args.real, args.offset_s)
        print(compare(args.sim, args.real, offset, args.output))
        return 0
    from dex_manipulation.sensors.hardware_tactile import record

    output = args.output or root/'local/results/tactile'/('real_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    try:
        result = asyncio.run(record(args.port, args.baudrate_enum, args.slave_id, output, args.seconds, args.hz))
    except (ImportError, RuntimeError, ValueError, OSError) as error:
        parser.exit(1, f'Tactile recording failed: {error}\n')
    print(result)
    return 0
