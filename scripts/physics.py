#!/usr/bin/env python3
"""View reference motion with gravity/contact, no trained policy or early-failure reset."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('robot',choices=('floating','arm'))
    parser.add_argument('--loops',type=int,default=3,help='Repeat complete demonstrations; 0 repeats until stopped')
    parser.add_argument('--speed',type=float,default=1.0,help='Floating reference speed multiplier; physics dt stays fixed')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--config',type=Path,default=ROOT/'config/policy.json',help='Floating physics configuration')
    parser.add_argument('--arm-config',type=Path,default=ROOT/'config/ik.json',help='Demo input, arm model and placement')
    parser.add_argument('--arm-reference',type=Path,default=ROOT/'local/results/ik/full/trajectory.npz')
    args=parser.parse_args()
    if not math.isfinite(args.speed) or args.speed<=0:parser.error('--speed must be positive and finite')
    if args.robot=='arm' and args.speed!=1.0:parser.error('--speed currently applies only to floating playback')
    if args.loops<0:parser.error('--loops must be nonnegative (0 repeats until stopped)')
    output=args.output or ROOT/'local/results/physics'/args.robot
    if args.robot=='arm':
        from dex_manipulation.launch import arm_reference_error
        reason=arm_reference_error(ROOT,args.arm_reference,args.arm_config)
        if reason:
            parser.error(f'Arm reference unavailable/stale: {reason}. Use ./run.sh arm retarget for automatic IK preparation.')
        from scripts.replay import main as replay
        sys.argv=[sys.argv[0],'--mode','targets','--rate','120','--loops',str(args.loops),'--output',str(output),
                  '--reference',str(args.arm_reference),'--config',str(args.arm_config),'--physics-config',str(args.config)]
        sys.argv+=['--headless'] if args.headless else ['--realtime']
        return replay()
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':args.headless,'multi_gpu':False,'enable_crashreporter':False,
        'extra_args':['--enable','isaacsim.core.api','--enable','isaacsim.core.prims']})
    code=0
    try:
        from dex_manipulation.sim import replay_floating
        replay_floating(ROOT,output,args.loops,render=not args.headless,
                        config_path=args.config,arm_config_path=args.arm_config,is_running=app.is_running,speed=args.speed)
    except Exception:
        import traceback
        traceback.print_exc();code=1
    finally:
        app.close(exit_code=code)
    return code


if __name__=='__main__':sys.exit(main())
