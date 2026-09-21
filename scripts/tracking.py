#!/usr/bin/env python3
"""Plan and physically replay a frozen policy rollout without online inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    plan=sub.add_parser('plan',help='Convert a strict floating capture into saved arm+finger targets')
    plan.add_argument('--capture',type=Path,required=True)
    plan.add_argument('--arm-config',type=Path,required=True)
    plan.add_argument('--kind',choices=('command','measured'),default='command')
    plan.add_argument('--time-scale',type=float,default=1.)
    plan.add_argument('--output',type=Path,required=True)
    play=sub.add_parser('replay',help='Drive the physical arm using the saved joint trajectory only')
    play.add_argument('--reference',type=Path,required=True)
    play.add_argument('--checkpoint',type=Path,required=True,help='Restore saved physical properties only; no actor inference')
    play.add_argument('--output',type=Path,required=True)
    play.add_argument('--repeats',type=int,default=3)
    play.add_argument('--hold',type=float,default=1.)
    play.add_argument('--no-velocity-feedforward',action='store_true')
    play.add_argument('--headless',action='store_true')
    args=parser.parse_args()
    if args.mode=='plan':
        from dex_manipulation.policy.tracking import plan as make_plan
        report=make_plan(ROOT,args.capture,args.arm_config,args.output,args.kind,args.time_scale)
        print(json.dumps({k:report[k] for k in ('success_count','frame_count','continuous_reference_valid','failed_frame_indices')},indent=2))
        return 0 if report['continuous_reference_valid'] else 2
    from dex_manipulation.reference import JointReference
    JointReference(args.reference)  # Reject failures before starting the simulator.
    manifest=json.loads((args.reference.parent/'frozen.json').read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    config_path=args.output/'physics_config.json'
    arm_path=args.output/'arm_config.json'
    config_path.write_text(json.dumps(manifest['config'],indent=2)+'\n')
    arm_path.write_text(json.dumps(manifest['arm_config'],indent=2)+'\n')
    from scripts.policy import parse_args,launch
    sim_args=parse_args(['--mode','play','--robot','arm','--episodes','1','--evaluation-protocol','strict',
        '--checkpoint',str(args.checkpoint),'--config',str(config_path),'--arm-config',str(arm_path),
        '--output',str(args.output)]+(['--headless'] if args.headless else []))
    def ready(env,metadata):
        from dex_manipulation.policy.tracking import replay
        return replay(env,metadata,ROOT,args.reference,args.checkpoint,args.output,
                      args.repeats,args.hold,not args.no_velocity_feedforward)
    return launch(sim_args,on_ready=ready)


if __name__=='__main__':
    sys.exit(main())
