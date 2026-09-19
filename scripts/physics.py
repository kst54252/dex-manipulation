#!/usr/bin/env python3
"""View reference motion with gravity/contact, no trained policy or early-failure reset."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('robot',choices=('floating','arm'))
    parser.add_argument('--loops',type=int,default=3,help='Repeat complete demonstrations, resetting between repetitions')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.loops<1:parser.error('--loops must be positive')
    output=args.output or ROOT/'local/results/physics'/args.robot
    if args.robot=='arm':
        import hashlib
        import numpy as np
        from dex_manipulation.reference import JointReference
        config=json.loads((ROOT/'config/ik.json').read_text())
        path=ROOT/'local/results/ik/full/trajectory.npz'
        if not path.exists():
            parser.error('Generate the arm reference first: python scripts/ik.py solve')
        reference=JointReference(path)
        from dex_manipulation.scene import Workcell
        workcell=Workcell.load(ROOT/config['workcell'])
        alignment=workcell.resolve_alignment(json.loads((ROOT/config['alignment']).read_text()))
        workcell.validate_reference(reference.metadata, alignment)
        if (reference.metadata['input_sha256']!=hashlib.sha256((ROOT/config['input']).read_bytes()).hexdigest()
                or not np.allclose(reference.metadata['base_from_source'],alignment['base_from_source'],atol=1e-10,rtol=0)
                or reference.metadata.get('trajectory_substeps',1)!=config.get('trajectory_substeps',1)):
            parser.error('Arm reference is stale after dataset/placement changes. Run: python scripts/ik.py solve')
        from scripts.replay import main as replay
        sys.argv=[sys.argv[0],'--mode','targets','--rate','120','--loops',str(args.loops),'--output',str(output)]
        sys.argv+=['--headless'] if args.headless else ['--realtime']
        return replay()
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':args.headless,'multi_gpu':False,'enable_crashreporter':False,
        'extra_args':['--enable','isaacsim.core.api','--enable','isaacsim.core.prims']})
    code=0
    try:
        from dex_manipulation.sim import replay_floating
        replay_floating(ROOT,output,args.loops,render=not args.headless)
    except Exception:
        import traceback
        traceback.print_exc();code=1
    finally:
        app.close(exit_code=code)
    return code


if __name__=='__main__':sys.exit(main())
