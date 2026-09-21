"""Prepare a derived contact reference; preserve original demo/asset files."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dex_manipulation.contact import prepare_grasp_reference
from dex_manipulation.fk import HandModel

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start-frame',type=int,required=True);p.add_argument('--blend-start',type=float,required=True)
    p.add_argument('--table-margin',type=float,default=.008)
    p.add_argument('--side-contact',action='store_true')
    a=p.parse_args()
    r=prepare_grasp_reference(a.source,a.output,HandModel.load(ROOT/'assets/models/revo2.json'),
        ROOT/'assets/models/can_mesh.json',start_frame_id=a.start_frame,blend_start_s=a.blend_start,
        table_margin_m=a.table_margin,side_contact=a.side_contact)
    print(json.dumps({k:v for k,v in r.items() if k not in ('frames','transitions')},indent=2))
