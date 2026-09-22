"""Receive ROS joint state into an independent kinematic Isaac USD scene."""
import json
import time

import numpy as np

from .ros_state import RobotState, UsdStateMirror, subscribe_state
from .scene import Workcell


def run(root,settings,seconds,headless,output):
    from isaacsim import SimulationApp
    app=SimulationApp(dict(headless=headless,multi_gpu=False,enable_crashreporter=False,
                           hide_ui=headless,disable_viewport_updates=headless))
    code=0
    try:
        import rclpy
        import omni.usd
        from pxr import UsdGeom, UsdLux
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.viewports import set_camera_view
        rclpy.init(args=[])
        config=json.loads((root/settings['arm_config']).read_text())
        state=RobotState(root,config);workcell=Workcell.load(root/config['workcell'])
        omni.usd.get_context().new_stage()
        stage=omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
        add_reference_to_stage(str(root/config['usd']),'/Robot')
        workcell.create_usd(stage,collision=False)
        UsdLux.DomeLight.Define(stage,'/Light').CreateIntensityAttr(900)
        mirror=UsdStateMirror(stage,state,'/Robot',workcell.world_from_base)
        UsdGeom.Imageable(stage.GetPrimAtPath('/Robot')).MakeInvisible()
        set_camera_view(eye=np.array([1.65,-1.9,1.25]),target=np.array([.35,0,.05]))
        node=subscribe_state(settings,state,'robot_state_mirror')
        print(f'Mirror: joint-state display on {settings["namespace"]}; '
              'feedback sources are listed in diagnostics',flush=True)
        started=time.monotonic();last_stamp=None;updates=0;fresh_before=None;last_error=0.
        previous_q=None;maximum_step=0.
        try:
            while app.is_running() and (seconds==0 or time.monotonic()-started<seconds):
                tick=time.monotonic()
                rclpy.spin_once(node,timeout_sec=0.)
                fresh=state.fresh(settings['state_timeout_s'])
                if fresh and state.stamp_ns!=last_stamp:
                    if previous_q is not None:
                        maximum_step=max(maximum_step,float(np.max(np.abs(state.q-previous_q))))
                    previous_q=state.q.copy()
                    expected=mirror.apply(state.q)
                    UsdGeom.Imageable(stage.GetPrimAtPath('/Robot')).MakeVisible()
                    cache=UsdGeom.XformCache()
                    for name,pose in expected.items():
                        actual=np.asarray(cache.GetLocalToWorldTransform(mirror.prims[name])).T
                        last_error=max(last_error,float(np.max(np.abs(actual-workcell.world_from_base@pose))))
                    updates+=1;last_stamp=state.stamp_ns
                if fresh!=fresh_before:
                    print('Mirror: joint feedback live' if fresh else
                          'Mirror: feedback stale; holding last received pose',flush=True)
                    fresh_before=fresh
                app.update()
                time.sleep(max(0.,1/settings['mirror_rate_hz']-(time.monotonic()-tick)))
        finally:
            report=dict(messages=state.accepted,rejected=state.rejected,usd_updates=updates,
                        maximum_link_matrix_error=last_error,fresh=state.fresh(settings['state_timeout_s']),
                        maximum_measured_joint_step_rad=maximum_step,
                        latest_q_rad=state.q.tolist() if state.q is not None else None,
                        physical_simulation=False,robot_command_publishing=False,object_pose_received=False)
            if output:output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report),flush=True)
            node.destroy_node();rclpy.shutdown()
        if not updates:raise RuntimeError('No fresh ROS joint feedback was received')
    except Exception:
        import traceback
        traceback.print_exc();code=1
    finally:app.close(exit_code=code)
    return code
