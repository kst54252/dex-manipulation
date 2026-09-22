#!/usr/bin/env python3
"""ROS 2 bridge, recorded-motion client, measured state monitor and USD mirror."""
import argparse
import json
import math
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def make_goal(recording,hold):
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    from dex_manipulation.ros_bridge import goal_arrays
    goal=FollowJointTrajectory.Goal();goal.trajectory.joint_names=list(recording.names)
    times,positions=goal_arrays(recording,hold)
    for stamp,q in zip(times,positions):
        point=JointTrajectoryPoint();point.positions=q.tolist()
        ns=round(stamp*1e9);point.time_from_start.sec=ns//10**9;point.time_from_start.nanosec=ns%10**9
        goal.trajectory.points.append(point)
    return goal


def send(recording,settings,hold,wait):
    import rclpy
    from rclpy.action import ActionClient
    from control_msgs.action import FollowJointTrajectory
    from action_msgs.msg import GoalStatus
    node=rclpy.create_node('recorded_motion_client',namespace=settings['namespace'])
    client=ActionClient(node,FollowJointTrajectory,'follow_joint_trajectory')
    handle=None
    try:
        if not client.wait_for_server(timeout_sec=wait): raise TimeoutError('ROS action server not found')
        future=client.send_goal_async(make_goal(recording,hold))
        rclpy.spin_until_future_complete(node,future,timeout_sec=wait)
        if not future.done(): raise TimeoutError('Goal acknowledgment timed out')
        handle=future.result()
        if not handle.accepted: raise RuntimeError('Goal rejected; check bridge diagnostics and initial pose')
        result=handle.get_result_async()
        try:
            rclpy.spin_until_future_complete(node,result,timeout_sec=recording.duration+hold+wait)
            if not result.done(): raise TimeoutError('Motion result timed out')
        except (KeyboardInterrupt,TimeoutError):
            cancellation=handle.cancel_goal_async()
            rclpy.spin_until_future_complete(node,cancellation,timeout_sec=2.)
            rclpy.spin_until_future_complete(node,result,timeout_sec=3.)
            raise
        response=result.result()
        print(json.dumps(dict(status=response.status,error_code=response.result.error_code,
                              message=response.result.error_string),ensure_ascii=False,indent=2))
        return 0 if response.status==GoalStatus.STATUS_SUCCEEDED and response.result.error_code==0 else 1
    finally: client.destroy();node.destroy_node()


def monitor(root,settings,seconds,output):
    import rclpy
    from dex_manipulation.ros_state import RobotState, subscribe_state
    model=RobotState(root,json.loads((root/settings['arm_config']).read_text()))
    node=subscribe_state(settings,model,'robot_state_monitor')
    start=time.monotonic();last_print=0.;ages=[]
    try:
        while rclpy.ok() and (seconds==0 or time.monotonic()-start<seconds):
            rclpy.spin_once(node,timeout_sec=.02)
            if model.stamp_ns:
                ages.append((node.get_clock().now().nanoseconds-model.stamp_ns)*1e-9)
            if time.monotonic()-last_print>=1:
                last_print=time.monotonic()
                print(f'states={model.accepted} fresh={model.fresh(settings["state_timeout_s"])} joints=12',flush=True)
        report=dict(messages=model.accepted,rejected=model.rejected,elapsed_s=time.monotonic()-start,
                    fresh=model.fresh(settings['state_timeout_s']),
                    latest_q_rad=model.q.tolist() if model.q is not None else None,
                    observed_age_max_s=max(ages) if ages else None)
        print(json.dumps(report,indent=2))
        if output: output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
        return 0 if report['messages'] and report['fresh'] else 1
    finally: node.destroy_node()


def main(argv=None):
    parser=argparse.ArgumentParser(prog='./run.sh ros')
    parser.add_argument('mode',choices=('bridge','send','status','mirror'))
    parser.add_argument('--backend',choices=('mock','hardware'),default='mock')
    parser.add_argument('--enable-motion',action='store_true',help='Allow the selected recorded ROS goal')
    parser.add_argument('--config',type=Path,default=ROOT/'config/ros.json')
    parser.add_argument('--hardware-config',type=Path,default=ROOT/'config/hardware.example.json')
    parser.add_argument('--recording',type=Path)
    parser.add_argument('--hold',type=float)
    parser.add_argument('--seconds',type=float,default=0.,help='Duration for bridge/status/mirror; 0 keeps running')
    parser.add_argument('--wait',type=float,default=10.,help='Action server/result timeout margin')
    parser.add_argument('--headless',action='store_true',help='Isaac mirror without a window')
    parser.add_argument('--output',type=Path,help='Local JSON report for status/mirror')
    args=parser.parse_args(argv)
    try:
        if args.enable_motion and (args.mode!='bridge' or args.backend!='hardware'):
            raise ValueError('--enable-motion requires bridge --backend hardware')
        for key in ('seconds','wait'):
            if not math.isfinite(getattr(args,key)) or getattr(args,key)<0: raise ValueError('Invalid duration')
        settings=json.loads((ROOT/args.config).read_text())
        for key in ('state_rate_hz','state_timeout_s','mirror_rate_hz'):
            if not math.isfinite(settings[key]) or settings[key]<=0: raise ValueError(f'Invalid {key}')
        output=(ROOT/args.output).resolve() if args.output else None
        if output and (not output.is_relative_to(ROOT/'local') or output.exists()):
            raise ValueError('Use a new report path under local/')
        if args.mode=='mirror':
            from dex_manipulation.ros_mirror import run
            return run(ROOT,settings,args.seconds,args.headless,output)
        import rclpy
        rclpy.init(args=[])
        try:
            if args.mode=='status': return monitor(ROOT,settings,args.seconds,output)
            from dex_manipulation.execution import RecordedCommands
            defaults=json.loads((ROOT/'config/execution.json').read_text())
            recording=RecordedCommands(ROOT/(args.recording or defaults['recording']))
            hold=args.hold if args.hold is not None else defaults['hold_s']
            if not math.isfinite(hold) or hold<0: raise ValueError('Hold must be finite and nonnegative')
            if args.mode=='send': return send(recording,settings,hold,args.wait)
            from rclpy.executors import MultiThreadedExecutor
            from dex_manipulation.ros_bridge import make_node
            config=json.loads((ROOT/args.hardware_config).read_text())
            config['simulation_validation']=str(ROOT/config['simulation_validation'])
            node=make_node(ROOT,settings,recording,config,backend=args.backend,
                           enable_motion=args.enable_motion,hold_s=hold)
            executor=MultiThreadedExecutor(num_threads=3);executor.add_node(node)
            try:
                node.worker.start()
                print(f'ROS ready: backend={args.backend}, motion={node.motion_enabled}, '
                      f'action={settings["namespace"]}/follow_joint_trajectory',flush=True)
                started=time.monotonic()
                while rclpy.ok() and (args.seconds==0 or time.monotonic()-started<args.seconds):
                    executor.spin_once(timeout_sec=.05)
            finally:
                node.cancel.set();node.worker.close();executor.shutdown(timeout_sec=5.)
                node.action.destroy();node.destroy_node()
            return 0
        finally:
            if rclpy.ok():rclpy.shutdown()
    except KeyboardInterrupt:return 130
    except (OSError,ValueError,RuntimeError,ImportError,TimeoutError) as error:
        print(f'ROS 실행 실패: {error}',flush=True);return 2
