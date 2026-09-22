"""ROS 2 action for a recorded 12-axis motion and continuous measured feedback.

One I/O owner shares the two vendor connections between idle feedback and motion.
Only the selected, validated recording can move hardware; ROS cannot overwrite
its calibration, timing, initial-state guard or prevalidated command sequence.
"""
import asyncio
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import datetime
import json
from pathlib import Path
import threading
import time

import numpy as np

from .execution import MockBackend, check_state, stream
from .hardware import RBPodoStark, connection_plan, hardware_plan
from .ros_state import RobotState


class MotionCancelled(RuntimeError): pass


def goal_arrays(recording, hold_s):
    return (np.r_[recording.times, recording.duration+hold_s],
            np.vstack([recording.q, recording.q[-1]]))


def validate_goal(request, recording, hold_s, guard):
    """Accept a named permutation of the recording, never silently retime it."""
    trajectory = request.trajectory
    names = list(trajectory.joint_names)
    if len(names) != 12 or set(names) != set(recording.names):
        raise ValueError('Expected all twelve arm and finger joint names')
    if (trajectory.header.stamp.sec or trajectory.header.stamp.nanosec or
            request.multi_dof_trajectory.points or request.multi_dof_trajectory.joint_names or
            request.component_path_tolerance or request.component_goal_tolerance or
            request.goal_time_tolerance.sec or request.goal_time_tolerance.nanosec):
        raise ValueError('Only immediate single-DoF goals with fixed recording timing are supported')
    times, positions = goal_arrays(recording, hold_s)
    if len(trajectory.points) != len(times):
        raise ValueError('Goal must contain the selected recording plus its final hold endpoint')
    indices = [names.index(n) for n in recording.names]
    for point, stamp, q in zip(trajectory.points, times, positions):
        values = np.asarray(point.positions, float)
        if (values.shape != (12,) or not np.isfinite(values).all() or
                point.velocities or point.accelerations or point.effort or
                abs(point.time_from_start.sec+point.time_from_start.nanosec*1e-9-stamp)>2e-9 or
                not np.allclose(values[indices], q, atol=1e-8, rtol=0)):
            raise ValueError('Goal differs from the configured recording; select/validate that recording first')
    tolerances = []
    for rows in (request.path_tolerance, request.goal_tolerance):
        limit = np.deg2rad(guard['tracking_tolerance_deg']).copy()
        seen = set()
        for row in rows:
            if (row.name not in recording.names or row.name in seen or
                    not np.isfinite([row.position,row.velocity,row.acceleration]).all() or
                    row.position<0 or row.velocity or row.acceleration):
                raise ValueError('Only named position tolerances may tighten the configured guard')
            seen.add(row.name)
            if row.position:
                i=recording.names.index(row.name)
                limit[i]=min(limit[i],row.position)
        tolerances.append(limit)
    return tolerances


class ObservedBackend:
    def __init__(self, worker, cancel, feedback):
        self.worker, self.cancel, self.feedback = worker, cancel, feedback
        self.hardware = worker.backend.hardware
        self.feedback_source = getattr(worker.backend,'feedback_source',None)
        self.desired = worker.recording.initial.copy()
        self.started = time.monotonic()

    async def connect(self): pass  # Connection belongs to the worker lifetime.
    async def close(self): pass

    def check_cancel(self):
        if self.cancel.is_set() or self.worker.shutdown.is_set():
            raise MotionCancelled('ROS goal canceled/stopped')

    async def read(self):
        self.check_cancel()
        state = await self.worker.read()
        self.feedback(state, self.desired, time.monotonic()-self.started)
        return state

    async def send(self, q, velocity, dt):
        self.check_cancel()
        await self.worker.backend.send(q, velocity, dt)
        self.desired = q.copy()

    async def stop(self):
        await self.worker.backend.stop()
        if hasattr(self.worker.backend,'sent'): self.worker.backend.sent=False


class DeviceWorker:
    def __init__(self, recording, backend, guard, rate, publish):
        self.recording, self.backend, self.guard = recording, backend, guard
        self.rate, self.publish = rate, publish
        self.shutdown = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.last = None
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._thread, name='robot_io', daemon=True)

    def _thread(self):
        asyncio.set_event_loop(self.loop)
        try: self.loop.run_until_complete(self._session())
        finally: self.loop.close()

    async def _session(self):
        self.lock = asyncio.Lock()
        try:
            await self.backend.connect()
            await self.read()
            self.ready.set()
            while not self.shutdown.is_set():
                started=time.monotonic()
                async with self.lock:
                    if not self.shutdown.is_set(): await self.read()
                await asyncio.sleep(max(0., 1/self.rate-(time.monotonic()-started)))
        except Exception as error:
            self.error=f'{type(error).__name__}: {error}'
            self.ready.set()
        finally:
            for operation in (self.backend.stop,self.backend.close):
                try: await operation()
                except Exception as error: self.error=f'{operation.__name__}: {error}'

    async def read(self):
        state=await self.backend.read()
        check_state(dict(state,ready=True), np.asarray(state['q_rad']), np.full(12,np.inf),
                    self.guard['maximum_feedback_age_s'],time.monotonic)
        self.last=state
        self.publish(state)
        return state

    def start(self):
        self.thread.start()
        if not self.ready.wait(5): raise TimeoutError('Robot connection/initial feedback timed out')
        if self.error: raise RuntimeError(self.error)

    async def _motion(self, output, hold, cancel, feedback, tolerances):
        async with self.lock:
            observed=ObservedBackend(self,cancel,feedback)
            try:
                result=await stream(self.recording,observed,output,
                    start_tolerance_rad=np.deg2rad(self.guard['start_tolerance_deg']),
                    tracking_tolerance_rad=tolerances[0],
                    maximum_feedback_age_s=self.guard['maximum_feedback_age_s'],
                    maximum_lateness_s=self.guard['maximum_lateness_s'],hold_s=hold)
            except Exception:
                report=Path(output)/'report.json'
                if report.exists() and json.loads(report.read_text()).get('cleanup_errors'):
                    self.error='Robot stop acknowledgment failed; restart after checking physical state'
                    raise RuntimeError(self.error)
                raise
            check_state(await self.read(),self.recording.q[-1],tolerances[1],
                        self.guard['maximum_feedback_age_s'],time.monotonic)
            return result

    def submit(self, *args):
        if self.error or not self.thread.is_alive(): raise RuntimeError(self.error or 'Device worker stopped')
        return asyncio.run_coroutine_threadsafe(self._motion(*args),self.loop)

    def close(self):
        self.shutdown.set()
        self.thread.join(5)
        if self.thread.is_alive(): raise RuntimeError('Robot I/O did not shut down within five seconds')


def make_node(root, settings, recording, hardware_config, *, backend='mock', enable_motion=False, hold_s=1.):
    import rclpy
    from rclpy.action import ActionServer, CancelResponse, GoalResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.time import Time
    from control_msgs.action import FollowJointTrajectory
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import JointState
    from std_srvs.srv import Trigger
    from scipy.spatial.transform import Rotation

    if backend=='hardware':
        connection_plan(recording.names,hardware_config)
        if enable_motion: hardware_plan(recording,hardware_config)
    elif backend=='vcb':
        from .vcb import VCBBackend
    elif backend!='mock': raise ValueError('Unknown ROS device backend')
    guard=hardware_config['guard']
    arm_config=json.loads((root/settings['arm_config']).read_text())
    model=RobotState(root,arm_config)
    if model.names != recording.names: raise ValueError('Recording and current model joint names differ')

    class RobotNode(Node):
        def __init__(self):
            super().__init__('rb3_revo2',namespace=settings['namespace'])
            if self.get_parameter('use_sim_time').value: raise ValueError('Hardware feedback requires wall clock, not /clock')
            self.group=ReentrantCallbackGroup()
            qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)
            self.states=self.create_publisher(JointState,'joint_states',qos)
            self.full_states=self.create_publisher(JointState,'model_joint_states',qos)
            self.wrist=self.create_publisher(PoseStamped,'wrist_pose',qos)
            self.diagnostics=self.create_publisher(DiagnosticArray,'diagnostics',10)
            self.motion_enabled=backend=='mock' or enable_motion
            self.busy=False;self.goal_lock=threading.Lock();self.cancel=threading.Event()
            self.last_error='';self.samples=0
            if backend=='mock': device=MockBackend(recording.initial)
            elif backend=='vcb': device=VCBBackend(recording,hardware_config)
            else: device=RBPodoStark(recording.names,hardware_config)
            self.worker=DeviceWorker(recording,device,guard,settings['state_rate_hz'],self.publish_state)
            self.action=ActionServer(self,FollowJointTrajectory,'follow_joint_trajectory',
                goal_callback=self.accept_goal,cancel_callback=self.cancel_goal,
                execute_callback=self.execute_goal,callback_group=self.group)
            self.stop_service=self.create_service(Trigger,'stop',self.stop_motion,callback_group=self.group)
            self.timer=self.create_timer(.2,self.publish_diagnostics,callback_group=self.group)

        def publish_state(self,state):
            age=time.monotonic()-state['sample_time_s']
            stamp=Time(nanoseconds=self.get_clock().now().nanoseconds-int(age*1e9)).to_msg()
            message=JointState();message.header.stamp=stamp;message.header.frame_id=settings['frame_id']
            message.name=list(model.names);message.position=np.asarray(state['q_rad']).tolist()
            self.states.publish(message)
            full=JointState();full.header=message.header;full.name=list(model.model_names)
            full.position=model.expand(np.asarray(state['q_rad'])).tolist();self.full_states.publish(full)
            pose=model.arm.pose(np.asarray(state['q_rad'])[:6]);quat=Rotation.from_matrix(pose[:3,:3]).as_quat()
            wrist=PoseStamped();wrist.header=message.header
            wrist.pose.position.x,wrist.pose.position.y,wrist.pose.position.z=map(float,pose[:3,3])
            wrist.pose.orientation.x,wrist.pose.orientation.y,wrist.pose.orientation.z,wrist.pose.orientation.w=map(float,quat)
            self.wrist.publish(wrist);self.samples+=1

        def publish_diagnostics(self):
            state=self.worker.last
            age=time.monotonic()-state['sample_time_s'] if state else float('inf')
            status=DiagnosticStatus();status.name='rb3_revo2';status.hardware_id=backend
            error=self.worker.error or self.last_error
            ready=bool(state and state['ready'])
            status.level=DiagnosticStatus.ERROR if error or not ready or age>settings['state_timeout_s'] else DiagnosticStatus.OK
            status.message=error or ('feedback stale' if age>settings['state_timeout_s'] else
                                    'robot not ready' if not ready else 'moving' if self.busy else 'ready')
            values=dict(backend=backend,motion_enabled=self.motion_enabled,busy=self.busy,
                ready=ready,feedback_age_s=age,state_samples=self.samples,command_hz=1/recording.dt,
                feedback_rate_requested_hz=settings['state_rate_hz'],
                dependent_fingers='derived from coupling; not measured',
                effort='unavailable; JointState.effort empty')
            values['feedback_source']=getattr(self.worker.backend,'feedback_source',None)
            if state:
                for key in ('rb3_device_time_s','rb3_status','revo2_status','read_window_s'):
                    if key in state: values[key]=state[key]
            status.values=[KeyValue(key=k,value=json.dumps(v)) for k,v in values.items()]
            msg=DiagnosticArray();msg.header.stamp=self.get_clock().now().to_msg();msg.status=[status]
            self.diagnostics.publish(msg)

        def accept_goal(self,request):
            try:
                validate_goal(request,recording,hold_s,guard)
                if not self.motion_enabled: raise ValueError('Read-only bridge; --enable-motion was not set')
                if self.worker.error or self.worker.last is None: raise ValueError('Feedback is unavailable')
                check_state(self.worker.last,recording.initial,np.deg2rad(guard['start_tolerance_deg']),
                            guard['maximum_feedback_age_s'],time.monotonic)
                with self.goal_lock:
                    if self.busy: raise ValueError('Another motion is active; cancel/finish it first')
                    self.busy=True;self.cancel.clear();self.last_error=''
                return GoalResponse.ACCEPT
            except (ValueError,RuntimeError,KeyError) as error:
                self.get_logger().warning(str(error));return GoalResponse.REJECT

        def cancel_goal(self,handle):
            self.cancel.set();return CancelResponse.ACCEPT

        def stop_motion(self,request,response):
            self.cancel.set();response.success=True
            response.message='Cancellation requested; completion/stop acknowledgment appears in action result and diagnostics'
            return response

        def execute_goal(self,handle):
            result=FollowJointTrajectory.Result()
            def feedback(state,desired,elapsed):
                if not handle.is_active: return
                msg=FollowJointTrajectory.Feedback();msg.header.stamp=self.get_clock().now().to_msg()
                msg.joint_names=list(recording.names)
                msg.desired.positions=desired.tolist();msg.actual.positions=np.asarray(state['q_rad']).tolist()
                msg.error.positions=(desired-np.asarray(state['q_rad'])).tolist()
                for point in (msg.desired,msg.actual,msg.error):
                    ns=int(elapsed*1e9);point.time_from_start.sec=ns//10**9;point.time_from_start.nanosec=ns%10**9
                handle.publish_feedback(msg)
            try:
                output=root/'local/results/ros'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                limits=validate_goal(handle.request,recording,hold_s,guard)
                future=self.worker.submit(output,hold_s,self.cancel,feedback,limits)
                while True:
                    try: future.result(timeout=.05);break
                    except FutureTimeout:
                        if future.done(): raise
                        if handle.is_cancel_requested or not rclpy.ok(): self.cancel.set()
                handle.succeed();result.error_code=result.SUCCESSFUL
                result.error_string=f'Completed; log: {output}'
            except MotionCancelled as error:
                if handle.is_cancel_requested: handle.canceled()
                else: handle.abort()
                result.error_code=result.PATH_TOLERANCE_VIOLATED;result.error_string=str(error)
            except Exception as error:
                handle.abort();self.last_error=str(error)
                result.error_code=result.PATH_TOLERANCE_VIOLATED;result.error_string=str(error)
            finally:
                with self.goal_lock: self.busy=False
            return result

        def close(self):
            self.cancel.set();self.worker.close();self.action.destroy();self.destroy_node()

    return RobotNode()
