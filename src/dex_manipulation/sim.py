"""Isaac boundary adapter; pure IK/reference modules do not import this module."""
import numpy as np


class IsaacJointAdapter:
    def __init__(self, articulation, arm, hand):
        self.articulation, self.arm, self.hand = articulation, arm, hand
        self.command_names = arm.active_names + hand.full_names
        self.indices = np.array([articulation.dof_names.index(name) for name in self.command_names])

    def positions(self, target):
        arm = target.ordered(self.arm.active_names)
        finger = self.hand.expand(target.ordered(self.hand.active_names))
        return np.r_[arm, finger]

    def set_target(self, target):
        """Position-control boundary; keeps USD gains. Tracking is a separate concern."""
        from isaacsim.core.utils.types import ArticulationAction
        self.articulation.apply_action(ArticulationAction(joint_positions=self.positions(target), joint_indices=self.indices))

    def set_state(self, target):
        """Kinematic playback/FK verification only, not drive-tracking validation."""
        self.articulation.set_joint_positions(self.positions(target), joint_indices=self.indices)
        self.articulation.set_joint_velocities(np.zeros(len(self.indices)), joint_indices=self.indices)


def replay_floating(root, output, loops=3, render=True, *, observer=None):
    """Physical floating-hand demonstration: zero residual, no learner/early reset.

    Use the REGRIND pose-PD controller shared with RL: root force and distributed
    link torque, with hand gravity disabled by config. The can retains gravity
    and contact. Only startup/reset assigns body/joint poses directly.
    """
    import json
    from pathlib import Path
    import torch
    from .fk import HandModel
    from .policy.trajectory import ReferenceMotion
    from .policy.env import PhysxResidualEnv
    from .coordinates import collision_bottom
    from .transforms import transform
    from scipy.spatial.transform import Rotation
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg=json.loads((root/'config/policy.json').read_text())
    for name in ('rsi','augmentation','gravity_curriculum','domain_randomization','push_curriculum'):
        cfg[name]['enabled']=False
    # Respect hand_gravity=False, as in REGRIND's floating-hand asset config.
    # Disabling the curriculum below sets normal world/object gravity for PLAY.
    cfg['reset_perturbation']=False
    model=HandModel.load(root/cfg['model'])
    reference=ReferenceMotion(root/cfg['reference'],model,root/cfg['object_geometry'],cfg['world_frame'])
    # Preserve the current input segment's timing rather than invoking RL's retiming.
    cfg['episode_length_s']=reference.duration
    cfg['augmentation']['blend_start_s']=reference.duration*cfg['augmentation']['phase_start_ratio']
    cfg['augmentation']['blend_end_s']=reference.duration*cfg['augmentation']['phase_end_ratio']
    torch.set_num_threads(1)
    env=PhysxResidualEnv(root,model,reference,cfg,num_envs=1,render=render)
    rows=[]; failures=[]; initial_bottoms=[]; physics_steps=[]
    try:
        env.set_training(False)
        for loop in range(loops):
            env.reset(randomize=False)
            initial=env.state()
            if observer is not None:observer(env,initial,0.,loop,'reset')
            pose=transform(Rotation.from_quat(initial['object_quaternion'][0].cpu().numpy()).as_matrix(),
                           initial['object_position'][0].cpu().numpy())
            initial_bottoms.append(collision_bottom(pose, reference.collision_shapes))
            if abs(initial_bottoms[-1]) > 1e-6:
                raise RuntimeError('Initial can is not on the Z=0 tabletop')
            reset_count=env.object_reset_count
            start_physics_step=env.world.current_time_step_index
            first_failure=None
            for step in range(int(np.ceil(reference.duration/env.task.dt))+1):
                _,_,terminated,truncated,info=env.step(torch.zeros((1,12),device=env.device),auto_reset=False)
                elapsed_physics_steps=env.world.current_time_step_index-start_physics_step
                if elapsed_physics_steps!=(step+1)*cfg['control_decimation']:
                    raise RuntimeError('Physics/control clock mismatch: rendering must not advance extra physics')
                if env.object_reset_count!=reset_count:
                    raise RuntimeError('Unexpected object reset during demonstration')
                state=info['state']
                if not all(torch.isfinite(value).all() for value in state.values()):
                    raise RuntimeError('Nonfinite simulated state')
                if bool(info['metrics']['early_failure'][0]) and first_failure is None:
                    first_failure=step
                if loop==0:
                    # State is observed AFTER physics; compare it with that time,
                    # while preserving the controller's pre-step target separately.
                    time_s=float(env.time[0]); sampled=reference.sample(time_s)
                    rows.append(dict(timestamp_s=time_s,command_time_s=float(info['reference_time'][0]),
                        physics_time_s=elapsed_physics_steps*cfg['physics_dt'],
                        q_finger=state['q'][0].cpu().numpy(),wrist_position=state['wrist_position'][0].cpu().numpy(),
                        wrist_quaternion=state['wrist_quaternion'][0].cpu().numpy(),
                        object_position=state['object_position'][0].cpu().numpy(),
                        object_quaternion=state['object_quaternion'][0].cpu().numpy(),
                        reference_object_position=sampled['object_position'][0],
                        reference_wrist_position=sampled['wrist_position'][0],
                        reference_wrist_quaternion=sampled['wrist_quaternion'][0]))
                if observer is not None:observer(env,state,float(env.time[0]),loop,'step')
            failures.append(first_failure)
            physics_steps.append(elapsed_physics_steps)
        np.savez_compressed(output/'measurements.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
        errors=np.array([row['wrist_position']-row['reference_wrist_position'] for row in rows])
        angles=np.array([(Rotation.from_quat(row['wrist_quaternion'])*Rotation.from_quat(row['reference_wrist_quaternion']).inv()).magnitude() for row in rows])
        report=dict(robot='floating',samples=len(rows),loops=loops,duration_s=reference.duration,
                    object_geometry_fingerprint=reference.metadata['object_geometry_fingerprint'],
                    gravity_m_s2=env.gravity.value,hand_gravity=cfg['hand_gravity'],object_gravity=True,object_kinematic=False,
                    object_pose_writes_during_demonstration=0,continue_after_grasp_failure=True,
                    first_failure_step_by_loop=failures,policy_loaded=False,optimizer_updates=0,
                    initial_can_bottom_z_m_by_loop=initial_bottoms, tabletop_z_m=0.,
                    floor_top_z_m=None, surface=env.surface.metadata(),
                    physics_steps_by_loop=physics_steps,physics_duration_s_by_loop=[n*cfg['physics_dt'] for n in physics_steps],
                    rsi=False,augmentation=False,randomization=False,pushes=False,
                    controller=env.metadata['controller'],
                    wrist_tracking=dict(mean_position_error_m=float(np.linalg.norm(errors,axis=1).mean()),
                        max_position_error_m=float(np.linalg.norm(errors,axis=1).max()),
                        mean_z_error_m=float(errors[:,2].mean()),max_abs_z_error_m=float(np.abs(errors[:,2]).max()),
                        max_orientation_error_rad=float(angles.max())),
                    control='REGRIND pose PD: root force + mass-proportional link torques; zero residual, original gains',
                    interpretation='Full physical motion preview; no claim of grasp or accurate tracking',
                    physics=env.metadata)
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('[physics]',json.dumps({k:v for k,v in report.items() if k!='physics'}),flush=True)
    finally:
        env.close()
