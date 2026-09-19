"""REGRIND objective/action contract, adapted only for Revo2 coupling and can geometry."""
import torch
from .math3d import from_rotvec, quat_multiply, rotation_error


class ResidualTask:
    action_size = 12
    observation_size = 67
    critic_observation_size = 88

    def __init__(self, model, reference, config, device='cpu'):
        if len(model.active_names) != 6:
            raise ValueError('Expected the six independent Revo2 joints')
        self.model, self.reference, self.config, self.device = model, reference, config, torch.device(device)
        self.dt = config['physics_dt']*config['control_decimation']
        def tensor(v): return torch.as_tensor(v,dtype=torch.float32,device=device)
        self.lower,self.upper,self.velocity = map(tensor,(model.lower,model.upper,model.velocity))
        self.coupling,self.offset = map(tensor,(model.coupling,model.offset))
        self.full_lower,self.full_upper,self.full_velocity = map(tensor,(model.full_lower,model.full_upper,model.full_velocity))
        self.scales = tensor([config['residual_translation_m']]*3+[config['residual_rotation_rad']]*3+[config['residual_joint_rad']]*6)

    def targets(self,ref,actions,previous_q=None,default_offset=None):
        if actions.shape != (len(ref['q']),12): raise ValueError('Expected [env,12] residual action')
        delta=actions.clamp(-1,1)*self.scales
        q=(ref['q']+delta[:,6:]+(0 if default_offset is None else default_offset)).clamp(self.lower,self.upper)
        # Retained hardware constraint: Revo2 has coupled low-speed joints.
        if previous_q is not None:
            q=q.clamp(previous_q-self.velocity*self.dt,previous_q+self.velocity*self.dt)
        return dict(position=ref['wrist_position']+delta[:,:3],quaternion=quat_multiply(from_rotvec(delta[:,3:6]),ref['wrist_quaternion']),
                    active_q=q,full_q=q@self.coupling.T+self.offset)

    def score(self,state,ref,action,previous_action,demo_end=None,timed_out=None):
        c=self.config['reward']
        points=self.reference.points(state['object_position'],state['object_quaternion'])
        targets=self.reference.points(ref['object_position'],ref['object_quaternion'])
        eo=(points-targets).norm(dim=-1).mean(-1)
        ep=(state['wrist_position']-ref['wrist_position']).norm(dim=-1)
        er=rotation_error(ref['wrist_quaternion'],state['wrist_quaternion']).norm(dim=-1)
        ev=(state['object_velocity']-ref['object_velocity']).square().sum(-1)
        ew=(state['object_angular_velocity']-ref['object_angular_velocity']).square().sum(-1)
        eq=(state['q']-ref['q']).square().mean(-1).sqrt()
        coupling=(state['full_q']-state['q']@self.coupling.T-self.offset).abs().amax(-1)
        lower=torch.as_tensor(self.config['workspace']['lower_m'],device=self.device)
        upper=torch.as_tensor(self.config['workspace']['upper_m'],device=self.device)
        outside=((state['object_position']<lower)|(state['object_position']>upper)).any(-1)
        far=(state['wrist_position']-state['object_position']).norm(dim=-1)>self.config['failure_hand_object_distance_m']
        failure=(eo>self.config['failure_object_error_m'])|outside|far
        # Coupling violation is a model validity failure specific to Revo2.
        failure |= coupling>.1
        demo_end=ref['time']>=self.reference.duration-1e-6 if demo_end is None else demo_end
        timed_out=torch.zeros_like(failure) if timed_out is None else timed_out
        term=failure|demo_end
        terms=dict(object=c['object_weight']*torch.exp(-eo/c['object_std_m']),
                   object_linear_velocity=c['object_linear_velocity_weight']*torch.exp(-ev/c['object_linear_velocity_std']**2),
                   object_angular_velocity=c['object_angular_velocity_weight']*torch.exp(-ew/c['object_angular_velocity_std']**2),
                   wrist=c['wrist_position_weight']*torch.exp(-ep/c['wrist_position_std_m']),
                   orientation=c['wrist_rotation_weight']*torch.exp(-er/c['wrist_rotation_std_rad']),
                   residual=c['action_l2_weight']*torch.exp(-action.square().mean(-1)/c['action_l2_std']**2),
                   action_rate=c['action_rate_weight']*torch.exp(-(action-previous_action).square().mean(-1)/c['action_rate_std']**2),
                   action_bounds=c['action_out_of_bounds_weight']*torch.exp(-(action.abs()-1).clamp_min(0).sum(-1)/c['action_out_of_bounds_std']),
                   early_failure=c['early_failure_weight']*(failure&~demo_end).float())
        reward=sum(terms.values())*self.dt
        metrics=dict(object_keypoint_error_m=eo,wrist_error_m=ep,wrist_rotation_error_rad=er,joint_rmse_rad=eq,coupling_error_rad=coupling,
                     object_linear_velocity_error_m_s=ev.sqrt(),object_angular_velocity_error_rad_s=ew.sqrt(),
                     object_height_m=state['object_position'][:,2],target_object_height_m=ref['object_position'][:,2],
                     early_failure=failure,demo_end=demo_end,workspace_violation=outside,hand_object_separation_failure=far,
                     joint_limit_violation_rad=torch.maximum(self.full_lower-state['full_q'],state['full_q']-self.full_upper).clamp_min(0).amax(-1),
                     joint_velocity_violation_rad_s=(state['full_q_velocity'].abs()-self.full_velocity).clamp_min(0).amax(-1),
                     **{'reward_'+k:v for k,v in terms.items()})
        return reward,term,timed_out,metrics
