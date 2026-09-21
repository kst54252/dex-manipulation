"""Asymmetric REGRIND observation recipe with device-resident delay/history.

No source code is imported. Delay interpolation, grouped random lags and raw
policy noise are independently implemented; repeated reads do not resample.
"""
import torch
from tensordict import TensorDict

from .math3d import rotation6d, from_rotvec, quat_multiply, slerp, uniform


class ObservationHistory:
    def __init__(self, count, dofs, config, device, generator):
        self.count, self.dofs, self.cfg, self.device, self.generator = count, dofs, config, device, generator
        self.include_object_velocity=config.get('include_object_velocity',False)
        self.actor_size, self.critic_size = 43+4*dofs, 43+5*dofs+15+6*int(self.include_object_velocity)
        self.lags = torch.zeros((count, 3), device=device)
        self.default_q = torch.zeros((count, dofs), device=device)
        self.position_offset = torch.zeros((count,3),device=device)
        self.buffers = {name: torch.zeros((count, 3, width), device=device) for name, width in
                        (("object_position",3),("object_quaternion",4),("wrist_position",3),("wrist_quaternion",4),("q",dofs))}
        self.histories = {name: torch.zeros((count,2,width),device=device) for name,width in
                          (("policy_position",3),("policy_rotation",6),("policy_q",dofs),
                           ("critic_position",3),("critic_rotation",6),("critic_q",dofs))}
        self.actor = torch.zeros((count,self.actor_size),device=device)
        self.critic = torch.zeros((count,self.critic_size),device=device)
        self.delayed = {}

    def reset_lags(self, ids, noisy):
        self.lags[ids] = uniform((len(ids),3),0,self.cfg["max_delay_steps"],self.device,self.generator) if noisy else 0

    def update(self, state, ids, reset=False, noisy=True):
        if reset:
            self.reset_lags(ids,noisy)
        for name, buffer in self.buffers.items():
            value = state[name][ids].clone()
            if name.endswith('position'):value-=self.position_offset[ids]
            if noisy:
                if name.endswith("quaternion"):
                    axis = torch.randn((len(ids),3),device=self.device,generator=self.generator)
                    axis /= axis.norm(dim=-1,keepdim=True).clamp_min(1e-12)
                    angle = uniform((len(ids),1),-self.cfg["rotation_noise_rad"],self.cfg["rotation_noise_rad"],self.device,self.generator)
                    value = quat_multiply(from_rotvec(axis*angle),value)
                else:
                    scale = self.cfg["joint_noise_rad"] if name=="q" else self.cfg["position_noise_m"]
                    value += uniform(value.shape,-scale,scale,self.device,self.generator)
            if name=="q": value -= self.default_q[ids]
            if reset: buffer[ids] = value[:,None]
            else:
                buffer[ids,2] = buffer[ids,1].clone()
                buffer[ids,1] = buffer[ids,0].clone()
                buffer[ids,0] = value
            group = 0 if name.startswith("object") else (2 if name=="q" else 1)
            lag = self.lags[ids,group]
            lower = lag.floor().long().clamp(0,2)
            upper = (lower+1).clamp(max=2)
            a,b = buffer[ids,lower],buffer[ids,upper]
            blend = lag-lower
            delayed = slerp(a,b,blend) if name.endswith("quaternion") else torch.lerp(a,b,blend[:,None])
            if name not in self.delayed: self.delayed[name]=torch.zeros_like(state[name])
            self.delayed[name][ids]=delayed
        critic_q=state['q'][ids]
        if self.cfg.get('critic_joint_relative_to_default',False):critic_q=critic_q-self.default_q[ids]
        values = dict(policy_position=self.delayed["wrist_position"][ids],policy_rotation=rotation6d(self.delayed["wrist_quaternion"][ids]),policy_q=self.delayed["q"][ids],
                      critic_position=state["wrist_position"][ids]-self.position_offset[ids],critic_rotation=rotation6d(state["wrist_quaternion"][ids]),critic_q=critic_q)
        for name,value in values.items():
            history=self.histories[name]
            if reset: history[ids]=value[:,None]
            else:
                history[ids,0]=history[ids,1].clone()
                history[ids,1]=value

    def assemble(self,state,ref,action,phase,ids=None):
        ids=torch.arange(self.count,device=self.device) if ids is None else ids
        common=[action[ids],phase[ids,None],ref["wrist_position"][ids]-self.position_offset[ids],rotation6d(ref["wrist_quaternion"][ids]),ref["q"][ids]]
        def history(prefix):
            return [self.histories[prefix+name][ids].flatten(1) for name in ("_position","_rotation","_q")]
        self.actor[ids]=torch.cat([self.delayed["object_position"][ids],rotation6d(self.delayed["object_quaternion"][ids]),*history("policy"),*common],-1)
        object_velocity=[state['object_velocity'][ids],state['object_angular_velocity'][ids]] if self.include_object_velocity else []
        self.critic[ids]=torch.cat([state["object_position"][ids]-self.position_offset[ids],rotation6d(state["object_quaternion"][ids]),*history("critic"),*common,*object_velocity,
                                  (state["fingertips"][ids]-self.position_offset[ids,None]).flatten(1),state["q_velocity"][ids]],-1)
        return self.get()

    def get(self):
        return TensorDict({"policy":self.actor.clone(),"critic":self.critic.clone()},batch_size=[self.count])
