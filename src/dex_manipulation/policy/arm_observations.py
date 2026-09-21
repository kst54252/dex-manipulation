"""Measured arm context appended to the existing hand/object observations."""
import torch
from tensordict import TensorDict
from .observations import ObservationHistory


class ArmObservations:
    extra_size=20

    def __init__(self,count,config,arm,device,generator,max_failures):
        self.base=ObservationHistory(count,6,config,device,generator)
        self.count,self.device,self.max_failures=count,device,max_failures
        tensor=lambda v:torch.as_tensor(v,device=device,dtype=torch.float32)
        self.center=tensor((arm.lower+arm.upper)/2);self.half=tensor((arm.upper-arm.lower)/2)
        self.velocity=tensor(arm.velocity)
        self.error_scale=tensor([.02]*3+[.2]*3)
        self.actor_size=self.base.actor_size+self.extra_size
        self.critic_size=self.base.critic_size+self.extra_size
        self.actor=torch.zeros(count,self.actor_size,device=device)
        self.critic=torch.zeros(count,self.critic_size,device=device)

    def __getattr__(self,name):
        return getattr(self.base,name)

    def update(self,*args,**kwargs):
        self.base.update(*args,**kwargs)

    def assemble(self,state,ref,action,phase,ids=None):
        ids=torch.arange(self.count,device=self.device) if ids is None else ids
        self.base.assemble(state,ref,action,phase,ids)
        extra=torch.cat(((state['q_arm'][ids]-self.center)/self.half,
                         (state['q_arm_velocity'][ids]/self.velocity).clamp(-5,5),
                         (state['arm_last_ik_error'][ids]/self.error_scale).clamp(-5,5),
                         state['arm_last_ik_success'][ids,None].float(),
                         state['arm_ik_failure_streak'][ids,None].float()/self.max_failures),-1)
        self.actor[ids]=torch.cat((self.base.actor[ids],extra),-1)
        self.critic[ids]=torch.cat((self.base.critic[ids],extra),-1)
        return self.get()

    def get(self):
        return TensorDict({'policy':self.actor.clone(),'critic':self.critic.clone()},batch_size=[self.count])
