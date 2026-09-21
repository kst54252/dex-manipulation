"""Batched bounded arm FK/IK; pure Torch, independent of any simulator API."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .ik import IKOptions
from .policy.math3d import quat_apply, quat_inverse, quat_multiply, from_rotvec, rotation_error


def pose_tensor(matrix, device='cpu', dtype=torch.float32):
    matrix=np.asarray(matrix)
    return torch.as_tensor(np.r_[matrix[:3,3],Rotation.from_matrix(matrix[:3,:3]).as_quat()],device=device,dtype=dtype)


def compose(a,b):
    aq,bq=torch.broadcast_tensors(a[...,3:],b[...,3:])
    return torch.cat((a[...,:3]+quat_apply(aq,b[...,:3].expand_as(aq[...,:3])),quat_multiply(aq,bq)),-1)


def inverse(a):
    q=quat_inverse(a[...,3:])
    return torch.cat((quat_apply(q,-a[...,:3]),q),-1)


class BatchedArmIK:
    """Local continuation, adaptive damped least squares and strict FK checks.

    Failed candidates never replace accepted targets. A separate candidate
    residual remains available to RL, including at step/velocity bounds.
    """
    def __init__(self,model,options=None,*,device='cpu',dtype=torch.float32,iterations=12,compile_fk=False):
        self.model,self.options=model,options or IKOptions()
        self.device,self.dtype=torch.device(device),dtype
        self.iterations=int(iterations)
        if self.iterations<1:raise ValueError('IK iterations must be positive')
        tensor=lambda v:torch.as_tensor(v,device=device,dtype=dtype)
        self.lower,self.upper,self.velocity=map(tensor,(model.lower,model.upper,model.velocity))
        self.lo=self.lower+self.options.limit_margin_rad;self.hi=self.upper-self.options.limit_margin_rad
        self.scale=tensor([1/self.options.jacobian_length_m]*3+[1.]*3)
        self.identity=torch.eye(6,device=device,dtype=dtype)
        self.root=tensor([0.,0.,0.,0.,0.,0.,1.])
        self.mount=pose_tensor(model.mount,device,dtype)
        self.frames=[]
        for j in model.joints:
            a,b=pose_tensor(j['frame0'],device,dtype),pose_tensor(j['frame1'],device,dtype)
            rev=j.get('reversed',False)
            self.frames.append((j['parent'],j['child'],b if rev else a,inverse(a if rev else b),
                None if j['kind']=='fixed' else tensor(j['axis'])*(-1 if rev else 1),
                None if j['kind']=='fixed' else model.active_names.index(j['name'])))
        self.evaluate=self.fk_jacobian
        if compile_fk:self.evaluate=torch.compile(self.fk_jacobian,fullgraph=True,dynamic=True)

    def fk_jacobian(self,q):
        links={self.model.root:self.root.expand(len(q),7)}
        axes=[];origins=[]
        for parent,child,f0,f1,axis,index in self.frames:
            frame=compose(links[parent],f0)
            if index is not None:
                origins.append(frame[:,:3]);axes.append(quat_apply(frame[:,3:],axis.expand(len(q),3)))
                motion=torch.cat((torch.zeros_like(q[:,:3]),from_rotvec(q[:,index:index+1]*axis)),-1)
                frame=compose(frame,motion)
            links[child]=compose(frame,f1)
        flange,wrist=links[self.model.flange],links[self.model.wrist]
        axis=torch.stack(axes,-1);origin=torch.stack(origins,1)
        linear=torch.cross(axis.transpose(1,2),flange[:,None,:3]-origin,dim=-1).transpose(1,2)
        return wrist,flange,torch.cat((linear,axis),1)

    def error(self,target,pose):
        return torch.cat((target[:,:3]-pose[:,:3],rotation_error(target[:,3:],pose[:,3:])),-1)

    def singularity(self,jacobian):
        values=torch.linalg.svdvals(jacobian*self.scale[None,:,None])
        return values[:,-1],values[:,0]/values[:,-1].clamp_min(1e-12)

    @torch.no_grad()
    def solve(self,target,seed,*,previous=None,dt=None,transition=True):
        if target.ndim!=2 or target.shape!=(len(seed),7) or seed.shape[1:]!=(6,):
            raise ValueError('Expected target [N,7] XYZW and arm seed [N,6]')
        if not torch.isfinite(target).all() or not torch.isfinite(seed).all():
            raise ValueError('Nonfinite arm IK input')
        if (target[:,3:].norm(dim=-1)-1).abs().max()>1e-3:
            raise ValueError('Arm IK target must use unit XYZW quaternions')
        low,high=self.lo.expand_as(seed),self.hi.expand_as(seed)
        if previous is not None:
            if previous.shape!=seed.shape or not torch.isfinite(previous).all():raise ValueError('Invalid previous arm target')
            if dt is None or not np.isfinite(dt) or dt<=0:raise ValueError('Streaming IK requires actual positive dt')
            step=torch.minimum(self.velocity*dt,torch.full_like(self.velocity,self.options.max_joint_step_rad))
            low=torch.maximum(low,previous-step);high=torch.minimum(high,previous+step)
        if (low>high).any():raise ValueError('Empty IK step/limit interval')
        q=seed.clamp(low,high);preferred=q.clone()
        flange_target=compose(target,inverse(self.mount))
        peak=torch.zeros(len(q),device=self.device,dtype=self.dtype)
        for _ in range(self.iterations):
            wrist,flange,jac=self.evaluate(q)
            error=self.error(flange_target,flange)*self.scale
            j=jac*self.scale[None,:,None]
            sigma,_=self.singularity(jac)
            damping=self.options.damping_min+self.options.damping_max*(1-sigma/self.options.near_sigma_min).clamp_min(0).square()
            # Float32 normal equations need a small nonzero numerical floor.
            damping=damping.clamp_min(1e-4 if self.dtype==torch.float32 else 1e-8)
            peak=torch.maximum(peak,damping)
            distance=torch.minimum(q-self.lower,self.upper-q).clamp_min(1e-5)
            weights=1+(self.options.limit_warning_rad/distance).square().clamp_max(100)
            bias=(.05*((self.lower+self.upper)/2-q)/(self.upper-self.lower)+.1*(preferred-q)).clamp(-.1,.1)
            regularizer=damping[:,None].square()*weights
            normal=j.transpose(1,2)@j+torch.diag_embed(regularizer)
            rhs=(j.transpose(1,2)@error[:,:,None]).squeeze(-1)+regularizer*bias
            delta=torch.linalg.solve(normal,rhs)
            delta*= (self.options.iteration_step_rad/delta.abs().amax(-1).clamp_min(1e-12)).clamp_max(1)[:,None]
            # Trust-region trial; keep the previous candidate if a step worsens pose cost.
            trial=(q+delta).clamp(low,high)
            _,tf,_=self.evaluate(trial)
            old_cost=error.square().sum(-1)
            new_cost=(self.error(flange_target,tf)*self.scale).square().sum(-1)
            quarter=(q+.25*delta).clamp(low,high)
            _,qf,_=self.evaluate(quarter)
            quarter_cost=(self.error(flange_target,qf)*self.scale).square().sum(-1)
            improved=torch.where((quarter_cost<old_cost)[:,None],quarter,q)
            q=torch.where((new_cost<=old_cost+1e-14)[:,None],trial,improved)
        wrist,flange,jac=self.evaluate(q)
        error=self.error(target,wrist);sigma,condition=self.singularity(jac)
        pos,rot=error[:,:3].norm(dim=-1),error[:,3:].norm(dim=-1)
        margin=torch.minimum(q-self.lower,self.upper-q).amin(-1)
        success=(pos<=self.options.position_tolerance_m)&(rot<=self.options.orientation_tolerance_rad)
        success&=(sigma>=self.options.singular_sigma_min)&(condition<=self.options.max_condition)
        success&=(margin>=self.options.limit_margin_rad-1e-6)&torch.isfinite(q).all(-1)
        transition_ok=torch.ones_like(success)
        if previous is not None and transition:
            # Sample the actual accepted command interval, as in the CPU solver.
            alpha=torch.linspace(0,1,self.options.transition_samples+1,device=self.device,dtype=self.dtype)
            samples=previous[:,None]+alpha[None,:,None]*(q-previous)[:,None]
            _,_,tj=self.evaluate(samples.reshape(-1,6))
            ts,tc=self.singularity(tj)
            transition_ok=((ts>=self.options.singular_sigma_min)&(tc<=self.options.max_condition)).reshape(len(q),-1).all(-1)
            success&=transition_ok
        accepted=torch.where(success[:,None],q,seed if previous is None else previous)
        return dict(q=accepted,candidate_q=q,candidate_pose=wrist,error=error,position_error_m=pos,
                    orientation_error_rad=rot,sigma_min=sigma,condition=condition,success=success,
                    joint_limit_distance_rad=margin,transition_valid=transition_ok,damping_max=peak,
                    velocity=torch.zeros_like(q) if previous is None else (accepted-previous)/dt)
