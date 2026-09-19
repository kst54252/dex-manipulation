"""Startup physics randomization and interval velocity disturbances.

Ranges follow the chosen source recipe; Revo2 joint/body mappings use its actual
asset. Physics properties are read back for validation.
"""
import numpy as np
import torch
from .math3d import uniform


def apply_startup_randomization(env):
    cfg=env.cfg['domain_randomization']
    env.default_q_offset=torch.zeros((env.num_envs,6),device=env.device)
    views={'hand':env.robot._physics_view,'object':env.can._physics_view,'table':env.table._physics_view}
    env.nominal_physics={}
    for name,view in views.items():
        env.nominal_physics[name]={'materials':view.get_material_properties().clone().cpu(), 'masses':view.get_masses().clone().cpu()}
    env.nominal_physics['hand']['inertias']=env.robot._physics_view.get_inertias().clone().cpu()
    env.nominal_physics['object']['inertias']=env.can._physics_view.get_inertias().clone().cpu()
    env.nominal_physics['object']['coms']=env.can._physics_view.get_coms().clone().cpu()
    env.nominal_physics['hand']['stiffness']=env.robot._physics_view.get_dof_stiffnesses().clone().cpu()
    env.nominal_physics['hand']['damping']=env.robot._physics_view.get_dof_dampings().clone().cpu()
    if not cfg['enabled']:
        env.randomized_physics={name:{key:value.clone() for key,value in item.items()} for name,item in env.nominal_physics.items()}
        return
    rng=env.rng
    env.randomized_physics={name:{} for name in views}
    for name,view in views.items():
        nominal=env.nominal_physics[name]
        low,high=cfg[name+'_friction']
        buckets=rng.uniform(low,high,(cfg['material_buckets'],2))
        buckets[:,1]=np.minimum(buckets[:,1],buckets[:,0])
        materials=nominal['materials'].clone()
        indices=rng.integers(0,len(buckets),materials.shape[:-1])
        materials[...,:2]=torch.from_numpy(buckets[indices]).float()
        materials[...,2]=0
        env.randomized_physics[name]['materials']=materials
        if name!='table':
            mass=nominal['masses']
            factor=torch.tensor(rng.uniform(*cfg[name+'_mass_scale'],mass.shape),dtype=mass.dtype)
            env.randomized_physics[name]['masses']=mass*factor
            # Inertia scales with mass; this is also the general Isaac Lab mass randomizer's behavior.
            inertia=nominal['inertias']
            env.randomized_physics[name]['inertias']=inertia*factor.reshape(*inertia.shape[:-1],1)
    hand=env.randomized_physics['hand']
    for name in ('stiffness','damping'):
        nominal=env.nominal_physics['hand'][name]
        factor=torch.tensor(np.exp(rng.uniform(np.log(cfg['gain_scale'][0]),np.log(cfg['gain_scale'][1]),nominal.shape)),dtype=nominal.dtype)
        hand[name]=nominal*factor
    com=env.nominal_physics['object']['coms'].clone()
    extent=np.array(cfg['object_com_range_m'])
    com[...,:3]+=torch.tensor(rng.uniform(-extent,extent,com[...,:3].shape),dtype=com.dtype)
    env.randomized_physics['object']['coms']=com
    amount=cfg['joint_default_offset_rad']
    env.default_q_offset[:]=torch.tensor(rng.uniform(-amount,amount,(env.num_envs,6)),device=env.device,dtype=torch.float32)
    set_physics_properties(env,env.randomized_physics)


def set_physics_properties(env,properties):
    indices=torch.arange(env.num_envs,dtype=torch.int32,device='cpu')
    for name,values in properties.items():
        view={'hand':env.robot._physics_view,'object':env.can._physics_view,'table':env.table._physics_view}[name]
        for key,value in values.items():
            method={'materials':'set_material_properties','masses':'set_masses','inertias':'set_inertias','coms':'set_coms','stiffness':'set_dof_stiffnesses','damping':'set_dof_dampings'}[key]
            getattr(view,method)(value.contiguous(),indices)
    env.refresh_physics_properties()


def property_report(env):
    report={}
    for name,view in [('hand',env.robot._physics_view),('object',env.can._physics_view),('table',env.table._physics_view)]:
        m=view.get_material_properties()
        mass=view.get_masses()
        report[name]=dict(static_friction_range=[float(m[...,0].min()),float(m[...,0].max())],dynamic_friction_range=[float(m[...,1].min()),float(m[...,1].max())],
                          mass_range_kg=[float(mass.min()),float(mass.max())])
    com=env.can._physics_view.get_coms().cpu()[...,:3]-env.nominal_physics['object']['coms'][...,:3]
    report['object']['com_offset_ranges_m']=torch.stack((com.amin(0),com.amax(0))).tolist()
    report['joint_default_offset_ranges_rad']=torch.stack((env.default_q_offset.amin(0),env.default_q_offset.amax(0))).cpu().tolist()
    return report


class PushSchedule:
    def __init__(self,count,config,device,generator):
        self.cfg,self.device,self.generator=config,device,generator
        self.remaining=torch.zeros((count,2),device=device)
        self.total_events=0
        self.last_delta=torch.zeros((count,2,6),device=device)
        self.reset(torch.arange(count,device=device))

    def reset(self,ids):
        self.remaining[ids]=uniform((len(ids),2),*self.cfg['interval_s'],self.device,self.generator)

    def advance(self,env):
        self.last_delta.zero_()
        if not env.training or not self.cfg['enabled']: return
        self.remaining-=env.task.dt
        stage=max(row for row in self.cfg['stages'] if row[0]<=env.control_steps)
        for slot,asset in enumerate((env.robot,env.can)):
            ids=torch.nonzero(self.remaining[:,slot]<=0,as_tuple=False).flatten()
            if not len(ids): continue
            amplitudes=torch.tensor([stage[1]]*3+[stage[2]]*3,device=self.device)
            delta=uniform((len(ids),6),-amplitudes,amplitudes,self.device,self.generator)
            velocities=asset.get_velocities()[ids]+delta
            asset.set_velocities(velocities,indices=ids)
            self.last_delta[ids,slot]=delta
            self.remaining[ids,slot]=uniform((len(ids),),*self.cfg['interval_s'],self.device,self.generator)
            if stage[1]>0: self.total_events+=len(ids)
