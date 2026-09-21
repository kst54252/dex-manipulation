"""Dense pad approach shaping plus measured five-pad contact rewards.

Proximity is a shaping signal, never a contact or collision certificate.
Only filtered PhysX pad/can forces count toward grasp/contact metrics.
"""
import copy
import numpy as np
import torch

from .grasp import SCHEMA, ContactWorld, PadCanContacts, contact_schedule, contact_terms
from .math3d import quat_apply, quat_inverse
from ..materials import PAD_BODIES

SCHEMA_V2='pad_approach_and_contact_v2'


def cylinder_distance(points,center,radius,half_height):
    """Signed distance to a finite Z cylinder; works on arbitrary batch dims."""
    p=points-center
    d=torch.stack((p[...,:2].norm(dim=-1)-radius,p[...,2].abs()-half_height),-1)
    return d.clamp_min(0).norm(dim=-1)+d.amax(-1).clamp_max(0)


class PadApproach:
    def __init__(self,env):
        self.ids=torch.tensor([env.robot.body_names.index(n) for n in PAD_BODIES],device=env.device)
        samples=[]
        for name in PAD_BODIES:
            collider=next(c for c in env.model.colliders if c['link']==name)
            vertices=np.array(collider['vertices']);faces=np.array(collider['faces'])
            # Triangle centroids include face interiors; FPS fixes 32 local
            # surface samples per physical pad, independent of object pose.
            cloud=np.concatenate((vertices,vertices[faces].mean(1)))
            selected=[int(np.argmax(np.linalg.norm(cloud-cloud.mean(0),axis=-1)))]
            distance=np.full(len(cloud),np.inf)
            for _ in range(31):
                distance=np.minimum(distance,np.square(cloud-cloud[selected[-1]]).sum(-1))
                selected.append(int(np.argmax(distance)))
            samples.append(cloud[selected])
        self.samples=torch.tensor(np.array(samples),device=env.device,dtype=torch.float32)
        self.shapes=[]
        for shape in env.reference.object_geometry['collision_shapes']:
            transform=np.array(shape['transform'])
            if shape['type']!='cylinder' or not np.allclose(transform[:3,:3],np.eye(3)):
                raise ValueError('Pad approach shaping currently requires object-local Z cylinders')
            self.shapes.append((torch.tensor(transform[:3,3],device=env.device,dtype=torch.float32),shape['radius'],shape['height']/2))

    def gap(self,state):
        pose=state['link_transforms'][:,self.ids]
        shape=(len(pose),5,self.samples.shape[1],3)
        points=quat_apply(pose[:,:,None,3:].expand(*shape[:-1],4),self.samples[None].expand(shape))+pose[:,:,None,:3]
        local=quat_apply(quat_inverse(state['object_quaternion'])[:,None,None].expand(*shape[:-1],4),
                         points-state['object_position'][:,None,None])
        signed=torch.stack([cylinder_distance(local,*s) for s in self.shapes]).amin(0)
        return signed.clamp_min(0).amin(-1)


def approach_terms(gap,forces,time,object_error,early_failure,settings,schedule):
    start=schedule['start_time_s'];lead=settings['approach_lead_s']
    on=((time+1e-6>=start-lead)&(time<=schedule['end_time_s']+1e-6)).float()
    distance=torch.exp(-gap/settings['distance_scale_m'])
    quality=torch.exp(-object_error/settings['tracking_std_m'])*(~early_failure).float()
    proximity=on*quality*settings['proximity_weight']*distance.mean(-1)
    weakest=on*quality*settings['weakest_pad_weight']*distance.amin(-1)
    contacts=(forces>=settings['minimum_force_n']).float()
    opposition=(contacts[:,:,0]*contacts[:,:,1:].amax(-1)).mean(0)
    grasp_on=((time+1e-6>=start)&(time<=schedule['end_time_s']+1e-6)).float()
    opposed=grasp_on*quality*settings['opposition_weight']*opposition
    pressure=(forces/settings['pressure_target_n']).clamp(0,1).mean((0,2))
    pressure_bonus=grasp_on*quality*settings['pressure_weight']*pressure
    return proximity+weakest+opposed+pressure_bonus,dict(
        reward_pad_approach=proximity,reward_pad_weakest=weakest,reward_pad_opposition=opposed,
        reward_pad_pressure=pressure_bonus,pad_mean_gap_m=gap.mean(-1),pad_max_gap_m=gap.amax(-1),
        pad_thumb_gap_m=gap[:,0],grasp_opposition_duty=opposition)


class GraspTask:
    def __init__(self,base,contacts,approach,settings,schedule):
        self.base,self.contacts,self.approach=base,contacts,approach
        self.settings,self.schedule=settings,schedule
        self.preload=torch.tensor(settings['preload_rad'],device=base.device)

    def __getattr__(self,name):return getattr(self.base,name)

    def targets(self,ref,actions,previous_q=None,default_offset=None):
        target=self.base.targets(ref,actions,None,default_offset)
        lead=self.settings['preload_ramp_s'];start=self.schedule['start_time_s']
        alpha=((ref['time']-(start-lead))/lead).clamp(0,1)
        alpha=alpha*alpha*(3-2*alpha)
        target['active_q']=(target['active_q']+alpha[:,None]*self.preload).clamp(self.base.lower,self.base.upper)
        if previous_q is not None and self.base.config.get('joint_target_velocity_limit',True):
            step=self.base.velocity*self.base.dt
            target['active_q']=target['active_q'].clamp(previous_q-step,previous_q+step)
        target['full_q']=target['active_q']@self.base.coupling.T+self.base.offset
        return target

    def score(self,state,ref,*args,**kwargs):
        reward,term,trunc,metrics=self.base.score(state,ref,*args,**kwargs)
        forces=self.contacts.consume()
        extra,contact=contact_terms(forces,ref['time'],metrics['object_keypoint_error_m'],metrics['early_failure'],self.settings,self.schedule)
        dense,proximity=approach_terms(self.approach.gap(state),forces,ref['time'],metrics['object_keypoint_error_m'],metrics['early_failure'],self.settings,self.schedule)
        metrics.update(contact);metrics.update(proximity)
        return reward+(extra+dense)*self.dt,term,trunc,metrics


def attach_grasp_task(env):
    settings=copy.deepcopy(env.cfg['grasp_task'])
    if settings['schema']!=SCHEMA_V2:raise ValueError('Unknown grasp task schema')
    schedule=contact_schedule(env.reference,dict(settings,schema=SCHEMA))
    for key in ('approach_lead_s','distance_scale_m','pressure_target_n','preload_ramp_s'):
        if not np.isfinite(settings[key]) or settings[key]<=0:raise ValueError(f'Invalid {key}')
    for key in ('proximity_weight','weakest_pad_weight','opposition_weight','pressure_weight'):
        if not np.isfinite(settings[key]) or settings[key]<0:raise ValueError(f'Invalid {key}')
    preload=np.array(settings['preload_rad'])
    if preload.shape!=(6,) or not np.isfinite(preload).all() or np.abs(preload).max()>.1:
        raise ValueError('Preload requires six bounded joint offsets in radians')
    contacts=PadCanContacts(env)
    env.world=ContactWorld(env.world,contacts)
    env.task=GraspTask(env.task,contacts,PadApproach(env),settings,schedule)
    env.metadata['grasp_task']=dict(settings=settings,schedule=schedule,contact_sensor=contacts.metadata,
        proximity='32 fixed pad-collision surface samples against actual finite can cylinders; shaping only',
        preload='Small nominal finger target compression; no motor strength/friction change; reset reference stays geometric')
    return contacts
