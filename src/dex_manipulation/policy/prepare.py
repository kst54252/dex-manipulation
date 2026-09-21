"""Prepare a contact reference from this project's own physical rollout.

Only the wrist and fingers are changed. The original object targets, their
timestamps and the source human points remain intact. No external retargeter
or learned weights are used by this offline, simulator-independent operation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp

from ..geometry import CollisionScene


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _transforms(position, quaternion):
    p,q=np.asarray(position),np.asarray(quaternion)
    if not np.isfinite(p).all() or not np.isfinite(q).all() or np.any(abs(np.linalg.norm(q,axis=-1)-1)>1e-3):
        raise ValueError('Invalid measured pose')
    out=np.tile(np.eye(4),(len(p),1,1))
    out[:,:3,3]=p;out[:,:3,:3]=Rotation.from_quat(q).as_matrix()
    return out


def align_contact_poses(data,rollout,environment=0):
    """Match each measured hand/object relationship to the unchanged target.

    Accept the single-environment output from policy play/evaluate as well as
    batched diagnostic rollouts. Require a measured sample at every input pose;
    target equality checks prevent silently using a different demonstration.
    """
    n=len(data['timestamps_s'])
    batched=rollout['q'].ndim==3
    count=rollout['q'].shape[1] if batched else 1
    if rollout['q'].ndim not in (2,3) or not 0<=environment<count:
        raise ValueError('Invalid rollout shape or environment index')
    def all_values(key):return rollout[key][:,environment] if batched else rollout[key]
    failure_key='early_failure' if 'early_failure' in rollout else 'metric_early_failure'
    if failure_key in rollout and all_values(failure_key).any():
        raise ValueError('A failed physical rollout cannot seed this contact reference')
    indices=np.arange(n)
    if len(rollout['q'])!=n:
        if 'time_s' not in rollout:raise ValueError('Dense rollouts require command timestamps')
        time=all_values('time_s')
        if len(time)<n or np.any(np.diff(time)<=0):raise ValueError('Invalid rollout timestamps')
        phase=(data['timestamps_s']-data['timestamps_s'][0])/np.ptp(data['timestamps_s'])
        requested=time[0]+phase*(time[-1]-time[0])
        indices=np.abs(time[:,None]-requested[None]).argmin(0)
        if not np.allclose(time[indices],requested,atol=1e-6,rtol=0):
            raise ValueError('Rollout has no exact samples at the original reference poses')
    def values(key):return all_values(key)[indices]
    target=_transforms(values('reference_object_position'),values('reference_object_quaternion'))
    if not np.allclose(target,data['object_transform'],atol=2e-6,rtol=0):
        raise ValueError('Rollout object targets differ from the original reference')
    if not np.allclose(values('reference_q'),data['active_q_rad'],atol=2e-6,rtol=0):
        raise ValueError('Rollout finger targets differ from the original reference')
    actual_object=_transforms(values('object_position'),values('object_quaternion'))
    actual_wrist=_transforms(values('wrist_position'),values('wrist_quaternion'))
    return data['object_transform']@np.linalg.inv(actual_object)@actual_wrist,values('q').copy()


def prepare_contact_reference(source,rollout_path,output,model,object_geometry,*,environment=0,
                              clearance_m=.0001,projection_translation_m=.002,
                              projection_rotation_rad=.02,projection_joint_rad=.02,
                              transition_substeps=20,ground_z_m=0.):
    """Collision-project measured contact poses, then verify all interpolants.

    The small projection removes PhysX contact penetration and FK/coupling
    discrepancies. Solver status and measured constraint feasibility are
    recorded separately. Substep checks are not a continuous collision proof.
    """
    source,rollout_path,output=map(Path,(source,rollout_path,output))
    if output.suffix!='.npz':raise ValueError('The output filename must end in .npz')
    if output.resolve() in (source.resolve(),rollout_path.resolve()) or output.exists():
        raise ValueError('Choose a new output; never overwrite the reference or rollout')
    parameters=[clearance_m,projection_translation_m,projection_rotation_rad,projection_joint_rad]
    if not np.isfinite(parameters).all() or min(parameters)<=0 or transition_substeps<1:
        raise ValueError('Positive finite projection parameters and substeps are required')
    with np.load(source,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
    with np.load(rollout_path,allow_pickle=False) as f:rollout={k:f[k].copy() for k in f.files}
    if list(data['active_joint_names'])!=model.active_names:
        raise ValueError('Reference/model joint order mismatch')
    if not data['valid'].all() or not data['transition_valid'][1:].all():
        raise ValueError('The original reference contains unverified frames or transitions')
    wrist,q=align_contact_poses(data,rollout,environment)
    if not np.isfinite(q).all() or max(np.max(model.lower-q),np.max(q-model.upper))>5e-4:
        raise ValueError('Measured joints exceed the model limits')
    q=np.clip(q,model.lower,model.upper)
    scene=CollisionScene(model,object_geometry)
    if str(data['object_geometry_fingerprint'])!=scene.fingerprint:
        raise ValueError('Collision geometry differs from the source reference')
    scale=np.array([projection_translation_m]*3+[projection_rotation_rad]*3+[projection_joint_rad]*6)
    records=[]
    def clearance(j,t,obj):
        links=model.link_transforms(j,t)
        ground=min((np.asarray(c['vertices'])@links[c['link']][:3,:3].T+links[c['link']][:3,3])[:,2].min()
                   for c in model.colliders)-ground_z_m
        return scene.distances(links,obj),float(ground)
    for i in range(len(q)):
        base,j,obj=wrist[i],q[i],data['object_transform'][i]
        def pose(x):
            t=base.copy();t[:3,3]+=x[:3]*scale[:3]
            t[:3,:3]=Rotation.from_rotvec(x[3:6]*scale[3:6]).as_matrix()@base[:3,:3]
            return t,j+x[6:]*scale[6:]
        def constraints(x):
            t,a=pose(x);ds,z=clearance(a,t,obj)
            return np.r_[ds-clearance_m,z]/projection_translation_m
        bounds=[(-1.,1.)]*6+list(zip(np.maximum(-1,(model.lower-j)/scale[6:]),np.minimum(1,(model.upper-j)/scale[6:])))
        result=minimize(lambda x:float(x@x),np.zeros(12),method='SLSQP',bounds=bounds,
            constraints=[dict(type='ineq',fun=constraints)],options={'maxiter':300,'ftol':1e-8})
        t,a=pose(result.x);ds,z=clearance(a,t,obj)
        feasible=bool(np.isfinite(result.x).all() and ds.min()>=clearance_m*.9 and z>=0 and model.limit_violation(a)<=1e-8)
        records.append(dict(frame_id=int(data['frame_ids'][i]),solver_success=bool(result.success),
            solver_message=str(result.message),geometry_valid=feasible,minimum_distance_m=float(ds.min()),
            ground_clearance_m=z,projection=result.x.tolist()))
        if not feasible:raise ValueError(f'Contact projection failed: {records[-1]}')
        data['wrist_transform'][i]=t;data['active_q_rad'][i]=a;data['full_q_rad'][i]=model.expand(a)
        data['wrist_translation_m'][i]=t[:3,3];data['wrist_quaternion_xyzw'][i]=Rotation.from_matrix(t[:3,:3]).as_quat()
        data['robot_keypoints'][i]=model.keypoint_positions(a,t);data['valid'][i]=True
    transition=[]
    for i in range(1,len(q)):
        keys=['wrist_transform','object_transform']
        rotations=[Slerp([0,1],Rotation.from_matrix(data[k][i-1:i+1,:3,:3])) for k in keys]
        minimum=ground_minimum=float('inf')
        for fraction in np.linspace(0,1,transition_substeps+1):
            poses=[]
            for key,rot in zip(keys,rotations):
                t=np.eye(4);t[:3,:3]=rot(fraction).as_matrix()
                t[:3,3]=(1-fraction)*data[key][i-1,:3,3]+fraction*data[key][i,:3,3];poses.append(t)
            j=(1-fraction)*data['active_q_rad'][i-1]+fraction*data['active_q_rad'][i]
            ds,z=clearance(j,poses[0],poses[1]);minimum=min(minimum,float(ds.min()));ground_minimum=min(ground_minimum,z)
        valid=minimum>=0 and ground_minimum>=0
        data['transition_valid'][i]=valid
        transition.append(dict(frame_id=int(data['frame_ids'][i]),minimum_distance_m=minimum,ground_clearance_m=ground_minimum,valid=valid))
        if not valid:raise ValueError(f'Contact transition failed: {transition[-1]}')
    velocity=np.abs(np.diff(data['full_q_rad'],axis=0)/np.diff(data['timestamps_s'])[:,None])
    if np.any(velocity>model.full_velocity+1e-6):raise ValueError('Prepared reference exceeds input-time joint velocity limits')
    metadata=dict(method='own physical hand/object contact alignment with fixed original object targets',
        source=str(source),source_sha256=_digest(source),rollout=str(rollout_path),rollout_sha256=_digest(rollout_path),
        environment_index=environment,object_targets_changed=False,timestamps_changed=False,
        clearance_m=clearance_m,projection_scales=scale.tolist(),transition_substeps=transition_substeps,
        frames=records,transitions=transition,
        limitations=['sampled hand-object and hand-ground geometry checks, not continuous collision proof',
                     'does not certify physical grasp success; evaluate the newly trained policy'])
    data['policy_reference_alignment_json']=np.array(json.dumps(metadata,sort_keys=True))
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**data)
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata
