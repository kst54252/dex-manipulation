"""Device-resident PhysX environment with an independent REGRIND task adapter.

The reference can pose is written only at episode reset. Interval pushes change
velocity (never teleport pose). Source USD/data remain read-only.
"""
from pathlib import Path
import copy
import time
import math
import numpy as np
import torch
from .task import ResidualTask
from .reference import TensorReference
from .math3d import quat_apply, from_rotvec, quat_multiply, uniform
from ..control import wrist_pd_wrenches
from .observations import ObservationHistory
from .curriculum import ReferenceStateSampler,GravityCurriculum
from .randomization import apply_startup_randomization,set_physics_properties,property_report,PushSchedule


class PhysxResidualEnv:
    def __init__(self,root,model,reference,config,num_envs=8,render=False):
        from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,UsdShade,UsdLux,Gf
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation,RigidPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from ..scene import FloatingSurface
        started=time.monotonic()
        self.startup_timings={}
        def progress(phase):
            elapsed=time.monotonic()-started
            self.startup_timings[phase]=elapsed
            print(f"[startup] {phase} | envs={num_envs} elapsed={elapsed:.1f}s",flush=True)
        progress("create physics world")
        self.model,self.reference,self.cfg=model,reference,config
        self.surface=FloatingSurface(config['surface'], reference.object[0,:3,3])
        from .trajectory import digest
        expected = reference.metadata.get('object_asset_sha256')
        if expected and digest(Path(root)/config['object_asset']) != expected:
            raise ValueError('Object USD differs from the extracted geometry; rebuild object and reference')
        self.device=torch.device(config['physics_device'])
        self.num_envs,self.render=num_envs,render
        self.all_ids=torch.arange(num_envs,device=self.device,dtype=torch.long)
        self.generator=torch.Generator(device=self.device).manual_seed(config['seed'])
        self.rng=np.random.default_rng(config['seed'])
        self.motion=TensorReference(reference,num_envs,config['augmentation'],self.device)
        self.task=ResidualTask(model,self.motion,config,self.device)
        self.sampler=ReferenceStateSampler(reference.duration,self.task.dt,config['rsi'],self.device)
        self.gravity=GravityCurriculum(config['gravity_curriculum'],config['gravity'])
        self.training,self.control_steps=False,0
        self.evaluation_protocol='strict'
        width=math.ceil(math.sqrt(num_envs))
        scene_origins=torch.tensor([[config['env_spacing_m']*(i%width),config['env_spacing_m']*(i//width),0] for i in range(num_envs)],device=self.device)
        # Task/reference coordinates remain unchanged. Translate only at the
        # simulator boundary so the nominal can starts at the support's center.
        self.origins=scene_origins+torch.tensor(self.surface.translation,device=self.device,dtype=torch.float32)
        self.time=torch.zeros(num_envs,device=self.device)
        self.episode_length_buf=torch.zeros(num_envs,dtype=torch.long,device=self.device)
        self.max_episode_length=int(round(reference.duration/self.task.dt))+1
        self.last_action=torch.zeros(num_envs,12,device=self.device)
        self.last_q_target=torch.zeros(num_envs,6,device=self.device)
        self.object_reset_count=0
        # Match Isaac Lab's headless Fabric output policy. SimulationApp's
        # generic defaults also export joint/sensor/velocity state for display,
        # costing CPU readback even when World.step(render=False) is used.
        # All learning state is read from PhysX tensors; these are output flags,
        # not switches for collision solving, joint constraints or integration.
        import carb
        output_settings=carb.settings.get_settings()
        fabric_output_keys=('fabricUpdateTransformations','fabricUpdateVelocities',
                            'fabricUpdateForceSensors','fabricUpdateJointStates')
        if not render:
            for key in fabric_output_keys:output_settings.set_bool('/physics/'+key,False)
            output_settings.set_bool('/physics/fabricUseGPUInterop',self.device.type=='cuda')
        self.world=World(stage_units_in_meters=1.,physics_dt=config['physics_dt'],rendering_dt=self.task.dt,backend='torch',device=str(self.device))
        context=self.world.get_physics_context()
        context.set_gravity(-config['gravity'])
        context.set_solver_type('TGS')
        context.set_gpu_max_rigid_contact_count(2**23)
        context.set_gpu_max_rigid_patch_count(2**23)
        # These are inherited Isaac Lab PhysxCfg defaults in the source recipe;
        # legacy World otherwise allocates too few aggregate pairs at 4096 envs.
        context.set_gpu_found_lost_pairs_capacity(2**21)
        context.set_gpu_found_lost_aggregate_pairs_capacity(2**25)
        context.set_gpu_total_aggregate_pairs_capacity(2**21)
        context.set_gpu_collision_stack_size(2**28)
        context.set_gpu_max_num_partitions(1)
        stage=self.world.stage
        scene_api=PhysxSchema.PhysxSceneAPI.Apply(context.get_current_physics_scene_prim())
        scene_api.CreateMaxPositionIterationCountAttr(192)
        scene_api.CreateMaxVelocityIterationCountAttr(1)
        scene_api.CreateBounceThresholdAttr(.2)
        scene_api.CreateFrictionOffsetThresholdAttr(.01)
        scene_api.CreateFrictionCorrelationDistanceAttr(.00625)
        UsdLux.DomeLight.Define(stage,'/World/Light').CreateIntensityAttr(900)
        if render:
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(eye=np.array([.6,-.6,.5]),target=np.array([0,0,.12]))
        material=UsdShade.Material.Define(stage,'/World/ContactMaterial')
        mat=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        mat.CreateStaticFrictionAttr(1.);mat.CreateDynamicFrictionAttr(1.);mat.CreateRestitutionAttr(0.)
        self.mimic_specs=[]
        progress("build source environment")
        from isaacsim.core.cloner import Cloner
        cloner=Cloner(stage=stage)
        cloner.define_base_env('/World/envs')
        self.env_paths=cloner.generate_paths('/World/envs/env',num_envs)
        prefix=self.env_paths[0]
        UsdGeom.Xform.Define(stage,prefix)
        hand_path=prefix+'/Hand'
        add_reference_to_stage(str(Path(root)/config['hand_asset']),hand_path)
        add_reference_to_stage(str(Path(root)/config['object_asset']),prefix+'/Can')
        geometry=self.surface.create_usd(stage,prefix)
        table=geometry['Table']
        body=UsdPhysics.RigidBodyAPI.Apply(table);body.CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(table).CreateMassAttr(1.)
        tablephys=PhysxSchema.PhysxRigidBodyAPI.Apply(table);tablephys.CreateDisableGravityAttr(True)
        tablephys.CreateSolverPositionIterationCountAttr(8);tablephys.CreateSolverVelocityIterationCountAttr(1)
        tablephys.CreateMaxDepenetrationVelocityAttr(1.)
        col=PhysxSchema.PhysxCollisionAPI.Apply(table);col.CreateContactOffsetAttr(.005);col.CreateRestOffsetAttr(0.)
        col.CreateTorsionalPatchRadiusAttr(.02);col.CreateMinTorsionalPatchRadiusAttr(.005)
        art=PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(hand_path))
        art.CreateSolverPositionIterationCountAttr(32);art.CreateSolverVelocityIterationCountAttr(1)
        art.CreateEnabledSelfCollisionsAttr(config['self_collision'])
        art.CreateSleepThresholdAttr(.005);art.CreateStabilizationThresholdAttr(.0005)
        joints={p.GetName():p for p in Usd.PrimRange(stage.GetPrimAtPath(hand_path)) if p.IsA(UsdPhysics.RevoluteJoint)}
        for j in model.moving:
            prim=joints[j['name']];drive=UsdPhysics.DriveAPI.Apply(prim,'angular')
            if j.get('mimic'):
                m=j['mimic']
                api=PhysxSchema.PhysxMimicJointAPI.Apply(prim,'rot'+str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()))
                api.CreateReferenceJointRel().SetTargets([joints[m['leader']].GetPath()])
                api.CreateReferenceJointAxisAttr('rot'+str(UsdPhysics.RevoluteJoint(joints[m['leader']]).GetAxisAttr().Get()))
                api.CreateGearingAttr(-m['multiplier']);api.CreateOffsetAttr(-np.rad2deg(m['offset']))
                drive.CreateStiffnessAttr(0.);drive.CreateDampingAttr(0.)
                self.mimic_specs.append(dict(follower=j['name'],**m))
            else:
                drive.CreateStiffnessAttr(config['joint_stiffness']);drive.CreateDampingAttr(config['joint_damping'])
        for asset in (hand_path,prefix+'/Can',prefix+'/Table'):
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(asset)).Bind(material,materialPurpose='physics')
            for prim in Usd.PrimRange(stage.GetPrimAtPath(asset)):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rb=PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                    if asset==hand_path: rb.CreateDisableGravityAttr(not config['hand_gravity'])
                    if asset.endswith('/Can'):
                        rb.CreateDisableGravityAttr(False);rb.CreateEnableGyroscopicForcesAttr(True)
                        rb.CreateSolverPositionIterationCountAttr(8);rb.CreateSolverVelocityIterationCountAttr(0)
                        rb.CreateSleepThresholdAttr(.005);rb.CreateStabilizationThresholdAttr(.0025)
                    if not asset.endswith('/Table'):rb.CreateMaxDepenetrationVelocityAttr(5.)
                if prim.HasAPI(UsdPhysics.CollisionAPI) and not asset.endswith('/Table'):
                    col=PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    col.CreateContactOffsetAttr(config['contact_offset_m']);col.CreateRestOffsetAttr(config['rest_offset_m'])
        cache=UsdGeom.XformCache()
        palm=next(p for p in Usd.PrimRange(stage.GetPrimAtPath(hand_path)) if p.GetName()==model.root)
        relative=cache.GetLocalToWorldTransform(palm)*cache.GetLocalToWorldTransform(stage.GetPrimAtPath(hand_path)).GetInverse()
        if not np.allclose(np.array(relative),np.eye(4),atol=1e-6):raise ValueError('Explicit root/palm transform adapter required')
        # Author the first pose once. Replication carries it to every env;
        # tensor reset later sets the actual per-env joint/reference state.
        from scipy.spatial.transform import Rotation
        for path,pose in ((hand_path,reference.wrist[0]),(prefix+'/Can',reference.object[0])):
            xform=UsdGeom.Xformable(stage.GetPrimAtPath(path));xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*(pose[:3,3]+self.surface.translation)))
            q=Rotation.from_matrix(pose[:3,:3]).as_quat()
            xform.AddOrientOp().Set(Gf.Quatf(float(q[3]),Gf.Vec3f(*q[:3])))
            xform.AddScaleOp().Set(Gf.Vec3f(1.))
        progress("clone USD and register physics replication")
        cloner.clone(source_prim_path=prefix,prim_paths=self.env_paths,
            positions=scene_origins,replicate_physics=num_envs>1,
            copy_from_source=False,enable_env_ids=True)
        progress("create articulation view")
        # Do not normalize/write thousands of USD poses again after cloning.
        # The source already has translate/orient/scale ops and world poses.
        cloner.disable_change_listener()
        try:
            self.robot=self.world.scene.add(Articulation('/World/envs/env_.*/Hand',name='residual_hands',reset_xform_properties=False))
            progress("create object and support views")
            self.can=self.world.scene.add(RigidPrim('/World/envs/env_.*/Can',name='dynamic_cans',reset_xform_properties=False,track_contact_forces=True))
            self.table=self.world.scene.add(RigidPrim('/World/envs/env_.*/Table',name='tables',reset_xform_properties=False))
        finally:
            cloner.enable_change_listener()
        progress("initialize PhysX (world.reset)")
        self.world.reset()
        progress("initialize randomized physics and controller")
        self.full_ids=torch.tensor([self.robot.dof_names.index(n) for n in model.full_names],device=self.device)
        self.active_ids=torch.tensor([self.robot.dof_names.index(n) for n in model.active_names],device=self.device)
        self.palm_id=self.robot.body_names.index(model.root)
        self.root_id=self.robot.body_names.index('Hand')
        self.kp_ids=torch.tensor([self.robot.body_names.index(k['link']) for k in model.keypoints],device=self.device)
        self.kp_local=torch.tensor([k['xyz'] for k in model.keypoints],device=self.device)
        self.tip_ids=torch.tensor([model.semantic_names.index(k+'_tip') for k in ('thumb','index','middle','ring','little')],device=self.device)
        self.robot.set_max_joint_velocities(self.task.full_velocity[None].expand(num_envs,-1),joint_indices=self.full_ids)
        apply_startup_randomization(self)
        self.refresh_physics_properties()
        self.obs_history=ObservationHistory(num_envs,6,config['observation'],self.device,self.generator)
        self.obs_history.default_q.copy_(self.default_q_offset)
        self.push=PushSchedule(num_envs,config['push_curriculum'],self.device,self.generator)
        self.metadata=dict(backend='Isaac Sim 6 PhysX '+str(self.device)+' / Torch device-resident control',num_envs=num_envs,
            physics_device=str(self.device),body_names=self.robot.body_names,dof_names=self.robot.dof_names,mimic_constraints=self.mimic_specs,
            self_collision=config['self_collision'],hand_gravity=config['hand_gravity'],object_kinematic=False,
            object_control='gravity/contact plus source velocity-push curriculum; pose only reset',root_control='PD root force + mass-proportional distributed torque; hand gravity '+('enabled' if config['hand_gravity'] else 'disabled'),
            physics_dt=config['physics_dt'],control_dt=self.task.dt,randomization=property_report(self),
            active_effort_limits_nm=self.robot.get_max_efforts()[0,self.active_ids].cpu().tolist(),
            actor_observation_size=67,critic_observation_size=88)
        self.metadata.update(controller='regrind_pose_pd',gravity_compensation=False,reference_velocity_feedforward=False,
                             stepping='one explicit physics step per PD update; rendering advances no physics')
        self.metadata.update(surface=self.surface.metadata(),
            state_coordinate_frame='unchanged task/reference frame; add task_origins_world_m to positions for simulator world',
            task_origins_world_m=self.origins.cpu().tolist())
        self.metadata['gpu_pair_capacities']=dict(
            found_lost=context.get_gpu_found_lost_pairs_capacity(),
            found_lost_aggregate=context.get_gpu_found_lost_aggregate_pairs_capacity(),
            total_aggregate=context.get_gpu_total_aggregate_pairs_capacity())
        self.reset(randomize=False)
        progress("environment ready")
        self.metadata.update(startup_timings_s=self.startup_timings,
            environment_creation="single USD source + physics replication",
            inter_environment_collision_filter="PhysX environment IDs",
            gpu_dynamics_enabled=bool(scene_api.GetEnableGPUDynamicsAttr().Get()),
            broadphase=scene_api.GetBroadphaseTypeAttr().Get(),
            scene_query_support=bool(scene_api.GetEnableSceneQuerySupportAttr().Get()),
            fabric_output_settings={key:output_settings.get('/physics/'+key)
                for key in (*fabric_output_keys,'fabricUseGPUInterop')})

    def refresh_physics_properties(self):
        self.body_com_local=self.robot.get_body_coms()[0].to(self.device).clone()
        self.can_com_local=self.can.get_coms()[0].to(self.device).reshape(self.num_envs,3).clone()
        self.body_mass=self.robot.get_body_masses().to(self.device).clone()
        self.total_mass=self.body_mass.sum(1)

    def _apply_gravity(self,magnitude):
        import carb
        self.gravity.value=float(magnitude)
        self.world.get_physics_context().set_gravity(-float(magnitude))
        self.world.physics_sim_view.set_gravity(carb.Float3(0,0,-float(magnitude)))

    def set_training(self,training):
        changed=self.training!=bool(training);self.training=bool(training)
        if changed or not training:self._apply_gravity(self.gravity.sample(self.control_steps,self.rng,self.training))
        # Startup DR remains fixed in both train/eval, matching source PLAY inheritance.

    def training_state_dict(self):
        return dict(schema=2,control_steps=self.control_steps,training=self.training,evaluation_protocol=self.evaluation_protocol,rng=copy.deepcopy(self.rng.bit_generator.state),
                    device_rng=self.generator.get_state().cpu(),sampler=self.sampler.state_dict(),gravity=self.gravity.state_dict(),
                    randomized_physics=self.randomized_physics,default_q_offset=self.default_q_offset.cpu())

    def load_training_state_dict(self,state):
        if state['schema']!=2:raise ValueError('Incompatible curriculum checkpoint')
        self.control_steps=int(state['control_steps']);self.training=bool(state['training'])
        self.evaluation_protocol=state.get('evaluation_protocol','strict')
        self.rng.bit_generator.state=copy.deepcopy(state['rng']);self.generator.set_state(state['device_rng'].cpu())
        self.sampler.load_state_dict(state['sampler']);self.gravity.load_state_dict(state['gravity']);self._apply_gravity(self.gravity.value)
        # Environment count may differ for evaluation: repeat the saved per-env startup distribution.
        self.randomized_physics={asset:{k:v[torch.arange(self.num_envs)%len(v)].clone() for k,v in vals.items()} for asset,vals in state['randomized_physics'].items()}
        set_physics_properties(self,self.randomized_physics)
        offsets=state['default_q_offset'];self.default_q_offset.copy_(offsets[torch.arange(self.num_envs)%len(offsets)].to(self.device))
        self.obs_history.default_q.copy_(self.default_q_offset)

    def reset(self,indices=None,randomize=True,times=None):
        ids=self.all_ids if indices is None else torch.as_tensor(indices,device=self.device,dtype=torch.long)
        if not len(ids):return
        learning=self.training and randomize
        if times is None:times=self.sampler.sample(len(ids),self.generator) if learning else torch.zeros(len(ids),device=self.device)
        times=torch.as_tensor(times,device=self.device,dtype=torch.float32)
        if times.shape!=(len(ids),) or not torch.isfinite(times).all() or (times<0).any() or (times>self.reference.duration).any():raise ValueError('Invalid reset times')
        if self.training:self._apply_gravity(self.gravity.sample(self.control_steps,self.rng,True))
        self.motion.reset(ids,self.generator,randomize and (self.training or self.evaluation_protocol=='source'))
        ref=self.motion.sample(times,ids)
        q=ref['q'].clone()
        poses={k:ref[k].clone() for k in ('wrist_position','wrist_quaternion','object_position','object_quaternion')}
        if learning and self.cfg['reset_perturbation']:
            noise=self.cfg['rsi']
            q=(q+uniform(q.shape,-noise['joint_noise_rad'],noise['joint_noise_rad'],self.device,self.generator)).clamp(self.task.lower,self.task.upper)
            for name in ('wrist','object'):
                poses[name+'_position']+=uniform((len(ids),3),-noise['position_noise_m'],noise['position_noise_m'],self.device,self.generator)
                axis=torch.randn((len(ids),3),device=self.device,generator=self.generator)
                axis/=axis.norm(dim=-1,keepdim=True).clamp_min(1e-12)
                rv=axis*uniform((len(ids),1),-noise['rotation_noise_rad'],noise['rotation_noise_rad'],self.device,self.generator)
                poses[name+'_quaternion']=quat_multiply(from_rotvec(rv),poses[name+'_quaternion'])
        self.time[ids]=times;self.episode_length_buf[ids]=0;self.last_action[ids]=0;self.last_q_target[ids]=q
        moving=(times>1e-8)[:,None].float() if self.cfg['rsi']['initialize_velocity'] else torch.zeros((len(ids),1),device=self.device)
        self.robot.set_world_poses(poses['wrist_position']+self.origins[ids],poses['wrist_quaternion'][:,[3,0,1,2]],indices=ids)
        com=quat_apply(poses['wrist_quaternion'],self.body_com_local[ids,self.root_id])
        root_v=ref['wrist_velocity']+torch.cross(ref['wrist_angular_velocity'],com,dim=-1)
        self.robot.set_velocities(torch.cat((root_v,ref['wrist_angular_velocity']),-1)*moving,indices=ids)
        full=q@self.task.coupling.T+self.task.offset
        self.robot.set_joint_positions(full,indices=ids,joint_indices=self.full_ids)
        self.robot.set_joint_velocities((ref['q_velocity']*moving)@self.task.coupling.T,indices=ids,joint_indices=self.full_ids)
        self.robot.set_joint_position_targets(full,indices=ids,joint_indices=self.full_ids)
        self.can.set_world_poses(poses['object_position']+self.origins[ids],poses['object_quaternion'][:,[3,0,1,2]],indices=ids)
        com=quat_apply(poses['object_quaternion'],self.can_com_local[ids])
        can_v=ref['object_velocity']+torch.cross(ref['object_angular_velocity'],com,dim=-1)
        self.can.set_velocities(torch.cat((can_v,ref['object_angular_velocity']),-1)*moving,indices=ids)
        self.object_reset_count+=len(ids)
        self.world.physics_sim_view.update_articulations_kinematic()
        self.push.reset(ids)
        state=self.state()
        self.obs_history.update(state,ids,reset=True,noisy=self.training or self.evaluation_protocol=='source')
        self.obs_history.assemble(state,self.motion.sample(self.time),self.last_action,self.phase(),ids)

    def phase(self):return (self.time/self.task.dt)/(int(round(self.reference.duration/self.task.dt))+1)

    def state(self):
        links=self.robot._physics_view.get_link_transforms().clone()
        velocities=self.robot._physics_view.get_link_velocities().clone()
        offsets=quat_apply(links[:,:,3:7],self.body_com_local)
        velocities[:,:,:3]-=torch.cross(velocities[:,:,3:],offsets,dim=-1)
        q,qv=self.robot.get_joint_positions(),self.robot.get_joint_velocities()
        cp,cqw=self.can.get_world_poses();cq=cqw[:,[1,2,3,0]]
        cv=self.can.get_velocities().clone();cv[:,:3]-=torch.cross(cv[:,3:],quat_apply(cq,self.can_com_local),dim=-1)
        palm,pv=links[:,self.palm_id],velocities[:,self.palm_id]
        kp=quat_apply(links[:,self.kp_ids,3:7],self.kp_local[None].expand(self.num_envs,-1,-1))+links[:,self.kp_ids,:3]-self.origins[:,None]
        task_links=links.clone();task_links[:,:,:3]-=self.origins[:,None]
        return dict(q=q[:,self.active_ids],full_q=q[:,self.full_ids],q_velocity=qv[:,self.active_ids],full_q_velocity=qv[:,self.full_ids],
            wrist_position=palm[:,:3]-self.origins,wrist_quaternion=palm[:,3:7],wrist_velocity=pv[:,:3],wrist_angular_velocity=pv[:,3:],
            object_position=cp-self.origins,object_quaternion=cq,object_velocity=cv[:,:3],object_angular_velocity=cv[:,3:],
            link_transforms=task_links,link_velocities=velocities,robot_keypoints=kp,fingertips=kp[:,self.tip_ids])

    def observation(self):return self.obs_history.get()

    def step(self,actions,auto_reset=True):
        started=time.monotonic()
        actions=torch.as_tensor(actions,device=self.device,dtype=torch.float32)
        if actions.shape!=(self.num_envs,12) or not torch.isfinite(actions).all():raise ValueError('Invalid residual actions')
        # Current command drives physics and reward; command advances afterwards as in ManagerBasedRLEnv.
        ref=self.motion.sample(self.time)
        target=self.task.targets(ref,actions,self.last_q_target,self.default_q_offset)
        self.last_q_target.copy_(target['active_q'])
        self.robot.set_joint_position_targets(target['full_q'],joint_indices=self.full_ids)
        for substep in range(self.cfg['control_decimation']):
            s=self.state()
            all_forces,all_torques=wrist_pd_wrenches(s,target,self.body_mass,self.root_id,self.cfg)
            self.robot._physics_view.apply_forces_and_torques_at_position(all_forces,all_torques,None,self.all_ids.to(torch.int32),True)
            # Legacy World.step(render=True) advances an app/render interval,
            # which can contain multiple physics steps with a stale wrench.
            # As in Isaac Lab's RL loop, step physics explicitly and render alone.
            self.world.step(render=False)
        if self.render:self.world.render()
        if self.render:time.sleep(max(0,self.task.dt-(time.monotonic()-started)))
        self.episode_length_buf+=1
        state=self.state()
        demo_end=self.time>=self.reference.duration-1e-6
        timed_out=self.episode_length_buf>=self.max_episode_length
        reward,term,trunc,metrics=self.task.score(state,ref,actions,self.last_action,demo_end,timed_out)
        metrics['gravity_m_s2']=torch.full((self.num_envs,),self.gravity.value,device=self.device)
        if self.training:
            self.control_steps+=1
            self.sampler.record(self.time,term,demo_end,self.num_envs,self.max_episode_length)
        self.last_action.copy_(actions)
        self.obs_history.update(state,self.all_ids,noisy=self.training or self.evaluation_protocol=='source')
        next_time=(self.time+self.task.dt).clamp(max=self.reference.duration)
        next_phase=(next_time/self.task.dt)/self.max_episode_length
        # Timeout value must see the next command too, exactly as a continuing
        # policy observation would, before any auto-reset overwrites this state.
        final_obs=self.obs_history.assemble(state,self.motion.sample(next_time),self.last_action,next_phase)
        info=dict(metrics=metrics,final_observation=final_obs,state=state,reference=ref,applied_targets=target,object_reset_count=self.object_reset_count,reference_time=self.time.clone())
        self.time=next_time
        done=torch.nonzero(term|trunc,as_tuple=False).flatten()
        if auto_reset and len(done):self.reset(done,randomize=True)
        self.push.advance(self)
        # Assemble new command without pushing the history or drawing noise a second time.
        self.obs_history.assemble(self.state() if auto_reset and len(done) else state,self.motion.sample(self.time),self.last_action,self.phase())
        return self.observation(),reward,term,trunc,info

    def close(self):self.world.stop()
