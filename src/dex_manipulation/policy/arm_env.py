"""Inference-only RB3+Revo2 physics adapter for a floating-hand checkpoint.

Policy observations stay in the training/source frame. Only wrist commands cross
the explicit workcell transform into strict RB3 IK. The can is a dynamic body.
"""
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ..control import MotionController
from ..coordinates import collision_bottom
from ..fk import ArmModel
from ..ik import IKOptions, pose_error
from ..scene import Workcell
from ..transforms import transform
from .arm import PolicyArmBridge, PolicyFrame, named_material_transfer
from .env import PhysxResidualEnv
from .observations import ObservationHistory
from .reference import TensorReference
from .task import ResidualTask
from .trajectory import digest
from .math3d import rotation_error, quat_apply


class ArmPolicyEnv(PhysxResidualEnv):
    """Single physical arm, closed-loop policy inference, never PPO training."""
    def __init__(self, root, model, reference, config, arm_config, render=False):
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, UsdLux, Gf
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation, RigidPrim
        from isaacsim.core.utils.stage import add_reference_to_stage

        if config.get('motion_control', {}).get('enabled', False):
            raise ValueError('Arm policy play requires a checkpoint trained without the optional motion governor')
        root = Path(root)
        self.model, self.reference, self.cfg = model, reference, config
        self.arm_control = dict(velocity_feedforward=True, restore_hand_materials=True,
                                wrist_response='training_pd',contact_feedback_gain=.25,contact_filter_alpha=.5)
        self.arm_control.update(arm_config.get('policy_control', {}))
        if set(self.arm_control)!={'velocity_feedforward','restore_hand_materials','wrist_response','contact_feedback_gain','contact_filter_alpha'}:
            raise ValueError('Unknown arm policy-control setting')
        if any(not isinstance(self.arm_control[name],bool) for name in ('velocity_feedforward','restore_hand_materials')):
            raise ValueError('Arm feedforward and material restoration settings must be booleans')
        if self.arm_control['wrist_response'] not in ('training_pd','direct'):
            raise ValueError('Unknown arm policy wrist-response mode')
        for name in ('contact_feedback_gain','contact_filter_alpha'):
            value=self.arm_control[name]
            if isinstance(value,bool) or not np.isfinite(value) or not 0<=value<=1:
                raise ValueError(f'Arm {name} must lie in [0,1]')
        self.wrist_response=None
        self.hand_contacts=None
        self.contact_wrench=np.zeros(6)
        self.hand_asset=root/config['hand_asset']
        self.arm = ArmModel.load(root / arm_config['arm_model'])
        if self.arm.wrist != model.root:
            raise ValueError('Arm mount wrist and hand root differ')
        if digest(root / arm_config['usd']) != self.arm.description['source_sha256']:
            raise ValueError('Arm USD changed; extract the arm model again')
        for path, expected in self.arm.description['layer_sha256'].items():
            if not Path(path).is_file() or digest(path) != expected:
                raise ValueError(f'Arm USD layer changed/missing: {path}')
        if digest(root / config['object_asset']) != reference.metadata['object_asset_sha256']:
            raise ValueError('Object asset differs from the checkpoint reference')
        self.workcell = Workcell.load(root / arm_config['workcell'])
        alignment = self.workcell.resolve_alignment(json.loads((root / arm_config['alignment']).read_text()))
        self.world_from_source = np.asarray(alignment['world_from_source'])
        self.device = torch.device(config['physics_device'])
        self.frame = PolicyFrame(self.world_from_source, self.device)
        self.bridge = PolicyArmBridge(self.arm, alignment['base_from_source'], arm_config['seed_q_rad'],
                                      IKOptions(**arm_config['solver']))
        self.num_envs, self.render, self.training = 1, render, False
        self.evaluation_protocol = 'strict'
        self.all_ids = torch.arange(1, device=self.device, dtype=torch.long)
        self.origins = torch.zeros((1, 3), device=self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(config['seed'])
        self.motion = TensorReference(reference, 1, config['augmentation'], self.device,
                                      velocity_mode=config.get('reference_velocity_mode', 'segment'),
                                      control_dt=config['physics_dt'] * config['control_decimation'])
        self.task = ResidualTask(model, self.motion, config, self.device)
        self.motion_controller = MotionController(model, 1, config['physics_dt'], config.get('motion_control'), self.device)
        self.gravity = SimpleNamespace(value=float(config['gravity']))
        self.time = torch.zeros(1, device=self.device)
        self.reference_frame_count = int(round(reference.duration / self.task.dt)) + 1
        self.max_episode_length = (int(np.ceil(config['episode_length_s'] / self.task.dt))
                                   if 'reference_timing' in config else self.reference_frame_count)
        self.episode_length_buf = torch.zeros(1, device=self.device, dtype=torch.long)
        self.last_action = torch.zeros((1, 12), device=self.device)
        self.last_q_target = torch.zeros((1, 6), device=self.device)
        self.default_q_offset = torch.zeros((1, 6), device=self.device)
        self.object_reset_count = 0
        self.control_steps = 0
        self.world = World(stage_units_in_meters=1., physics_dt=config['physics_dt'],
                           rendering_dt=self.task.dt, backend='torch', device=str(self.device))
        context = self.world.get_physics_context()
        context.set_gravity(-self.gravity.value)
        context.set_solver_type('TGS')
        stage = self.world.stage
        physics = config.get('solver_iterations', {})
        scene_api = PhysxSchema.PhysxSceneAPI.Apply(context.get_current_physics_scene_prim())
        scene_api.CreateMaxPositionIterationCountAttr(physics.get('scene_position', 64))
        scene_api.CreateMaxVelocityIterationCountAttr(physics.get('scene_velocity', 4))
        scene_api.CreateBounceThresholdAttr(.2)
        scene_api.CreateFrictionOffsetThresholdAttr(.01)
        scene_api.CreateFrictionCorrelationDistanceAttr(.00625)
        for name,value in config.get('scene_physics',{}).items():
            attribute=scene_api.GetPrim().GetAttribute('physxScene:'+name)
            if not attribute or not attribute.Set(value):
                raise ValueError(f'Unsupported PhysX scene option: {name}')
        from .arm_scene import build_arm_scene
        scene=build_arm_scene(stage,root,model,reference,config,arm_config,self.arm,self.workcell,self.world_from_source)
        articulation_resolution=scene['resolution']
        self.initial_can_pose=scene['initial_can_pose']
        hand_links={model.root}|{j['child'] for j in model.joints}
        UsdLux.DomeLight.Define(stage,'/Light').CreateIntensityAttr(900)
        self.robot=self.world.scene.add(Articulation(scene['robot'],name='policy_arm',reset_xform_properties=False))
        self.can=self.world.scene.add(RigidPrim(scene['can'],name='policy_can',reset_xform_properties=False))
        self.table=self.world.scene.add(RigidPrim(scene['table'],name='policy_table',reset_xform_properties=False))
        self.world.reset()
        if self.robot.count != 1 or not set(self.arm.active_names + model.full_names).issubset(self.robot.dof_names):
            raise RuntimeError(f'PhysX did not form one combined arm/hand articulation: count={self.robot.count}, joints={self.robot.dof_names}')
        materials = self.robot._physics_view.get_material_properties().detach().cpu().clone()
        materials[..., :2] = config['friction']; materials[..., 2] = 0.
        self.robot._physics_view.set_material_properties(materials.contiguous(), torch.zeros(1, dtype=torch.int32))
        self.full_ids = torch.tensor([self.robot.dof_names.index(n) for n in model.full_names], device=self.device)
        self.active_ids = torch.tensor([self.robot.dof_names.index(n) for n in model.active_names], device=self.device)
        self.arm_ids = torch.tensor([self.robot.dof_names.index(n) for n in self.arm.active_names], device=self.device)
        self.body_shape_counts = [self.world.physics_sim_view.create_rigid_body_view(path).max_shapes
                                  for path in self.robot._physics_view.link_paths[0]]
        if sum(self.body_shape_counts) != self.robot._physics_view.max_shapes:
            raise RuntimeError('Could not verify the assembled collider material layout')
        self.base_id = self.robot.body_names.index(self.arm.root)
        self.palm_id = self.robot.body_names.index(model.root)
        self.kp_ids = torch.tensor([self.robot.body_names.index(k['link']) for k in model.keypoints], device=self.device)
        self.kp_local = torch.tensor([k['xyz'] for k in model.keypoints], device=self.device)
        self.tip_ids = torch.tensor([model.semantic_names.index(k + '_tip') for k in ('thumb', 'index', 'middle', 'ring', 'little')], device=self.device)
        self.tip_body_ids = self.kp_ids[self.tip_ids]
        self.configure_table_safety()
        self.robot.set_max_joint_velocities(torch.tensor(self.arm.velocity[None], device=self.device, dtype=torch.float32), joint_indices=self.arm_ids)
        self.robot.set_max_joint_velocities(torch.full((1, len(model.full_names)), config['simulation_joint_velocity_rad_s'], device=self.device), joint_indices=self.full_ids)
        self.refresh_physics_properties()
        if self.arm_control['wrist_response']=='training_pd' and self.arm_control['contact_feedback_gain']>0:
            paths=[path for name,path in zip(self.robot.body_names,self.robot._physics_view.link_paths[0])
                   if name in hand_links]
            self.hand_contacts=RigidPrim(paths,name='policy_hand_contacts',reset_xform_properties=False,
                track_contact_forces=True,contact_filter_prim_paths_expr=[[] for _ in paths],disable_stablization=False)
            self.hand_contacts.initialize(self.world.physics_sim_view)
        self.obs_history = ObservationHistory(1, 6, config['observation'], self.device, self.generator)
        self.metadata = dict(robot='RB3-730 + Revo2', num_envs=1, workcell=self.workcell.metadata(),
                             articulation_resolution=articulation_resolution,
                             arm_model_sha256=digest(root / arm_config['arm_model']),
                             assembled_usd_sha256=self.arm.description['source_sha256'],
                             arm_joint_names=self.arm.active_names, finger_joint_names=model.active_names,
                             full_finger_joint_names=model.full_names,
                             world_from_source=self.world_from_source.tolist(), base_from_source=alignment['base_from_source'],
                             body_names=self.robot.body_names, dof_names=self.robot.dof_names,
                             body_shape_counts=self.body_shape_counts,
                             arm_control=self.arm_control,
                             scene_physics={name:scene_api.GetPrim().GetAttribute('physxScene:'+name).Get()
                                            for name in config.get('scene_physics',{})},
                             self_collision=bool(self.robot.get_enabled_self_collisions()[0]),
                             hand_gravity=config['hand_gravity'], arm_gravity=True, object_kinematic=False,
                             physics_dt=config['physics_dt'], control_dt=self.task.dt,
                             root_control='none; fixed-base articulation receives arm joint targets from strict IK',
                             arm_drives='USD gains/effort limits; bounded IK position and velocity targets',
                             friction='named checkpoint hand materials, can and tabletop materials; nominal arm friction',
                             support_physics='fixed kinematic table; source contact/solver properties, workcell size and placement',
                             training_environment_changed=True,
                             ik_failure='hold previous joint target, record failure, terminate episode',
                             collision_validation='PhysX collision response enabled; no collision-free trajectory certification')
        if self.metadata['self_collision'] != config['self_collision']:
            raise RuntimeError('Requested articulation self-collision was not applied')
        if render:
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(eye=np.array([1.65, -1.9, 1.25]), target=np.array([.35, 0, .05]))
        self.reset(randomize=False)

    def set_training(self, training):
        if training:
            raise ValueError('The arm adapter supports checkpoint inference only')
        self.training = False
        self.world.get_physics_context().set_gravity(-self.cfg['gravity'])
        self.gravity.value = self.cfg['gravity']

    def load_training_state_dict(self, state):
        """Restore compatible hand/can properties by name; never map arm by index."""
        if state.get('schema') != 2:
            raise ValueError('Unknown saved physical-state schema')
        saved = state['randomized_physics']
        names = self.checkpoint_physics_names
        indices = torch.zeros(1, dtype=torch.int32)
        view = self.robot._physics_view
        for key, getter, setter, source_names, target_names in (
                ('masses', 'get_masses', 'set_masses', names['body_names'], self.robot.body_names),
                ('inertias', 'get_inertias', 'set_inertias', names['body_names'], self.robot.body_names),
                ('stiffness', 'get_dof_stiffnesses', 'set_dof_stiffnesses', names['dof_names'], self.robot.dof_names),
                ('damping', 'get_dof_dampings', 'set_dof_dampings', names['dof_names'], self.robot.dof_names)):
            values = getattr(view, getter)().detach().cpu().clone()
            source = saved['hand'][key][0].detach().cpu()
            for i, name in enumerate(source_names):
                if name == 'Hand':
                    continue  # Floating-only carrier body does not exist on the arm.
                if name not in target_names:
                    raise ValueError(f'Saved hand body/joint is missing in assembly: {name}')
                values[0, target_names.index(name)] = source[i]
            getattr(view, setter)(values.contiguous(), indices)
        if self.arm_control['restore_hand_materials']:
            # Each extracted collider must correspond to the same runtime
            # shape count on the shared hand asset; never assume a +7 offset.
            from collections import Counter
            counts=Counter(collider['link'] for collider in self.model.colliders)
            source_counts=names.get('body_shape_counts',[counts[name] for name in names['body_names']])
            materials=named_material_transfer(names['body_names'],source_counts,
                saved['hand']['materials'][0].detach().cpu().numpy(),
                self.robot.body_names,self.body_shape_counts,
                view.get_material_properties()[0].detach().cpu().numpy())
            view.set_material_properties(torch.as_tensor(materials[None]).contiguous(),indices)
        methods = dict(materials='set_material_properties', masses='set_masses', inertias='set_inertias', coms='set_coms')
        for key, values in saved['object'].items():
            getattr(self.can._physics_view, methods[key])(values[:1].detach().cpu().contiguous(), indices)
        self.table._physics_view.set_material_properties(saved['table']['materials'][:1].detach().cpu().contiguous(),indices)
        self.default_q_offset.copy_(state['default_q_offset'][:1].to(self.device))
        self.obs_history.default_q.copy_(self.default_q_offset)
        self.refresh_physics_properties()
        self.metadata['restored_physics'] = 'first supplied checkpoint environment: named hand masses/inertias/drives/default offsets, verified collider materials, can properties and tabletop material'
        if self.arm_control.get('wrist_response','direct')=='training_pd':
            from pxr import Usd,UsdGeom,UsdPhysics
            from .arm_control import WristResponse
            # The absent floating carrier is a real authored inertial body in
            # the training asset. Read its frame/COM; never invent its mass.
            stage=Usd.Stage.Open(str(self.hand_asset))
            root=stage.GetDefaultPrim()
            palm=next(p for p in Usd.PrimRange(root) if p.GetName()==self.model.root)
            cache=UsdGeom.XformCache()
            relative=cache.GetLocalToWorldTransform(palm)*cache.GetLocalToWorldTransform(root).GetInverse()
            if not np.allclose(np.asarray(relative),np.eye(4),atol=1e-6,rtol=0):
                raise ValueError('Training carrier and palm frames are not coincident')
            api=UsdPhysics.MassAPI(root)
            center=api.GetCenterOfMassAttr().Get();axes=api.GetPrincipalAxesAttr().Get()
            if center is None or axes is None or not np.isfinite(np.asarray(center)).all():
                raise ValueError('Authored training carrier COM/principal axes are required')
            carrier_com=np.r_[np.asarray(center),np.asarray(axes.GetImaginary()),axes.GetReal()]
            actual_coms=view.get_coms()[0].detach().cpu().numpy()
            source_coms=[]
            for name in names['body_names']:
                source_coms.append(carrier_com if name=='Hand' else actual_coms[self.robot.body_names.index(name)])
            self.wrist_response=WristResponse(self.model,names['body_names'],
                saved['hand']['masses'][0].detach().cpu().numpy(),saved['hand']['inertias'][0].detach().cpu().numpy(),
                source_coms,self.cfg,self.cfg['physics_dt'])
            self.metadata['wrist_response']=dict(mode='training_pd',body_names=names['body_names'],
                total_mass_kg=float(self.wrist_response.mass.sum()),carrier_com=carrier_com.tolist(),
                physics_dt=self.cfg['physics_dt'],hand_asset_sha256=digest(self.hand_asset),
                approximation='Rigid composite at measured finger pose; filtered measured contact forces, COM approximation of contact moment; internal finger momentum not predicted',
                command_semantics='Actor pose target -> training PD response -> strict arm IK -> position/velocity drive targets')

    def reset(self, indices=None, randomize=False, times=None):
        if times is not None or self.evaluation_protocol != 'strict':
            raise ValueError('Arm policy playback starts at frame zero with strict observations')
        self.motion.reset(self.all_ids, self.generator, False)
        self.contact_wrench[:]=0
        self.time.zero_(); self.last_action.zero_(); self.episode_length_buf.zero_()
        ref = self.motion.sample(self.time)
        initial = self.bridge.reset(ref['wrist_position'][0].cpu().numpy(), ref['wrist_quaternion'][0].cpu().numpy())
        qa = torch.tensor(initial.q[None], device=self.device, dtype=torch.float32)
        q = ref['q'].clone()
        full = q @ self.task.coupling.T + self.task.offset
        for ids, values in ((self.arm_ids, qa), (self.full_ids, full)):
            self.robot.set_joint_positions(values, joint_indices=ids)
            self.robot.set_joint_velocities(torch.zeros_like(values), joint_indices=ids)
            self.robot.set_joint_position_targets(values, joint_indices=ids)
        self.robot.set_joint_velocity_targets(torch.zeros_like(qa),joint_indices=self.arm_ids)
        self.last_q_target.copy_(q)
        self.last_full_target = full.clone()
        p = torch.tensor(self.initial_can_pose[None, :3, 3], device=self.device, dtype=torch.float32)
        qw = Rotation.from_matrix(self.initial_can_pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
        self.can.set_world_poses(p, torch.tensor(qw[None], device=self.device, dtype=torch.float32))
        self.can.set_velocities(torch.zeros((1, 6), device=self.device))
        self.object_reset_count += 1
        self.world.physics_sim_view.update_articulations_kinematic()
        state = self.state()
        if self.wrist_response is not None:
            self.wrist_response.reset(state['wrist_position'][0].cpu().numpy(),state['wrist_quaternion'][0].cpu().numpy())
        self.motion_controller.reset(self.all_ids, state)
        self.obs_history.update(state, self.all_ids, reset=True, noisy=False)
        self.obs_history.assemble(state, ref, self.last_action, self.phase())

    def state(self):
        world = super().state()
        base = world['link_transforms'][0, self.base_id].detach().cpu().numpy()
        base_pose = transform(Rotation.from_quat(base[3:]).as_matrix(), base[:3])
        if not np.allclose(base_pose, self.workcell.world_from_base, atol=1e-5, rtol=0):
            raise RuntimeError('RB3 base moved relative to its fixed workcell mount')
        out = self.frame.state(world)
        out['q_arm'] = self.robot.get_joint_positions()[:, self.arm_ids].clone()
        out['q_arm_velocity'] = self.robot.get_joint_velocities()[:, self.arm_ids].clone()
        return out

    def step(self, actions, auto_reset=True):
        started = time.monotonic()
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if actions.shape != (1, 12) or not torch.isfinite(actions).all():
            raise ValueError('Invalid residual policy output')
        ref = self.motion.sample(self.time)
        target = self.task.targets(ref, actions, self.last_q_target, self.default_q_offset)
        table=getattr(self,'table_safety',None)
        protected=target
        if table is not None:
            table.begin(target,self.state())
            protected=table.guard(target,self.state())
        position=protected['position'][0].cpu().numpy();quaternion=protected['quaternion'][0].cpu().numpy()
        response=getattr(self,'wrist_response',None)
        if response is not None:
            snapshot=response.snapshot()
            contact=np.zeros(6)
            if self.hand_contacts is not None and float(self.time[0])>0:
                forces=self.frame.rotate(self.hand_contacts.get_net_contact_forces(dt=self.cfg['physics_dt'])).cpu().numpy()
                wp,wq=self.hand_contacts.get_world_poses();cp,_=self.hand_contacts.get_coms()
                centers=self.frame.position(wp+quat_apply(wq[:,[1,2,3,0]],cp.reshape(-1,3).to(wp.device))).cpu().numpy()
                measured=np.r_[forces.sum(0),np.cross(centers-response.position,forces).sum(0)]
                alpha=self.arm_control['contact_filter_alpha']
                self.contact_wrench=alpha*measured+(1-alpha)*self.contact_wrench
                contact=self.contact_wrench*self.arm_control['contact_feedback_gain']
            position,quaternion=response.step(position,quaternion,
                self.robot.get_joint_positions()[0,self.active_ids].cpu().numpy(),self.cfg['control_decimation'],contact[:3],contact[3:])
        if table is not None:
            response_target=dict(target,position=torch.as_tensor(position[None],device=self.device,dtype=torch.float32),
                                 quaternion=torch.as_tensor(quaternion[None],device=self.device,dtype=torch.float32))
            safe_response=table.guard(response_target,self.state())
            corrected=safe_response['position'][0].cpu().numpy()
            if response is not None and corrected[2]>position[2]+1e-7:
                response.position=corrected.copy()
                response.velocity[2]=max(0.,response.velocity[2])
            position=corrected
        ik = self.bridge.solve(position,quaternion,self.task.dt)
        if response is not None and not ik['success']:response.restore(snapshot)
        qa = torch.tensor(ik['q_arm'][None], device=self.device, dtype=torch.float32)
        if ik['success']:
            self.last_q_target.copy_(target['active_q']); self.last_full_target.copy_(target['full_q'])
        self.robot.set_joint_position_targets(qa, joint_indices=self.arm_ids)
        qva=torch.tensor(ik['q_arm_velocity'][None],device=self.device,dtype=torch.float32)
        self.robot.set_joint_velocity_targets(qva if self.arm_control['velocity_feedforward'] else torch.zeros_like(qva),
                                             joint_indices=self.arm_ids)
        self.robot.set_joint_position_targets(self.last_full_target, joint_indices=self.full_ids)
        # Never apply the floating root force/torque to a fixed-base robot arm.
        for _ in range(self.cfg['control_decimation']):
            self.world.step(render=False)
            if table is not None: table.observe(self.state())
        self.episode_length_buf += 1
        state = self.state()
        if not all(torch.isfinite(v).all() for v in state.values()):
            raise RuntimeError('Nonfinite assembled-robot physics state')
        # Strict arm replay always starts at zero. Count commands rather than
        # accumulating float32 dt, which otherwise repeats the final command.
        demo_end = self.episode_length_buf >= self.reference_frame_count
        reward, term, trunc, metrics = self.task.score(state, ref, actions, self.last_action, demo_end,
                                                       self.episode_length_buf >= self.max_episode_length,
                                                       table_metrics=table.metrics() if table is not None else None)
        result = ik['result']
        scalar = lambda v: torch.tensor([v], device=self.device, dtype=torch.float32)
        measured_arm = state['q_arm'][0].cpu().numpy()
        actual_error = pose_error(ik['wrist_target'], self.arm.pose(measured_arm))
        measured_pose=self.bridge.target(state['wrist_position'][0].cpu().numpy(),
                                         state['wrist_quaternion'][0].cpu().numpy())
        consistency_error=pose_error(measured_pose,self.arm.pose(measured_arm))
        actual_sigma, actual_condition = self.bridge.solver.singularity(measured_arm)
        metrics.update(gravity_m_s2=scalar(self.gravity.value), arm_ik_success=scalar(ik['success']),
                       arm_ik_position_error_m=scalar(result.position_error_m),
                       arm_ik_orientation_error_rad=scalar(result.orientation_error_rad),
                       arm_sigma_min=scalar(result.sigma_min), arm_condition=scalar(result.condition),
                       arm_joint_step_rad=scalar(ik['joint_step_rad'].max()),
                       arm_joint_limit_distance_rad=scalar(result.joint_limit_distance_rad),
                       arm_actual_sigma_min=scalar(actual_sigma), arm_actual_condition=scalar(actual_condition),
                       arm_joint_limit_violation_rad=scalar(max(0., np.max(self.arm.lower-measured_arm), np.max(measured_arm-self.arm.upper))),
                       arm_joint_tracking_error_rad=scalar(np.max(np.abs(measured_arm-ik['q_arm']))),
                       arm_joint_velocity_violation_rad_s=(state['q_arm_velocity'].abs() - torch.tensor(self.arm.velocity, device=self.device)).clamp_min(0).amax(-1),
                       arm_tracking_position_error_m=scalar(np.linalg.norm(actual_error[:3])),
                       arm_tracking_orientation_error_rad=scalar(np.linalg.norm(actual_error[3:])),
                       policy_wrist_position_error_m=(target['position']-state['wrist_position']).norm(dim=-1),
                       policy_wrist_orientation_error_rad=rotation_error(target['quaternion'],state['wrist_quaternion']).norm(dim=-1),
                       arm_fk_consistency_position_error_m=scalar(np.linalg.norm(consistency_error[:3])),
                       arm_fk_consistency_orientation_error_rad=scalar(np.linalg.norm(consistency_error[3:])),
                       filtered_contact_force_n=scalar(np.linalg.norm(getattr(self,'contact_wrench',np.zeros(6))[:3])),
                       filtered_contact_moment_nm=scalar(np.linalg.norm(getattr(self,'contact_wrench',np.zeros(6))[3:])))
        term = term | torch.tensor([not ik['success']], device=self.device)
        if not ik['success']:
            print('[arm IK failure]', json.dumps(dict(time_s=float(self.time[0]), reason=ik['reason'],
                  position_error_m=result.position_error_m, orientation_error_rad=result.orientation_error_rad)), flush=True)
        applied = dict(target, active_q=self.last_q_target.clone(), full_q=self.last_full_target.clone(), q_arm=qa,
                       q_arm_velocity=qva if self.arm_control['velocity_feedforward'] else torch.zeros_like(qva))
        applied_pose = self.frame.source_from_world @ self.workcell.world_from_base @ self.arm.pose(ik['q_arm'])
        applied['position'] = torch.tensor(applied_pose[None, :3, 3], device=self.device, dtype=torch.float32)
        applied['quaternion'] = torch.tensor(Rotation.from_matrix(applied_pose[:3, :3]).as_quat()[None], device=self.device, dtype=torch.float32)
        self.last_action.copy_(actions)
        self.obs_history.update(state, self.all_ids, noisy=False)
        next_time = (self.time + self.task.dt).clamp(max=self.reference.duration)
        final_obs = self.obs_history.assemble(state, self.motion.sample(next_time), self.last_action, self.phase_at(next_time))
        info = dict(state=state, reference=ref, raw_targets=target, applied_targets=applied, metrics=metrics,
                    final_observation=final_obs, object_reset_count=self.object_reset_count, reference_time=self.time.clone())
        self.time = next_time
        if auto_reset and bool((term | trunc)[0]):
            self.reset(randomize=False)
        if self.render:
            self.world.render()
            time.sleep(max(0., self.task.dt - (time.monotonic() - started)))
        return self.observation(), reward, term, trunc, info
