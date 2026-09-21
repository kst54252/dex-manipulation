"""Common assembled robot/workcell USD authoring for play and training.

Original assets are read-only; runtime drives/materials are authored on the stage.
"""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from ..coordinates import collision_bottom


def build_arm_scene(stage,root,model,reference,config,arm_config,arm,workcell,world_from_source,*,prefix=''):
    from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,UsdShade,Gf
    from isaacsim.core.utils.stage import add_reference_to_stage
    robot_path,workcell_path,can_path=(prefix+'/Robot',prefix+'/Workcell',prefix+'/DynamicCan')
    physics=config.get('solver_iterations',{})
    def prims():
        return Usd.PrimRange(stage.GetPrimAtPath(prefix)) if prefix else stage.Traverse()
    add_reference_to_stage(str(root / arm_config['usd']), robot_path)
    source_stage = Usd.Stage.Open(str(root / arm_config['usd']))
    source_root = str(source_stage.GetDefaultPrim().GetPath())
    base_path = robot_path + arm_config['base_path'][len(source_root):]
    workcell.mount_robot(stage, robot_path, base_path)
    workcell_prims=workcell.create_usd(stage, prefix=workcell_path, collision=True)
    table_prim=workcell_prims['Table']
    UsdPhysics.RigidBodyAPI.Apply(table_prim).CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(table_prim).CreateMassAttr(1.)
    table_physics=PhysxSchema.PhysxRigidBodyAPI.Apply(table_prim)
    table_physics.CreateDisableGravityAttr(True)
    table_physics.CreateSolverPositionIterationCountAttr(8)
    table_physics.CreateSolverVelocityIterationCountAttr(1)
    table_physics.CreateMaxDepenetrationVelocityAttr(1.)
    material = UsdShade.Material.Define(stage, prefix+'/ContactMaterial')
    mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    mat.CreateStaticFrictionAttr(config['friction'])
    mat.CreateDynamicFrictionAttr(config['friction'])
    mat.CreateRestitutionAttr(0.)
    for path in (robot_path, workcell_path):
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(material, materialPurpose='physics')
    joints = {p.GetName(): p for p in prims() if p.IsA(UsdPhysics.RevoluteJoint)}
    if not set(arm.active_names + model.full_names).issubset(joints):
        raise ValueError('USD articulation is missing named arm/hand joints')
    for joint in model.moving:
        prim = joints[joint['name']]
        drive = UsdPhysics.DriveAPI.Apply(prim, 'angular')
        drive.CreateTypeAttr(config.get('joint_drive_type', 'force'))
        scale = np.pi / 180 if config.get('joint_gain_units') == 'si' else 1.
        driven = not joint.get('mimic') or config.get('mimic_mode') == 'constraint_with_drives'
        drive.CreateStiffnessAttr(config['joint_stiffness'] * scale if driven else 0.)
        drive.CreateDampingAttr(config['joint_damping'] * scale if driven else 0.)
        drive.CreateMaxForceAttr(config['joint_effort_limit_nm'])
        if joint.get('mimic'):
            policy = config.get('mimic_schema_policy')
            if policy == 'asset':
                if 'NewtonMimicAPI' not in prim.GetAppliedSchemas():
                    raise ValueError('The native hand mimic schema is not registered in this Isaac runtime')
                for axis in ('rotX', 'rotY', 'rotZ'):
                    if prim.HasAPI(PhysxSchema.PhysxMimicJointAPI, axis):
                        prim.RemoveAPI(PhysxSchema.PhysxMimicJointAPI, axis)
            elif policy == 'single':
                prim.RemoveAppliedSchema('NewtonMimicAPI')
                spec = joint['mimic']; leader = joints[spec['leader']]
                mimic = PhysxSchema.PhysxMimicJointAPI.Apply(prim, 'rot' + str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()))
                mimic.CreateReferenceJointRel().SetTargets([leader.GetPath()])
                mimic.CreateReferenceJointAxisAttr('rot' + str(UsdPhysics.RevoluteJoint(leader).GetAxisAttr().Get()))
                mimic.CreateGearingAttr(-spec['multiplier']); mimic.CreateOffsetAttr(-np.rad2deg(spec['offset']))
            else:
                raise ValueError('Arm policy play requires one explicit or native mimic representation')
    hand_links = {model.root} | {j['child'] for j in model.description['joints']}
    for prim in prims():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rb.CreateDisableGravityAttr(not config['hand_gravity'] if prim.GetName() in hand_links else False)
            if prim.GetName() in hand_links:
                rb.CreateMaxDepenetrationVelocityAttr(config['motion_control']['contact_correction_velocity_m_s'])
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            col.CreateContactOffsetAttr(config['contact_offset_m'])
            col.CreateRestOffsetAttr(config['rest_offset_m'])
        if prim.IsA(UsdGeom.Sphere) and '/kp_' in str(prim.GetPath()):
            UsdGeom.Imageable(prim).CreateVisibilityAttr('invisible')
    from ..sim import prepare_arm_articulation
    table_collision=PhysxSchema.PhysxCollisionAPI.Apply(table_prim)
    table_collision.CreateContactOffsetAttr(.005)
    table_collision.CreateRestOffsetAttr(0.)
    table_collision.CreateTorsionalPatchRadiusAttr(config.get('table_torsional_patch_radius_m',0.))
    table_collision.CreateMinTorsionalPatchRadiusAttr(config.get('table_min_torsional_patch_radius_m',0.))
    wrist_path = robot_path + arm_config['wrist_path'][len(source_root):]
    articulation_root, articulation_resolution = prepare_arm_articulation(stage, robot_path, base_path, wrist_path)
    print('[articulation]', json.dumps(articulation_resolution), flush=True)
    art = PhysxSchema.PhysxArticulationAPI.Apply(articulation_root)
    art.CreateEnabledSelfCollisionsAttr(config['self_collision'])
    art.CreateSolverPositionIterationCountAttr(physics.get('hand_position', 32))
    art.CreateSolverVelocityIterationCountAttr(physics.get('hand_velocity', 2))
    add_reference_to_stage(str(root / config['object_asset']), can_path)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(can_path)).Bind(material, materialPurpose='physics')
    for prim in Usd.PrimRange(stage.GetPrimAtPath(can_path)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rb.CreateDisableGravityAttr(False); rb.CreateEnableGyroscopicForcesAttr(True)
            rb.CreateMaxDepenetrationVelocityAttr(config['object_max_depenetration_velocity_m_s'])
            rb.CreateSolverPositionIterationCountAttr(physics.get('object_position', 16))
            rb.CreateSolverVelocityIterationCountAttr(physics.get('object_velocity', 2))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            col.CreateContactOffsetAttr(config.get('object_contact_offset_m', config['contact_offset_m']))
            col.CreateRestOffsetAttr(config['rest_offset_m'])
    from ..materials import bind_pad_material
    pad_material = bind_pad_material(stage, robot_path, config, prefix+'/Revo2PadMaterial')
    initial_can_pose = world_from_source @ reference.object[0]
    if abs(collision_bottom(initial_can_pose, reference.collision_shapes)) > 1e-6:
        raise ValueError('Arm workcell placement does not put the initial can bottom on z=0')
    q = Rotation.from_matrix(initial_can_pose[:3, :3]).as_quat()
    can_xform = UsdGeom.Xformable(stage.GetPrimAtPath(can_path))
    can_xform.ClearXformOpOrder()
    can_xform.AddTranslateOp().Set(Gf.Vec3d(*initial_can_pose[:3, 3]))
    can_xform.AddOrientOp().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))
    can_xform.AddScaleOp().Set(Gf.Vec3f(1.))
    return dict(robot=str(articulation_root.GetPath()),can=can_path,table=str(table_prim.GetPath()),
                resolution=articulation_resolution,initial_can_pose=initial_can_pose,pad_material_binding=pad_material)
