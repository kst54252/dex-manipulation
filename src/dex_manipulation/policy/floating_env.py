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
from ..control import MotionController, wrist_pd_wrenches
from .observations import ObservationHistory
from .curriculum import ReferenceStateSampler, GravityCurriculum
from .randomization import (
    apply_startup_randomization,
    set_physics_properties,
    property_report,
    PushSchedule,
)


class PhysxResidualEnv:
    def __init__(
        self,
        root,
        model,
        reference,
        config,
        num_envs=8,
        render=False,
        *,
        motion_control_override=None,
    ):
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, UsdLux, Gf
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation, RigidPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from ..scene import FloatingSurface

        started = time.monotonic()
        self.startup_timings = {}

        def progress(phase):
            elapsed = time.monotonic() - started
            self.startup_timings[phase] = elapsed
            print(f"[startup] {phase} | envs={num_envs} elapsed={elapsed:.1f}s", flush=True)

        progress("create physics world")
        self.model, self.reference, self.cfg = model, reference, config
        gain_units = config.get("joint_gain_units", "usd_degrees")
        if gain_units not in ("si", "usd_degrees"):
            raise ValueError("joint_gain_units must be si or usd_degrees")
        drive_type = config.get("joint_drive_type")
        if drive_type not in (None, "force", "acceleration"):
            raise ValueError("Unknown joint_drive_type")
        mimic_mode = config.get("mimic_mode", "constraint")
        if mimic_mode not in ("constraint", "constraint_with_drives", "drive"):
            raise ValueError("Unknown mimic_mode")
        mimic_schema_policy = config.get("mimic_schema_policy", "legacy_mixed")
        if mimic_schema_policy not in ("asset", "single", "legacy_mixed"):
            raise ValueError("Unknown mimic_schema_policy")
        physics = config.get("solver_iterations", {})
        self.surface = FloatingSurface(config["surface"], reference.object[0, :3, 3])
        from .trajectory import digest

        expected = reference.metadata.get("object_asset_sha256")
        if expected and digest(Path(root) / config["object_asset"]) != expected:
            raise ValueError(
                "Object USD differs from the extracted geometry; rebuild object and reference"
            )
        self.device = torch.device(config["physics_device"])
        self.num_envs, self.render = num_envs, render
        self.all_ids = torch.arange(num_envs, device=self.device, dtype=torch.long)
        self.generator = torch.Generator(device=self.device).manual_seed(config["seed"])
        self.rng = np.random.default_rng(config["seed"])
        self.motion = TensorReference(
            reference,
            num_envs,
            config["augmentation"],
            self.device,
            velocity_mode=config.get("reference_velocity_mode", "segment"),
            control_dt=config["physics_dt"] * config["control_decimation"],
        )
        self.task = ResidualTask(model, self.motion, config, self.device)
        controller_config = (
            config.get("motion_control")
            if motion_control_override is None
            else motion_control_override
        )
        self.motion_controller = MotionController(
            model, num_envs, config["physics_dt"], controller_config, self.device
        )
        self.contact_correction_velocity = float(
            (controller_config or {}).get(
                "contact_correction_velocity_m_s", config.get("max_depenetration_velocity_m_s", 5.0)
            )
        )
        self.object_contact_correction_velocity = float(
            config.get("object_max_depenetration_velocity_m_s", self.contact_correction_velocity)
        )
        if (
            not np.isfinite(self.object_contact_correction_velocity)
            or self.object_contact_correction_velocity <= 0
        ):
            raise ValueError("Object contact correction velocity must be positive and finite")
        self.sampler = ReferenceStateSampler(
            reference.duration, self.task.dt, config["rsi"], self.device
        )
        self.gravity = GravityCurriculum(config["gravity_curriculum"], config["gravity"])
        self.training, self.control_steps = False, 0
        self.evaluation_protocol = "strict"
        width = math.ceil(math.sqrt(num_envs))
        scene_origins = torch.tensor(
            [
                [config["env_spacing_m"] * (i % width), config["env_spacing_m"] * (i // width), 0]
                for i in range(num_envs)
            ],
            device=self.device,
        )
        # Task/reference coordinates remain unchanged. Translate only at the
        # simulator boundary so the nominal can starts at the support's center.
        self.origins = scene_origins + torch.tensor(
            self.surface.translation, device=self.device, dtype=torch.float32
        )
        self.time = torch.zeros(num_envs, device=self.device)
        self.last_reset_frame = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.reference_frame_count = int(round(reference.duration / self.task.dt)) + 1
        self.max_episode_length = (
            int(np.ceil(config["episode_length_s"] / self.task.dt))
            if "reference_timing" in config
            else self.reference_frame_count
        )
        self.last_action = torch.zeros(num_envs, 12, device=self.device)
        self.last_q_target = torch.zeros(num_envs, 6, device=self.device)
        self.object_reset_count = 0
        # Match Isaac Lab's headless Fabric output policy. SimulationApp's
        # generic defaults also export joint/sensor/velocity state for display,
        # costing CPU readback even when World.step(render=False) is used.
        # All learning state is read from PhysX tensors; these are output flags,
        # not switches for collision solving, joint constraints or integration.
        import carb

        output_settings = carb.settings.get_settings()
        fabric_output_keys = (
            "fabricUpdateTransformations",
            "fabricUpdateVelocities",
            "fabricUpdateForceSensors",
            "fabricUpdateJointStates",
        )
        if not render:
            for key in fabric_output_keys:
                output_settings.set_bool("/physics/" + key, False)
            output_settings.set_bool("/physics/fabricUseGPUInterop", self.device.type == "cuda")
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=config["physics_dt"],
            rendering_dt=self.task.dt,
            backend="torch",
            device=str(self.device),
        )
        context = self.world.get_physics_context()
        context.set_gravity(-config["gravity"])
        context.set_solver_type("TGS")
        context.set_gpu_max_rigid_contact_count(2**23)
        context.set_gpu_max_rigid_patch_count(2**23)
        # These are inherited Isaac Lab PhysxCfg defaults in the source recipe;
        # legacy World otherwise allocates too few aggregate pairs at 4096 envs.
        context.set_gpu_found_lost_pairs_capacity(2**21)
        context.set_gpu_found_lost_aggregate_pairs_capacity(2**25)
        context.set_gpu_total_aggregate_pairs_capacity(2**21)
        context.set_gpu_collision_stack_size(2**28)
        context.set_gpu_max_num_partitions(1)
        stage = self.world.stage
        scene_api = PhysxSchema.PhysxSceneAPI.Apply(context.get_current_physics_scene_prim())
        scene_api.CreateMaxPositionIterationCountAttr(physics.get("scene_position", 192))
        scene_api.CreateMaxVelocityIterationCountAttr(physics.get("scene_velocity", 1))
        scene_api.CreateBounceThresholdAttr(0.2)
        scene_api.CreateFrictionOffsetThresholdAttr(0.01)
        scene_api.CreateFrictionCorrelationDistanceAttr(0.00625)
        for name, value in config.get("scene_physics", {}).items():
            # Explicitly author the source solver switches; World defaults differ.
            attribute = scene_api.GetPrim().GetAttribute("physxScene:" + name)
            if not attribute or not attribute.Set(value):
                raise ValueError(f"Unsupported PhysX scene option: {name}")
        UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(900)
        if render:
            from isaacsim.core.utils.viewports import set_camera_view

            set_camera_view(eye=np.array([0.6, -0.6, 0.5]), target=np.array([0, 0, 0.12]))
        material = UsdShade.Material.Define(stage, "/World/ContactMaterial")
        mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        mat.CreateStaticFrictionAttr(config.get("friction", 1.0))
        mat.CreateDynamicFrictionAttr(config.get("friction", 1.0))
        mat.CreateRestitutionAttr(0.0)
        self.mimic_specs = []
        progress("build source environment")
        from isaacsim.core.cloner import Cloner

        cloner = Cloner(stage=stage)
        cloner.define_base_env("/World/envs")
        self.env_paths = cloner.generate_paths("/World/envs/env", num_envs)
        prefix = self.env_paths[0]
        UsdGeom.Xform.Define(stage, prefix)
        hand_path = prefix + "/Hand"
        add_reference_to_stage(str(Path(root) / config["hand_asset"]), hand_path)
        add_reference_to_stage(str(Path(root) / config["object_asset"]), prefix + "/Can")
        geometry = self.surface.create_usd(stage, prefix)
        table = geometry["Table"]
        body = UsdPhysics.RigidBodyAPI.Apply(table)
        body.CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(table).CreateMassAttr(1.0)
        tablephys = PhysxSchema.PhysxRigidBodyAPI.Apply(table)
        tablephys.CreateDisableGravityAttr(True)
        tablephys.CreateSolverPositionIterationCountAttr(8)
        tablephys.CreateSolverVelocityIterationCountAttr(1)
        tablephys.CreateMaxDepenetrationVelocityAttr(1.0)
        col = PhysxSchema.PhysxCollisionAPI.Apply(table)
        col.CreateContactOffsetAttr(0.005)
        col.CreateRestOffsetAttr(0.0)
        col.CreateTorsionalPatchRadiusAttr(config.get("table_torsional_patch_radius_m", 0.02))
        col.CreateMinTorsionalPatchRadiusAttr(
            config.get("table_min_torsional_patch_radius_m", 0.005)
        )
        art = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(hand_path))
        art.CreateSolverPositionIterationCountAttr(physics.get("hand_position", 32))
        art.CreateSolverVelocityIterationCountAttr(physics.get("hand_velocity", 1))
        art.CreateEnabledSelfCollisionsAttr(config["self_collision"])
        art.CreateSleepThresholdAttr(0.005)
        art.CreateStabilizationThresholdAttr(0.0005)
        joints = {
            p.GetName(): p
            for p in Usd.PrimRange(stage.GetPrimAtPath(hand_path))
            if p.IsA(UsdPhysics.RevoluteJoint)
        }
        for j in model.moving:
            prim = joints[j["name"]]
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            if j.get("mimic") and mimic_schema_policy == "single":
                # Isaac Sim 6 recognizes the imported Newton schema as well.
                # Use one explicit constraint representation in this adapter.
                prim.RemoveAppliedSchema("NewtonMimicAPI")
            if drive_type is not None:
                drive.CreateTypeAttr(drive_type)
            if config.get("joint_effort_limit_nm") is not None:
                drive.CreateMaxForceAttr(float(config["joint_effort_limit_nm"]))
            # USD angular drives consume degrees; PhysX tensors and the task use
            # radians. Convert once at the USD boundary, before world.reset.
            gain_scale = np.pi / 180 if gain_units == "si" else 1.0
            if j.get("mimic") and mimic_mode != "drive":
                m = j["mimic"]
                if mimic_schema_policy == "asset":
                    if "NewtonMimicAPI" not in prim.GetAppliedSchemas():
                        raise ValueError(
                            f"Registered native Newton mimic schema required: {prim.GetPath()}"
                        )
                    # The source adapter retains the imported native constraint.
                    # Do not add another representation on top of it.
                    for axis in ("rotX", "rotY", "rotZ"):
                        if prim.HasAPI(PhysxSchema.PhysxMimicJointAPI, axis):
                            prim.RemoveAPI(PhysxSchema.PhysxMimicJointAPI, axis)
                    implemented = "NewtonMimicAPI"
                else:
                    api = PhysxSchema.PhysxMimicJointAPI.Apply(
                        prim, "rot" + str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get())
                    )
                    api.CreateReferenceJointRel().SetTargets([joints[m["leader"]].GetPath()])
                    api.CreateReferenceJointAxisAttr(
                        "rot"
                        + str(UsdPhysics.RevoluteJoint(joints[m["leader"]]).GetAxisAttr().Get())
                    )
                    api.CreateGearingAttr(-m["multiplier"])
                    api.CreateOffsetAttr(-np.rad2deg(m["offset"]))
                    implemented = "PhysxMimicJointAPI"
                drive.CreateStiffnessAttr(
                    config["joint_stiffness"] * gain_scale
                    if mimic_mode == "constraint_with_drives"
                    else 0.0
                )
                drive.CreateDampingAttr(
                    config["joint_damping"] * gain_scale
                    if mimic_mode == "constraint_with_drives"
                    else 0.0
                )
                self.mimic_specs.append(
                    dict(follower=j["name"], **m, implemented_schema=implemented)
                )
            else:
                if j.get("mimic"):
                    for axis in ("rotX", "rotY", "rotZ"):
                        if prim.HasAPI(PhysxSchema.PhysxMimicJointAPI, axis):
                            prim.RemoveAPI(PhysxSchema.PhysxMimicJointAPI, axis)
                drive.CreateStiffnessAttr(config["joint_stiffness"] * gain_scale)
                drive.CreateDampingAttr(config["joint_damping"] * gain_scale)
        for asset in (hand_path, prefix + "/Can", prefix + "/Table"):
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(asset)).Bind(
                material, materialPurpose="physics"
            )
            for prim in Usd.PrimRange(stage.GetPrimAtPath(asset)):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                    if asset == hand_path:
                        rb.CreateDisableGravityAttr(not config["hand_gravity"])
                    if asset.endswith("/Can"):
                        rb.CreateDisableGravityAttr(False)
                        rb.CreateEnableGyroscopicForcesAttr(True)
                        rb.CreateSolverPositionIterationCountAttr(physics.get("object_position", 8))
                        rb.CreateSolverVelocityIterationCountAttr(physics.get("object_velocity", 0))
                        rb.CreateSleepThresholdAttr(0.005)
                        rb.CreateStabilizationThresholdAttr(0.0025)
                    if not asset.endswith("/Table"):
                        rb.CreateMaxDepenetrationVelocityAttr(
                            self.object_contact_correction_velocity
                            if asset.endswith("/Can")
                            else self.contact_correction_velocity
                        )
                if prim.HasAPI(UsdPhysics.CollisionAPI) and not asset.endswith("/Table"):
                    col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    col.CreateContactOffsetAttr(
                        config.get("object_contact_offset_m", config["contact_offset_m"])
                        if asset.endswith("/Can")
                        else config["contact_offset_m"]
                    )
                    col.CreateRestOffsetAttr(config["rest_offset_m"])
        from ..materials import bind_pad_material

        pad_material = bind_pad_material(stage, hand_path, config, "/World/Revo2PadMaterial")
        cache = UsdGeom.XformCache()
        palm = next(
            p for p in Usd.PrimRange(stage.GetPrimAtPath(hand_path)) if p.GetName() == model.root
        )
        relative = (
            cache.GetLocalToWorldTransform(palm)
            * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(hand_path)).GetInverse()
        )
        if not np.allclose(np.array(relative), np.eye(4), atol=1e-6):
            raise ValueError("Explicit root/palm transform adapter required")
        # Author the first pose once. Replication carries it to every env;
        # tensor reset later sets the actual per-env joint/reference state.
        from scipy.spatial.transform import Rotation

        for path, pose in ((hand_path, reference.wrist[0]), (prefix + "/Can", reference.object[0])):
            xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*(pose[:3, 3] + self.surface.translation)))
            q = Rotation.from_matrix(pose[:3, :3]).as_quat()
            xform.AddOrientOp().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))
            xform.AddScaleOp().Set(Gf.Vec3f(1.0))
        progress("clone USD and register physics replication")
        cloner.clone(
            source_prim_path=prefix,
            prim_paths=self.env_paths,
            positions=scene_origins,
            replicate_physics=num_envs > 1,
            copy_from_source=False,
            enable_env_ids=True,
        )
        progress("create articulation view")
        # Do not normalize/write thousands of USD poses again after cloning.
        # The source already has translate/orient/scale ops and world poses.
        cloner.disable_change_listener()
        try:
            self.robot = self.world.scene.add(
                Articulation(
                    "/World/envs/env_.*/Hand", name="residual_hands", reset_xform_properties=False
                )
            )
            progress("create object and support views")
            self.can = self.world.scene.add(
                RigidPrim(
                    "/World/envs/env_.*/Can",
                    name="dynamic_cans",
                    reset_xform_properties=False,
                    track_contact_forces=True,
                )
            )
            self.table = self.world.scene.add(
                RigidPrim("/World/envs/env_.*/Table", name="tables", reset_xform_properties=False)
            )
        finally:
            cloner.enable_change_listener()
        progress("initialize PhysX (world.reset)")
        self.world.reset()
        progress("initialize randomized physics and controller")
        self.full_ids = torch.tensor(
            [self.robot.dof_names.index(n) for n in model.full_names], device=self.device
        )
        self.active_ids = torch.tensor(
            [self.robot.dof_names.index(n) for n in model.active_names], device=self.device
        )
        self.palm_id = self.robot.body_names.index(model.root)
        self.root_id = self.robot.body_names.index("Hand")
        self.kp_ids = torch.tensor(
            [self.robot.body_names.index(k["link"]) for k in model.keypoints], device=self.device
        )
        self.kp_local = torch.tensor([k["xyz"] for k in model.keypoints], device=self.device)
        self.tip_ids = torch.tensor(
            [
                model.semantic_names.index(k + "_tip")
                for k in ("thumb", "index", "middle", "ring", "little")
            ],
            device=self.device,
        )
        self.tip_body_ids = self.kp_ids[self.tip_ids]
        self.configure_table_safety()
        velocity = self.task.full_velocity[None].expand(num_envs, -1)
        if config.get("simulation_coupled_velocity_limits", False):
            velocity = (self.task.coupling.abs() @ self.task.velocity)[None].expand(num_envs, -1)
        if config.get("simulation_joint_velocity_rad_s") is not None:
            velocity = torch.full_like(velocity, float(config["simulation_joint_velocity_rad_s"]))
        self.robot.set_max_joint_velocities(velocity, joint_indices=self.full_ids)
        self.force_body_id = self.robot.body_names.index(config.get("root_force_body", "Hand"))
        if gain_units == "si":
            nominal_kp, nominal_kd = self.robot.get_gains(joint_indices=self.active_ids)
            torch.testing.assert_close(
                nominal_kp,
                torch.full_like(nominal_kp, config["joint_stiffness"]),
                rtol=1e-5,
                atol=1e-6,
            )
            torch.testing.assert_close(
                nominal_kd,
                torch.full_like(nominal_kd, config["joint_damping"]),
                rtol=1e-5,
                atol=1e-6,
            )
        apply_startup_randomization(self)
        self.refresh_physics_properties()
        self.obs_history = ObservationHistory(
            num_envs, 6, config["observation"], self.device, self.generator
        )
        self.obs_history.default_q.copy_(self.default_q_offset)
        self.push = PushSchedule(num_envs, config["push_curriculum"], self.device, self.generator)
        self.metadata = dict(
            backend="Isaac Sim 6 PhysX " + str(self.device) + " / Torch device-resident control",
            num_envs=num_envs,
            physics_device=str(self.device),
            body_names=self.robot.body_names,
            dof_names=self.robot.dof_names,
            mimic_constraints=self.mimic_specs,
            self_collision=config["self_collision"],
            hand_gravity=config["hand_gravity"],
            object_kinematic=False,
            object_control="gravity/contact plus source velocity-push curriculum; pose only reset",
            root_control="PD root force + mass-proportional distributed torque; hand gravity "
            + ("enabled" if config["hand_gravity"] else "disabled"),
            physics_dt=config["physics_dt"],
            control_dt=self.task.dt,
            randomization=property_report(self),
            active_effort_limits_nm=self.robot.get_max_efforts()[0, self.active_ids].cpu().tolist(),
            actor_observation_size=self.task.observation_size,
            critic_observation_size=self.task.critic_observation_size,
        )
        actual_kp, actual_kd = self.robot.get_gains(joint_indices=self.active_ids)
        self.metadata["joint_drives"] = dict(
            input_units=gain_units,
            drive_type=drive_type or "asset",
            mimic_mode=mimic_mode,
            mimic_schema_policy=mimic_schema_policy,
            stiffness_si_min=float(actual_kp.min()),
            stiffness_si_max=float(actual_kp.max()),
            damping_si_min=float(actual_kd.min()),
            damping_si_max=float(actual_kd.max()),
            force_body=self.robot.body_names[self.force_body_id],
            simulation_velocity_limits_rad_s=velocity[0].cpu().tolist(),
        )
        self.metadata["pad_material_binding"] = pad_material
        self.metadata.update(
            controller="regrind_pose_pd",
            gravity_compensation=False,
            reference_velocity_feedforward=False,
            stepping="one explicit physics step per PD update; rendering advances no physics",
        )
        self.metadata["motion_control"] = copy.deepcopy(self.motion_controller.config)
        self.metadata["contact_correction_velocity_m_s"] = self.contact_correction_velocity
        self.metadata["object_contact_correction_velocity_m_s"] = (
            self.object_contact_correction_velocity
        )
        self.metadata["solver_iterations"] = dict(
            scene_position=scene_api.GetMaxPositionIterationCountAttr().Get(),
            scene_velocity=scene_api.GetMaxVelocityIterationCountAttr().Get(),
            hand_position=art.GetSolverPositionIterationCountAttr().Get(),
            hand_velocity=art.GetSolverVelocityIterationCountAttr().Get(),
        )
        self.metadata["scene_physics"] = {
            name: scene_api.GetPrim().GetAttribute("physxScene:" + name).Get()
            for name in (
                "enableExternalForcesEveryIteration",
                "enableStabilization",
                "solveArticulationContactLast",
                "gpuMaxNumPartitions",
            )
        }
        self.metadata["newton_mimic_registered"] = bool(
            Usd.SchemaRegistry.GetTypeFromSchemaTypeName("NewtonMimicAPI")
        )
        self.metadata.update(
            surface=self.surface.metadata(),
            state_coordinate_frame="unchanged task/reference frame; add task_origins_world_m to positions for simulator world",
            task_origins_world_m=self.origins.cpu().tolist(),
        )
        self.metadata["gpu_pair_capacities"] = dict(
            found_lost=context.get_gpu_found_lost_pairs_capacity(),
            found_lost_aggregate=context.get_gpu_found_lost_aggregate_pairs_capacity(),
            total_aggregate=context.get_gpu_total_aggregate_pairs_capacity(),
        )
        self.reset(randomize=False)
        progress("environment ready")
        self.metadata.update(
            startup_timings_s=self.startup_timings,
            environment_creation="single USD source + physics replication",
            inter_environment_collision_filter="PhysX environment IDs",
            gpu_dynamics_enabled=bool(scene_api.GetEnableGPUDynamicsAttr().Get()),
            broadphase=scene_api.GetBroadphaseTypeAttr().Get(),
            scene_query_support=bool(scene_api.GetEnableSceneQuerySupportAttr().Get()),
            fabric_output_settings={
                key: output_settings.get("/physics/" + key)
                for key in (*fabric_output_keys, "fabricUseGPUInterop")
            },
        )

    def refresh_physics_properties(self):
        self.body_com_local = self.robot.get_body_coms()[0].to(self.device).clone()
        self.can_com_local = (
            self.can.get_coms()[0].to(self.device).reshape(self.num_envs, 3).clone()
        )
        self.body_mass = self.robot.get_body_masses().to(self.device).clone()
        self.total_mass = self.body_mass.sum(1)

    def _apply_gravity(self, magnitude):
        import carb

        self.gravity.value = float(magnitude)
        self.world.get_physics_context().set_gravity(-float(magnitude))
        self.world.physics_sim_view.set_gravity(carb.Float3(0, 0, -float(magnitude)))

    def configure_motion_control(self, config):
        """Explicit deployment override; checkpoint/training configuration stays intact."""
        if self.training:
            raise ValueError(
                "Change the training config to train with a different motion controller"
            )
        correction = float(
            (config or {}).get(
                "contact_correction_velocity_m_s",
                self.cfg.get("max_depenetration_velocity_m_s", 5.0),
            )
        )
        if correction != self.contact_correction_velocity:
            raise ValueError(
                "Contact correction settings must be selected before PhysX initialization"
            )
        self.motion_controller = MotionController(
            self.model, self.num_envs, self.cfg["physics_dt"], config, self.device
        )
        self.motion_controller.reset(self.all_ids, self.state())
        self.metadata["motion_control"] = copy.deepcopy(config)
        self.metadata["contact_correction_velocity_m_s"] = correction

    def set_training(self, training):
        changed = self.training != bool(training)
        self.training = bool(training)
        if changed or not training:
            self._apply_gravity(self.gravity.sample(self.control_steps, self.rng, self.training))
        # Startup DR remains fixed in both train/eval, matching source PLAY inheritance.

    def training_state_dict(self):
        return dict(
            schema=2,
            control_steps=self.control_steps,
            training=self.training,
            evaluation_protocol=self.evaluation_protocol,
            rng=copy.deepcopy(self.rng.bit_generator.state),
            device_rng=self.generator.get_state().cpu(),
            sampler=self.sampler.state_dict(),
            gravity=self.gravity.state_dict(),
            randomized_physics=self.randomized_physics,
            default_q_offset=self.default_q_offset.cpu(),
        )

    def load_training_state_dict(self, state):
        if state["schema"] != 2:
            raise ValueError("Incompatible curriculum checkpoint")
        self.control_steps = int(state["control_steps"])
        self.training = bool(state["training"])
        self.evaluation_protocol = state.get("evaluation_protocol", "strict")
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.generator.set_state(state["device_rng"].cpu())
        self.sampler.load_state_dict(state["sampler"])
        self.gravity.load_state_dict(state["gravity"])
        self._apply_gravity(self.gravity.value)
        # Environment count may differ for evaluation: repeat the saved per-env startup distribution.
        self.randomized_physics = {
            asset: {k: v[torch.arange(self.num_envs) % len(v)].clone() for k, v in vals.items()}
            for asset, vals in state["randomized_physics"].items()
        }
        set_physics_properties(self, self.randomized_physics)
        offsets = state["default_q_offset"]
        self.default_q_offset.copy_(
            offsets[torch.arange(self.num_envs) % len(offsets)].to(self.device)
        )
        self.obs_history.default_q.copy_(self.default_q_offset)

    def reset(self, indices=None, randomize=True, times=None):
        ids = (
            self.all_ids
            if indices is None
            else torch.as_tensor(indices, device=self.device, dtype=torch.long)
        )
        if not len(ids):
            return
        learning = self.training and randomize
        if times is None:
            times = (
                self.sampler.sample(len(ids), self.generator)
                if learning
                else torch.zeros(len(ids), device=self.device)
            )
        times = torch.as_tensor(times, device=self.device, dtype=torch.float32)
        if (
            times.shape != (len(ids),)
            or not torch.isfinite(times).all()
            or (times < 0).any()
            or (times > self.reference.duration).any()
        ):
            raise ValueError("Invalid reset times")
        if self.training:
            self._apply_gravity(self.gravity.sample(self.control_steps, self.rng, True))
        self.motion.reset(
            ids,
            self.generator,
            randomize and (self.training or self.evaluation_protocol == "source"),
        )
        if self.cfg["observation"].get("canonicalize_translation", False):
            if self.motion.mode != "rigid_sequence" or self.motion.yaw[ids].abs().max() > 1e-8:
                raise ValueError(
                    "Translation canonicalization requires constant translation and zero yaw"
                )
            self.obs_history.position_offset[ids] = self.motion.translation[ids]
        ref = self.motion.sample(times, ids)
        q = ref["q"].clone()
        poses = {
            k: ref[k].clone()
            for k in ("wrist_position", "wrist_quaternion", "object_position", "object_quaternion")
        }
        if learning and self.cfg["reset_perturbation"]:
            noise = self.cfg["rsi"]
            q = (
                q
                + uniform(
                    q.shape,
                    -noise["joint_noise_rad"],
                    noise["joint_noise_rad"],
                    self.device,
                    self.generator,
                )
            ).clamp(self.task.lower, self.task.upper)
            # Original floating Revo2 perturbs only finger positions. Its wrist
            # and grounded can start exactly at the selected reference pose.
            noisy_bodies = (
                () if noise.get("reset_noise_mode") == "source_joint_only" else ("wrist", "object")
            )
            for name in noisy_bodies:
                poses[name + "_position"] += uniform(
                    (len(ids), 3),
                    -noise["position_noise_m"],
                    noise["position_noise_m"],
                    self.device,
                    self.generator,
                )
                axis = torch.randn((len(ids), 3), device=self.device, generator=self.generator)
                axis /= axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                rv = axis * uniform(
                    (len(ids), 1),
                    -noise["rotation_noise_rad"],
                    noise["rotation_noise_rad"],
                    self.device,
                    self.generator,
                )
                poses[name + "_quaternion"] = quat_multiply(
                    from_rotvec(rv), poses[name + "_quaternion"]
                )
        self.time[ids] = times
        self.episode_length_buf[ids] = 0
        self.last_action[ids] = 0
        self.last_q_target[ids] = q
        self.last_reset_frame[ids] = (times / self.task.dt).round().long()
        moving = (
            (times > 1e-8)[:, None].float()
            if self.cfg["rsi"]["initialize_velocity"]
            else torch.zeros((len(ids), 1), device=self.device)
        )
        self.robot.set_world_poses(
            poses["wrist_position"] + self.origins[ids],
            poses["wrist_quaternion"][:, [3, 0, 1, 2]],
            indices=ids,
        )
        com = quat_apply(poses["wrist_quaternion"], self.body_com_local[ids, self.root_id])
        root_v = ref["wrist_velocity"] + torch.cross(ref["wrist_angular_velocity"], com, dim=-1)
        self.robot.set_velocities(
            torch.cat((root_v, ref["wrist_angular_velocity"]), -1) * moving, indices=ids
        )
        full = q @ self.task.coupling.T + self.task.offset
        self.robot.set_joint_positions(full, indices=ids, joint_indices=self.full_ids)
        self.robot.set_joint_velocities(
            (ref["q_velocity"] * moving) @ self.task.coupling.T,
            indices=ids,
            joint_indices=self.full_ids,
        )
        self.robot.set_joint_position_targets(full, indices=ids, joint_indices=self.full_ids)
        self.can.set_world_poses(
            poses["object_position"] + self.origins[ids],
            poses["object_quaternion"][:, [3, 0, 1, 2]],
            indices=ids,
        )
        com = quat_apply(poses["object_quaternion"], self.can_com_local[ids])
        can_v = ref["object_velocity"] + torch.cross(ref["object_angular_velocity"], com, dim=-1)
        self.can.set_velocities(
            torch.cat((can_v, ref["object_angular_velocity"]), -1) * moving, indices=ids
        )
        self.object_reset_count += len(ids)
        self.world.physics_sim_view.update_articulations_kinematic()
        self.push.reset(ids)
        state = self.state()
        self.motion_controller.reset(ids, state)
        self.obs_history.update(
            state, ids, reset=True, noisy=self.training or self.evaluation_protocol == "source"
        )
        self.obs_history.assemble(
            state, self.motion.sample(self.time), self.last_action, self.phase(), ids
        )

    def phase_at(self, times):
        count = getattr(self, "reference_frame_count", self.max_episode_length)
        denominator = (
            count - 1 if self.cfg.get("reference_phase") == "endpoint_normalized" else count
        )
        return ((times / self.task.dt) / max(denominator, 1)).clamp(0, 1)

    def phase(self):
        return self.phase_at(self.time)

    def state(self):
        links = self.robot._physics_view.get_link_transforms().clone()
        velocities = self.robot._physics_view.get_link_velocities().clone()
        offsets = quat_apply(links[:, :, 3:7], self.body_com_local)
        velocities[:, :, :3] -= torch.cross(velocities[:, :, 3:], offsets, dim=-1)
        q, qv = self.robot.get_joint_positions(), self.robot.get_joint_velocities()
        cp, cqw = self.can.get_world_poses()
        cq = cqw[:, [1, 2, 3, 0]]
        cv = self.can.get_velocities().clone()
        cv[:, :3] -= torch.cross(cv[:, 3:], quat_apply(cq, self.can_com_local), dim=-1)
        palm, pv = links[:, self.palm_id], velocities[:, self.palm_id]
        kp = (
            quat_apply(
                links[:, self.kp_ids, 3:7], self.kp_local[None].expand(self.num_envs, -1, -1)
            )
            + links[:, self.kp_ids, :3]
            - self.origins[:, None]
        )
        task_links = links.clone()
        task_links[:, :, :3] -= self.origins[:, None]
        return dict(
            q=q[:, self.active_ids],
            full_q=q[:, self.full_ids],
            q_velocity=qv[:, self.active_ids],
            full_q_velocity=qv[:, self.full_ids],
            wrist_position=palm[:, :3] - self.origins,
            wrist_quaternion=palm[:, 3:7],
            wrist_velocity=pv[:, :3],
            wrist_angular_velocity=pv[:, 3:],
            object_position=cp - self.origins,
            object_quaternion=cq,
            object_velocity=cv[:, :3],
            object_angular_velocity=cv[:, 3:],
            link_transforms=task_links,
            link_velocities=velocities,
            robot_keypoints=kp,
            fingertips=(
                task_links[:, self.tip_body_ids, :3]
                if self.cfg["observation"].get("fingertip_frame") == "link_origin"
                else kp[:, self.tip_ids]
            ),
        )

    def observation(self):
        return self.obs_history.get()

    def configure_table_safety(self, override=None):
        from .table import TableClearance

        settings = self.cfg.get("table_safety", {}) if override is None else override
        if self.cfg["surface"]["top_z_m"] != 0:
            raise ValueError("Table protection requires source tabletop Z=0")
        if hasattr(self, "frame") and not np.allclose(
            self.frame.world_from_source[2], [0, 0, 1, 0], atol=1e-8
        ):
            raise ValueError("Table protection requires arm placement to preserve tabletop Z=0")
        self.table_safety = (
            TableClearance(self.model, self.robot.body_names, settings, self.device)
            if settings.get("enabled", False)
            else None
        )

    def step(self, actions, auto_reset=True):
        started = time.monotonic()
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if actions.shape != (self.num_envs, 12) or not torch.isfinite(actions).all():
            raise ValueError("Invalid residual actions")
        # Current command drives physics and reward; command advances afterwards as in ManagerBasedRLEnv.
        ref = self.motion.sample(self.time)
        target = self.task.targets(ref, actions, self.last_q_target, self.default_q_offset)
        if self.task.finger_tracking.enabled:
            target = self.task.protect_fingers(target, self.state()["q"], self.last_q_target)
        table = getattr(self, "table_safety", None)
        if table is not None:
            table.begin(target, self.state())
        self.last_q_target.copy_(target["active_q"])
        if not self.motion_controller.enabled:
            self.robot.set_joint_position_targets(target["full_q"], joint_indices=self.full_ids)
        applied_target = target
        for substep in range(self.cfg["control_decimation"]):
            applied_target = target
            if self.motion_controller.enabled:
                applied_target = self.motion_controller.step(target)
                self.robot.set_joint_position_targets(
                    applied_target["full_q"], joint_indices=self.full_ids
                )
            s = self.state()
            if table is not None:
                table.observe(s)
                applied_target = table.guard(applied_target, s)
            all_forces, all_torques = wrist_pd_wrenches(
                s, applied_target, self.body_mass, self.force_body_id, self.cfg
            )
            external = getattr(self, "external_wrench_provider", None)
            if external is not None:
                if self.training:
                    raise RuntimeError(
                        "Trajectory contact guidance must not leak into policy training"
                    )
                hand_force, hand_torque, object_force, object_torque = external(self, s, substep)
                for supplied, expected in ((hand_force, all_forces), (hand_torque, all_torques)):
                    if supplied.shape != expected.shape or not torch.isfinite(supplied).all():
                        raise ValueError("Invalid contact-guidance articulation wrench")
                for supplied in (object_force, object_torque):
                    if supplied.shape != (self.num_envs, 3) or not torch.isfinite(supplied).all():
                        raise ValueError("Invalid contact-guidance object wrench")
                all_forces = all_forces + hand_force
                all_torques = all_torques + hand_torque
                self.can._physics_view.apply_forces_and_torques_at_position(
                    object_force, object_torque, None, self.all_ids.to(torch.int32), True
                )
            self.robot._physics_view.apply_forces_and_torques_at_position(
                all_forces, all_torques, None, self.all_ids.to(torch.int32), True
            )
            # Legacy World.step(render=True) advances an app/render interval,
            # which can contain multiple physics steps with a stale wrench.
            # As in Isaac Lab's RL loop, step physics explicitly and render alone.
            self.world.step(render=False)
        self.episode_length_buf += 1
        state = self.state()
        if table is not None:
            table.observe(state)
        demo_end = self.time >= self.reference.duration - 1e-6
        timed_out = self.episode_length_buf >= self.max_episode_length
        reward, term, trunc, metrics = self.task.score(
            state,
            ref,
            actions,
            self.last_action,
            demo_end,
            timed_out,
            table_metrics=table.metrics() if table is not None else None,
            finger_target=applied_target,
        )
        metrics["gravity_m_s2"] = torch.full(
            (self.num_envs,), self.gravity.value, device=self.device
        )
        if self.training:
            self.control_steps += 1
            self.sampler.record(self.time, term, demo_end, self.num_envs, self.max_episode_length)
        self.last_action.copy_(actions)
        self.obs_history.update(
            state, self.all_ids, noisy=self.training or self.evaluation_protocol == "source"
        )
        next_time = (self.time + self.task.dt).clamp(max=self.reference.duration)
        next_phase = self.phase_at(next_time)
        # Timeout value must see the next command too, exactly as a continuing
        # policy observation would, before any auto-reset overwrites this state.
        final_obs = self.obs_history.assemble(
            state, self.motion.sample(next_time), self.last_action, next_phase
        )
        info = dict(
            metrics=metrics,
            final_observation=final_obs,
            state=state,
            reference=ref,
            raw_targets=target,
            applied_targets=applied_target,
            object_reset_count=self.object_reset_count,
            reference_time=self.time.clone(),
        )
        self.time = next_time
        done = torch.nonzero(term | trunc, as_tuple=False).flatten()
        if auto_reset and len(done):
            self.reset(done, randomize=True)
            if self.cfg.get("advance_command_after_auto_reset", False):
                # Isaac Lab calls command_manager.compute AFTER resetting done
                # environments. Their bodies start at RSI frame k, but the
                # next observation/action uses command k+1. Initial env.reset()
                # outside step() still exposes k, as in the source lifecycle.
                self.time[done] = (self.time[done] + self.task.dt).clamp(
                    max=self.reference.duration
                )
        self.push.advance(self)
        # Assemble new command without pushing the history or drawing noise a second time.
        self.obs_history.assemble(
            self.state() if auto_reset and len(done) else state,
            self.motion.sample(self.time),
            self.last_action,
            self.phase(),
        )
        # UI callbacks can stop/close the timeline and invalidate PhysX views.
        # Finish all physics reads before dispatching those callbacks.
        if self.render:
            self.world.render()
        if self.render:
            time.sleep(max(0, self.task.dt - (time.monotonic() - started)))
        return self.observation(), reward, term, trunc, info

    def close(self):
        self.world.stop()
