"""Batched RB3+Revo2 learning with wrist/finger residuals and online strict IK.

Every observation/reward uses the physical assembled articulation. There is no
floating carrier, root wrench, virtual wrist response or object pose override
between resets. Training and deployment use this same adapter.
"""

from ..configuration import read_config
import math
import time
from pathlib import Path
import numpy as np
import torch

from ..fk import ArmModel
from ..ik import IKOptions
from ..ik_torch import BatchedArmIK, pose_tensor, compose
from ..scene import Workcell
from ..control import MotionController
from .arm_ik import PolicyFrame
from .arm_scene import build_arm_scene
from .arm_observations import ArmObservations
from .floating_env import PhysxResidualEnv
from .task import ResidualTask
from .reference import TensorReference
from .curriculum import ReferenceStateSampler, GravityCurriculum
from .randomization import apply_startup_randomization, PushSchedule, property_report
from .math3d import quat_apply, rotation_error, uniform
from .trajectory import digest


class ArmTrainingEnv(PhysxResidualEnv):
    def __init__(self, root, model, reference, config, arm_config, num_envs=128, render=False):
        from pxr import UsdGeom, PhysxSchema, UsdLux
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation, RigidPrim
        from isaacsim.core.cloner import Cloner
        import carb

        root = Path(root)
        started = time.monotonic()

        def progress(message):
            print(
                f"[startup arm] {message} | envs={num_envs} elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

        self.model, self.reference, self.cfg = model, reference, config
        self.arm_cfg = config["arm_training"]
        if not self.arm_cfg["enabled"]:
            raise ValueError("Arm training contract must be explicitly enabled")
        if config.get("motion_control", {}).get("enabled", False):
            raise ValueError("Arm learning uses direct pose targets plus tabletop protection")
        self.arm = ArmModel.load(root / arm_config["arm_model"])
        if self.arm.wrist != model.root:
            raise ValueError("Arm mount and hand model differ")
        if digest(root / arm_config["usd"]) != self.arm.description["source_sha256"]:
            raise ValueError("Arm USD changed; re-extract model")
        for path, sha in self.arm.description["layer_sha256"].items():
            if digest(path) != sha:
                raise ValueError(f"Arm USD layer changed: {path}")
        if digest(root / config["object_asset"]) != reference.metadata["object_asset_sha256"]:
            raise ValueError("Object asset changed")
        self.workcell = Workcell.load(root / arm_config["workcell"])
        alignment = self.workcell.resolve_alignment(read_config(root / arm_config["alignment"]))
        self.world_from_source = np.asarray(alignment["world_from_source"])
        self.device = torch.device(config["physics_device"])
        self.frame = PolicyFrame(self.world_from_source, self.device)
        self.to_world = PolicyFrame(np.linalg.inv(self.world_from_source), self.device)
        self.base_from_source = pose_tensor(alignment["base_from_source"], self.device)
        self.from_base = PolicyFrame(np.asarray(alignment["base_from_source"]), self.device)
        self.num_envs, self.render, self.training = num_envs, render, False
        self.evaluation_protocol = "strict"
        self.all_ids = torch.arange(num_envs, device=self.device, dtype=torch.long)
        self.generator = torch.Generator(device=self.device).manual_seed(config["seed"])
        self.rng = np.random.default_rng(config["seed"])
        self.motion = TensorReference(
            reference,
            num_envs,
            config["augmentation"],
            self.device,
            velocity_mode=config["reference_velocity_mode"],
            control_dt=config["physics_dt"] * config["control_decimation"],
        )
        self.task = ResidualTask(model, self.motion, config, self.device)
        self.motion_controller = MotionController(
            model, num_envs, config["physics_dt"], config.get("motion_control"), self.device
        )
        self.sampler = ReferenceStateSampler(
            reference.duration, self.task.dt, config["rsi"], self.device
        )
        self.gravity = GravityCurriculum(config["gravity_curriculum"], config["gravity"])
        self.control_steps = 0
        self.object_reset_count = 0
        self.reset_ik_rejections = 0
        self.time = torch.zeros(num_envs, device=self.device)
        self.episode_length_buf = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.last_reset_frame = torch.zeros_like(self.episode_length_buf)
        self.reference_frame_count = reference.metadata["control_reference_frames"]
        self.max_episode_length = int(np.ceil(config["episode_length_s"] / self.task.dt))
        self.last_action = torch.zeros((num_envs, 12), device=self.device)
        self.last_q_target = torch.zeros((num_envs, 6), device=self.device)
        self.last_arm_target = torch.zeros_like(self.last_q_target)
        self.last_ik_error = torch.zeros_like(self.last_q_target)
        self.last_ik_success = torch.ones(num_envs, device=self.device, dtype=torch.bool)
        self.ik_failure_streak = torch.zeros_like(self.episode_length_buf)
        width = math.ceil(math.sqrt(num_envs))
        self.origins = torch.tensor(
            [
                [config["env_spacing_m"] * (i % width), config["env_spacing_m"] * (i // width), 0.0]
                for i in range(num_envs)
            ],
            device=self.device,
            dtype=torch.float32,
        )
        settings = carb.settings.get_settings()
        if not render:
            for key in (
                "fabricUpdateTransformations",
                "fabricUpdateVelocities",
                "fabricUpdateForceSensors",
                "fabricUpdateJointStates",
            ):
                settings.set_bool("/physics/" + key, False)
            settings.set_bool("/physics/fabricUseGPUInterop", self.device.type == "cuda")
        progress("create physics world")
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
        context.set_gpu_found_lost_pairs_capacity(2**21)
        context.set_gpu_found_lost_aggregate_pairs_capacity(2**25)
        context.set_gpu_total_aggregate_pairs_capacity(2**21)
        context.set_gpu_collision_stack_size(2**28)
        stage = self.world.stage
        scene_api = PhysxSchema.PhysxSceneAPI.Apply(context.get_current_physics_scene_prim())
        physics = config["solver_iterations"]
        scene_api.CreateMaxPositionIterationCountAttr(physics["scene_position"])
        scene_api.CreateMaxVelocityIterationCountAttr(physics["scene_velocity"])
        scene_api.CreateBounceThresholdAttr(0.2)
        scene_api.CreateFrictionOffsetThresholdAttr(0.01)
        scene_api.CreateFrictionCorrelationDistanceAttr(0.00625)
        for name, value in config.get("scene_physics", {}).items():
            attr = scene_api.GetPrim().GetAttribute("physxScene:" + name)
            if not attr or not attr.Set(value):
                raise ValueError(f"Unsupported PhysX setting: {name}")
        UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(900)
        cloner = Cloner(stage=stage)
        cloner.define_base_env("/World/envs")
        self.env_paths = cloner.generate_paths("/World/envs/env", num_envs)
        prefix = self.env_paths[0]
        UsdGeom.Xform.Define(stage, prefix)
        progress("build assembled source robot and table")
        scene = build_arm_scene(
            stage,
            root,
            model,
            reference,
            config,
            arm_config,
            self.arm,
            self.workcell,
            self.world_from_source,
            prefix=prefix,
        )
        progress("clone and replicate physics")
        cloner.clone(
            source_prim_path=prefix,
            prim_paths=self.env_paths,
            positions=self.origins,
            replicate_physics=num_envs > 1,
            copy_from_source=False,
            enable_env_ids=True,
        )
        cloner.disable_change_listener()
        try:

            def expression(path):
                return path.replace(prefix, "/World/envs/env_.*", 1)

            self.robot = self.world.scene.add(
                Articulation(
                    expression(scene["robot"]), name="train_arms", reset_xform_properties=False
                )
            )
            self.can = self.world.scene.add(
                RigidPrim(expression(scene["can"]), name="train_cans", reset_xform_properties=False)
            )
            self.table = self.world.scene.add(
                RigidPrim(
                    expression(scene["table"]), name="train_tables", reset_xform_properties=False
                )
            )
        finally:
            cloner.enable_change_listener()
        progress("initialize PhysX")
        self.world.reset()
        if self.robot.count != num_envs:
            raise ValueError("Assembled articulation replication count mismatch")

        def tensor_ids(names):
            return torch.tensor([self.robot.dof_names.index(n) for n in names], device=self.device)

        self.full_ids = tensor_ids(model.full_names)
        self.active_ids = tensor_ids(model.active_names)
        self.arm_ids = tensor_ids(self.arm.active_names)
        self.palm_id = self.robot.body_names.index(model.root)
        self.base_id = self.robot.body_names.index(self.arm.root)
        self.kp_ids = torch.tensor(
            [self.robot.body_names.index(k["link"]) for k in model.keypoints], device=self.device
        )
        self.kp_local = torch.tensor([k["xyz"] for k in model.keypoints], device=self.device)
        self.tip_ids = torch.tensor(
            [
                model.semantic_names.index(n + "_tip")
                for n in ("thumb", "index", "middle", "ring", "little")
            ],
            device=self.device,
        )
        self.tip_body_ids = self.kp_ids[self.tip_ids]
        self.body_shape_counts = [
            self.world.physics_sim_view.create_rigid_body_view(p).max_shapes
            for p in self.robot._physics_view.link_paths[0]
        ]
        if sum(self.body_shape_counts) != self.robot._physics_view.max_shapes:
            raise ValueError("Unknown arm/hand material shape layout")
        self.robot.set_max_joint_velocities(
            torch.as_tensor(self.arm.velocity, device=self.device, dtype=torch.float32)[
                None
            ].expand(num_envs, -1),
            joint_indices=self.arm_ids,
        )
        self.robot.set_max_joint_velocities(
            torch.full(
                (num_envs, len(model.full_names)),
                config["simulation_joint_velocity_rad_s"],
                device=self.device,
            ),
            joint_indices=self.full_ids,
        )
        self.configure_table_safety()
        apply_startup_randomization(self)
        self.refresh_physics_properties()
        self.obs_history = ArmObservations(
            num_envs,
            config["observation"],
            self.arm,
            self.device,
            self.generator,
            self.arm_cfg["failure_grace_steps"],
        )
        self.obs_history.default_q.copy_(self.default_q_offset)
        self.push = PushSchedule(num_envs, config["push_curriculum"], self.device, self.generator)
        self.ik = BatchedArmIK(
            self.arm,
            IKOptions(**arm_config["solver"]),
            device=self.device,
            iterations=self.arm_cfg["ik_iterations"],
            compile_fk=self.arm_cfg.get("compile_fk", False) and self.device.type == "cuda",
        )
        self.reset_ik = BatchedArmIK(
            self.arm,
            self.ik.options,
            device=self.device,
            iterations=self.arm_cfg["reset_ik_iterations"],
            compile_fk=self.arm_cfg.get("compile_fk", False) and self.device.type == "cuda",
        )
        progress("solve reference reset poses")
        times = np.arange(self.reference_frame_count) * self.task.dt
        anchors = reference.sample(times)
        poses = torch.tensor(
            np.c_[anchors["wrist_position"], anchors["wrist_quaternion"]],
            device=self.device,
            dtype=torch.float32,
        )
        seed = torch.tensor(
            arm_config["seed_q_rad"], device=self.device, dtype=torch.float32
        ).expand(len(poses), 6)
        solved = self.reset_ik.solve(compose(self.base_from_source, poses), seed)
        if not solved["success"].all():
            raise ValueError("Nominal arm RSI reference contains unreachable/singular poses")
        self.reset_q = solved["q"].clone()
        self.metadata = dict(
            robot="RB3-730 + Revo2 trained with online IK",
            num_envs=num_envs,
            physics_device=str(self.device),
            workcell=self.workcell.metadata(),
            arm_training=dict(self.arm_cfg),
            articulation_resolution=scene["resolution"],
            pad_material_binding=scene["pad_material_binding"],
            arm_model_sha256=digest(root / arm_config["arm_model"]),
            assembled_usd_sha256=digest(root / arm_config["usd"]),
            world_from_source=self.world_from_source.tolist(),
            base_from_source=alignment["base_from_source"],
            arm_joint_names=self.arm.active_names,
            body_names=self.robot.body_names,
            dof_names=self.robot.dof_names,
            body_shape_counts=self.body_shape_counts,
            self_collision=config["self_collision"],
            hand_gravity=config["hand_gravity"],
            arm_gravity=True,
            object_kinematic=False,
            physics_dt=config["physics_dt"],
            control_dt=self.task.dt,
            randomization=property_report(self),
            actor_observation_size=self.task.observation_size,
            critic_observation_size=self.task.critic_observation_size,
            root_control="none: only physical arm/finger joint position and velocity targets",
            arm_drives="original USD gains and effort limits; hand-only startup gain randomization",
            command_semantics="actor wrist/finger residual -> tabletop guard -> bounded strict IK -> actual assembled robot",
            arm_observation="hand/object observation plus normalized arm q/qdot, previous IK error, success, failure streak",
            ik_failure="hold previous arm AND finger targets; penalize each failure; terminate at configured consecutive count",
            gpu_dynamics_enabled=bool(scene_api.GetEnableGPUDynamicsAttr().Get()),
            broadphase=scene_api.GetBroadphaseTypeAttr().Get(),
            inter_environment_collision_filter="PhysX environment IDs",
            motion_control=self.motion_controller.config,
            collision_validation="Physical contact response and conservative tabletop guard, not a general noncollision certificate",
        )
        if render:
            from isaacsim.core.utils.viewports import set_camera_view

            set_camera_view(eye=np.array([1.65, -1.9, 1.25]), target=np.array([0.35, 0, 0.05]))
        self.reset(randomize=False)
        progress("environment ready")

    def state(self):
        world = super().state()
        out = self.frame.state(world)
        q = self.robot.get_joint_positions()
        qv = self.robot.get_joint_velocities()
        out.update(
            q_arm=q[:, self.arm_ids].clone(),
            q_arm_velocity=qv[:, self.arm_ids].clone(),
            arm_last_ik_error=self.last_ik_error.clone(),
            arm_last_ik_success=self.last_ik_success.clone(),
            arm_ik_failure_streak=self.ik_failure_streak.clone(),
        )
        return out

    def reset(self, indices=None, randomize=True, times=None):
        ids = (
            self.all_ids
            if indices is None
            else torch.as_tensor(indices, device=self.device, dtype=torch.long)
        )
        if not len(ids):
            return
        learning = self.training and randomize
        times = (
            self.sampler.sample(len(ids), self.generator)
            if times is None and learning
            else (torch.zeros(len(ids), device=self.device) if times is None else times)
        )
        times = torch.as_tensor(times, device=self.device, dtype=torch.float32)
        if (
            times.shape != (len(ids),)
            or not torch.isfinite(times).all()
            or (times < 0).any()
            or (times > self.reference.duration).any()
        ):
            raise ValueError("Invalid arm RSI times")
        if self.training:
            self._apply_gravity(self.gravity.sample(self.control_steps, self.rng, True))
        augment = randomize and (self.training or self.evaluation_protocol == "source")
        self.motion.reset(ids, self.generator, augment)
        frames = (times / self.task.dt).round().long().clamp(0, len(self.reset_q) - 1)
        ref = self.motion.sample(times, ids)
        for attempt in range(4):
            goal = compose(
                self.base_from_source,
                torch.cat((ref["wrist_position"], ref["wrist_quaternion"]), -1),
            )
            result = self.reset_ik.solve(goal, self.reset_q[frames])
            bad = ~result["success"]
            if not bad.any():
                break
            self.reset_ik_rejections += int(bad.sum())
            if not augment or attempt == 3:
                raise ValueError("Arm RSI placement could not pass strict IK")
            # Rejection sampling preserves the chosen RSI frame. A final nominal
            # placement is explicit fallback, counted in reset diagnostics.
            self.motion.reset(ids[bad], self.generator, attempt < 2)
            ref = self.motion.sample(times, ids)
        if self.cfg["observation"].get("canonicalize_translation", False):
            if self.motion.mode != "rigid_sequence" or self.motion.yaw[ids].abs().max() > 1e-8:
                raise ValueError(
                    "Arm observation canonicalization requires translation-only augmentation"
                )
            self.obs_history.position_offset[ids] = self.motion.translation[ids]
        q = ref["q"].clone()
        if learning and self.cfg["reset_perturbation"]:
            if self.cfg["rsi"]["reset_noise_mode"] != "source_joint_only":
                raise ValueError("Arm RSI supports source_joint_only reset noise")
            q = (
                q
                + uniform(
                    q.shape,
                    -self.cfg["rsi"]["joint_noise_rad"],
                    self.cfg["rsi"]["joint_noise_rad"],
                    self.device,
                    self.generator,
                )
            ).clamp(self.task.lower, self.task.upper)
        qa = result["q"]
        full = q @ self.task.coupling.T + self.task.offset
        moving = (
            (times > 1e-8)[:, None].float()
            if self.cfg["rsi"]["initialize_velocity"]
            else torch.zeros((len(ids), 1), device=self.device)
        )
        wrist, flange, jac = self.reset_ik.evaluate(qa)
        jac = jac.clone()
        jac[:, :3] += torch.cross(
            jac[:, 3:].transpose(1, 2),
            (wrist[:, :3] - flange[:, :3])[:, None].expand(-1, 6, -1),
            dim=-1,
        ).transpose(1, 2)
        twist = torch.cat(
            (
                quat_apply(self.base_from_source[3:].expand(len(ids), 4), ref["wrist_velocity"]),
                quat_apply(
                    self.base_from_source[3:].expand(len(ids), 4), ref["wrist_angular_velocity"]
                ),
            ),
            -1,
        )
        qva = (
            torch.linalg.solve(jac, twist).clamp(-self.reset_ik.velocity, self.reset_ik.velocity)
            * moving
        )
        for joint_ids, position, velocity in (
            (self.arm_ids, qa, qva),
            (self.full_ids, full, (ref["q_velocity"] * moving) @ self.task.coupling.T),
        ):
            self.robot.set_joint_positions(position, indices=ids, joint_indices=joint_ids)
            self.robot.set_joint_velocities(velocity, indices=ids, joint_indices=joint_ids)
            self.robot.set_joint_position_targets(position, indices=ids, joint_indices=joint_ids)
        self.robot.set_joint_velocity_targets(qva, indices=ids, joint_indices=self.arm_ids)
        wp = self.to_world.position(ref["object_position"])
        wq = self.to_world.orientation(ref["object_quaternion"])
        self.can.set_world_poses(wp + self.origins[ids], wq[:, [3, 0, 1, 2]], indices=ids)
        omega = self.to_world.rotate(ref["object_angular_velocity"])
        velocity = self.to_world.rotate(ref["object_velocity"])
        velocity += torch.cross(omega, quat_apply(wq, self.can_com_local[ids]), dim=-1)
        self.can.set_velocities(torch.cat((velocity, omega), -1) * moving, indices=ids)
        self.time[ids] = times
        self.episode_length_buf[ids] = 0
        self.last_reset_frame[ids] = frames
        self.last_action[ids] = 0
        self.last_q_target[ids] = q
        self.last_arm_target[ids] = qa
        self.last_ik_error[ids] = 0
        self.last_ik_success[ids] = True
        self.ik_failure_streak[ids] = 0
        self.object_reset_count += len(ids)
        self.world.physics_sim_view.update_articulations_kinematic()
        self.push.reset(ids)
        state = self.state()
        # Verify actual replicated fixed-base positions before any learning.
        raw = self.robot._physics_view.get_link_transforms()[:, self.base_id, :3] - self.origins
        expected = torch.as_tensor(
            self.workcell.world_from_base[:3, 3], device=self.device, dtype=torch.float32
        )
        if not torch.allclose(raw[ids], expected.expand(len(ids), 3), atol=1e-5, rtol=0):
            raise RuntimeError("Cloned RB3 bases moved or fixed-joint anchors are wrong")
        measured = compose(
            self.base_from_source,
            torch.cat((state["wrist_position"][ids], state["wrist_quaternion"][ids]), -1),
        )
        error = self.reset_ik.error(measured, wrist)
        if (error[:, :3].norm(dim=-1) > 1e-4).any() or (error[:, 3:].norm(dim=-1) > 5e-4).any():
            raise RuntimeError("Physical assembled wrist disagrees with extracted FK at reset")
        self.motion_controller.reset(ids, state)
        self.obs_history.update(
            state, ids, reset=True, noisy=self.training or self.evaluation_protocol == "source"
        )
        self.obs_history.assemble(
            state, self.motion.sample(self.time), self.last_action, self.phase(), ids
        )

    def step(self, actions, auto_reset=True):
        started = time.monotonic()
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if actions.shape != (self.num_envs, 12) or not torch.isfinite(actions).all():
            raise ValueError("Invalid residual actions")
        ref = self.motion.sample(self.time)
        before = self.state()
        raw = self.task.targets(ref, actions, self.last_q_target, self.default_q_offset)
        target = raw
        table = self.table_safety
        if table is not None:
            table.begin(raw, before)
            target = table.guard(raw, before)
        goal = compose(
            self.base_from_source, torch.cat((target["position"], target["quaternion"]), -1)
        )
        result = self.ik.solve(
            goal, self.last_arm_target, previous=self.last_arm_target, dt=self.task.dt
        )
        valid = result["success"]
        qa = result["q"]
        arm_step = (qa - self.last_arm_target).abs().amax(-1)
        self.last_ik_error[:, :3] = self.from_base.rotate(result["error"][:, :3])
        self.last_ik_error[:, 3:] = self.from_base.rotate(result["error"][:, 3:])
        self.last_ik_success.copy_(valid)
        self.ik_failure_streak = torch.where(valid, 0, self.ik_failure_streak + 1)
        self.last_arm_target.copy_(qa)
        self.last_q_target.copy_(
            torch.where(valid[:, None], target["active_q"], self.last_q_target)
        )
        full = self.last_q_target @ self.task.coupling.T + self.task.offset
        self.robot.set_joint_position_targets(qa, joint_indices=self.arm_ids)
        self.robot.set_joint_velocity_targets(result["velocity"], joint_indices=self.arm_ids)
        self.robot.set_joint_position_targets(full, joint_indices=self.full_ids)
        for _ in range(self.cfg["control_decimation"]):
            self.world.step(render=False)
            if table is not None:
                table.observe(self.state())
        self.episode_length_buf += 1
        state = self.state()
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
        )
        actual, _, jac = self.ik.evaluate(state["q_arm"])
        sigma, condition = self.ik.singularity(jac)
        tracking_p = (state["wrist_position"] - target["position"]).norm(dim=-1)
        tracking_r = rotation_error(target["quaternion"], state["wrist_quaternion"]).norm(dim=-1)
        margin = torch.minimum(state["q_arm"] - self.ik.lower, self.ik.upper - state["q_arm"]).amin(
            -1
        )
        c = self.arm_cfg["reward"]
        terms = dict(
            arm_ik=-c["ik_weight"]
            * (
                (result["position_error_m"] / c["ik_position_std_m"]).clamp_max(3).square()
                + (result["orientation_error_rad"] / c["ik_rotation_std_rad"]).clamp_max(3).square()
            ),
            arm_tracking=-c["tracking_weight"]
            * (
                (tracking_p / c["tracking_position_std_m"]).clamp_max(3).square()
                + (tracking_r / c["tracking_rotation_std_rad"]).clamp_max(3).square()
            ),
            arm_ik_failure=-c["ik_failure_weight"] * (~valid).float(),
            arm_singularity=-c["singularity_weight"]
            * (1 - sigma / self.ik.options.near_sigma_min).clamp_min(0).square(),
            arm_limit=-c["limit_weight"]
            * (1 - margin / self.ik.options.limit_warning_rad).clamp(0, 2).square(),
        )
        arm_failure = (
            (self.ik_failure_streak >= self.arm_cfg["failure_grace_steps"])
            | (sigma < self.ik.options.singular_sigma_min)
            | (condition > self.ik.options.max_condition)
            | (margin < -0.001)
        )
        terms["arm_termination"] = (
            self.cfg["reward"]["early_failure_weight"]
            * (arm_failure & ~metrics["early_failure"] & ~demo_end).float()
        )
        reward += sum(terms.values()) * self.task.dt
        term |= arm_failure
        metrics["early_failure"] |= arm_failure
        measured_source = torch.cat((state["wrist_position"], state["wrist_quaternion"]), -1)
        measured_base = compose(self.base_from_source, measured_source)
        consistency = self.ik.error(measured_base, actual)
        metrics.update(
            arm_ik_success=valid,
            arm_ik_position_error_m=result["position_error_m"],
            arm_ik_orientation_error_rad=result["orientation_error_rad"],
            arm_ik_failure_streak=self.ik_failure_streak.clone(),
            arm_constraint_failure=arm_failure,
            arm_joint_step_rad=arm_step,
            arm_joint_tracking_error_rad=(state["q_arm"] - qa).abs().amax(-1),
            arm_sigma_min=result["sigma_min"],
            arm_condition=result["condition"],
            arm_actual_sigma_min=sigma,
            arm_actual_condition=condition,
            arm_joint_limit_distance_rad=margin,
            arm_joint_limit_violation_rad=(-margin).clamp_min(0),
            arm_joint_velocity_violation_rad_s=(state["q_arm_velocity"].abs() - self.ik.velocity)
            .clamp_min(0)
            .amax(-1),
            arm_tracking_position_error_m=tracking_p,
            arm_tracking_orientation_error_rad=tracking_r,
            arm_fk_consistency_position_error_m=consistency[:, :3].norm(dim=-1),
            arm_fk_consistency_orientation_error_rad=consistency[:, 3:].norm(dim=-1),
            gravity_m_s2=torch.full((self.num_envs,), self.gravity.value, device=self.device),
            **{"reward_" + name: value for name, value in terms.items()},
        )
        if self.training:
            self.control_steps += 1
            self.sampler.record(self.time, term, demo_end, self.num_envs, self.max_episode_length)
        self.last_action.copy_(actions)
        self.obs_history.update(
            state, self.all_ids, noisy=self.training or self.evaluation_protocol == "source"
        )
        next_time = (self.time + self.task.dt).clamp(max=self.reference.duration)
        final_obs = self.obs_history.assemble(
            state, self.motion.sample(next_time), self.last_action, self.phase_at(next_time)
        )
        accepted_pose, _, _ = self.ik.evaluate(qa)
        applied = dict(
            target,
            position=self.from_base.position(accepted_pose[:, :3]),
            quaternion=self.from_base.orientation(accepted_pose[:, 3:]),
            active_q=self.last_q_target.clone(),
            full_q=full,
            q_arm=qa.clone(),
            q_arm_velocity=result["velocity"],
        )
        info = dict(
            state=state,
            reference=ref,
            raw_targets=raw,
            applied_targets=applied,
            metrics=metrics,
            final_observation=final_obs,
            object_reset_count=self.object_reset_count,
            reference_time=self.time.clone(),
        )
        self.time = next_time
        done = torch.nonzero(term | trunc, as_tuple=False).flatten()
        if auto_reset and len(done):
            self.reset(done, randomize=True)
            if self.cfg.get("advance_command_after_auto_reset", False):
                self.time[done] = (self.time[done] + self.task.dt).clamp(
                    max=self.reference.duration
                )
        self.push.advance(self)
        self.obs_history.assemble(
            self.state() if auto_reset and len(done) else state,
            self.motion.sample(self.time),
            self.last_action,
            self.phase(),
        )
        if self.render:
            self.world.render()
            time.sleep(max(0, self.task.dt - (time.monotonic() - started)))
        return self.observation(), reward, term, trunc, info
