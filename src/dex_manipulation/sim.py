"""Isaac boundary adapter; pure IK/reference modules do not import this module."""

from .configuration import read_config
import numpy as np


def prepare_arm_articulation(stage, assembly_path, base_path, wrist_path):
    """Keep the base's root and merge only joint-connected imported hand roots.

    Isaac's registered NewtonArticulationRootAPI includes PhysicsArticulationRootAPI
    implicitly. A standalone USD reader can therefore see fewer roots than Isaac.
    Author overrides in the live stage only; never edit referenced asset layers.
    """
    from pxr import Usd, UsdPhysics

    assembly = stage.GetPrimAtPath(assembly_path)
    base = stage.GetPrimAtPath(base_path)
    wrist = stage.GetPrimAtPath(wrist_path)
    if not assembly or not base or not wrist:
        raise ValueError("Missing assembly/base/wrist prim for articulation resolution")
    if not base.HasAPI(UsdPhysics.RigidBodyAPI) or not wrist.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError("Configured base/wrist must be actual USD rigid bodies")

    def root_schemas(prim):
        authored = prim.GetMetadata("apiSchemas")
        schemas = set(prim.GetAppliedSchemas())
        if authored is not None:
            schemas.update(authored.GetAppliedItems())
        return schemas & {"PhysicsArticulationRootAPI", "NewtonArticulationRootAPI"}

    prims = list(Usd.PrimRange(assembly))
    roots = [p for p in prims if root_schemas(p)]
    ancestors = [p for p in roots if base.GetPath().HasPrefix(p.GetPath())]
    if len(ancestors) != 1:
        raise ValueError(
            f"Expected one articulation ancestor of {base_path}; found {[str(p.GetPath()) for p in ancestors]}"
        )
    chosen = ancestors[0]
    bodies = {str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    graph = {p: set() for p in bodies}

    def rigid_owner(targets):
        if not targets:
            return None
        if len(targets) != 1:
            raise ValueError("Joint body relationship must have at most one target")
        prim = stage.GetPrimAtPath(targets[0])
        while prim and prim.GetPath().HasPrefix(assembly.GetPath()):
            if str(prim.GetPath()) in bodies:
                return str(prim.GetPath())
            prim = prim.GetParent()
        return None  # World/static anchor, not another articulated rigid body.

    anchored = False
    for prim in prims:
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        if (
            joint.GetJointEnabledAttr().Get() is False
            or joint.GetExcludeFromArticulationAttr().Get()
        ):
            continue
        a, b = (
            rigid_owner(joint.GetBody0Rel().GetTargets()),
            rigid_owner(joint.GetBody1Rel().GetTargets()),
        )
        if a is not None and b is not None:
            graph[a].add(b)
            graph[b].add(a)
        elif prim.IsA(UsdPhysics.FixedJoint) and str(base.GetPath()) in (a, b):
            anchored = True
    connected, pending = set(), [str(base.GetPath())]
    while pending:
        body = pending.pop()
        if body not in connected:
            connected.add(body)
            pending.extend(graph[body] - connected)
    if not anchored or str(wrist.GetPath()) not in connected:
        raise ValueError(
            "Expected a world-fixed arm base with an enabled joint path to the mounted hand"
        )
    removed = []
    # Validate every extra root before mutating any schema.
    for prim in roots:
        if prim == chosen:
            continue
        members = {p for p in bodies if stage.GetPrimAtPath(p).GetPath().HasPrefix(prim.GetPath())}
        if not members or not members.issubset(connected):
            raise ValueError(f"Refusing to merge disconnected articulation root: {prim.GetPath()}")
        removed.append(prim)
    for prim in removed:
        for schema in (
            "NewtonArticulationRootAPI",
            "PhysicsArticulationRootAPI",
            "PhysxArticulationAPI",
        ):
            prim.RemoveAppliedSchema(schema)
    remaining = [str(p.GetPath()) for p in prims if root_schemas(p)]
    if remaining != [str(chosen.GetPath())]:
        raise ValueError(f"Articulation root resolution did not converge: {remaining}")
    return chosen, dict(
        root=str(chosen.GetPath()),
        roots_before=[str(p.GetPath()) for p in roots],
        removed_connected_roots=[str(p.GetPath()) for p in removed],
        connected_rigid_bodies=len(connected),
        fixed_base=anchored,
    )


class IsaacJointAdapter:
    def __init__(self, articulation, arm, hand):
        self.articulation, self.arm, self.hand = articulation, arm, hand
        self.command_names = arm.active_names + hand.full_names
        self.indices = np.array([articulation.dof_names.index(name) for name in self.command_names])

    def positions(self, target):
        arm = target.ordered(self.arm.active_names)
        finger = self.hand.expand(target.ordered(self.hand.active_names))
        return np.r_[arm, finger]

    def set_target(self, target):
        """Position-control boundary; keeps USD gains. Tracking is a separate concern."""
        from isaacsim.core.utils.types import ArticulationAction

        self.articulation.apply_action(
            ArticulationAction(joint_positions=self.positions(target), joint_indices=self.indices)
        )

    def set_state(self, target):
        """Kinematic playback/FK verification only, not drive-tracking validation."""
        self.articulation.set_joint_positions(self.positions(target), joint_indices=self.indices)
        self.articulation.set_joint_velocities(
            np.zeros(len(self.indices)), joint_indices=self.indices
        )


def replay_floating(
    root,
    output,
    loops=3,
    render=True,
    *,
    observer=None,
    config_path=None,
    arm_config_path=None,
    is_running=None,
    speed=1.0,
):
    """Physical floating-hand demonstration: zero residual, no learner/early reset.

    Use the REGRIND pose-PD controller shared with RL: root force and distributed
    link torque, with hand gravity disabled by config. The can retains gravity
    and contact. Only startup/reset assigns body/joint poses directly.
    """
    import json
    import time
    from itertools import count
    from pathlib import Path
    import torch
    from .fk import HandModel
    from .policy.trajectory import ReferenceMotion
    from .floating_env import PhysxResidualEnv
    from .coordinates import collision_bottom
    from .transforms import transform
    from scipy.spatial.transform import Rotation

    if loops < 0:
        raise ValueError("loops must be nonnegative; 0 repeats until stopped")
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = read_config(Path(config_path or root / "config/tasks/can_pick/policy_demo2.json"))
    # This command previews geometric retargeting, like the arm preview. A
    # separately prepared policy contact reference belongs to policy.py.
    cfg.pop("table_safety", None)  # Policy safety is separate from raw retargeting diagnostics.
    cfg["reference"] = read_config(
        Path(arm_config_path or root / "config/tasks/can_pick/ik_demo2.json")
    )["input"]
    for name in (
        "rsi",
        "augmentation",
        "gravity_curriculum",
        "domain_randomization",
        "push_curriculum",
    ):
        cfg[name]["enabled"] = False
    # Respect hand_gravity=False, as in REGRIND's floating-hand asset config.
    # Disabling the curriculum below sets normal world/object gravity for PLAY.
    cfg["reset_perturbation"] = False
    model = HandModel.load(root / cfg["model"])
    reference = ReferenceMotion(
        root / cfg["reference"], model, root / cfg["object_geometry"], cfg["world_frame"]
    )
    from .policy.playback import prepare_playback

    reference, cfg, playback_timing = prepare_playback(reference, cfg, speed)
    # Preserve the current input segment's timing rather than invoking RL's retiming.
    cfg["episode_length_s"] = reference.duration
    if cfg["augmentation"].get("mode", "fade") == "fade":
        cfg["augmentation"]["blend_start_s"] = (
            reference.duration * cfg["augmentation"]["phase_start_ratio"]
        )
        cfg["augmentation"]["blend_end_s"] = (
            reference.duration * cfg["augmentation"]["phase_end_ratio"]
        )
    torch.set_num_threads(1)
    env = PhysxResidualEnv(root, model, reference, cfg, num_envs=1, render=render)
    rows = []
    failures = []
    initial_bottoms = []
    physics_steps = []
    running = is_running or (lambda: True)
    try:
        env.set_training(False)
        for loop in range(loops) if loops else count():
            if not running():
                break
            env.reset(randomize=False)
            initial = env.state()
            if observer is not None:
                observer(env, initial, 0.0, loop, "reset")
            pose = transform(
                Rotation.from_quat(initial["object_quaternion"][0].cpu().numpy()).as_matrix(),
                initial["object_position"][0].cpu().numpy(),
            )
            initial_bottoms.append(collision_bottom(pose, reference.collision_shapes))
            if abs(initial_bottoms[-1]) > 1e-6:
                raise RuntimeError("Initial can is not on the Z=0 tabletop")
            reset_count = env.object_reset_count
            start_physics_step = env.world.current_time_step_index
            first_failure = None
            elapsed_physics_steps = 0
            started = time.monotonic()
            for step in range(int(np.ceil(reference.duration / env.task.dt)) + 1):
                if not running():
                    break
                _, _, terminated, truncated, info = env.step(
                    torch.zeros((1, 12), device=env.device), auto_reset=False
                )
                elapsed_physics_steps = env.world.current_time_step_index - start_physics_step
                if elapsed_physics_steps != (step + 1) * cfg["control_decimation"]:
                    raise RuntimeError(
                        "Physics/control clock mismatch: rendering must not advance extra physics"
                    )
                if env.object_reset_count != reset_count:
                    raise RuntimeError("Unexpected object reset during demonstration")
                state = info["state"]
                if not all(torch.isfinite(value).all() for value in state.values()):
                    raise RuntimeError("Nonfinite simulated state")
                if bool(info["metrics"]["early_failure"][0]) and first_failure is None:
                    first_failure = step
                if loop == 0:
                    # State is observed AFTER physics; compare it with that time,
                    # while preserving the controller's pre-step target separately.
                    time_s = float(env.time[0])
                    sampled = reference.sample(time_s)
                    rows.append(
                        dict(
                            timestamp_s=time_s,
                            command_time_s=float(info["reference_time"][0]),
                            physics_time_s=elapsed_physics_steps * cfg["physics_dt"],
                            q_finger=state["q"][0].cpu().numpy(),
                            wrist_position=state["wrist_position"][0].cpu().numpy(),
                            wrist_quaternion=state["wrist_quaternion"][0].cpu().numpy(),
                            object_position=state["object_position"][0].cpu().numpy(),
                            object_quaternion=state["object_quaternion"][0].cpu().numpy(),
                            reference_object_position=sampled["object_position"][0],
                            reference_wrist_position=sampled["wrist_position"][0],
                            reference_wrist_quaternion=sampled["wrist_quaternion"][0],
                        )
                    )
                if observer is not None:
                    observer(env, state, float(env.time[0]), loop, "step")
                if render:
                    delay = started + (step + 1) * env.task.dt - time.monotonic()
                    if delay > 0:
                        time.sleep(min(delay, env.task.dt))
            failures.append(first_failure)
            physics_steps.append(elapsed_physics_steps)
        if not rows:
            (output / "report.json").write_text(
                json.dumps(dict(robot="floating", samples=0, interrupted=True)) + "\n"
            )
            return
        np.savez_compressed(
            output / "measurements.npz", **{k: np.asarray([r[k] for r in rows]) for k in rows[0]}
        )
        errors = np.array([row["wrist_position"] - row["reference_wrist_position"] for row in rows])
        angles = np.array(
            [
                (
                    Rotation.from_quat(row["wrist_quaternion"])
                    * Rotation.from_quat(row["reference_wrist_quaternion"]).inv()
                ).magnitude()
                for row in rows
            ]
        )
        report = dict(
            robot="floating",
            samples=len(rows),
            loops=len(failures),
            requested_loops=loops,
            duration_s=reference.duration,
            playback_timing=playback_timing,
            reference=cfg["reference"],
            interrupted=not running(),
            object_geometry_fingerprint=reference.metadata["object_geometry_fingerprint"],
            gravity_m_s2=env.gravity.value,
            hand_gravity=cfg["hand_gravity"],
            object_gravity=True,
            object_kinematic=False,
            object_pose_writes_during_demonstration=0,
            continue_after_grasp_failure=True,
            first_failure_step_by_loop=failures,
            policy_loaded=False,
            optimizer_updates=0,
            policy_table_safety_applied=False,
            initial_can_bottom_z_m_by_loop=initial_bottoms,
            tabletop_z_m=0.0,
            floor_top_z_m=None,
            surface=env.surface.metadata(),
            physics_steps_by_loop=physics_steps,
            physics_duration_s_by_loop=[n * cfg["physics_dt"] for n in physics_steps],
            rsi=False,
            augmentation=False,
            randomization=False,
            pushes=False,
            controller=env.metadata["controller"],
            motion_control=env.motion_controller.config,
            wrist_tracking=dict(
                mean_position_error_m=float(np.linalg.norm(errors, axis=1).mean()),
                max_position_error_m=float(np.linalg.norm(errors, axis=1).max()),
                mean_z_error_m=float(errors[:, 2].mean()),
                max_abs_z_error_m=float(np.abs(errors[:, 2]).max()),
                max_orientation_error_rad=float(angles.max()),
            ),
            control="REGRIND pose PD: root force + mass-proportional link torques; zero residual, configured gains",
            interpretation="Full physical motion preview; no claim of grasp or accurate tracking",
            physics=env.metadata,
        )
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            "[physics]", json.dumps({k: v for k, v in report.items() if k != "physics"}), flush=True
        )
    finally:
        env.close()
