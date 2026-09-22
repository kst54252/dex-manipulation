#!/usr/bin/env python3
"""Replay a validated arm+hand joint reference in Isaac Sim (kinematic by default)."""

import argparse
from itertools import count
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dex_manipulation.configuration import read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=ROOT / "local/results/ik/full/trajectory.npz"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/tasks/can_pick/ik_demo2.json")
    parser.add_argument(
        "--physics-config",
        type=Path,
        default=ROOT / "config/tasks/can_pick/policy_demo2.json",
        help="Localized fingertip contact settings for physical target replay",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "local/results/ik/replay")
    parser.add_argument("--mode", choices=("kinematic", "targets"), default="kinematic")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rate", type=float, default=60.0)
    parser.add_argument("--loops", type=int, default=1, help="Repetitions; 0 repeats until stopped")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument(
        "--capture", action="store_true", help="Save one viewport PNG per original reference frame"
    )
    args = parser.parse_args()
    if args.rate <= 0 or args.loops < 0:
        parser.error("rate must be positive; loops must be nonnegative (0 repeats until stopped)")
    if args.capture and args.headless:
        parser.error("--capture requires a visible viewport")
    from dex_manipulation.joint_trajectory import JointReference

    reference = JointReference(args.reference)  # Refuse failures before starting Isaac.
    config = read_config(args.config)
    physics_config = read_config(args.physics_config) if args.mode == "targets" else {}
    from dex_manipulation.scene import Workcell

    workcell = Workcell.load(ROOT / config["workcell"])
    workcell.validate_reference(reference.metadata, read_config(ROOT / config["alignment"]))
    args.output.mkdir(parents=True, exist_ok=True)
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": args.headless,
            "multi_gpu": False,
            "enable_crashreporter": False,
            "renderer": "RaytracedLighting",
            "width": 1280,
            "height": 900,
            "extra_args": ["--enable", "isaacsim.core.api", "--enable", "isaacsim.core.prims"],
        }
    )
    code = 0
    try:
        import numpy as np
        from scipy.spatial.transform import Rotation, Slerp
        from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, PhysxSchema, Gf
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, RigidPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.viewports import set_camera_view
        from dex_manipulation.fk import ArmModel, HandModel
        from dex_manipulation.ik import pose_error, model_fingerprint
        from dex_manipulation.transforms import transform, inverse
        from dex_manipulation.sim import IsaacJointAdapter, prepare_arm_articulation
        from dex_manipulation.coordinates import collision_bottom

        can_geometry = read_config(ROOT / "assets/models/can_mesh.json")
        can_shapes = can_geometry["collision_shapes"]
        if reference.metadata.get("object_geometry_fingerprint") != can_geometry.get("fingerprint"):
            raise ValueError(
                "Arm reference uses stale object geometry. Regenerate grounding and IK."
            )
        import hashlib

        if (
            can_geometry.get("collision_source_sha256")
            != hashlib.sha256((ROOT / can_geometry["collision_source"]).read_bytes()).hexdigest()
        ):
            raise ValueError(
                "Object USD changed since model extraction. Rebuild object and reference."
            )
        arm = ArmModel.load(ROOT / config["arm_model"])
        hand = HandModel.load(ROOT / config["hand_model"])
        if (
            model_fingerprint(arm) != reference.metadata["arm_fingerprint"]
            or model_fingerprint(hand) != reference.metadata["hand_fingerprint"]
        ):
            raise ValueError("Reference arm/mount/hand model differs from replay models")
        import hashlib

        if (
            hashlib.sha256((ROOT / config["usd"]).read_bytes()).hexdigest()
            != arm.description["source_sha256"]
        ):
            raise ValueError("USD changed since arm extraction; extract and solve again")
        for path, digest in arm.description["layer_sha256"].items():
            # Extraction provenance contains resolved layer paths. Current workspace
            # extraction is required after relocating/replacing the USD assembly.
            if (
                not Path(path).exists()
                or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest
            ):
                raise ValueError("Composed USD layer changed/missing since arm extraction")
        world = World(
            stage_units_in_meters=1.0, physics_dt=1 / args.rate, rendering_dt=1 / args.rate
        )
        world.get_physics_context().set_gravity(0.0 if args.mode == "kinematic" else -9.81)
        add_reference_to_stage(str(ROOT / config["usd"]), "/Robot")
        stage = world.stage
        if args.mode == "targets":
            # Use the same verified contact solver as the floating/arm policy
            # environments, including external forces on every TGS iteration.
            world.get_physics_context().set_solver_type("TGS")
            solver = physics_config.get("solver_iterations", {})
            api = PhysxSchema.PhysxSceneAPI.Apply(
                world.get_physics_context().get_current_physics_scene_prim()
            )
            api.CreateMaxPositionIterationCountAttr(solver.get("scene_position", 64))
            api.CreateMaxVelocityIterationCountAttr(solver.get("scene_velocity", 4))
            for name, value in physics_config.get("scene_physics", {}).items():
                api.GetPrim().GetAttribute("physxScene:" + name).Set(value)
        # USD /World is mapped to /Robot by the reference's default prim.
        source_stage = Usd.Stage.Open(str(ROOT / config["usd"]))
        source_root = str(source_stage.GetDefaultPrim().GetPath())
        base_path = "/Robot" + config["base_path"][len(source_root) :]
        workcell.mount_robot(stage, "/Robot", base_path)
        world_from_base = workcell.world_from_base
        UsdLux.DomeLight.Define(stage, "/Light").CreateIntensityAttr(800)
        workcell.create_usd(stage, collision=args.mode == "targets")
        for prim in stage.Traverse():
            if args.mode == "targets" and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(False)
            if prim.HasAPI(UsdPhysics.CollisionAPI) and args.mode == "kinematic":
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            if prim.IsA(UsdPhysics.RevoluteJoint) and args.mode == "kinematic":
                drive = UsdPhysics.DriveAPI(prim, "angular")
                if drive:
                    drive.CreateStiffnessAttr(0.0)
                    drive.CreateDampingAttr(0.0)
            if prim.IsA(UsdGeom.Sphere) and "/kp_" in str(prim.GetPath()):
                UsdGeom.Imageable(prim).CreateVisibilityAttr("invisible")
        mimic_count = 0
        if args.mode == "targets":
            # Use one explicit coupling representation. Isaac Sim 6 also reads
            # native Newton mimic, so retaining it would duplicate constraints.
            joints = {p.GetName(): p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
            for joint in hand.moving:
                if not joint.get("mimic"):
                    continue
                spec = joint["mimic"]
                prim = joints[joint["name"]]
                leader = joints[spec["leader"]]
                prim.RemoveAppliedSchema("NewtonMimicAPI")
                axis = "rot" + str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get())
                api = PhysxSchema.PhysxMimicJointAPI.Apply(prim, axis)
                api.CreateReferenceJointRel().SetTargets([leader.GetPath()])
                api.CreateReferenceJointAxisAttr(
                    "rot" + str(UsdPhysics.RevoluteJoint(leader).GetAxisAttr().Get())
                )
                api.CreateGearingAttr(-spec["multiplier"])
                api.CreateOffsetAttr(-np.rad2deg(spec["offset"]))
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateStiffnessAttr(0.0)
                drive.CreateDampingAttr(0.0)
                mimic_count += 1
        wrist_path = "/Robot" + config["wrist_path"][len(source_root) :]
        root, articulation_resolution = prepare_arm_articulation(
            stage, "/Robot", base_path, wrist_path
        )
        if args.mode == "targets":
            api = PhysxSchema.PhysxArticulationAPI.Apply(root)
            api.CreateSolverPositionIterationCountAttr(solver.get("hand_position", 32))
            api.CreateSolverVelocityIterationCountAttr(solver.get("hand_velocity", 2))
        robot = world.scene.add(SingleArticulation(prim_path=str(root.GetPath()), name="rb3_revo2"))
        object_data = reference.data.get("object_transform")
        can_ops = None
        dynamic_can = None
        initial_can_pose = None
        if object_data is not None:
            if args.mode == "targets":
                add_reference_to_stage(str(ROOT / can_geometry["collision_source"]), "/DynamicCan")
                initial_can_pose = world_from_base @ object_data[0]
                quat = Rotation.from_matrix(initial_can_pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
                dynamic_can = world.scene.add(
                    RigidPrim(
                        "/DynamicCan",
                        name="free_can",
                        positions=initial_can_pose[None, :3, 3],
                        orientations=quat[None],
                    )
                )
            else:
                add_reference_to_stage(str(ROOT / can_geometry["mesh_source"]), "/ReferenceCan")
                for prim in Usd.PrimRange(stage.GetPrimAtPath("/ReferenceCan")):
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
                    if prim.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                can = UsdGeom.Xformable(stage.GetPrimAtPath("/ReferenceCan"))
                can.ClearXformOpOrder()
                can_ops = can.AddTransformOp()
                object_rotation = Slerp(
                    reference.times, Rotation.from_matrix(object_data[:, :3, :3])
                )
        pad_binding = None
        if args.mode == "targets":
            from dex_manipulation.materials import bind_pad_material

            pad_binding = bind_pad_material(stage, "/Robot", physics_config, "/Revo2PadMaterial")
            (args.output / "contact_materials.json").write_text(
                json.dumps(pad_binding, indent=2) + "\n"
            )
        world.reset()
        adapter = IsaacJointAdapter(robot, arm, hand)
        tensor = robot._articulation_view._physics_view
        bodies = robot._articulation_view.body_names
        if arm.wrist not in bodies:
            raise ValueError(f"Mounted hand is not in RB3 articulation: {bodies}")
        set_camera_view(eye=np.array([1.65, -1.9, 1.25]), target=np.array([0.35, 0, -0.05]))
        if args.capture:
            (args.output / "frames").mkdir(exist_ok=True)
            from omni.kit.viewport.utility import (
                get_active_viewport,
                get_active_viewport_window,
                capture_viewport_to_file,
            )
            import asyncio

            viewport = get_active_viewport()
            viewport.updates_enabled = True
            get_active_viewport_window().visible = True
            viewport.resolution = (1280, 900)
        targets = list(reference.iter_targets(args.rate))
        source_mask = reference.data.get("source_frame_mask", np.ones(len(reference.times), bool))
        records = []
        frame_records = []
        initial_bottoms = []
        maximum_base_error = 0.0
        # Initialization is explicitly not a trajectory from the robot's current state.
        adapter.set_state(targets[0])
        adapter.set_target(targets[0])
        if args.mode == "kinematic":
            for _ in range(5):
                world.step(render=not args.headless)
        started = time.monotonic()
        for loop in range(args.loops) if args.loops else count():
            if not app.is_running():
                break
            if args.mode == "targets":
                # Reset only between complete demonstrations, never on grasp failure.
                adapter.set_state(targets[0])
                adapter.set_target(targets[0])
                if dynamic_can is not None:
                    quat = Rotation.from_matrix(initial_can_pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
                    dynamic_can.set_world_poses(initial_can_pose[None, :3, 3], quat[None])
                    dynamic_can.set_velocities(np.zeros((1, 6), np.float32))
                    pos, orientation = dynamic_can.get_world_poses()
                    initial = transform(
                        Rotation.from_quat(orientation[0][[1, 2, 3, 0]]).as_matrix(), pos[0]
                    )
                    initial_bottoms.append(collision_bottom(initial, can_shapes))
                    if (
                        reference.metadata.get("alignment_status")
                        == "simulation_ground_placement_not_calibration"
                        and abs(initial_bottoms[-1]) > 1e-6
                    ):
                        raise RuntimeError("Initial can is not on the Z=0 tabletop")
            for index, target in enumerate(targets):
                if not app.is_running():
                    break
                if args.mode == "kinematic":
                    adapter.set_state(target)
                else:
                    adapter.set_target(target)
                if can_ops is not None:
                    pos = np.array(
                        [
                            np.interp(target.timestamp_s, reference.times, object_data[:, i, 3])
                            for i in range(3)
                        ]
                    )
                    pose = transform(object_rotation([target.timestamp_s]).as_matrix()[0], pos)
                    pose = world_from_base @ pose
                    can_ops.Set(Gf.Matrix4d(pose.T.tolist()))
                world.step(render=not args.headless)
                poses = np.asarray(tensor.get_link_transforms())[0]

                def body(name):
                    p = poses[bodies.index(name)]
                    return transform(Rotation.from_quat(p[3:7]).as_matrix(), p[:3])

                measured_base = body(arm.root)
                base_error = float(np.max(np.abs(measured_base - world_from_base)))
                maximum_base_error = max(maximum_base_error, base_error)
                if base_error > 1e-5:
                    raise RuntimeError("Simulated RB3 base differs from workcell mount")
                measured = inverse(measured_base) @ body(arm.wrist)
                expected = arm.pose(target.ordered(arm.active_names))
                error = pose_error(expected, measured)
                commanded = adapter.positions(target)
                actual = robot.get_joint_positions()[adapter.indices]
                if not np.isfinite(actual).all() or not np.isfinite(measured).all():
                    raise RuntimeError("Non-finite simulated robot state")
                row = dict(
                    time_s=target.timestamp_s,
                    position_error_m=float(np.linalg.norm(error[:3])),
                    orientation_error_rad=float(np.linalg.norm(error[3:])),
                    joint_error_rad=float(np.max(np.abs(commanded - actual))),
                )
                if dynamic_can is not None:
                    position = dynamic_can.get_world_poses()[0][0]
                    if not np.isfinite(position).all():
                        raise RuntimeError("Non-finite can state")
                    row.update(
                        can_x_m=float(position[0]),
                        can_y_m=float(position[1]),
                        can_z_m=float(position[2]),
                    )
                if loop == 0:
                    records.append(row)
                    matches = np.flatnonzero(
                        np.isclose(reference.times, target.timestamp_s, atol=1e-8)
                    )
                    matches = matches[source_mask[matches]]
                    if len(matches):
                        i = int(matches[0])
                        e = pose_error(reference.data["wrist_target_pose"][i], measured)
                        frame_records.append(
                            dict(
                                frame_id=int(reference.data["frame_ids"][i]),
                                position_error_m=float(np.linalg.norm(e[:3])),
                                orientation_error_rad=float(np.linalg.norm(e[3:])),
                            )
                        )
                        if args.capture:
                            for _ in range(3):
                                world.render()
                            file = (
                                args.output
                                / "frames"
                                / f"frame_{int(reference.data['frame_ids'][i]):06d}.png"
                            )
                            file.unlink(missing_ok=True)
                            capture = capture_viewport_to_file(viewport, str(file.resolve()))
                            future = asyncio.ensure_future(capture.wait_for_result())
                            deadline = time.monotonic() + 20.0
                            while time.monotonic() < deadline:
                                world.render()
                                # The capture delegate can finish before renderer
                                # file I/O; wait for both rather than closing Kit early.
                                if future.done():
                                    future.result()
                                    if file.exists() and file.stat().st_size > 8:
                                        break
                                time.sleep(0.01)
                            if not future.done() or not file.exists():
                                raise RuntimeError("Viewport capture did not complete")
                if args.realtime:
                    deadline = (
                        started
                        + loop * (reference.times[-1] - reference.times[0])
                        + target.timestamp_s
                        - reference.times[0]
                    )
                    delay = deadline - time.monotonic()
                    if delay > 0:
                        time.sleep(min(delay, 1 / args.rate))
            if not app.is_running():
                break
        if not records:
            raise RuntimeError("No frames replayed")
        success = (
            len(records) == len(targets)
            and len(frame_records) == int(source_mask.sum())
            and all(
                r["position_error_m"] < 3e-5
                and r["orientation_error_rad"] < 3e-4
                and r["joint_error_rad"] < 1e-5
                for r in records
            )
        )
        report = dict(
            mode=args.mode,
            samples=len(records),
            source_frames=len(frame_records),
            ik_samples=len(reference.times),
            loops_requested=args.loops,
            articulation_resolution=articulation_resolution,
            contact_materials=pad_binding,
            contact_solver=(
                dict(
                    type="TGS",
                    iterations=physics_config.get("solver_iterations", {}),
                    scene=physics_config.get("scene_physics", {}),
                )
                if args.mode == "targets"
                else None
            ),
            object_geometry_fingerprint=can_geometry["fingerprint"],
            physics_vs_fk_passed=success,
            maximum_position_error_m=max(r["position_error_m"] for r in records),
            maximum_orientation_error_rad=max(r["orientation_error_rad"] for r in records),
            maximum_joint_error_rad=max(r["joint_error_rad"] for r in records),
            frame_measurements=frame_records,
            alignment_status=reference.metadata["alignment_status"],
            interpretation="kinematic articulation-state replay; no drive tuning, collision or physical grasp validation"
            if args.mode == "kinematic"
            else "existing USD position-drive targets; no gain tuning",
            gravity_m_s2=0.0 if args.mode == "kinematic" else 9.81,
            floor_top_z_m=workcell.description["floor_z"],
            tabletop_z_m=0.0,
            workcell=workcell.metadata(),
            maximum_base_mount_matrix_error=maximum_base_error,
            initial_can_bottom_z_m_by_loop=initial_bottoms,
            object_control="reset only; free rigid body under gravity/contact"
            if dynamic_can is not None
            else "kinematic reference visualization",
            object_pose_writes_during_demonstration=0 if dynamic_can is not None else len(records),
            continue_after_grasp_failure=args.mode == "targets",
            physx_finger_mimic_constraints=mimic_count,
            target_count=len(targets),
        )
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(
            args.output / "measurements.npz",
            **{k: np.array([r[k] for r in records]) for k in records[0]},
        )
        print("[replay]", json.dumps(report), flush=True)
        if args.mode == "kinematic" and not success:
            code = 2
        if args.mode == "targets" and len(records) != len(targets):
            code = 2
    except Exception:
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        app.close(exit_code=code)
    return code


if __name__ == "__main__":
    sys.exit(main())
