"""Sequential SLSQP with fixed-object interaction mesh and geometric constraints."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from .transforms import transform, apply, rigid_fit
from .mesh import interaction_graph, deformation
from .metrics import finger_edges, directions, direction_error


@dataclass
class SolverOptions:
    laplacian_norm: str = "regrind"
    smoothness: float = 0.02
    translation_scale_m: float = 0.1
    collision_margin_m: float = 0.0005
    wrist_linear_velocity_m_s: float = 1.0
    wrist_angular_velocity_rad_s: float = 8.0
    max_iterations: int = 400
    ftol: float = 1e-6
    feasibility_tolerance: float = 2e-6
    max_attempts: int = 3
    collision_substeps: int = 10
    finger_direction_weight: float = 0.5
    keypoint_position_weight: float = 200.0


def solve_sequence(
    model,
    sequence,
    object_points,
    collision_scene,
    output_dir,
    options=None,
    geometric_debug=False,
    max_frames=None,
    *,
    contact_constraints=None,
    initial_trajectory=None,
):
    options = options or SolverOptions()
    if (
        min(
            options.smoothness,
            options.collision_margin_m,
            options.finger_direction_weight,
            options.keypoint_position_weight,
        )
        < 0
    ):
        raise ValueError("Invalid solver weights/margin")
    if options.max_attempts < 1 or options.collision_substeps < 1:
        raise ValueError("Attempts and collision substeps must be positive")
    if not geometric_debug and (sequence.times is None or collision_scene is None):
        raise ValueError("Constrained retargeting needs actual timing and collision geometry")
    frame_count = (
        len(sequence.frame_ids) if max_frames is None else min(max_frames, len(sequence.frame_ids))
    )
    if frame_count < 1:
        raise ValueError("Empty sequence")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if contact_constraints is not None and (
        not np.array_equal(contact_constraints.plan.frame_ids, sequence.frame_ids)
        or sequence.times is None
        or not np.allclose(contact_constraints.plan.times, sequence.times, atol=1e-9, rtol=0)
    ):
        raise ValueError("Contact plan must match source frame IDs and timestamps")
    if contact_constraints is not None and geometric_debug:
        raise ValueError("Hard contact requires full collision checks")
    if initial_trajectory is not None and (
        len(initial_trajectory["active_q_rad"]) != len(sequence.frame_ids)
        or list(initial_trajectory["active_joint_names"]) != model.active_names
    ):
        raise ValueError("Initial trajectory does not match source/model")
    pair_margins = (
        options.collision_margin_m
        if contact_constraints is None
        else contact_constraints.margins(options.collision_margin_m)
    )
    hand_count = len(model.keypoints)
    object_points = np.asarray(object_points)
    if object_points.shape != (50, 3):
        raise ValueError("Expected 50 fixed object-local points")
    n = len(model.active_names)
    wrists = np.full((frame_count, 4, 4), np.nan)
    active = np.full((frame_count, n), np.nan)
    full = np.full((frame_count, len(model.full_names)), np.nan)
    robot_points = np.full((frame_count, hand_count, 3), np.nan)
    object_world = np.stack([apply(t, object_points) for t in sequence.object_poses[:frame_count]])
    valid = np.zeros(frame_count, dtype=bool)
    records, graph_records = [], []
    previous = None
    previous_index = None
    semantic = model.semantic_names
    articulation_edges = finger_edges(semantic)
    palm_indices = [
        semantic.index(s) for s in ("wrist", "index_mcp", "middle_mcp", "ring_mcp", "little_mcp")
    ]

    for frame in range(frame_count):
        started = time.perf_counter()
        if not sequence.valid[frame]:
            records.append(
                dict(
                    frame_id=int(sequence.frame_ids[frame]),
                    valid=False,
                    reason="invalid source annotations",
                )
            )
            graph_records.append([])
            continue
        human = sequence.hand[frame]
        source_directions = directions(human, articulation_edges)
        can_pose = sequence.object_poses[frame]
        source = np.concatenate([human, object_world[frame]])
        try:
            laplacian, edges = interaction_graph(source)
        except Exception as error:
            records.append(
                dict(
                    frame_id=int(sequence.frame_ids[frame]),
                    valid=False,
                    reason=f"Delaunay failed: {error}",
                )
            )
            graph_records.append([])
            continue
        graph_records.append(edges.tolist())
        if previous is None:
            q0 = model.lower + 0.2 * (model.upper - model.lower)
            local = model.keypoint_positions(q0)
            pose0 = rigid_fit(local[palm_indices], human[palm_indices])
            if initial_trajectory is not None:
                pose0 = initial_trajectory["wrist_transform"][frame].copy()
                q0 = initial_trajectory["active_q_rad"][frame].copy()
            dt = None
        else:
            pose0, q0 = previous[0].copy(), previous[1].copy()
            dt = (
                None
                if sequence.times is None
                else float(sequence.times[frame] - sequence.times[previous_index])
            )
            if dt is not None and dt <= 0:
                raise ValueError("Nonpositive frame interval")
        base_rotation = pose0[:3, :3].copy()
        x0 = np.r_[pose0[:3, 3], np.zeros(3), q0]
        lower_q, upper_q = model.lower.copy(), model.upper.copy()
        if dt is not None:
            lower_q = np.maximum(lower_q, q0 - model.velocity * dt)
            upper_q = np.minimum(upper_q, q0 + model.velocity * dt)
        bounds = [(None, None)] * 3 + [(-np.pi, np.pi)] * 3 + list(zip(lower_q, upper_q))
        if contact_constraints is not None:
            # These component bounds are implied by the existing velocity
            # norms. They also keep infeasible SQP line searches numerically sane.
            translation_radius = 0.3 if dt is None else options.wrist_linear_velocity_m_s * dt
            rotation_radius = (
                np.pi if dt is None else min(np.pi, options.wrist_angular_velocity_rad_s * dt)
            )
            bounds[:3] = list(
                zip(pose0[:3, 3] - translation_radius, pose0[:3, 3] + translation_radius)
            )
            bounds[3:6] = [(-rotation_radius, rotation_radius)] * 3
        cached_x, cached = None, None
        alphas = (
            np.array([1.0])
            if previous is None
            else np.arange(1, options.collision_substeps + 1) / options.collision_substeps
        )
        collision_object_poses = []
        for alpha in alphas:
            if previous is None:
                collision_object_poses.append(can_pose)
            else:
                old_object = sequence.object_poses[previous_index]
                delta_rotation = Rotation.from_matrix(
                    can_pose[:3, :3] @ old_object[:3, :3].T
                ).as_rotvec()
                collision_object_poses.append(
                    transform(
                        Rotation.from_rotvec(alpha * delta_rotation).as_matrix()
                        @ old_object[:3, :3],
                        (1 - alpha) * old_object[:3, 3] + alpha * can_pose[:3, 3],
                    )
                )

        def kinematics(x, alpha=1.0):
            wrist = transform(
                Rotation.from_rotvec(alpha * x[3:6]).as_matrix() @ base_rotation,
                (1 - alpha) * pose0[:3, 3] + alpha * x[:3],
            )
            return wrist, model.link_transforms((1 - alpha) * q0 + alpha * x[6:], wrist)

        def evaluate(x):
            nonlocal cached_x, cached
            if cached_x is not None and np.array_equal(x, cached_x):
                return cached
            wrist, links = kinematics(x)
            points = model.keypoint_positions(x[6:], links=links)
            target = np.concatenate([points, object_world[frame]])
            energy, residual = deformation(laplacian, source, target, norm=options.laplacian_norm)
            direction_energy = options.finger_direction_weight * np.sum(
                (directions(points, articulation_edges) - source_directions) ** 2
            )
            position_energy = options.keypoint_position_weight * np.sum((points - human) ** 2)
            smoothing = 0.0
            if previous is not None:
                smoothing = options.smoothness * (
                    np.sum(((x[:3] - pose0[:3, 3]) / options.translation_scale_m) ** 2)
                    + np.sum(x[3:6] ** 2)
                    + np.sum((x[6:] - q0) ** 2)
                )
            distances, collision_samples = [], []
            if collision_scene is not None:
                for alpha, object_pose in zip(alphas, collision_object_poses):
                    sampled_links = links if alpha == 1 else kinematics(x, alpha)[1]
                    sample_distance, witnesses = collision_scene.distances(
                        sampled_links, object_pose, with_witnesses=True
                    )
                    distances.extend(sample_distance)
                    collision_samples.append((alpha, sampled_links, sample_distance, witnesses))
            distances = np.array(distances)
            cached_x = x.copy()
            cached = (
                energy
                + smoothing
                + direction_energy
                + position_energy
                + (
                    0.0
                    if contact_constraints is None
                    else contact_constraints.guidance(frame, links, can_pose)
                ),
                energy,
                smoothing,
                residual,
                distances,
                wrist,
                points,
                links,
                collision_samples,
            )
            return cached

        def inequalities(x):
            result = []
            if collision_scene is not None:
                result.extend(
                    evaluate(x)[4] - np.tile(pair_margins, len(alphas))
                    if np.ndim(pair_margins)
                    else evaluate(x)[4] - pair_margins
                )
            if dt is not None:
                result.extend(
                    [
                        options.wrist_linear_velocity_m_s
                        - np.linalg.norm(x[:3] - pose0[:3, 3]) / dt,
                        options.wrist_angular_velocity_rad_s - np.linalg.norm(x[3:6]) / dt,
                    ]
                )
            return np.asarray(result)

        def inequality_jacobian(x):
            values = evaluate(x)
            distances, collision_samples = values[4], values[8]
            jac = np.zeros((len(distances) + (2 if dt is not None else 0), len(x)))
            if len(distances):
                # Differentiate motion of the current closest points, rather
                # than subtracting noisy GJK distances. This is the local
                # signed-distance derivative, with subgradients at feature ties.
                names = [name for name, _ in collision_scene.hand for _ in collision_scene.objects]
                for sample_index, (alpha, links, sample_distance, witnesses) in enumerate(
                    collision_samples
                ):
                    rows = slice(sample_index * len(names), (sample_index + 1) * len(names))
                    delta = witnesses[:, 0] - witnesses[:, 1]
                    normals = delta / np.maximum(np.linalg.norm(delta, axis=1)[:, None], 1e-15)
                    normals *= np.where(sample_distance < 0, -1.0, 1.0)[:, None]
                    local_points = [
                        links[name][:3, :3].T @ (point - links[name][:3, 3])
                        for name, point in zip(names, witnesses[:, 0])
                    ]
                    jac[rows, :3] = alpha * normals
                    for column in range(3, len(x)):
                        offset = np.zeros_like(x)
                        offset[column] = 1e-6
                        plus, minus = (
                            kinematics(x + offset, alpha)[1],
                            kinematics(x - offset, alpha)[1],
                        )
                        point_velocity = np.array(
                            [
                                (apply(plus[name], point) - apply(minus[name], point)) / 2e-6
                                for name, point in zip(names, local_points)
                            ]
                        )
                        jac[rows, column] = np.einsum("ij,ij->i", normals, point_velocity)
            if dt is not None:
                delta = x[:3] - pose0[:3, 3]
                jac[-2, :3] = -delta / max(np.linalg.norm(delta), 1e-15) / dt
                jac[-1, 3:6] = -x[3:6] / max(np.linalg.norm(x[3:6]), 1e-15) / dt
            return jac

        constraints = []
        if collision_scene is not None or dt is not None:
            constraints = [dict(type="ineq", fun=inequalities, jac=inequality_jacobian)]

        def contact_sample_frame(alpha):
            if previous_index is None:
                return frame
            sample_time = (1 - alpha) * sequence.times[previous_index] + alpha * sequence.times[
                frame
            ]
            return int(np.searchsorted(sequence.times, sample_time + 1e-9, side="right") - 1)

        def extra_constraints(x):
            return np.concatenate(
                [
                    contact_constraints.constraints(
                        contact_sample_frame(alpha), kinematics(x, alpha)[1], pose
                    )
                    for alpha, pose in zip(alphas, collision_object_poses)
                ]
            )

        def extra_jacobian(x):
            return np.concatenate(
                [
                    contact_constraints.jacobian(
                        contact_sample_frame(alpha),
                        lambda candidate: kinematics(candidate, alpha),
                        x,
                        pose,
                    )
                    for alpha, pose in zip(alphas, collision_object_poses)
                ]
            )

        if contact_constraints is not None:
            constraints.append(dict(type="ineq", fun=extra_constraints, jac=extra_jacobian))
        initial_objective = evaluate(x0)[0]
        # Large unreachable jumps can make the extra squared point loss dwarf
        # the constraint derivatives. A fixed positive framewise scale changes
        # conditioning, never the minimizer or the physical constraint checks.
        objective_scale = max(1.0, initial_objective / 10.0)
        attempts = []
        seed = x0.copy()
        # Contact distances are piecewise smooth. A failed line search or
        # iteration limit gets a bounded SQP restart at its candidate, resetting
        # the Hessian approximation. Constraints and acceptance never relax.
        for attempt in range(options.max_attempts):
            result = minimize(
                lambda x: evaluate(x)[0] / objective_scale,
                seed,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options=dict(
                    maxiter=options.max_iterations, ftol=options.ftol, eps=1e-6, disp=False
                ),
            )
            ineq = inequalities(result.x)
            if contact_constraints is not None:
                ineq = np.r_[ineq, extra_constraints(result.x)]
            minimum = float(ineq.min()) if len(ineq) else None
            attempts.append(
                dict(
                    success=bool(result.success),
                    status=int(result.status),
                    message=str(result.message),
                    iterations=int(result.nit),
                    objective=float(result.fun * objective_scale),
                    minimum_inequality=minimum,
                )
            )
            if result.success and (minimum is None or minimum >= -options.feasibility_tolerance):
                break
            if not np.isfinite(result.x).all():
                break
            seed = result.x.copy()
            if contact_constraints is not None and initial_trajectory is not None:
                baseline_pose = initial_trajectory["wrist_transform"][frame]
                seed = np.r_[
                    baseline_pose[:3, 3],
                    Rotation.from_matrix(baseline_pose[:3, :3] @ base_rotation.T).as_rotvec(),
                    initial_trajectory["active_q_rad"][frame],
                ]
                seed = np.clip(seed, [b[0] for b in bounds], [b[1] for b in bounds])
        contact_recovery = False
        if (
            contact_constraints is not None
            and initial_trajectory is not None
            and (
                not result.success
                or np.min(np.r_[inequalities(result.x), extra_constraints(result.x)])
                < -options.feasibility_tolerance
            )
        ):
            # A feasibility restoration around the immutable input is preferable
            # to accepting an infeasible objective minimizer. This is recorded;
            # it does not claim convergence of the Laplacian objective.
            recovery_seed = seed.copy()
            scale = np.r_[[0.05] * 3, [0.3] * 3, [0.3] * n]
            restored = minimize(
                lambda x: np.sum(((x - recovery_seed) / scale) ** 2),
                recovery_seed,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options=dict(maxiter=options.max_iterations, ftol=options.ftol, eps=1e-6),
            )
            minimum = float(np.min(np.r_[inequalities(restored.x), extra_constraints(restored.x)]))
            attempts.append(
                dict(
                    success=bool(restored.success),
                    status=int(restored.status),
                    message=str(restored.message),
                    iterations=int(restored.nit),
                    objective=float(evaluate(restored.x)[0]),
                    minimum_inequality=minimum,
                    purpose="feasibility restoration around input reference",
                )
            )
            if restored.success and minimum >= -options.feasibility_tolerance:
                result = restored
                contact_recovery = True
        objective, energy, smoothing, residual, distances, wrist, points, links, _ = evaluate(
            result.x
        )
        q = result.x[6:]
        violation = model.limit_violation(q)
        velocity_violation = 0.0
        full_speed_max = 0.0
        wrist_linear_speed = wrist_angular_speed = 0.0
        if dt is not None:
            speed = np.abs(model.expand(q) - model.expand(q0)) / dt
            full_speed_max = float(speed.max())
            wrist_linear_speed = float(np.linalg.norm(wrist[:3, 3] - pose0[:3, 3]) / dt)
            wrist_angular_speed = float(
                Rotation.from_matrix(wrist[:3, :3] @ pose0[:3, :3].T).magnitude() / dt
            )
            velocity_violation = float(
                max(
                    0,
                    np.max(speed - model.full_velocity),
                    wrist_linear_speed - options.wrist_linear_velocity_m_s,
                    wrist_angular_speed - options.wrist_angular_velocity_rad_s,
                )
            )
        separation = (
            None
            if not len(distances)
            else float(distances[-len(collision_scene.pair_names) :].min())
        )
        path_separation = None if not len(distances) else float(distances.min())
        collision_flags = (
            None if collision_scene is None else collision_scene.collision_flags(links, can_pose)
        )
        feasible = (
            violation <= options.feasibility_tolerance
            and velocity_violation <= options.feasibility_tolerance
            and (
                path_separation is None
                or np.min(inequalities(result.x)) >= -options.feasibility_tolerance
            )
            and (collision_flags is None or not collision_flags.any())
            and (
                contact_constraints is None
                or extra_constraints(result.x).min() >= -options.feasibility_tolerance
            )
        )
        accepted = bool(result.success and feasible and np.isfinite(result.x).all())
        wrists[frame], active[frame], full[frame], robot_points[frame] = (
            wrist,
            q,
            model.expand(q),
            points,
        )
        # Debug trajectories can be inspected but are never marked fully valid.
        valid[frame] = accepted and not geometric_debug
        records.append(
            dict(
                frame_id=int(sequence.frame_ids[frame]),
                contact=None
                if contact_constraints is None
                else contact_constraints.report(frame, links, can_pose),
                valid=bool(valid[frame]),
                optimizer_success=bool(result.success),
                contact_feasibility_restoration=contact_recovery,
                feasible=bool(feasible),
                status=int(result.status),
                message=str(result.message),
                iterations=sum(a["iterations"] for a in attempts),
                attempts=attempts,
                elapsed_s=time.perf_counter() - started,
                objective=float(objective),
                initial_objective=float(initial_objective),
                objective_scale=objective_scale,
                laplacian_energy=float(energy),
                smoothness_energy=float(smoothing),
                finger_direction_energy=float(
                    options.finger_direction_weight
                    * np.sum((directions(points, articulation_edges) - source_directions) ** 2)
                ),
                keypoint_position_energy=float(
                    options.keypoint_position_weight * np.sum((points - human) ** 2)
                ),
                finger_direction_error=direction_error(human, points, articulation_edges),
                hand_laplacian_rms_m=float(np.sqrt(np.mean(residual[:hand_count] ** 2))),
                object_laplacian_rms_m=float(np.sqrt(np.mean(residual[hand_count:] ** 2))),
                hand_keypoint_rmse_m=float(np.sqrt(np.mean(np.sum((points - human) ** 2, axis=1)))),
                joint_limit_violation_rad=violation,
                velocity_limit_violation=velocity_violation,
                full_joint_max_speed_rad_s=full_speed_max,
                wrist_linear_speed_m_s=wrist_linear_speed,
                wrist_angular_speed_rad_s=wrist_angular_speed,
                min_collision_distance_m=separation,
                collision_distances_m=[]
                if collision_scene is None
                else distances[-len(collision_scene.pair_names) :].tolist(),
                path_min_collision_distance_m=path_separation,
                path_collision_alphas=alphas.tolist(),
                path_collision_distances_m=distances.tolist(),
                collision_flags=None if collision_flags is None else collision_flags.tolist(),
                dt_s=dt,
                temporal_reference_frame=None
                if previous_index is None
                else int(sequence.frame_ids[previous_index]),
            )
        )
        print(
            f"[retarget] {sequence.frame_ids[frame]} success={result.success} feasible={feasible} "
            f"E={energy:.5g} gap={separation} iter={result.nit}",
            flush=True,
        )
        # Failed solutions remain in output but do not silently become warm starts.
        if accepted:
            previous = wrist.copy(), q.copy()
            previous_index = frame

    if contact_constraints is not None:
        from .contact_retargeting import validate_contact_path

        valid, dense_records = validate_contact_path(
            model,
            collision_scene,
            contact_constraints,
            wrists,
            active,
            sequence.object_poses[:frame_count],
            sequence.times[:frame_count],
            valid,
            options.collision_margin_m,
            options.feasibility_tolerance,
            max(10, 2 * options.collision_substeps),
        )
        for index, check in enumerate(dense_records):
            records[index]["dense_validation"] = check
            if check is not None and not check["passed"]:
                records[index]["valid"] = False
                records[index]["feasible"] = False
                records[index]["rejection"] = (
                    "denser interpolation check failed; pose retained for diagnosis"
                )

    # A transition after any failed/missing source frame is not a contiguous valid trajectory segment.
    transition_valid = np.zeros(frame_count, dtype=bool)
    if frame_count > 1:
        transition_valid[1:] = valid[1:] & valid[:-1]
    quaternions = np.full((frame_count, 4), np.nan)
    for i, wrist in enumerate(wrists):
        if np.isfinite(wrist).all():
            quaternions[i] = Rotation.from_matrix(wrist[:3, :3]).as_quat()
            if (
                i
                and np.isfinite(quaternions[i - 1]).all()
                and np.dot(quaternions[i], quaternions[i - 1]) < 0
            ):
                quaternions[i] *= -1
    frame_arrays = {}
    if "frame_metadata" in sequence.metadata:
        frame_arrays["frame_metadata_json"] = np.array(
            json.dumps(sequence.metadata["frame_metadata"])
        )
    np.savez_compressed(
        output_dir / "trajectory.npz",
        frame_ids=sequence.frame_ids[:frame_count],
        timestamps_s=np.full(frame_count, np.nan)
        if sequence.times is None
        else sequence.times[:frame_count],
        wrist_transform=wrists,
        wrist_translation_m=wrists[:, :3, 3],
        wrist_quaternion_xyzw=quaternions,
        active_joint_names=np.asarray(model.active_names),
        full_joint_names=np.asarray(model.full_names),
        active_q_rad=active,
        full_q_rad=full,
        semantic_names=np.asarray(model.semantic_names),
        human_keypoints=sequence.hand[:frame_count],
        robot_keypoints=robot_points,
        object_points_local=object_points,
        object_keypoints=object_world,
        object_transform=sequence.object_poses[:frame_count],
        valid=valid,
        transition_valid=transition_valid,
        object_geometry_fingerprint=np.array(
            "" if collision_scene is None else collision_scene.fingerprint
        ),
        **frame_arrays,
    )
    report = dict(
        classification="geometric_debug"
        if geometric_debug
        else "constrained_geometric_retargeting",
        options=asdict(options),
        contact_method=None if contact_constraints is None else "hard_object_anchor_to_pad_surface",
        collision_margins_m=np.asarray(pair_margins).tolist(),
        contact_scope=None
        if contact_constraints is None
        else "fixed object anchors to pad convex surfaces; phase-active endpoints and interpolation samples; table half-space at same samples",
        sequence_metadata=sequence.metadata,
        frames=records,
        frame_count=frame_count,
        valid_count=int(valid.sum()),
        failed_frame_ids=sequence.frame_ids[:frame_count][~valid].tolist(),
        collision_pairs=[] if collision_scene is None else collision_scene.pair_names,
        object_geometry_fingerprint=None
        if collision_scene is None
        else collision_scene.fingerprint,
        collision_scope=f"all {len(model.colliders)} hand convex meshes vs all {0 if collision_scene is None else len(collision_scene.objects)} object colliders; frames plus {options.collision_substeps} samples per transition",
        not_validated=[
            "continuous-time collision",
            "hand self-collision",
            "table/environment collision",
            "physical grasp",
            "dynamics",
            "RB3 IK",
        ],
        graph=dict(
            vertex_order="semantic robot order, then fixed object sample order",
            source_edges=graph_records,
        ),
    )
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
