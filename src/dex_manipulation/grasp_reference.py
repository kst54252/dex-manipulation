"""Contact-conditioned hand references with unchanged demonstration objects.

Independent FK/FCL constrained optimization. A fixed object-relative grasp
prevents the retargeted hand from opening or sliding away during the lift.
The output is geometric; physical success must be checked separately.
"""

from .configuration import read_config
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp

from dex_manipulation.geometry import CollisionScene, can_orientation_report
from .materials import PAD_BODIES


def prepare_grasp_reference(
    source,
    output,
    model,
    object_geometry,
    *,
    start_frame_id,
    blend_start_s,
    clearance_m=0.0005,
    table_margin_m=0.004,
    transition_substeps=20,
    side_contact=False,
):
    source, output = Path(source), Path(output)
    if output.exists() or output.resolve() == source.resolve():
        raise ValueError("Contact preparation requires a new derived output")
    with np.load(source, allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    where = np.flatnonzero(data["frame_ids"] == start_frame_id)
    if len(where) != 1:
        raise ValueError("Grasp frame ID must occur exactly once")
    anchor = int(where[0])
    times = data["timestamps_s"] - data["timestamps_s"][0]
    if not 0 <= blend_start_s < times[anchor]:
        raise ValueError("Invalid approach blend interval")
    if list(data["active_joint_names"]) != model.active_names:
        raise ValueError("Joint order mismatch")
    geometry = read_config(Path(object_geometry))
    if not can_orientation_report(data["object_transform"], geometry)["initial_base_below_body"]:
        raise ValueError("The wider can base must start below the body")
    scene = CollisionScene(model, geometry)
    if str(data["object_geometry_fingerprint"]) != scene.fingerprint:
        raise ValueError("Stale collision geometry")
    pad_ids = [next(i for i, (n, _) in enumerate(scene.hand) if n == name) for name in PAD_BODIES]
    boxes = []
    for c in model.colliders:
        v = np.asarray(c["vertices"])
        low = v.min(0)
        high = v.max(0)
        boxes.append(
            (
                c["link"],
                np.array(
                    [
                        [x, y, z]
                        for x in (low[0], high[0])
                        for y in (low[1], high[1])
                        for z in (low[2], high[2])
                    ]
                ),
            )
        )
    base = data["wrist_transform"][anchor].copy()
    base_q = data["active_q_rad"][anchor].copy()
    obj = data["object_transform"][anchor]
    scale = np.array([0.05] * 3 + [0.3] * 3 + [0.3] * 6)

    def pose(x, base=base, q=base_q, scale=scale):
        t = base.copy()
        t[:3, 3] += x[:3] * scale[:3]
        t[:3, :3] = Rotation.from_rotvec(x[3:6] * scale[3:6]).as_matrix() @ base[:3, :3]
        return t, q + x[6:] * scale[6:]

    def inspect(w, q, o):
        links = model.link_transforms(q, w)
        distances = scene.distances(links, o).reshape(-1, len(scene.objects)).min(-1)
        # Enclosing-box clearance matches the conservative runtime guard.
        floor = min(
            (v @ links[name][:3, :3].T + links[name][:3, 3])[:, 2].min() for name, v in boxes
        )
        return distances, float(floor)

    def objective(x):
        ds, z = inspect(*pose(x), obj)
        return np.square((ds[pad_ids] - clearance_m) / 0.002).sum() + 0.001 * np.square(x).sum()

    def constraints(x):
        ds, z = inspect(*pose(x), obj)
        return np.r_[ds - clearance_m, z - table_margin_m] / 0.01

    bounds = [(-1.0, 1.0)] * 6 + list(
        zip(
            np.maximum(-1, (model.lower - base_q) / scale[6:]),
            np.minimum(1, (model.upper - base_q) / scale[6:]),
        )
    )
    initial = np.zeros(12)
    side_result = None
    if side_contact:
        # Aim pad surfaces at the body sidewall, below the top rim. These are
        # optimization targets, not fabricated contact sensor measurements.
        shapes = geometry["collision_shapes"]
        body = max(shapes, key=lambda s: s["height"])
        if body["type"] != "cylinder" or not np.allclose(
            np.array(body["transform"])[:3, :3], np.eye(3)
        ):
            raise ValueError("Side grasp currently requires an object-local Z cylinder")
        ds, pts = scene.distances(model.link_transforms(base_q, base), obj, with_witnesses=True)
        which = ds.reshape(-1, len(shapes)).argmin(1)
        targets = (
            pts.reshape(-1, len(shapes), 2, 3)[pad_ids, which[pad_ids], 1] - obj[:3, 3]
        ) @ obj[:3, :3]
        center = np.array(body["transform"])[:3, 3]
        radial = targets[:, :2] - center[:2]
        targets[:, :2] = center[:2] + body["radius"] * radial / np.linalg.norm(
            radial, axis=-1, keepdims=True
        )
        targets[:, 2] = center[2]
        targets = targets @ obj[:3, :3].T + obj[:3, 3]
        surfaces = []
        for name in PAD_BODIES:
            c = next(c for c in model.colliders if c["link"] == name)
            v = np.array(c["vertices"])
            surfaces.append(np.r_[v, v[np.array(c["faces"])].mean(1)])

        def side_objective(x):
            w, j = pose(x)
            links = model.link_transforms(j, w)
            return sum(
                np.square(v @ links[name][:3, :3].T + links[name][:3, 3] - p).sum(1).min()
                for name, v, p in zip(PAD_BODIES, surfaces, targets)
            ) / 0.005**2 + 0.0001 * (x @ x)

        side = minimize(
            side_objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=[dict(type="ineq", fun=constraints)],
            options=dict(maxiter=300, ftol=1e-8),
        )
        initial = side.x
        side_result = dict(
            success=bool(side.success), message=side.message, objective=float(side.fun)
        )
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=[dict(type="ineq", fun=constraints)],
        options=dict(maxiter=250, ftol=1e-8),
    )
    w, q = pose(result.x)
    ds, z = inspect(w, q, obj)
    if ds.min() < clearance_m * 0.85 or ds[pad_ids].max() > 0.0006 or z < table_margin_m - 1e-6:
        raise ValueError(
            f"Cannot form a safe five-pad grasp: {result.message}; gaps={ds[pad_ids]}, ground={z}"
        )
    relative = np.linalg.inv(obj) @ w
    originals = {k: data[k].copy() for k in ("wrist_transform", "active_q_rad")}
    records = []
    for i, time in enumerate(times):
        u = float(np.clip((time - blend_start_s) / (times[anchor] - blend_start_s), 0, 1))
        alpha = u * u * (3 - 2 * u)
        goal = data["object_transform"][i] @ relative
        original = originals["wrist_transform"][i]
        wrist = np.eye(4)
        wrist[:3, 3] = (1 - alpha) * original[:3, 3] + alpha * goal[:3, 3]
        wrist[:3, :3] = Slerp(
            [0, 1], Rotation.from_matrix(np.stack((original[:3, :3], goal[:3, :3])))
        )(alpha).as_matrix()
        joint = (1 - alpha) * originals["active_q_rad"][i] + alpha * q
        if i < anchor:
            # Project the approach and its incoming segment together. Endpoints
            # alone do not prevent a finger crossing the can between samples.
            seed_w, seed_q = wrist.copy(), joint.copy()
            projection_scale = np.array([0.01] * 3 + [0.08] * 3 + [0.15] * 6)

            def project_pose(x):
                return pose(x, seed_w, seed_q, projection_scale)

            def approach_constraints(x):
                t, j = project_pose(x)
                ds, floor = inspect(t, j, data["object_transform"][i])
                values = [ds - clearance_m, np.array([floor - table_margin_m])]
                if i:
                    prev = data["wrist_transform"][i - 1]
                    old_obj = data["object_transform"][i - 1]
                    wr = Slerp([0, 1], Rotation.from_matrix(np.stack((prev[:3, :3], t[:3, :3]))))
                    ob = Slerp(
                        [0, 1],
                        Rotation.from_matrix(data["object_transform"][i - 1 : i + 1, :3, :3]),
                    )
                    for a in (0.25, 0.5, 0.75):
                        wt, ot = np.eye(4), np.eye(4)
                        wt[:3, :3] = wr(a).as_matrix()
                        wt[:3, 3] = (1 - a) * prev[:3, 3] + a * t[:3, 3]
                        ot[:3, :3] = ob(a).as_matrix()
                        ot[:3, 3] = (1 - a) * old_obj[:3, 3] + a * data["object_transform"][
                            i, :3, 3
                        ]
                        ds, floor = inspect(wt, (1 - a) * data["active_q_rad"][i - 1] + a * j, ot)
                        values.extend((ds - clearance_m, np.array([floor - table_margin_m])))
                return np.concatenate(values) / 0.01

            if approach_constraints(np.zeros(12)).min() < 0:
                bd = [(-1, 1)] * 6 + list(
                    zip(
                        np.maximum(-1, (model.lower - seed_q) / projection_scale[6:]),
                        np.minimum(1, (model.upper - seed_q) / projection_scale[6:]),
                    )
                )
                projection = minimize(
                    lambda x: float(x @ x),
                    np.zeros(12),
                    method="SLSQP",
                    bounds=bd,
                    constraints=[dict(type="ineq", fun=approach_constraints)],
                    options=dict(maxiter=150, ftol=1e-8),
                )
                wrist, joint = project_pose(projection.x)
        ds, floor = inspect(wrist, joint, data["object_transform"][i])
        records.append(
            dict(
                frame_id=int(data["frame_ids"][i]),
                blend=alpha,
                pad_gaps_m=ds[pad_ids].tolist(),
                minimum_gap_m=float(ds.min()),
                table_clearance_m=floor,
            )
        )
        data["wrist_transform"][i] = wrist
        data["active_q_rad"][i] = joint
        data["full_q_rad"][i] = model.expand(joint)
        data["robot_keypoints"][i] = model.keypoint_positions(joint, wrist)
        data["wrist_translation_m"][i] = wrist[:3, 3]
        data["wrist_quaternion_xyzw"][i] = Rotation.from_matrix(wrist[:3, :3]).as_quat()
        data["valid"][i] = bool(
            ds.min() >= -1e-7 and floor >= 0 and model.limit_violation(joint) < 1e-7
        )
    transitions = []
    for i in range(1, len(times)):
        minimum = floor_min = float("inf")
        wr = Slerp([0, 1], Rotation.from_matrix(data["wrist_transform"][i - 1 : i + 1, :3, :3]))
        ob = Slerp([0, 1], Rotation.from_matrix(data["object_transform"][i - 1 : i + 1, :3, :3]))
        for a in np.linspace(0, 1, transition_substeps + 1):
            poses = []
            for key, r in [("wrist_transform", wr), ("object_transform", ob)]:
                t = np.eye(4)
                t[:3, :3] = r(a).as_matrix()
                t[:3, 3] = (1 - a) * data[key][i - 1, :3, 3] + a * data[key][i, :3, 3]
                poses.append(t)
            j = (1 - a) * data["active_q_rad"][i - 1] + a * data["active_q_rad"][i]
            ds, z = inspect(poses[0], j, poses[1])
            minimum = min(minimum, float(ds.min()))
            floor_min = min(floor_min, z)
        data["transition_valid"][i] = bool(minimum >= -1e-7 and floor_min >= 0)
        transitions.append(
            dict(
                frame_id=int(data["frame_ids"][i]),
                gap_m=minimum,
                table_m=floor_min,
                valid=bool(data["transition_valid"][i]),
            )
        )
    velocity = np.abs(np.diff(data["full_q_rad"], axis=0) / np.diff(times)[:, None])
    report = dict(
        schema="contact_reference_v1",
        source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        method="independent five-pad FK/FCL grasp + object-relative hold; cubic approach blend",
        start_frame_id=int(start_frame_id),
        start_time_s=float(times[anchor]),
        blend_start_s=blend_start_s,
        solver_success=bool(result.success),
        solver_message=result.message,
        side_contact=side_contact,
        side_seed_solver=side_result,
        grasp_relative=relative.tolist(),
        grasp_q=q.tolist(),
        object_targets_changed=False,
        timestamps_changed=False,
        original_human_points_changed=False,
        clearance_m=clearance_m,
        table_margin_m=table_margin_m,
        frames=records,
        transitions=transitions,
        max_velocity_violation_rad_s=float(np.maximum(velocity - model.full_velocity, 0).max()),
        geometry_valid=bool(
            data["valid"].all()
            and data["transition_valid"][1:].all()
            and np.all(velocity <= model.full_velocity + 1e-6)
        ),
        physical_grasp_verified=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["geometry_valid"]:
        raise ValueError(f"Contact path validation failed; see {output.with_suffix('.json')}")
    data["contact_reference_json"] = np.array(json.dumps(report, sort_keys=True))
    np.savez_compressed(output, **data)
    return report
