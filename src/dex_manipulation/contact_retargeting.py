"""Stable object-local contact proxies and opt-in hard pad-surface constraints.

Human skeleton tips are not skin contacts. The extraction explicitly records this
approximation; it never infers contact truth from a successful robot solve.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from .transforms import apply

FINGERS = ("thumb", "index", "middle", "ring", "little")
PAD_LINKS = tuple(
    "right_" + f + "_touch_link" for f in ("thumb", "index", "middle", "ring", "pinky")
)


@dataclass
class ContactPlan:
    frame_ids: np.ndarray
    times: np.ndarray
    anchors: np.ndarray
    active: np.ndarray
    metadata: dict

    def __post_init__(self):
        self.frame_ids = np.asarray(self.frame_ids)
        self.times = np.asarray(self.times, dtype=float)
        self.anchors = np.asarray(self.anchors, dtype=float)
        self.active = np.asarray(self.active, dtype=bool)
        count = len(self.frame_ids)
        if (
            count < 1
            or self.frame_ids.shape != (count,)
            or self.times.shape != (count,)
            or self.anchors.shape != (5, 3)
            or self.active.shape != (count, 5)
            or not np.isfinite(self.times).all()
            or not np.isfinite(self.anchors).all()
            or np.any(np.diff(self.times) <= 0)
            or np.any(np.diff(self.frame_ids) <= 0)
        ):
            raise ValueError("Invalid contact plan shape, timing or anchors")

    def save(self, path):
        np.savez_compressed(
            path,
            frame_ids=self.frame_ids,
            timestamps_s=self.times,
            object_anchors_local=self.anchors,
            active=self.active,
            fingers=np.array(FINGERS),
            pad_links=np.array(PAD_LINKS),
            metadata_json=np.array(json.dumps(self.metadata)),
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            if list(d["fingers"]) != list(FINGERS) or list(d["pad_links"]) != list(PAD_LINKS):
                raise ValueError("Contact semantic order mismatch")
            return cls(
                d["frame_ids"].copy(),
                d["timestamps_s"].copy(),
                d["object_anchors_local"].copy(),
                d["active"].copy(),
                json.loads(str(d["metadata_json"])),
            )


def extract_contacts(data, mesh_path, settings):
    """Project skeleton tips onto the real object mesh, cluster, select medoids.

    Contact phase is an explicit user annotation, not an image-based detector.
    Reject short/noisy/far candidates, recording why each was excluded. Keeping
    all five human requests in the hard solve is deliberate: infeasibility must
    not be hidden by dropping a finger after optimization.
    """
    import trimesh
    from sklearn.cluster import DBSCAN

    mesh_path = Path(mesh_path)
    with np.load(mesh_path, allow_pickle=False) as m:
        mesh = trimesh.Trimesh(m["vertices"], m["faces"], process=False)
    ids, times = data["frame_ids"], data["timestamps_s"]
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Finite increasing times are required for contact duration")
    mask = ids >= settings["start_frame_id"]
    if "end_frame_id" in settings:
        mask &= ids <= settings["end_frame_id"]
    if not mask.any():
        raise ValueError("Empty annotated contact phase")
    poses = data["object_transform"]
    anchors, records = [], []
    active = np.zeros((len(ids), 5), bool)
    names = list(data["semantic_names"])
    for k, finger in enumerate(FINGERS):
        world = data["human_keypoints"][:, names.index(finger + "_tip")]
        local = np.einsum("nji,nj->ni", poses[:, :3, :3], world - poses[:, :3, 3])
        points, gaps, _ = trimesh.proximity.closest_point_naive(mesh, local[mask])
        labels = DBSCAN(eps=settings["cluster_radius_m"], min_samples=2).fit_predict(points)
        groups = [v for v in np.unique(labels) if v >= 0]
        selected = max(groups, key=lambda v: int(np.sum(labels == v))) if groups else None
        cluster = points if selected is None else points[labels == selected]
        anchor = cluster[np.argmin(cdist(cluster, cluster).sum(axis=1))]
        drift = float(np.linalg.norm(points - anchor, axis=1).max())
        duration = float(times[mask][-1] - times[mask][0])
        reasons = []
        if selected is None:
            reasons.append("no stable cluster")
        if duration < settings["minimum_duration_s"]:
            reasons.append("contact phase too short")
        if drift > settings["maximum_drift_m"]:
            reasons.append("object-local drift too large")
        if float(np.median(gaps)) > settings["maximum_tip_gap_m"]:
            reasons.append("skeleton tip too far from object surface")
        active[:, k] = mask & (not reasons)
        anchors.append(anchor)
        records.append(
            dict(
                finger=finger,
                median_source_tip_gap_m=float(np.median(gaps)),
                max_object_local_drift_m=drift,
                duration_s=duration,
                cluster_size=len(cluster),
                accepted=not reasons,
                rejected_reasons=reasons,
            )
        )
    return ContactPlan(
        ids.copy(),
        times.copy(),
        np.array(anchors),
        active,
        dict(
            schema="object_local_contact_proxy_v1",
            settings=settings,
            source="explicit annotated phase; 21-point skeleton-tip projection onto current can mesh",
            limitation="proxy contacts, not measured MANO skin contacts or image/ray intersections",
            mesh_sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
            fingers=records,
            surface_correspondence="named human fingertips to actual Revo2 rubber pad collision meshes",
        ),
    )


def adapt_contact_anchors(plan, model, scene, data):
    """Optional morphology adaptation, explicitly distinct from human anchors.

    Preserve phase/finger identity, but choose the medoid of the existing robot's
    nearest object-surface witnesses. No failed hard solve is used as a seed and
    the baseline geometry must already be valid. This changes contact placement,
    so both the original anchors and every displacement are recorded.
    """
    if not data["valid"].all() or not data["transition_valid"][1:].all():
        raise ValueError("Robot contact adaptation requires a valid geometric reference")
    anchors = []
    for finger, link in enumerate(PAD_LINKS):
        points = []
        body = [n for n, _ in scene.hand].index(link)
        for frame in np.flatnonzero(plan.active[:, finger]):
            pose = data["object_transform"][frame]
            links = model.link_transforms(
                data["active_q_rad"][frame], data["wrist_transform"][frame]
            )
            gaps, witnesses = scene.distances(links, pose, with_witnesses=True)
            if gaps.min() < -2e-6:
                raise ValueError("Robot seed contains hand-object penetration")
            indices = np.arange(body * len(scene.objects), (body + 1) * len(scene.objects))
            pair = indices[np.argmin(gaps[indices])]
            points.append(pose[:3, :3].T @ (witnesses[pair, 1] - pose[:3, 3]))
        if not points:
            anchors.append(plan.anchors[finger])
        else:
            points = np.array(points)
            anchors.append(points[np.argmin(cdist(points, points).sum(axis=1))])
    anchors = np.asarray(anchors)
    metadata = dict(
        plan.metadata,
        anchor_mode="robot_surface_seed",
        original_human_anchors_local=plan.anchors.tolist(),
        anchor_displacement_m=np.linalg.norm(anchors - plan.anchors, axis=1).tolist(),
        adaptation="semantic phases retained; fixed object anchors reselected from valid robot seed contact witnesses",
    )
    return ContactPlan(plan.frame_ids, plan.times, anchors, plan.active, metadata)


class HardContacts:
    """Anchor-to-pad convex-surface distance bounded by a hard tolerance.

    Object anchors stay fixed through the phase; the contact location may slide
    on each pad surface. This avoids claiming a rigid 15-equality contact weld
    is feasible with only 12 controls. Full hand-object nonpenetration remains
    an independent solver constraint. A tiny FCL sphere is corrected to a point.
    """

    def __init__(self, model, scene, plan, tolerance_m=0.00075, table_margin_m=0.002):
        if not 0 < tolerance_m < 0.01 or table_margin_m < 0:
            raise ValueError("Invalid hard contact/table tolerance")
        self.scene, self.plan, self.tolerance, self.table_margin = (
            scene,
            plan,
            tolerance_m,
            table_margin_m,
        )
        self.pads = [next(obj for link, obj in scene.hand if link == name) for name in PAD_LINKS]
        self.radius = 1e-6
        self.point = scene.fcl.CollisionObject(scene.fcl.Sphere(self.radius))
        self.corners = []
        from itertools import product

        for c in model.colliders:
            v = np.asarray(c["vertices"])
            self.corners.append((c["link"], np.array(list(product(*zip(v.min(0), v.max(0)))))))

    def distances(self, links, object_pose, anchors=None):
        fcl = self.scene.fcl
        anchors = self.plan.anchors if anchors is None else anchors
        result, witnesses = [], []
        for name, pad, anchor in zip(PAD_LINKS, self.pads, apply(object_pose, anchors)):
            t = links[name]
            pad.setTransform(fcl.Transform(t[:3, :3], t[:3, 3]))
            self.point.setTransform(fcl.Transform(np.eye(3), anchor))
            d = fcl.DistanceResult()
            gap = fcl.distance(pad, self.point, self.scene.request, d) + self.radius
            result.append(gap)
            witnesses.append(d.nearest_points[0])
        return np.asarray(result), np.asarray(witnesses)

    def constraints(self, frame, links, object_pose):
        distances = self.distances(links, object_pose)[0]
        selected = distances[self.plan.active[frame]]
        floor = min(
            float(apply(links[name], corners)[:, 2].min()) for name, corners in self.corners
        )
        # Scaling improves SQP conditioning; acceptance uses these same bounds.
        return np.r_[(self.tolerance - np.abs(selected)) / 0.01, (floor - self.table_margin) / 0.01]

    def jacobian(self, frame, kinematics, x, object_pose):
        links = kinematics(x)[1]
        distance, witnesses = self.distances(links, object_pose)
        anchors = apply(object_pose, self.plan.anchors)
        selected = np.flatnonzero(self.plan.active[frame])
        local = [links[n][:3, :3].T @ (p - links[n][:3, 3]) for n, p in zip(PAD_LINKS, witnesses)]
        delta = witnesses - anchors
        # The absolute signed distance differentiates toward either side of a
        # convex surface. Whole-hand inequalities separately prevent penetration.
        normals = delta / np.maximum(np.linalg.norm(delta, axis=1)[:, None], 1e-12)
        floor_samples = [(n, p, apply(links[n], p)[:, 2]) for n, p in self.corners]
        floor_name, floor_points, floor_z = min(floor_samples, key=lambda v: float(v[2].min()))
        floor_local = floor_points[int(floor_z.argmin())]
        jac = np.zeros((len(selected) + 1, len(x)))
        for column in range(len(x)):
            offset = np.zeros_like(x)
            offset[column] = 1e-6
            plus, minus = kinematics(x + offset)[1], kinematics(x - offset)[1]
            velocity = np.array(
                [(apply(plus[n], p) - apply(minus[n], p)) / 2e-6 for n, p in zip(PAD_LINKS, local)]
            )
            jac[:-1, column] = -np.einsum("ij,ij->i", normals, velocity)[selected] / 0.01
            jac[-1, column] = (
                (apply(plus[floor_name], floor_local)[2] - apply(minus[floor_name], floor_local)[2])
                / 2e-6
                / 0.01
            )
        return jac

    def guidance(self, frame, links, object_pose):
        # Independent pre-contact approach prior; never replaces hard feasibility.
        starts = np.flatnonzero(self.plan.active.any(axis=1))
        if not len(starts) or frame >= starts[0]:
            return 0.0
        remaining = self.plan.times[starts[0]] - self.plan.times[frame]
        if remaining >= 0.2:
            return 0.0
        distances = self.distances(links, object_pose)[0]
        return float((1 - remaining / 0.2) ** 2 * np.sum((distances / 0.01) ** 2))

    def margins(self, default):
        # Positive 50 um prevents collision-query ambiguity at exact touching.
        return np.array(
            [
                min(default, 0.00005) if name in PAD_LINKS else default
                for name, _ in self.scene.hand
                for _ in self.scene.objects
            ]
        )

    def report(self, frame, links, pose):
        d = self.distances(links, pose)[0]
        return dict(
            active=self.plan.active[frame].tolist(),
            anchor_pad_surface_distance_m=d.tolist(),
            hard_tolerance_m=self.tolerance,
            table_min_z_m=min(float(apply(links[n], p)[:, 2].min()) for n, p in self.corners),
        )


def validate_contact_path(
    model,
    scene,
    hard,
    wrists,
    joints,
    object_poses,
    times,
    valid,
    collision_margin_m,
    feasibility_tolerance,
    substeps,
):
    """Densely recheck accepted paths; return conservative validity and diagnostics.

    Uses actual output interpolation, never a different object-relative curve.
    A denser failure invalidates the destination frame and its saved transitions.
    It does not silently loosen a contact tolerance or overwrite a pose.
    """
    from scipy.spatial.transform import Rotation
    from .transforms import transform

    if substeps < 2:
        raise ValueError("Dense validation needs at least two substeps")
    accepted = np.asarray(valid, dtype=bool).copy()
    checked = accepted.copy()
    records = [None] * len(accepted)
    for frame in np.flatnonzero(accepted):
        previous = frame - 1 if frame > 0 and accepted[frame - 1] else frame
        alphas = np.linspace(0, 1, substeps + 1) if previous != frame else [1.0]
        minimum_margin, minimum_contact = float("inf"), float("inf")
        collision = False
        for alpha in alphas:
            poses = []
            for sequence in (wrists, object_poses):
                a, b = sequence[previous], sequence[frame]
                rv = Rotation.from_matrix(b[:3, :3] @ a[:3, :3].T).as_rotvec()
                poses.append(
                    transform(
                        Rotation.from_rotvec(alpha * rv).as_matrix() @ a[:3, :3],
                        (1 - alpha) * a[:3, 3] + alpha * b[:3, 3],
                    )
                )
            wrist, obj = poses
            q = (1 - alpha) * joints[previous] + alpha * joints[frame]
            links = model.link_transforms(q, wrist)
            distance = scene.distances(links, obj)
            minimum_margin = min(
                minimum_margin, float(np.min(distance - hard.margins(collision_margin_m)))
            )
            time = (1 - alpha) * times[previous] + alpha * times[frame]
            contact_frame = int(np.searchsorted(hard.plan.times, time + 1e-9, side="right") - 1)
            minimum_contact = min(
                minimum_contact, float(hard.constraints(contact_frame, links, obj).min())
            )
            if np.any(distance <= 0):
                collision |= bool(scene.collision_flags(links, obj).any())
        passed = (
            minimum_margin >= -feasibility_tolerance
            and minimum_contact >= -feasibility_tolerance
            and not collision
        )
        checked[frame] = passed
        records[frame] = dict(
            passed=bool(passed),
            substeps=substeps,
            min_collision_margin_m=minimum_margin,
            min_scaled_contact_table_constraint=minimum_contact,
            collision=collision,
        )
    return checked, records
