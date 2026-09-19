"""Validated trajectory adapter, SI units, column transforms, XYZW quaternions."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ..transforms import inverse


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReferenceMotion:
    def __init__(self, path, model, geometry_path, world_frame):
        if world_frame not in ("grounded_dataset", "initial_object_frame_with_bottom_on_ground"):
            raise ValueError("Unknown world-frame convention")
        with np.load(path, allow_pickle=False) as data:
            self.data = {key: data[key].copy() for key in data.files}
        d = self.data
        if list(d["active_joint_names"]) != model.active_names or list(d["full_joint_names"]) != model.full_names:
            raise ValueError("Reference joint order differs from the extracted USD model")
        if list(d["semantic_names"]) != model.semantic_names:
            raise ValueError("Reference semantic correspondence differs from model")
        if not d["valid"].all() or not d["transition_valid"][1:].all():
            raise ValueError("Invalid reference frames/transitions cannot silently seed RL")
        self.times = np.asarray(d["timestamps_s"], dtype=float)
        if len(self.times) < 2 or not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0):
            raise ValueError("Reference requires strictly increasing, finite timestamps")
        self.times -= self.times[0]
        self.duration = float(self.times[-1])
        self.model = model
        geometry = json.loads(Path(geometry_path).read_text())
        from ..coordinates import collision_bottom, z_aligned_cylinders
        from ..geometry import geometry_fingerprint
        self.object_geometry = geometry
        self.collision_shapes = z_aligned_cylinders(geometry)
        fingerprint = geometry_fingerprint(geometry)
        if geometry.get('fingerprint') and (geometry['fingerprint'] != fingerprint
                or str(d.get('object_geometry_fingerprint', '')) != fingerprint):
            raise ValueError('Reference object geometry is stale; rerun retargeting and grounding')
        bottom = collision_bottom(np.eye(4), self.collision_shapes)
        if world_frame == 'grounded_dataset':
            frame = json.loads(str(d['frame_metadata_json']))
            if frame['coordinate_frame'] != 'ground' or abs(collision_bottom(d['object_transform'][0], self.collision_shapes)) > 1e-7:
                raise ValueError('Grounded reference must start with the can bottom at Z=0')
            # Historical attribute name: this maps INPUT coordinates to simulation.
            # The derived dataset is already in world coordinates: no second transform.
            self.camera_to_world = np.eye(4)
        else:
            if 'frame_metadata_json' in d:
                raise ValueError('Use grounded_dataset for already transformed data')
            self.camera_to_world = inverse(d["object_transform"][0])
            self.camera_to_world[2, 3] += -bottom + 0.001
        self.wrist = self.camera_to_world @ d["wrist_transform"]
        self.object = self.camera_to_world @ d["object_transform"]
        self.q = np.asarray(d["active_q_rad"], dtype=float)
        for arr in (self.wrist, self.object, self.q, d["object_points_local"]):
            if not np.isfinite(arr).all():
                raise ValueError("Nonfinite reference data")
        if np.any(self.q < model.lower - 1e-5) or np.any(self.q > model.upper + 1e-5):
            raise ValueError("Reference joint limits violated")
        self._slerp = {name: Slerp(self.times, Rotation.from_matrix(poses[:, :3, :3]))
                       for name, poses in (("wrist", self.wrist), ("object", self.object))}
        self.object_local = d["object_points_local"]
        self.metadata = dict(reference_sha256=digest(path), model_source_sha256=model.description["source_sha256"],
                             object_geometry_fingerprint=fingerprint, object_geometry_sha256=digest(geometry_path),
                             object_asset_sha256=geometry.get('collision_source_sha256'),
                             camera_to_world=self.camera_to_world.tolist(), quaternion_order="xyzw",
                             time_basis="input trajectory timestamps; currently user-authorized 5 Hz retiming",
                             duration_s=self.duration, frame_ids=d["frame_ids"].tolist(),
                             world_frame=world_frame, ground_z_m=0.0)
        if 'frame_metadata_json' in d:
            self.metadata['dataset_frame'] = json.loads(str(d['frame_metadata_json']))

    def sample(self, time):
        t = np.clip(np.atleast_1d(time).astype(float), 0, self.duration)
        hi = np.clip(np.searchsorted(self.times, t, side="right"), 1, len(self.times) - 1)
        lo = hi - 1
        dt = self.times[hi] - self.times[lo]
        a = (t - self.times[lo]) / dt
        q = self.q[lo] * (1 - a[:, None]) + self.q[hi] * a[:, None]
        out = dict(q=q, q_velocity=(self.q[hi] - self.q[lo]) / dt[:, None], time=t)
        for name, poses in (("wrist", self.wrist), ("object", self.object)):
            out[name + "_position"] = poses[lo, :3, 3] * (1 - a[:, None]) + poses[hi, :3, 3] * a[:, None]
            out[name + "_quaternion"] = self._slerp[name](t).as_quat()
            out[name + "_velocity"] = (poses[hi, :3, 3] - poses[lo, :3, 3]) / dt[:, None]
            out[name + "_angular_velocity"] = Rotation.from_matrix(
                poses[hi, :3, :3] @ poses[lo, :3, :3].transpose(0, 2, 1)).as_rotvec() / dt[:, None]
        return out

    def retime_for_control_horizon(self, episode_length_s, control_dt):
        """Source-style episode resampling without changing any source pose or file."""
        if not np.isfinite([episode_length_s, control_dt]).all() or episode_length_s <= 0 or control_dt <= 0:
            raise ValueError("Episode length and control dt must be positive and finite")
        count = int(np.ceil(episode_length_s / control_dt))
        if count < 2:
            raise ValueError("An episode must contain at least two control frames")
        previous_duration = self.duration
        self.duration = (count - 1) * control_dt
        self.times = self.times * (self.duration / previous_duration)
        self._slerp = {name: Slerp(self.times, Rotation.from_matrix(poses[:, :3, :3]))
                       for name, poses in (("wrist", self.wrist), ("object", self.object))}
        self.metadata.update(input_duration_s=previous_duration, duration_s=self.duration,
                             episode_length_s=episode_length_s, control_reference_frames=count,
                             time_basis="source-style episode resampling; original capture fps remains unknown")

    def points(self, position, quaternion):
        rotation = Rotation.from_quat(quaternion).as_matrix()
        return np.einsum("nij,kj->nki", rotation, self.object_local) + position[:, None]
