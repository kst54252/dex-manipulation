"""Device-resident reference interpolation and coherent episode augmentation."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .math3d import quat_apply, quat_multiply, from_rotvec, rotation_error, slerp, uniform


class TensorReference:
    def __init__(
        self, reference, num_envs, config, device, *, velocity_mode="segment", control_dt=None
    ):
        self.base, self.duration, self.cfg, self.device = (
            reference,
            reference.duration,
            config,
            torch.device(device),
        )
        self.times = torch.as_tensor(reference.times, dtype=torch.float32, device=device)
        self.q = torch.as_tensor(reference.q, dtype=torch.float32, device=device)
        self.poses = {}
        for name, data in (("wrist", reference.wrist), ("object", reference.object)):
            self.poses[name] = (
                torch.tensor(data[:, :3, 3], dtype=torch.float32, device=device),
                torch.tensor(
                    Rotation.from_matrix(data[:, :3, :3]).as_quat(),
                    dtype=torch.float32,
                    device=device,
                ),
            )
        self.object_local = torch.as_tensor(
            reference.object_local, dtype=torch.float32, device=device
        )
        self.translation = torch.zeros((num_envs, 3), device=device)
        self.yaw = torch.zeros(num_envs, device=device)
        self.mode = config.get("mode", "fade")
        if self.mode not in ("fade", "rigid_sequence"):
            raise ValueError("Unknown augmentation mode")
        self.start, self.end = (
            config.get("blend_start_s", 0.0),
            config.get("blend_end_s", self.duration),
        )
        if self.mode == "fade" and not 0 <= self.start < self.end <= self.duration:
            raise ValueError("Invalid augmentation fade interval")
        self.velocity_grid = None
        if velocity_mode not in ("segment", "control_difference"):
            raise ValueError("Unknown reference velocity mode")
        if velocity_mode == "control_difference":
            if control_dt is None or not np.isfinite(control_dt) or control_dt <= 0:
                raise ValueError("Control-grid velocities need a positive control_dt")
            count = round(self.duration / control_dt) + 1
            if count < 2 or not np.isclose((count - 1) * control_dt, self.duration, atol=1e-7):
                raise ValueError("Reference duration must span complete control intervals")
            self.velocity_dt = float(control_dt)
            grid = reference.sample(np.linspace(0, self.duration, count))
            prev = np.maximum(np.arange(count) - 1, 0)
            nxt = np.minimum(np.arange(count) + 1, count - 1)
            span = ((nxt - prev) * control_dt)[:, None]
            values = {}
            for name in ("q", "wrist", "object"):
                x = grid["q" if name == "q" else name + "_position"]
                v = (x[nxt] - x[prev]) / span
                v[0] = 0
                values[name + "_velocity"] = v
                if name != "q":
                    rotation = Rotation.from_quat(grid[name + "_quaternion"])
                    omega = (rotation[nxt] * rotation[prev].inv()).as_rotvec() / span
                    omega[0] = 0
                    values[name + "_angular_velocity"] = omega
            self.velocity_grid = {
                k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in values.items()
            }

    def reset(self, ids, generator, enabled):
        self.translation[ids], self.yaw[ids] = 0, 0
        if enabled and self.cfg["enabled"]:
            low = torch.tensor(self.cfg["translation_lower_m"], device=self.device)
            high = torch.tensor(self.cfg["translation_upper_m"], device=self.device)
            self.translation[ids] = uniform((len(ids), 3), low, high, self.device, generator)
            self.yaw[ids] = uniform((len(ids),), *self.cfg["yaw_range_rad"], self.device, generator)

    def sample(self, time, env_ids=None):
        t = (
            torch.as_tensor(time, device=self.device, dtype=torch.float32)
            .reshape(-1)
            .clamp(0, self.duration)
        )
        ids = torch.arange(len(self.yaw), device=self.device) if env_ids is None else env_ids
        if len(t) != len(ids):
            raise ValueError("Each reference time must have an environment ID")
        hi = torch.searchsorted(self.times, t, right=True).clamp(1, len(self.times) - 1)
        lo, dt = hi - 1, self.times[hi] - self.times[hi - 1]
        a = (t - self.times[lo]) / dt
        out = dict(
            time=t,
            q=torch.lerp(self.q[lo], self.q[hi], a[:, None]),
            q_velocity=(self.q[hi] - self.q[lo]) / dt[:, None],
        )
        for name, (position, quaternion) in self.poses.items():
            out[name + "_position"] = torch.lerp(position[lo], position[hi], a[:, None])
            out[name + "_quaternion"] = slerp(quaternion[lo], quaternion[hi], a)
            out[name + "_velocity"] = (position[hi] - position[lo]) / dt[:, None]
            out[name + "_angular_velocity"] = (
                rotation_error(quaternion[hi], quaternion[lo]) / dt[:, None]
            )
        if self.velocity_grid is not None:
            # Exact control-frame values for RSI; linear interpolation only for
            # off-grid diagnostic calls. Augmentation rotates these below.
            coordinate = (t / self.velocity_dt).clamp(max=len(self.velocity_grid["q_velocity"]) - 1)
            left = coordinate.floor().long()
            right = (left + 1).clamp(max=len(self.velocity_grid["q_velocity"]) - 1)
            blend = (coordinate - left)[:, None]
            for name, values in self.velocity_grid.items():
                out[name] = torch.lerp(values[left], values[right], blend)
        w = (
            torch.ones_like(t)
            if self.mode == "rigid_sequence"
            else ((self.end - t) / (self.end - self.start)).clamp(0, 1)
        )
        dw = (
            torch.zeros_like(t)
            if self.mode == "rigid_sequence"
            else torch.where((t >= self.start) & (t < self.end), -1 / (self.end - self.start), 0.0)
        )
        delta, yaw = self.translation[ids], self.yaw[ids]
        rv = torch.zeros((len(t), 3), device=self.device)
        rv[:, 2] = w * yaw
        rot = from_rotvec(rv)
        omega = torch.zeros_like(rv)
        omega[:, 2] = dw * yaw
        pivot, pivot_v = out["object_position"].clone(), out["object_velocity"].clone()
        if self.mode == "rigid_sequence":
            # One SE(3) transform of the complete demonstration. In particular,
            # augmentation cannot command a stationary can to slide before grasp.
            pivot = self.poses["object"][0][0].expand(len(t), -1)
            pivot_v = torch.zeros_like(pivot)
        arm = quat_apply(rot, out["wrist_position"] - pivot)
        out["wrist_position"] = pivot + w[:, None] * delta + arm
        out["wrist_velocity"] = (
            pivot_v
            + dw[:, None] * delta
            + quat_apply(rot, out["wrist_velocity"] - pivot_v)
            + torch.cross(omega, arm, dim=-1)
        )
        if self.mode == "rigid_sequence":
            out["object_position"] = pivot + delta + quat_apply(rot, out["object_position"] - pivot)
            out["object_velocity"] = quat_apply(rot, out["object_velocity"])
        else:
            out["object_position"] = pivot + w[:, None] * delta
            out["object_velocity"] = pivot_v + dw[:, None] * delta
        for name in ("wrist", "object"):
            out[name + "_quaternion"] = quat_multiply(rot, out[name + "_quaternion"])
            out[name + "_angular_velocity"] = omega + quat_apply(
                rot, out[name + "_angular_velocity"]
            )
        # Reference starts at rest, matching the source command's first-step convention.
        for name in (
            "q_velocity",
            "wrist_velocity",
            "object_velocity",
            "wrist_angular_velocity",
            "object_angular_velocity",
        ):
            out[name] = torch.where((t <= 1e-8)[:, None], 0.0, out[name])
        out["augmentation_weight"] = w
        return out

    def points(self, position, quaternion):
        return (
            quat_apply(quaternion[:, None], self.object_local[None].expand(len(position), -1, -1))
            + position[:, None]
        )
