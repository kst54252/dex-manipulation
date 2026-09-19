"""Device-resident reference interpolation and coherent episode augmentation."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .math3d import quat_apply, quat_multiply, from_rotvec, rotation_error, slerp, uniform


class TensorReference:
    def __init__(self, reference, num_envs, config, device):
        self.base, self.duration, self.cfg, self.device = reference, reference.duration, config, torch.device(device)
        self.times = torch.as_tensor(reference.times, dtype=torch.float32, device=device)
        self.q = torch.as_tensor(reference.q, dtype=torch.float32, device=device)
        self.poses = {}
        for name, data in (("wrist", reference.wrist), ("object", reference.object)):
            self.poses[name] = (torch.tensor(data[:, :3, 3], dtype=torch.float32, device=device),
                                torch.tensor(Rotation.from_matrix(data[:, :3, :3]).as_quat(), dtype=torch.float32, device=device))
        self.object_local = torch.as_tensor(reference.object_local, dtype=torch.float32, device=device)
        self.translation = torch.zeros((num_envs, 3), device=device)
        self.yaw = torch.zeros(num_envs, device=device)
        self.start, self.end = config["blend_start_s"], config["blend_end_s"]
        if not 0 <= self.start < self.end <= self.duration:
            raise ValueError("Invalid augmentation fade interval")

    def reset(self, ids, generator, enabled):
        self.translation[ids], self.yaw[ids] = 0, 0
        if enabled and self.cfg["enabled"]:
            low = torch.tensor(self.cfg["translation_lower_m"], device=self.device)
            high = torch.tensor(self.cfg["translation_upper_m"], device=self.device)
            self.translation[ids] = uniform((len(ids), 3), low, high, self.device, generator)
            self.yaw[ids] = uniform((len(ids),), *self.cfg["yaw_range_rad"], self.device, generator)

    def sample(self, time, env_ids=None):
        t = torch.as_tensor(time, device=self.device, dtype=torch.float32).reshape(-1).clamp(0, self.duration)
        ids = torch.arange(len(self.yaw), device=self.device) if env_ids is None else env_ids
        if len(t) != len(ids):
            raise ValueError("Each reference time must have an environment ID")
        hi = torch.searchsorted(self.times, t, right=True).clamp(1, len(self.times)-1)
        lo, dt = hi-1, self.times[hi]-self.times[hi-1]
        a = (t-self.times[lo])/dt
        out = dict(time=t, q=torch.lerp(self.q[lo], self.q[hi], a[:, None]), q_velocity=(self.q[hi]-self.q[lo])/dt[:, None])
        for name, (position, quaternion) in self.poses.items():
            out[name+"_position"] = torch.lerp(position[lo], position[hi], a[:, None])
            out[name+"_quaternion"] = slerp(quaternion[lo], quaternion[hi], a)
            out[name+"_velocity"] = (position[hi]-position[lo])/dt[:, None]
            out[name+"_angular_velocity"] = rotation_error(quaternion[hi], quaternion[lo])/dt[:, None]
        w = ((self.end-t)/(self.end-self.start)).clamp(0, 1)
        dw = torch.where((t>=self.start)&(t<self.end), -1/(self.end-self.start), 0.)
        delta, yaw = self.translation[ids], self.yaw[ids]
        rv = torch.zeros((len(t), 3), device=self.device)
        rv[:, 2] = w*yaw
        rot = from_rotvec(rv)
        omega = torch.zeros_like(rv)
        omega[:, 2] = dw*yaw
        pivot, pivot_v = out["object_position"].clone(), out["object_velocity"].clone()
        arm = quat_apply(rot, out["wrist_position"]-pivot)
        out["wrist_position"] = pivot+w[:, None]*delta+arm
        out["wrist_velocity"] = pivot_v+dw[:, None]*delta+quat_apply(rot,out["wrist_velocity"]-pivot_v)+torch.cross(omega,arm,dim=-1)
        out["object_position"] = pivot+w[:, None]*delta
        out["object_velocity"] = pivot_v+dw[:, None]*delta
        for name in ("wrist", "object"):
            out[name+"_quaternion"] = quat_multiply(rot,out[name+"_quaternion"])
            out[name+"_angular_velocity"] = omega+quat_apply(rot,out[name+"_angular_velocity"])
        # Reference starts at rest, matching the source command's first-step convention.
        for name in ("q_velocity", "wrist_velocity", "object_velocity", "wrist_angular_velocity", "object_angular_velocity"):
            out[name] = torch.where((t<=1e-8)[:, None], 0., out[name])
        out["augmentation_weight"] = w
        return out

    def points(self, position, quaternion):
        return quat_apply(quaternion[:, None], self.object_local[None].expand(len(position), -1, -1))+position[:, None]
