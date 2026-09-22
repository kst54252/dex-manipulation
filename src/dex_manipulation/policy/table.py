"""Collision-geometry tabletop clearance, shared by floating and arm control.

Link-local enclosing boxes are derived from EVERY extracted collision vertex.
The horizontal half-space is deliberately conservative outside the table edges.
This is a command safeguard, not a proof of collision-free dynamic execution.
"""

import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .math3d import quat_apply, quat_inverse, quat_multiply, from_rotvec


def compose(a, b):
    aq, bq = torch.broadcast_tensors(a[..., 3:], b[..., 3:])
    return torch.cat(
        (a[..., :3] + quat_apply(aq, b[..., :3].expand_as(aq[..., :3])), quat_multiply(aq, bq)), -1
    )


def invert(a):
    q = quat_inverse(a[..., 3:])
    return torch.cat((quat_apply(q, -a[..., :3]), q), -1)


class TableClearance:
    def __init__(self, model, body_names, config, device="cpu"):
        self.model, self.config = model, dict(config)
        self.device = torch.device(device)
        self.enabled = bool(config.get("enabled", False))
        self.margin = float(config.get("clearance_m", 0.004))
        self.lookahead = float(config.get("lookahead_s", 0.02))
        if not math.isfinite(self.margin) or self.margin <= 0 or not 0 <= self.lookahead <= 0.2:
            raise ValueError("Invalid table clearance/lookahead")
        if any(
            not math.isfinite(float(config.get(k, 0))) or float(config.get(k, 0)) < 0
            for k in ("actual_penalty_weight", "target_penalty_weight")
        ):
            raise ValueError("Table penalty weights must be finite and nonnegative")
        self.guard_enabled = self.enabled and config.get("guard_enabled", True)

        def tensor(v):
            return torch.as_tensor(np.asarray(v), device=device, dtype=torch.float32)

        if not model.colliders:
            raise ValueError("Table clearance requires extracted collision geometry")
        self.names = [c["link"] for c in model.colliders]
        self.ids = torch.tensor([body_names.index(n) for n in self.names], device=device)
        lows, highs = [], []
        for c in model.colliders:
            vertices = np.asarray(c["vertices"])
            if (
                vertices.ndim != 2
                or vertices.shape[1] != 3
                or not len(vertices)
                or not np.isfinite(vertices).all()
            ):
                raise ValueError("Invalid hand collision vertices")
            lows.append(vertices.min(0))
            highs.append(vertices.max(0))
        lo, hi = tensor(lows), tensor(highs)
        self.center, self.half = (hi + lo) / 2, (hi - lo) / 2
        self.radius = self.center.norm(dim=-1) + self.half.norm(dim=-1)
        self.coupling, self.offset = tensor(model.coupling), tensor(model.offset)
        self.z = tensor([0.0, 0.0, 1.0])
        self.frames = []
        for j in model.joints:

            def frame(key):
                a = np.asarray(j[key])
                return tensor(np.r_[a[:3, 3], Rotation.from_matrix(a[:3, :3]).as_quat()])

            self.frames.append(
                (
                    frame("frame0"),
                    invert(frame("frame1")),
                    tensor(j.get("axis", [0.0, 0.0, 0.0])),
                    model.full_names.index(j["name"]) if j["kind"] != "fixed" else None,
                )
            )

    def fk(self, target):
        """Batched SI FK with both joint frames, axis, affine coupling/reversal."""
        q = target["active_q"] @ self.coupling.T + self.offset
        root = torch.cat((target["position"], target["quaternion"]), -1)
        links = {self.model.root: root}
        for j, (f0, f1, axis, index) in zip(self.model.joints, self.frames):
            motion = torch.zeros_like(root)
            motion[:, 6] = 1.0
            if j["kind"] == "revolute":
                motion[:, 3:] = from_rotvec(q[:, index : index + 1] * axis)
            elif j["kind"] == "prismatic":
                motion[:, :3] = q[:, index : index + 1] * axis
            elif j["kind"] != "fixed":
                raise ValueError(f"Unsupported joint kind: {j['kind']}")
            relative = compose(compose(f0, motion), f1)
            if j.get("reversed", False):
                relative = invert(relative)
            links[j["child"]] = compose(links[j["parent"]], relative)
        return torch.stack([links[n] for n in self.names], 1)

    def lower(self, poses):
        normal = quat_apply(quat_inverse(poses[..., 3:]), self.z.expand_as(poses[..., :3]))
        return poses[..., 2] + (normal * self.center).sum(-1) - (normal.abs() * self.half).sum(-1)

    def target_clearance(self, target):
        return self.lower(self.fk(target)).amin(-1)

    def actual_clearance(self, state):
        return self.lower(state["link_transforms"][:, self.ids]).amin(-1)

    def begin(self, target, state):
        self.raw_target = target
        self.raw_clearance = self.target_clearance(target)
        self.minimum = self.actual_clearance(state)
        self.maximum_lift = torch.zeros_like(self.minimum)

    def observe(self, state):
        self.minimum = torch.minimum(self.minimum, self.actual_clearance(state))

    def guard(self, target, state):
        if not self.guard_enabled:
            return target
        # Respect the actual finger configuration too: a drive may lag its target.
        poses = state["link_transforms"][:, self.ids]
        wrist = torch.cat((state["wrist_position"], state["wrist_quaternion"]), -1)
        goal = torch.cat((target["position"], target["quaternion"]), -1)
        actual_at_goal = compose(goal[:, None], compose(invert(wrist)[:, None], poses))
        desired = self.raw_clearance if target is self.raw_target else self.target_clearance(target)
        command_lower = torch.minimum(desired, self.lower(actual_at_goal).amin(-1))
        velocity = state["link_velocities"][:, self.ids]
        down_speed = (-velocity[..., 2]).clamp_min(0) + velocity[..., 3:].norm(dim=-1) * self.radius
        predicted = (self.lower(poses) - self.lookahead * down_speed).amin(-1)
        # Preserve XY, orientation and fingers. Raise only the commanded wrist Z.
        current_to_target = target["position"][:, 2] - state["wrist_position"][:, 2]
        lift = torch.maximum(
            self.margin - command_lower, self.margin - predicted - current_to_target
        ).clamp_min(0)
        position = target["position"].clone()
        position[:, 2] += lift
        self.maximum_lift = torch.maximum(self.maximum_lift, lift)
        return dict(target, position=position)

    def metrics(self):
        return dict(
            table_clearance_m=self.minimum,
            table_raw_target_clearance_m=self.raw_clearance,
            table_target_lift_m=self.maximum_lift,
            table_clearance_violation_m=(self.margin - self.minimum).clamp_min(0),
        )
