"""Simulator-independent bridge from source-frame policy targets to RB3 joints."""

from dataclasses import replace
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ..ik import ArmIK, IKOptions, checked_pose
from ..transforms import inverse, transform
from .math3d import quat_apply, quat_multiply


class PolicyFrame:
    """Fixed, explicit world <- policy-source transform; velocities rotate too."""

    def __init__(self, world_from_source, device="cpu"):
        self.world_from_source = checked_pose(world_from_source).copy()
        self.source_from_world = inverse(self.world_from_source)
        self.translation = torch.tensor(
            self.source_from_world[:3, 3], dtype=torch.float32, device=device
        )
        self.quaternion = torch.tensor(
            Rotation.from_matrix(self.source_from_world[:3, :3]).as_quat(),
            dtype=torch.float32,
            device=device,
        )

    def rotate(self, value):
        return quat_apply(self.quaternion.expand(*value.shape[:-1], 4), value)

    def position(self, value):
        return self.rotate(value) + self.translation

    def orientation(self, value):
        return quat_multiply(self.quaternion.expand_as(value), value)

    def state(self, world_state):
        out = dict(world_state)
        for name in ("wrist", "object"):
            out[name + "_position"] = self.position(world_state[name + "_position"])
            out[name + "_quaternion"] = self.orientation(world_state[name + "_quaternion"])
            for suffix in ("_velocity", "_angular_velocity"):
                out[name + suffix] = self.rotate(world_state[name + suffix])
        for name in ("robot_keypoints", "fingertips"):
            out[name] = self.position(world_state[name])
        links = world_state["link_transforms"].clone()
        links[..., :3] = self.position(links[..., :3])
        links[..., 3:] = self.orientation(links[..., 3:])
        out["link_transforms"] = links
        velocities = world_state["link_velocities"].clone()
        velocities[..., :3] = self.rotate(velocities[..., :3])
        velocities[..., 3:] = self.rotate(velocities[..., 3:])
        out["link_velocities"] = velocities
        return out


class PolicyArmBridge:
    """Strict wrist/flange IK, bounded by the last accepted command and actual dt.

    Failed candidates never replace the previous command. No target projection,
    pose relaxation or floating root wrench is applied to the arm.
    """

    def __init__(self, model, base_from_source, seed, options=None):
        self.model = model
        self.base_from_source = checked_pose(base_from_source).copy()
        self.seed = np.asarray(seed, dtype=float).copy()
        # Continuation stays on the local branch; startup retains multistart.
        self.solver = ArmIK(model, replace(options or IKOptions(), continuation_seeds=1))
        self.previous = None

    def target(self, position, quaternion):
        return self.base_from_source @ transform(
            Rotation.from_quat(quaternion).as_matrix(), position
        )

    def reset(self, position, quaternion):
        target = self.target(position, quaternion)
        result = self.solver.solve(target, self.seed)
        if not result.success:
            raise ValueError(f"Initial policy wrist is not reachable: {result.reason}")
        self.previous = result.q.copy()
        return result

    def solve(self, position, quaternion, dt):
        if self.previous is None:
            raise RuntimeError("Initialize the arm bridge before streaming targets")
        target = self.target(position, quaternion)
        before = self.previous.copy()
        result = self.solver.solve(target, before, previous=before, dt=dt)
        transition = (
            self.solver.transition(before, result.q) if result.success else dict(valid=False)
        )
        valid = result.success and transition["valid"]
        if valid:
            self.previous = result.q.copy()
        return dict(
            q_arm=self.previous.copy(),
            wrist_target=target,
            result=result,
            success=valid,
            reason=result.reason
            if not result.success
            else "ok"
            if valid
            else "singular_transition",
            transition=transition,
            joint_step_rad=np.abs(self.previous - before),
            q_arm_velocity=(self.previous - before) / dt,
        )


def named_material_transfer(
    source_names, source_counts, source_materials, target_names, target_counts, target_materials
):
    """Map collider materials by body name and verified per-body shape counts.

    The tensor shape order is body order, then each body's shapes. Zero-shape
    floating carrier bodies may be absent on the arm. Arm-only bodies retain
    their original materials. Counts must come from the model/runtime, not a
    fixed hand offset in the assembled articulation.
    """
    source = np.asarray(source_materials)
    target = np.array(target_materials, copy=True)
    for names, counts, values in (
        (source_names, source_counts, source),
        (target_names, target_counts, target),
    ):
        if len(names) != len(set(names)) or len(names) != len(counts):
            raise ValueError("Material body names must be unique with one shape count each")
        if any(isinstance(n, bool) or int(n) != n or n < 0 for n in counts):
            raise ValueError("Invalid per-body shape counts")
        if values.shape != (sum(counts), 3) or not np.isfinite(values).all():
            raise ValueError("Material tensor does not match the verified collider layout")
    offsets = np.r_[0, np.cumsum(target_counts)]
    source_start = 0
    for name, count in zip(source_names, source_counts):
        if count:
            if name not in target_names:
                raise ValueError(f"Missing collider body: {name}")
            index = target_names.index(name)
            if count != target_counts[index]:
                raise ValueError(f"Collider shape count changed: {name}")
            target[offsets[index] : offsets[index + 1]] = source[
                source_start : source_start + count
            ]
        source_start += count
    return target
