"""Simulator-independent motion commands and floating wrist pose PD."""

import copy
import math
import torch
from .policy.math3d import rotation_error, from_rotvec, quat_multiply, quat_normalize


class MotionController:
    """Causal SE(3)/finger command governor, in SI units and world-frame XYZW.

    Policy targets arrive at control rate; commands evolve at physics rate.
    Velocity and acceleration limits constrain commands, not measured contact
    dynamics. No body pose or velocity is assigned by this class. Its output can
    also feed a separate arm IK solver; it does not assert arm reachability.
    """

    def __init__(self, model, count, dt, config, device="cpu"):
        self.config = copy.deepcopy(config)
        self.dt = float(dt)
        self.enabled = bool(config and config.get("enabled", True))
        self.device = device
        self.lower = torch.as_tensor(model.lower, dtype=torch.float32, device=device)
        self.upper = torch.as_tensor(model.upper, dtype=torch.float32, device=device)
        self.coupling = torch.as_tensor(model.coupling, dtype=torch.float32, device=device)
        self.offset = torch.as_tensor(model.offset, dtype=torch.float32, device=device)
        # Account for every coupled follower's velocity limit as well.
        coeff = self.coupling.abs()
        full_v = torch.as_tensor(model.full_velocity, dtype=torch.float32, device=device)
        coupled_v = torch.where(
            coeff > 1e-8, full_v[:, None] / coeff.clamp_min(1e-8), torch.inf
        ).amin(0)
        self.joint_velocity_limit = torch.minimum(
            coupled_v, torch.as_tensor(model.velocity, dtype=torch.float32, device=device)
        )
        if self.dt <= 0 or not math.isfinite(self.dt):
            raise ValueError("Motion controller dt must be positive and finite")
        if self.enabled:
            required = (
                "position_time_constant_s",
                "rotation_time_constant_s",
                "joint_time_constant_s",
                "linear_velocity_m_s",
                "linear_acceleration_m_s2",
                "angular_velocity_rad_s",
                "angular_acceleration_rad_s2",
                "joint_acceleration_rad_s2",
            )
            for name in required:
                value = config.get(name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(f"Invalid motion-control parameter: {name}")
            if min(config[k] for k in required[:3]) < self.dt:
                raise ValueError("Motion-control time constants must be at least one physics step")
            correction = config.get("contact_correction_velocity_m_s", 5.0)
            if isinstance(correction, bool) or not math.isfinite(correction) or correction <= 0:
                raise ValueError("Contact correction velocity must be positive and finite")
        self.position = torch.zeros(count, 3, device=device)
        self.quaternion = torch.zeros(count, 4, device=device)
        self.quaternion[:, 3] = 1
        self.q = torch.zeros(count, len(model.active_names), device=device)
        self.velocity = torch.zeros_like(self.position)
        self.omega = torch.zeros_like(self.position)
        self.q_velocity = torch.zeros_like(self.q)

    def reset(self, ids, state):
        """Reset only selected environments, starting from measured pose at rest."""
        self.position[ids] = state["wrist_position"][ids]
        self.quaternion[ids] = quat_normalize(state["wrist_quaternion"][ids])
        self.q[ids] = state["q"][ids].clamp(self.lower, self.upper)
        self.velocity[ids] = 0
        self.omega[ids] = 0
        self.q_velocity[ids] = 0

    @staticmethod
    def _vector_limit(value, limit):
        return value * (limit / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1)

    def _velocity(self, error, previous, maximum, acceleration, tau, vector):
        limit = self._vector_limit if vector else lambda x, bound: x.clamp(-bound, bound)
        # A braking envelope prevents the short-tau/saturated-acceleration
        # oscillation caused by repeatedly chasing a target at maximum speed.
        distance = error.norm(dim=-1, keepdim=True) if vector else error.abs()
        cap = torch.minimum(
            torch.as_tensor(maximum, device=error.device), (2 * acceleration * distance).sqrt()
        )
        desired = limit(error / tau, cap)
        delta = (desired - previous) * (-math.expm1(-self.dt / tau))
        return limit(previous + limit(delta, acceleration * self.dt), maximum)

    def step(self, target):
        if not self.enabled:
            return target
        c = self.config
        self.velocity = self._velocity(
            target["position"] - self.position,
            self.velocity,
            c["linear_velocity_m_s"],
            c["linear_acceleration_m_s2"],
            c["position_time_constant_s"],
            True,
        )
        self.omega = self._velocity(
            rotation_error(target["quaternion"], self.quaternion),
            self.omega,
            c["angular_velocity_rad_s"],
            c["angular_acceleration_rad_s2"],
            c["rotation_time_constant_s"],
            True,
        )
        self.position = self.position + self.velocity * self.dt
        self.quaternion = quat_normalize(
            quat_multiply(from_rotvec(self.omega * self.dt), self.quaternion)
        )
        acceleration = c["joint_acceleration_rad_s2"]
        self.q_velocity = self._velocity(
            target["active_q"].clamp(self.lower, self.upper) - self.q,
            self.q_velocity,
            self.joint_velocity_limit,
            acceleration,
            c["joint_time_constant_s"],
            False,
        )

        # Discrete braking viability: after this step enough distance remains to
        # stop with the specified acceleration. This avoids clipping positions
        # at a limit and silently introducing an unbounded velocity change.
        def stopping_speed(distance):
            return (
                (acceleration * self.dt) ** 2 + 2 * acceleration * distance.clamp_min(0)
            ).sqrt() - acceleration * self.dt

        self.q_velocity = self.q_velocity.clamp(
            -stopping_speed(self.q - self.lower), stopping_speed(self.upper - self.q)
        )
        self.q = self.q + self.q_velocity * self.dt
        return dict(
            target,
            position=self.position.clone(),
            quaternion=self.quaternion.clone(),
            active_q=self.q.clone(),
            full_q=self.q @ self.coupling.T + self.offset,
        )


def wrist_pd_wrenches(state, target, masses, root_id, gains):
    """World-frame force at root COM and mass-weighted torques on all links.

    Independently implement the local REGRIND pose-PD method. Targets are held
    for one control interval; feedback is evaluated each physics tick. Damping
    acts on measured velocity, with no gravity or reference-velocity feedforward.
    Gravity is disabled on hand bodies in the environment, not on the object.
    """
    force = (
        gains["root_kp_position"] * (target["position"] - state["wrist_position"])
        - gains["root_kd_position"] * state["wrist_velocity"]
    )
    torque = (
        gains["root_kp_rotation"] * rotation_error(target["quaternion"], state["wrist_quaternion"])
        - gains["root_kd_rotation"] * state["wrist_angular_velocity"]
    )
    forces = torch.zeros((*masses.shape, 3), device=force.device, dtype=force.dtype)
    forces[:, root_id] = force
    weights = masses / masses.sum(dim=1, keepdim=True)
    torques = weights[..., None] * torque[:, None]
    return forces, torques
