"""Opt-in measured-angle command governor and tracking costs (radians).

This limits issued drive targets, not force or the post-step physical angle.
A small nonzero lead leaves PD torque available for grasping. No hardware
calibration, motor strength or collision geometry is inferred here.
"""

import math
import torch


class FingerTracking:
    def __init__(self, model, settings, dt, device):
        self.enabled = bool(settings.get("enabled", False))
        if not self.enabled:
            return
        if settings.get("schema") != "measured_finger_tracking_v1":
            raise ValueError("Unknown finger tracking schema")
        limits = settings["max_error_rad"]
        if set(limits) != set(model.active_names):
            raise ValueError("Finger tracking limits must name every active joint exactly")
        values = [limits[name] for name in model.active_names]
        if any(not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError("Finger tracking limits must be finite positive radians")
        self.limit = torch.tensor(values, device=device, dtype=torch.float32)
        self.lower = torch.as_tensor(model.lower, device=device, dtype=torch.float32)
        self.upper = torch.as_tensor(model.upper, device=device, dtype=torch.float32)
        coupling = torch.as_tensor(model.coupling, device=device, dtype=torch.float32).abs()
        followers = torch.as_tensor(model.full_velocity, device=device, dtype=torch.float32)
        effective_velocity = torch.where(
            coupling > 1e-8, followers[:, None] / coupling.clamp_min(1e-8), torch.inf
        ).amin(0)
        self.step = torch.minimum(
            effective_velocity, torch.as_tensor(model.velocity, device=device, dtype=torch.float32)
        ) * dt
        if not bool(torch.isfinite(self.step).all() and (self.step > 0).all()):
            raise ValueError("Finger governor requires finite positive USD joint velocities")
        self.weights = [settings[name] for name in ("command_weight", "tracking_weight")]
        if any(not math.isfinite(x) or x < 0 for x in self.weights):
            raise ValueError("Finger tracking weights must be finite and nonnegative")

    def protect(self, target, measured, previous, coupling, offset):
        if not self.enabled:
            return target
        requested = target["active_q"]
        # Rate/joint limits take priority if an external disturbance makes the
        # measured-angle interval unreachable in one tick. Report that conflict;
        # never fix it with an instantaneous target jump.
        rate_lo = torch.maximum(self.lower, previous - self.step)
        rate_hi = torch.minimum(self.upper, previous + self.step)
        lower = torch.maximum(rate_lo, measured - self.limit)
        upper = torch.minimum(rate_hi, measured + self.limit)
        conflict = lower > upper
        bounded = torch.maximum(torch.minimum(requested, upper), lower)
        fallback = measured.clamp(rate_lo, rate_hi)
        q = torch.where(conflict, fallback, bounded)
        return dict(
            target,
            active_q=q,
            full_q=q @ coupling.T + offset,
            finger_requested_q=requested.clone(),
            finger_guard_conflict=conflict.any(-1),
            finger_guard_correction_rad=(q - requested).abs().amax(-1),
        )

    def score(self, target, measured):
        requested = target["finger_requested_q"]
        applied = target["active_q"]
        actual_error = (applied - measured).abs()
        requested_error = (requested - measured).abs()
        # Score the unprotected request too, so saturation cannot hide a policy
        # that keeps asking for a closed hand. Post-step actual error is logged
        # and penalized independently of whether the governor was active.
        command = ((requested_error - self.limit).clamp_min(0) / self.limit).square().mean(-1)
        tracking = ((actual_error - self.limit).clamp_min(0) / self.limit).square().mean(-1)
        terms = dict(
            finger_command=-self.weights[0] * command,
            finger_tracking=-self.weights[1] * tracking,
        )
        metrics = dict(
            finger_command_error_rad=requested_error.amax(-1),
            finger_tracking_error_rad=actual_error.amax(-1),
            finger_tracking_excess_rad=(actual_error - self.limit).clamp_min(0).amax(-1),
            finger_guard_conflict=target["finger_guard_conflict"].float(),
            finger_guard_correction_rad=target["finger_guard_correction_rad"],
        )
        return terms, metrics
