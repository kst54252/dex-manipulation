"""Deterministic CPU joint-servo mock with observable tracking lag, not contacts."""

import time
import numpy as np


class VirtualJointBackend:
    hardware = False
    feedback_source = dict(arm="virtual_joint_servo", hand="virtual_joint_servo")

    def __init__(self, initial, velocity, time_constant=0.025, clock=time.monotonic):
        self.q = np.asarray(initial, float).copy()
        self.target = self.q.copy()
        self.velocity = np.asarray(velocity, float)
        self.tau = float(time_constant)
        self.clock = clock
        if (
            self.q.shape != (12,)
            or self.velocity.shape != (12,)
            or not np.isfinite(np.r_[self.q, self.velocity, self.tau]).all()
            or self.tau <= 0
            or np.any(self.velocity <= 0)
        ):
            raise ValueError("Invalid virtual joint-servo configuration")
        self.last = clock()
        self.sent = False

    def advance(self):
        now = self.clock()
        dt = max(0.0, now - self.last)
        self.last = now
        step = (self.target - self.q) * (1 - np.exp(-dt / self.tau))
        self.q += np.clip(step, -self.velocity * dt, self.velocity * dt)

    async def connect(self):
        self.last = self.clock()

    async def read(self):
        self.advance()
        return dict(
            q_rad=self.q.copy(),
            sample_time_s=self.clock(),
            ready=True,
            feedback_source=self.feedback_source,
            physical_simulation=False,
        )

    async def send(self, q, arm_velocity, dt):
        self.advance()
        q = np.asarray(q, float)
        if q.shape != (12,) or not np.isfinite(q).all():
            raise ValueError("Invalid virtual command")
        self.target = q.copy()
        self.sent = True

    async def stop(self):
        self.advance()
        self.target = self.q.copy()
        self.sent = False

    async def close(self):
        pass
