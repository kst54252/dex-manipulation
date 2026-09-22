"""Simulator-independent response model for transferring a floating wrist policy.

The actor commands a compliant pose PD controller, not an instantaneous wrist
pose. Approximate that controller's rigid-composite response before strict arm
IK, optionally using a measured contact wrench. Contact generation and internal
finger momentum are not predicted by this model.
"""

import numpy as np
from scipy.spatial.transform import Rotation


class WristResponse:
    def __init__(self, hand_model, body_names, masses, inertias, coms, gains, physics_dt):
        self.hand = hand_model
        self.names = list(body_names)
        self.mass = np.asarray(masses, dtype=float).copy()
        self.inertia = np.asarray(inertias, dtype=float).reshape(-1, 3, 3).copy()
        self.coms = np.asarray(coms, dtype=float).copy()
        self.dt = float(physics_dt)
        count = len(self.names)
        if (
            len(set(self.names)) != count
            or self.mass.shape != (count,)
            or self.inertia.shape != (count, 3, 3)
            or self.coms.shape != (count, 7)
            or not np.isfinite(self.mass).all()
            or not np.isfinite(self.inertia).all()
            or not np.isfinite(self.coms).all()
            or np.any(self.mass <= 0)
            or np.any(np.linalg.eigvalsh(self.inertia) <= 0)
            or not np.isfinite(self.dt)
            or self.dt <= 0
        ):
            raise ValueError("Invalid wrist-response mass properties or physics dt")
        self.com_rotation = Rotation.from_quat(self.coms[:, 3:]).as_matrix()
        self.kp = float(gains["root_kp_position"])
        self.kd = float(gains["root_kd_position"])
        self.kr = float(gains["root_kp_rotation"])
        self.dr = float(gains["root_kd_rotation"])
        if not all(np.isfinite(v) and v > 0 for v in (self.kp, self.kd, self.kr, self.dr)):
            raise ValueError("Wrist response requires positive finite training PD gains")
        self.palm_index = self.names.index(hand_model.root)
        self.link_names = set(hand_model.link_transforms(np.zeros(len(hand_model.active_names))))
        self.carriers = [name for name in self.names if name not in self.link_names]
        if len(self.carriers) != 1:
            raise ValueError("Expected one validated, coincident floating carrier body")
        self.position = self.rotation = self.velocity = self.omega = None

    def reset(self, position, quaternion):
        self.position = np.asarray(position, dtype=float).copy()
        if self.position.shape != (3,) or not np.isfinite(self.position).all():
            raise ValueError("Invalid initial wrist position")
        self.rotation = Rotation.from_quat(quaternion).as_matrix()
        self.velocity = np.zeros(3)
        self.omega = np.zeros(3)

    def snapshot(self):
        return tuple(v.copy() for v in (self.position, self.rotation, self.velocity, self.omega))

    def restore(self, snapshot):
        self.position, self.rotation, self.velocity, self.omega = (v.copy() for v in snapshot)

    def composite(self, q):
        links = self.hand.link_transforms(q)
        centers = []
        inertias = []
        for i, name in enumerate(self.names):
            pose = np.eye(4) if name in self.carriers else links[name]
            centers.append(pose[:3, :3] @ self.coms[i, :3] + pose[:3, 3])
            rotation = pose[:3, :3] @ self.com_rotation[i]
            inertias.append(rotation @ self.inertia[i] @ rotation.T)
        centers = np.asarray(centers)
        total = self.mass.sum()
        center = np.sum(self.mass[:, None] * centers, axis=0) / total
        offsets = centers - center
        inertia = sum(
            body_inertia + mass * (np.eye(3) * np.dot(offset, offset) - np.outer(offset, offset))
            for mass, offset, body_inertia in zip(self.mass, offsets, inertias)
        )
        return total, center, inertia

    def step(
        self,
        position,
        quaternion,
        measured_fingers,
        substeps,
        contact_force=None,
        contact_torque=None,
    ):
        """Hold the actor target for one control interval, integrate at physics dt.

        No reference lookahead, object state, simulator state writes or learned
        rollout is used. Mass/inertia come from the checkpoint; shared hand COMs
        and the carrier frame are validated by the simulator adapter once.
        Optional contact force and moment are expressed in the policy frame;
        the moment is about the current wrist origin, not the composite COM.
        """
        if self.position is None:
            raise RuntimeError("Reset the wrist response first")
        if isinstance(substeps, bool) or int(substeps) != substeps or substeps < 1:
            raise ValueError("Response substeps must be a positive integer")
        target = np.asarray(position, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("Invalid wrist target")
        desired = Rotation.from_quat(quaternion)
        external_force = (
            np.zeros(3) if contact_force is None else np.asarray(contact_force, dtype=float)
        )
        external_torque = (
            np.zeros(3) if contact_torque is None else np.asarray(contact_torque, dtype=float)
        )
        if (
            external_force.shape != (3,)
            or external_torque.shape != (3,)
            or not np.isfinite(external_force).all()
            or not np.isfinite(external_torque).all()
        ):
            raise ValueError("Invalid contact wrench")
        mass, com, inertia = self.composite(measured_fingers)
        for _ in range(int(substeps)):
            rotation = self.rotation
            radius = rotation @ com
            force = self.kp * (target - self.position) - self.kd * self.velocity
            error = (desired * Rotation.from_matrix(rotation).inv()).as_rotvec()
            torque = self.kr * error - self.dr * self.omega
            # Floating training applies force at the palm COM, and torques
            # whose mass-weighted sum is the orientation PD torque.
            torque += np.cross(rotation @ self.coms[self.palm_index, :3] - radius, force)
            torque += external_torque - np.cross(radius, external_force)
            iw = rotation @ inertia @ rotation.T
            alpha = np.linalg.solve(iw, torque - np.cross(self.omega, iw @ self.omega))
            acceleration = (
                (force + external_force) / mass
                - np.cross(alpha, radius)
                - np.cross(self.omega, np.cross(self.omega, radius))
            )
            self.velocity += acceleration * self.dt
            self.position += self.velocity * self.dt
            self.omega += alpha * self.dt
            self.rotation = (
                Rotation.from_rotvec(self.omega * self.dt) * Rotation.from_matrix(rotation)
            ).as_matrix()
        if not all(np.isfinite(v).all() for v in self.snapshot()):
            raise FloatingPointError("Nonfinite virtual wrist state")
        return self.position.copy(), Rotation.from_matrix(self.rotation).as_quat()
