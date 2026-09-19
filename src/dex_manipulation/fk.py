"""Pure NumPy/SciPy FK. This module never imports USD or Isaac Sim."""
import json
from pathlib import Path
import numpy as np
from .transforms import inverse, rotation_about, apply


class HandModel:
    def __init__(self, description):
        self.description = description
        self.root = description["root_link"]
        self.joints = description["joints"]
        self.keypoints = description["keypoints"]
        self.colliders = description.get("colliders", [])
        self.moving = [j for j in self.joints if j["kind"] != "fixed"]
        self.full_names = [j["name"] for j in self.moving]
        self.active_names = [j["name"] for j in self.moving if j.get("mimic") is None]
        self.semantic_names = [k["semantic"] for k in self.keypoints]
        if len(set(self.semantic_names)) != len(self.semantic_names):
            raise ValueError("Duplicate keypoint semantics")
        self._by_name = {j["name"]: j for j in self.moving}
        if len(self._by_name) != len(self.moving):
            raise ValueError("Duplicate joint names")
        expressions = {}

        def resolve(name, visiting):
            if name in visiting:
                raise ValueError(f"Mimic cycle at {name}")
            if name in expressions:
                return expressions[name]
            joint = self._by_name[name]
            mimic = joint.get("mimic")
            if mimic is None:
                row = np.zeros(len(self.active_names))
                row[self.active_names.index(name)] = 1
                result = row, 0.0
            else:
                row, offset = resolve(mimic["leader"], visiting | {name})
                result = row * mimic["multiplier"], offset * mimic["multiplier"] + mimic["offset"]
            expressions[name] = result
            return result

        rows = [resolve(name, set()) for name in self.full_names]
        self.coupling = np.stack([x[0] for x in rows])
        self.offset = np.array([x[1] for x in rows])
        self.full_lower = np.array([j["lower"] for j in self.moving])
        self.full_upper = np.array([j["upper"] for j in self.moving])
        self.full_velocity = np.array([j["velocity"] for j in self.moving])
        self.lower = np.full(len(self.active_names), -np.inf)
        self.upper = np.full(len(self.active_names), np.inf)
        self.velocity = np.full(len(self.active_names), np.inf)
        for i, row in enumerate(self.coupling):
            indices = np.flatnonzero(row)
            if len(indices) == 0:
                if not self.full_lower[i] <= self.offset[i] <= self.full_upper[i]:
                    raise ValueError("Constant mimic violates joint limits")
                continue
            if len(indices) != 1:
                raise ValueError("Only affine single-leader mimic chains supported")
            k = indices[0]
            limits = (np.array([self.full_lower[i], self.full_upper[i]]) - self.offset[i]) / row[k]
            self.lower[k] = max(self.lower[k], limits.min())
            self.upper[k] = min(self.upper[k], limits.max())
            self.velocity[k] = min(self.velocity[k], self.full_velocity[i] / abs(row[k]))
        if np.any(self.lower > self.upper):
            raise ValueError("Inconsistent coupled limits")
        self._frames = [(np.asarray(j["frame0"]), inverse(np.asarray(j["frame1"]))) for j in self.joints]
        self._indices = {name: i for i, name in enumerate(self.full_names)}

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def expand(self, active):
        active = np.asarray(active, dtype=float)
        if active.shape != (len(self.active_names),) or not np.isfinite(active).all():
            raise ValueError("Invalid active joint vector")
        return self.coupling @ active + self.offset

    def link_transforms(self, active, wrist=None):
        full = self.expand(active)
        links = {self.root: np.eye(4) if wrist is None else np.asarray(wrist)}
        for joint, (f0, f1_inv) in zip(self.joints, self._frames):
            motion = np.eye(4)
            if joint["kind"] == "revolute":
                motion = rotation_about(joint["axis"], full[self._indices[joint["name"]]])
            elif joint["kind"] == "prismatic":
                motion[:3, 3] = np.array(joint["axis"]) * full[self._indices[joint["name"]]]
            relative = f0 @ motion @ f1_inv
            if joint.get("reversed", False):
                relative = inverse(relative)
            links[joint["child"]] = links[joint["parent"]] @ relative
        return links

    def keypoint_positions(self, active, wrist=None, links=None):
        links = self.link_transforms(active, wrist) if links is None else links
        return np.stack([apply(links[k["link"]], k["xyz"]) for k in self.keypoints])

    def limit_violation(self, active):
        q = self.expand(active)
        return float(max(0, np.max(self.full_lower - q), np.max(q - self.full_upper)))


class ArmModel(HandModel):
    """Serial arm plus fixed tool mount; SI units, base-relative poses.

    `flange` is the explicitly selected USD rigid link frame, not an inferred
    manufacturer flange-face/TCP frame. No simulator or USD dependency at runtime.
    """

    def __init__(self, description):
        super().__init__(description)
        self.flange = description["flange_link"]
        self.wrist = description["wrist_link"]
        self.mount = np.asarray(description["flange_to_wrist"], dtype=float)
        if len(self.active_names) != 6 or len(self.full_names) != 6:
            raise ValueError("Expected six independent arm joints")
        if any(j["kind"] != "revolute" for j in self.moving):
            raise ValueError("Expected six revolute arm joints")

    def pose(self, q, frame="wrist"):
        if frame not in ("wrist", "flange"):
            raise ValueError("Frame must be wrist or flange")
        return self.link_transforms(q)[getattr(self, frame)]

    def flange_target(self, wrist_target):
        return np.asarray(wrist_target) @ inverse(self.mount)

    def jacobian(self, q, frame="flange"):
        """Spatial geometric Jacobian [metres/rad; radians/rad] in base axes."""
        links = self.link_transforms(q)
        point = links[getattr(self, frame)][:3, 3]
        columns = []
        for joint in self.moving:
            reverse = joint.get("reversed", False)
            axis_frame = links[joint["parent"]] @ np.asarray(joint["frame1" if reverse else "frame0"])
            axis = axis_frame[:3, :3] @ np.asarray(joint["axis"]) * (-1 if reverse else 1)
            columns.append(np.r_[np.cross(axis, point - axis_frame[:3, 3]), axis])
        return np.stack(columns, axis=1)
