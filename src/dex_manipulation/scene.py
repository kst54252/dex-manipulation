"""Shared tabletop workcell geometry and explicit world/base frame boundary.

Only create_usd/mount_robot import USD. Layout and IK placement are pure NumPy.
Source robot assets and references are never edited by these adapters.
"""

from .configuration import read_config
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .transforms import transform, inverse


class FloatingSurface:
    """One finite horizontal support; no desk legs, pedestal or lower floor."""

    def __init__(self, description, initial_object_position):
        self.description = description
        size = np.asarray(description["size_xy_m"], float)
        center = np.asarray(description["center_xy_m"], float)
        initial = np.asarray(initial_object_position, float)
        top, thickness = description["top_z_m"], description["thickness_m"]
        if (
            size.shape != (2,)
            or center.shape != (2,)
            or initial.shape != (3,)
            or not np.isfinite(np.r_[size, center, initial, top, thickness]).all()
            or np.any(size <= 0)
            or thickness <= 0
            or top != 0
        ):
            raise ValueError("Floating surface needs positive finite dimensions and top Z=0")
        # Shift hand and object together. Never alter object-local geometry,
        # reference height, orientation or the relative hand/object motion.
        self.translation = np.r_[center - initial[:2], 0.0]
        self.fingerprint = hashlib.sha256(
            json.dumps(description, sort_keys=True).encode()
        ).hexdigest()

    def boxes(self):
        d = self.description
        return {
            "Table": dict(
                size=[*d["size_xy_m"], d["thickness_m"]],
                center=[*d["center_xy_m"], d["top_z_m"] - d["thickness_m"] / 2],
            )
        }

    def metadata(self):
        return dict(
            kind="finite_plane",
            fingerprint=self.fingerprint,
            tabletop_z_m=self.description["top_z_m"],
            lower_floor=False,
            floating_source_translation_m=self.translation.tolist(),
            geometry=self.boxes(),
        )

    def create_usd(self, stage, prefix, origin=(0.0, 0.0, 0.0)):
        from pxr import UsdGeom, UsdPhysics, Gf

        UsdGeom.Xform.Define(stage, prefix)
        box = self.boxes()["Table"]
        cube = UsdGeom.Cube.Define(stage, prefix + "/Table")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*(np.asarray(box["center"]) + origin)))
        cube.AddScaleOp().Set(Gf.Vec3d(*box["size"]))
        cube.CreateDisplayColorAttr([(0.24, 0.29, 0.34)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        return {"Table": cube.GetPrim()}


class Workcell:
    def __init__(self, description):
        self.description = description
        self.fingerprint = hashlib.sha256(
            json.dumps(description, sort_keys=True).encode()
        ).hexdigest()
        d = description
        if d["units"] != "m" or d["coordinate_frame"] != "tabletop_relative":
            raise ValueError("Workcell requires metres and tabletop-relative coordinates")
        base, table = d["robot_base"], d["table"]
        if not np.isclose(table["top_z"], 0.0, atol=1e-12):
            raise ValueError("Tabletop must define world Z=0")
        if (
            not np.isclose(base["center"][2] + base["size"][2] / 2, base["top_z"])
            or not np.isclose(base["center"][2] - base["size"][2] / 2, d["floor_z"])
            or not np.isclose(d["robot_mount"]["position"][2], base["top_z"])
        ):
            raise ValueError("Pedestal, floor and robot mount heights disagree")
        mount = d["robot_mount"]
        self.world_from_base = transform(
            Rotation.from_quat(mount["quaternion_xyzw"]).as_matrix(), mount["position"]
        )
        self.floating_translation = np.asarray(d["floating_source_translation_m"], float)
        if (
            self.floating_translation.shape != (3,)
            or not np.isfinite(self.floating_translation).all()
            or self.floating_translation[2] != 0
        ):
            raise ValueError(
                "Floating placement must be a finite XY translation; preserve tabletop Z=0"
            )
        for box in self.boxes().values():
            if not np.isfinite([box["size"], box["center"]]).all() or np.any(
                np.asarray(box["size"]) <= 0
            ):
                raise ValueError("Invalid workcell box")

    @classmethod
    def load(cls, path):
        return cls(read_config(Path(path)))

    def boxes(self):
        d = self.description
        table, base, floor = d["table"], d["robot_base"], d["display_floor"]
        top, thickness = table["top_z"], table["top_thickness"]
        result = {
            "Table": dict(
                size=[*table["size_xy"], thickness],
                center=[*table["center_xy"], top - thickness / 2],
            ),
            "Pedestal": dict(size=base["size"], center=base["center"]),
            "Floor": dict(
                size=[*floor["size_xy"], floor["thickness"]],
                center=[*floor["center_xy"], d["floor_z"] - floor["thickness"] / 2],
            ),
        }
        height = top - thickness - d["floor_z"]
        for i, xy in enumerate(table["leg_centers_xy"]):
            result[f"Leg{i + 1}"] = dict(
                size=[*table["leg_size_xy"], height], center=[*xy, d["floor_z"] + height / 2]
            )
        return result

    def metadata(self):
        return dict(
            fingerprint=self.fingerprint,
            world_from_base=self.world_from_base.tolist(),
            tabletop_z_m=self.description["table"]["top_z"],
            floor_z_m=self.description["floor_z"],
            floating_source_translation_m=self.floating_translation.tolist(),
            geometry=self.boxes(),
        )

    def resolve_alignment(self, alignment):
        """Convert a tabletop world placement to the physical robot base for IK.

        Explicit measured base_from_source inputs remain supported. Simulation
        placements are checked in world coordinates, not against base Z=0.
        """
        from .ik import checked_pose

        result = dict(alignment)
        if "world_from_source" in result:
            world = checked_pose(result["world_from_source"])
            base = inverse(self.world_from_base) @ world
            if "base_from_source" in result and not np.allclose(
                base, result["base_from_source"], atol=1e-10, rtol=0
            ):
                raise ValueError("World/base placements disagree with the workcell mount")
        else:
            base = checked_pose(result["base_from_source"])
            world = self.world_from_base @ base
        if result["status"].startswith("simulation_ground") and not np.allclose(
            world[2], [0, 0, 1, 0], atol=1e-9, rtol=0
        ):
            raise ValueError("Grounded simulation placement must preserve tabletop world Z=0")
        result.update(base_from_source=base.tolist(), world_from_source=world.tolist())
        result["scene_placement"] = dict(
            workcell_fingerprint=self.fingerprint,
            world_from_base=self.world_from_base.tolist(),
            world_from_source=world.tolist(),
        )
        return result

    def validate_reference(self, metadata, alignment):
        resolved = self.resolve_alignment(alignment)
        if metadata.get("scene_placement") != resolved["scene_placement"] or not np.allclose(
            metadata["base_from_source"], resolved["base_from_source"], atol=1e-10, rtol=0
        ):
            raise ValueError(
                "Arm reference predates this workcell/placement. Run: python scripts/ik.py solve"
            )

    def create_usd(self, stage, prefix="/Workcell", origin=(0.0, 0.0, 0.0), collision=True):
        """Seven boxes, same visible and collision geometry; no physics tuning."""
        from pxr import UsdGeom, UsdPhysics, Gf

        UsdGeom.Xform.Define(stage, prefix)
        colors = {
            "Table": (0.36, 0.25, 0.15),
            "Pedestal": (0.19, 0.23, 0.28),
            "Floor": (0.10, 0.12, 0.15),
        }
        result = {}
        for name, box in self.boxes().items():
            cube = UsdGeom.Cube.Define(stage, prefix + "/" + name)
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*(np.asarray(box["center"]) + origin)))
            cube.AddScaleOp().Set(Gf.Vec3d(*box["size"]))
            cube.CreateDisplayColorAttr([colors.get(name, (0.22, 0.24, 0.27))])
            if collision:
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            result[name] = cube.GetPrim()
        return result

    def mount_robot(self, stage, root_path, base_path):
        """Place the referenced assembly (including its static root-joint anchor).

        base_path is the mapped composed USD base, not an assumed asset origin.
        """
        from pxr import UsdGeom, Gf

        root = UsdGeom.Xformable(stage.GetPrimAtPath(root_path))
        base = stage.GetPrimAtPath(base_path)
        if not base:
            raise ValueError(f"Missing robot base: {base_path}")
        cache = UsdGeom.XformCache()
        old_world = np.asarray(cache.GetLocalToWorldTransform(base)).T
        root_world = np.asarray(cache.GetLocalToWorldTransform(root.GetPrim())).T
        parent_world = np.asarray(cache.GetLocalToWorldTransform(root.GetPrim().GetParent())).T
        pose = inverse(parent_world) @ self.world_from_base @ inverse(old_world) @ root_world
        root.ClearXformOpOrder()
        root.AddTransformOp(opSuffix="workcell").Set(Gf.Matrix4d(pose.T.tolist()))
        actual = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(base)).T
        if not np.allclose(actual, self.world_from_base, atol=1e-7, rtol=0):
            raise ValueError("Robot base did not inherit the workcell transform")
