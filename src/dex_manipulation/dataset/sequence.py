"""Common time/coordinate contract for any single manipulated rigid object."""

from dataclasses import dataclass
import json

import numpy as np
from scipy.spatial.transform import Rotation

from .hand_pose import validate_hands
from .schema import (
    CAMERA_AXES,
    HAND_NAMES,
    WORLD_AXES,
    bool_mask,
    faces,
    load_npz,
    metadata,
    same_frames,
    save_npz,
    timeline,
    transform_points,
    transforms,
)


@dataclass
class DemoSequence:
    arrays: dict
    meta: dict

    @classmethod
    def load(cls, path):
        arrays = load_npz(path)
        result = cls(arrays, metadata(arrays))
        result.validate()
        return result

    def validate(self):
        a = self.arrays
        ids, _ = timeline(a["frame_ids"], a["timestamps_s"])
        n = len(ids)
        hv = bool_mask(a["hand_valid"], (n, 2), "hand_valid")
        ov = bool_mask(a["object_valid"], (n,), "object_valid")
        cv = bool_mask(a["camera_valid"], (n,), "camera_valid")
        if (
            self.meta.get("schema") != 1
            or self.meta.get("length_unit") != "m"
            or self.meta.get("world_axes") != WORLD_AXES
            or self.meta.get("camera_axes") != CAMERA_AXES
            or self.meta.get("coordinate_frame") != "table"
            or not self.meta.get("world_frame_source")
        ):
            raise ValueError("Declare metres and the source of the Z-up table frame")
        if self.meta.get("geometry_status") not in (
            "metric_aligned",
            "monocular_estimate",
            "test_assumptions",
        ):
            raise ValueError(
                "Declare geometry_status=metric_aligned/monocular_estimate/test_assumptions"
            )
        if self.meta["geometry_status"] == "test_assumptions" and not self.meta.get("assumptions"):
            raise ValueError("Test geometry must list its assumptions")
        if self.meta.get("assumptions") and self.meta["geometry_status"] != "test_assumptions":
            raise ValueError("Assumed geometry cannot be labelled calibrated")
        if (
            self.meta.get("hand_estimator", {}).get("joints_are_proxies")
            or self.meta.get("hand_estimator", {}).get("landmarks_are_format_proxies")
        ) and self.meta["geometry_status"] != "test_assumptions":
            raise ValueError("Mesh landmark proxies require test_assumptions geometry status")
        if not self.meta.get("time_basis"):
            raise ValueError("Declare the capture time_basis")
        if (
            self.meta["geometry_status"] == "metric_aligned"
            and self.meta.get("hand_estimator", {}).get("metric_calibrated") is not True
        ):
            raise ValueError("Metric status requires declared hand calibration provenance")
        if list(a["joint_names"]) != list(HAND_NAMES) or list(a["hand_sides"]) != ["left", "right"]:
            raise ValueError("Unexpected canonical hand semantics")
        if a["hand_world_m"].shape != (n, 2, 21, 3) or not np.isfinite(a["hand_world_m"][hv]).all():
            raise ValueError("Invalid canonical hand shape/values")
        if a["T_world_object"].shape != (n, 4, 4) or a["T_world_camera"].shape != (n, 4, 4):
            raise ValueError("Camera/object poses must each be (T,4,4)")
        transforms(a["T_world_object"], "T_world_object", ov)
        transforms(a["T_world_camera"], "T_world_camera", cv)
        if np.any(ov & ~cv) or np.any(hv & ~cv[:, None]):
            raise ValueError("World annotations cannot be valid without camera extrinsics")
        for side, name in enumerate(("left", "right")):
            key = f"vertices_{name}_camera_m"
            if key in a:
                mesh = a[key]
                if (
                    mesh.ndim != 3
                    or mesh.shape[0] != n
                    or mesh.shape[2] != 3
                    or not np.isfinite(mesh[hv[:, side]]).all()
                ):
                    raise ValueError(f"Invalid visible {name} hand mesh")
                faces(a[f"faces_{name}"], mesh.shape[1])
        return self

    def save(self, path):
        self.validate()
        save_npz(path, **{**self.arrays, "metadata_json": np.array(json.dumps(self.meta))})

    def summary(self, hand_side="right"):
        a = self.arrays
        side = ["left", "right"].index(hand_side)
        valid = a["hand_valid"][:, side] & a["object_valid"] & a["camera_valid"]
        adjacent = valid[1:] & valid[:-1]
        dt = np.diff(a["timestamps_s"])
        translation = np.linalg.norm(np.diff(a["T_world_object"][:, :3, 3], axis=0), axis=1)
        wrist = np.linalg.norm(np.diff(a["hand_world_m"][:, side, 0], axis=0), axis=1)
        orientation = []
        for i in np.flatnonzero(adjacent):
            r0, r1 = a["T_world_object"][[i, i + 1], :3, :3]
            orientation.append(Rotation.from_matrix(r0.T @ r1).magnitude() / dt[i])
        return dict(
            frames=len(valid),
            valid_frames=int(valid.sum()),
            invalid_frame_ids=a["frame_ids"][~valid].tolist(),
            duration_s=float(a["timestamps_s"][-1] - a["timestamps_s"][0]),
            time_basis=self.meta["time_basis"],
            geometry_status=self.meta["geometry_status"],
            max_object_speed_m_s=float(np.max(translation[adjacent] / dt[adjacent]))
            if adjacent.any()
            else None,
            max_wrist_speed_m_s=float(np.max(wrist[adjacent] / dt[adjacent]))
            if adjacent.any()
            else None,
            max_object_angular_speed_rad_s=max(orientation) if orientation else None,
        )


def assemble(capture, hands, objects, calibration, camera, meta):
    ids, times = timeline(np.asarray(capture["frame_ids"]), capture["timestamps_s"])
    joints, hand_valid, hand_meta = validate_hands(hands, ids, require_metric=False)
    same_frames(objects, ids)
    object_meta = metadata(objects)
    if object_meta.get("camera_axes") != CAMERA_AXES or not object_meta.get("measurement_source"):
        raise ValueError("Object poses require camera axes and measurement_source")
    if capture["image_size"] != calibration["image_size"]:
        raise ValueError("Calibration resolution differs from the recording")
    object_valid = bool_mask(objects["valid"], (len(ids),), "object valid")
    object_pose = transforms(objects["T_camera_object"], "T_camera_object", object_valid)
    if object_pose.shape != (len(ids), 4, 4):
        raise ValueError("T_camera_object must be (T,4,4)")
    for name, archive in (("object", objects), ("hand", hands), ("camera", camera)):
        if archive is not None and "timestamps_s" in archive:
            if np.asarray(archive["timestamps_s"]).shape != times.shape or not np.allclose(
                archive["timestamps_s"], times, rtol=0, atol=1e-7
            ):
                raise ValueError(f"{name} timestamps differ from the RGB capture")
        if archive is not None:
            archive_source = metadata(archive).get("capture_sha256")
            capture_source = meta.get("sources", {}).get("capture", {}).get("sha256")
            if archive_source and capture_source and archive_source != capture_source:
                raise ValueError(f"{name} annotations belong to a different RGB capture")
    if camera is not None:
        same_frames(camera, ids)
        cm = metadata(camera)
        if cm.get("camera_axes") != CAMERA_AXES or not cm.get("measurement_source"):
            raise ValueError("Camera poses require camera axes and measurement_source")
        cv = bool_mask(camera["valid"], (len(ids),), "camera valid")
        wc = transforms(camera["T_world_camera"], "T_world_camera", cv)
        if wc.shape != (len(ids), 4, 4):
            raise ValueError("T_world_camera must be (T,4,4)")
        frame_source = cm["measurement_source"]
    else:
        if calibration.get("camera_motion") != "fixed" or calibration.get("T_world_camera") is None:
            raise ValueError(
                "Moving/unknown camera requires per-frame T_world_camera; no static fallback"
            )
        wc = np.repeat(transforms(calibration["T_world_camera"])[None], len(ids), axis=0)
        cv = np.ones(len(ids), bool)
        frame_source = calibration["extrinsics_source"]
    hv = hand_valid & cv[:, None]
    ov = object_valid & cv
    world_hands = transform_points(wc[:, None], joints)
    world_hands[~hv] = np.nan
    world_objects = wc @ object_pose
    world_objects[~ov] = np.nan
    a = dict(
        frame_ids=ids,
        timestamps_s=times,
        joint_names=np.array(HAND_NAMES),
        hand_sides=np.array(["left", "right"]),
        hand_world_m=world_hands,
        hand_valid=hv,
        T_world_object=world_objects,
        object_valid=ov,
        T_world_camera=wc,
        camera_valid=cv,
    )
    for name in ("reprojection_error_px",):
        if name in objects:
            a["object_" + name] = objects[name]
    for key, value in hands.items():
        if key.startswith("vertices_") or key.startswith("faces_") or key == "alignment_error_m":
            a[key] = value
    result_meta = dict(
        meta,
        schema=1,
        length_unit="m",
        camera_axes=CAMERA_AXES,
        world_axes=WORLD_AXES,
        coordinate_frame="table",
        world_frame_source=frame_source,
        time_basis=capture["time_basis"],
        hand_estimator=hand_meta,
        object_estimator=object_meta,
        temporal_processing="none",
        calibration=calibration,
        geometry_status="test_assumptions"
        if meta.get("assumptions")
        else "metric_aligned"
        if hand_meta.get("metric_calibrated") is True
        else "monocular_estimate",
    )
    return DemoSequence(a, result_meta).validate()
