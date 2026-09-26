"""Task-scoped paths, input readiness and offline build orchestration."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..tasks import load_task
from .calibration import load_calibration
from .capture import read_capture
from .object_pose import load_mesh
from .schema import fingerprint, load_npz, new_directory, read_json, save_npz, write_json
from .sequence import assemble


class DatasetProject:
    def __init__(self, root, task, demo, config=None):
        self.root = Path(root).resolve()
        self.task = load_task(self.root, task)
        if not str(demo).isdigit() or int(demo) <= 0:
            raise ValueError("Demo must be a positive integer")
        self.demo = str(int(demo))
        self.demo_dir = self.root / self.task.definition.get("data", {}).get(
            self.demo, f"data/{self.task.id}/demo{self.demo}"
        )
        self.work = self.root / "local/dataset" / self.task.id / f"demo{self.demo}"
        self.config_path = (
            Path(config) if config else self.root / "config/tasks" / self.task.id / "capture.json"
        )
        self.config = read_json(self.config_path)
        if (
            self.config.get("schema") != 1
            or self.config.get("task_id") != self.task.id
            or self.config.get("sensor") != "rgb"
        ):
            raise ValueError("Capture configuration must match task, schema=1 and sensor=rgb")

    def path(self, name):
        value = (
            self.config[name]
            .replace("{demo_dir}", str(self.demo_dir))
            .replace("{work_dir}", str(self.work))
        )
        return (self.root / value).resolve()

    def init(self):
        (self.demo_dir / "raw").mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        template = self.demo_dir / "raw/inputs.example.json"
        if not template.exists():
            write_json(
                template,
                dict(
                    sensor="rgb",
                    motion=self.config["motion"],
                    required=[
                        "RGB video with timestamps",
                        "calibration.json: K, distortion, image_size",
                        "known-size drill mesh and mesh_to_object",
                        "measured table/camera frame (static or per-frame)",
                    ],
                    hand_pose="HaMeR estimates or camera-frame annotations; metric status must be explicit",
                    object_pose="measured markers/CAD landmarks or externally estimated camera-frame poses",
                    depth_required=False,
                    fps=None,
                ),
            )
        return self.status()

    def status(self):
        files = {
            name: dict(path=str(self.path(name)), exists=self.path(name).is_file())
            for name in ("capture", "calibration", "boxes", "hands", "object_poses", "camera_poses")
        }
        missing = [
            name
            for name in ("capture", "calibration", "hands", "object_poses")
            if not files[name]["exists"]
        ]
        static = False
        if files["calibration"]["exists"]:
            try:
                c = load_calibration(self.path("calibration"))
                static = c.get("camera_motion") == "fixed" and c.get("T_world_camera") is not None
            except (ValueError, KeyError, TypeError):
                missing.append("valid_calibration")
        if not static and not files["camera_poses"]["exists"]:
            missing.append("per_frame_camera_poses_or_measured_fixed_camera")
        obj = self.config["object"]
        files["mesh"] = dict(
            path=str(self.root / obj["mesh"]), exists=(self.root / obj["mesh"]).is_file()
        )
        if not files["mesh"]["exists"]:
            missing.append("mesh")
        for name in ("mesh_unit_to_m", "mesh_to_object", "measurement_source"):
            if obj.get(name) is None:
                missing.append(name)
        backends = {}
        for name in ("hamer", "egophi"):
            cfg = read_json(self.path(name + "_config"))
            backends[name] = dict(
                repo_exists=(self.root / cfg["repo"]).is_dir(),
                checkpoint_exists=(self.root / cfg["checkpoint"]).is_file(),
                python_configured=bool(cfg.get("python")),
            )
        return dict(
            task=self.task.id,
            demo=self.demo,
            motion=self.config["motion"],
            sensor="rgb",
            depth_required=False,
            files=files,
            missing_inputs=missing,
            contact_context="EgoPHI defaults to skipping frames without both visible hand meshes; single-hand zero_context_debug stays invalid",
            inputs_present=not missing,
            backends=backends,
            data_directory=str(self.demo_dir),
            work_directory=str(self.work),
        )

    def build(self, output=None, hands=None):
        config = self.config
        obj = config["object"]
        if not obj.get("measurement_source") or obj.get("rigid_during_capture") is not True:
            raise ValueError(
                "Drill grasp requires measured mesh scale/frame and a rigid object during capture"
            )
        capture, _, _, _ = read_capture(self.path("capture"))
        calibration = load_calibration(self.path("calibration"))
        camera = (
            load_npz(self.path("camera_poses")) if self.path("camera_poses").is_file() else None
        )
        vertices, triangles, digest = load_mesh(
            self.root / obj["mesh"], obj["mesh_unit_to_m"], obj["mesh_to_object"]
        )
        source_paths = dict(
            capture=self.path("capture"),
            calibration=self.path("calibration"),
            hands=Path(hands) if hands else self.path("hands"),
            object_poses=self.path("object_poses"),
        )
        if camera is not None:
            source_paths["camera_poses"] = self.path("camera_poses")
        sequence = assemble(
            capture,
            load_npz(source_paths["hands"]),
            load_npz(source_paths["object_poses"]),
            calibration,
            camera,
            dict(
                task_id=self.task.id,
                demo_id=self.demo,
                motion=config["motion"],
                object_name=obj["name"],
                object_geometry=obj,
                object_source_sha256=digest,
                assumptions=config.get("assumptions", []),
                sources={
                    k: dict(path=str(p.resolve()), sha256=fingerprint(p))
                    for k, p in source_paths.items()
                },
            ),
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output = Path(output) if output else self.work / ("build_" + stamp)
        # Validate all inputs before allocating a result directory.
        new_directory(output)
        save_npz(
            output / "mesh.npz",
            vertices_m=vertices,
            faces=triangles,
            source_sha256=np.array(digest),
            vertex_ids=np.arange(len(vertices)),
        )
        sequence.meta["processed_mesh_sha256"] = fingerprint(output / "mesh.npz")
        sequence.save(output / "sequence.npz")
        write_json(output / "summary.json", sequence.summary(config["hand_side"]))
        write_json(self.work / "latest.json", dict(directory=str(output.resolve())))
        return output

    def latest(self):
        return Path(read_json(self.work / "latest.json")["directory"])
