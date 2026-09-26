"""Episode-boundary placement from verified tabletop IK samples (no simulator)."""

from ..configuration import read_config
from dataclasses import dataclass
import copy
from pathlib import Path
import secrets

import numpy as np

from ..coordinates import collision_bottom
from ..data import legacy_demo_paths, resolve_demo_path
from ..fk import ArmModel, HandModel
from ..ik import ArmIK, IKOptions, checked_pose, model_fingerprint
from ..scene import Workcell
from ..transforms import inverse
from dex_manipulation.policy.trajectory import ReferenceMotion, digest


@dataclass
class Placement:
    grid_index: tuple
    xy: np.ndarray
    world_from_source: np.ndarray
    base_from_source: np.ndarray
    initial_q: np.ndarray


class RandomCanPlacement:
    """Shuffle verified points, without immediate repeats even across cycles."""

    def __init__(self, candidates, metadata, seed=None):
        if not candidates:
            raise ValueError("No verified can placements remain")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise ValueError("Placement seed must be a nonnegative integer")
        self.candidates = candidates
        self.seed = secrets.randbits(63) if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        self.metadata = dict(
            metadata,
            enabled=True,
            seed=self.seed,
            count=len(candidates),
            sampling="shuffled verified grid points; no interpolation or immediate repeats",
            timing="episode reset only; no object writes during motion",
        )
        self.pending, self.previous = [], None

    def next(self):
        if not self.pending:
            self.pending = self.rng.permutation(len(self.candidates)).tolist()
            if len(self.pending) > 1 and self.pending[-1] == self.previous:
                self.pending[0], self.pending[-1] = self.pending[-1], self.pending[0]
        self.previous = self.pending.pop()
        return copy.deepcopy(self.candidates[self.previous])


def load_random_placement(root, config, arm_config, speed, seed=None, reference=None):
    """Fail closed on stale maps/assets, wrong demo, speed, or reference frame.

    The map certifies sampled *retarget* paths, not residual-policy rollouts.
    The policy must share its complete reference motion with that map.
    Subsequent residual commands still pass through the ordinary strict arm IK.
    """
    root = Path(root)
    catalog = read_config(root / "config/tasks/can_pick/play.json")
    demo = catalog["demos"]["2"]
    known = [
        read_config(root / demo[key])["reference"]
        for key in ("config", "training_config")
        if key in demo
    ]
    registered_reference = resolve_demo_path(config["reference"], root).resolve() in [
        resolve_demo_path(path, root).resolve() for path in known
    ]
    if config.get("arm_training", {}).get("enabled") or not (
        registered_reference or str(config.get("demo_id")) == "2"
    ):
        raise ValueError("--random-can supports only demo 2 floating-trained arm policy playback")
    reference_hash = digest(resolve_demo_path(config["reference"], root))
    maps = demo.get("random_can_policy_regions", {}).get(
        reference_hash, demo.get("random_can_regions", {})
    )
    path = maps.get(f"{speed:g}")
    if path is None:
        raise ValueError(
            "--random-can requires a verified IK map for this speed (available: "
            + ", ".join(maps)
            + ")"
        )
    directory = root / path
    if not (directory / "manifest.json").is_file():
        raise ValueError(
            f"Verified IK region is missing: {directory}. Restore or regenerate the tabletop IK report."
        )
    manifest = read_config(directory / "manifest.json")
    if manifest["playback_speed"] != speed:
        raise ValueError("IK region playback speed differs from this playback")
    hashes = manifest["input_hashes"]
    for path, expected in hashes.items():
        resolved = resolve_demo_path(path, root)
        if not resolved.is_file() or digest(resolved) != expected:
            raise ValueError(f"Stale IK region input: {path}. Regenerate the tabletop IK report.")
    scanned_config = read_config(root / manifest["config_path"])
    if (
        legacy_demo_paths({k: v for k, v in arm_config.items() if k != "input"})
        != legacy_demo_paths({k: v for k, v in scanned_config.items() if k != "input"})
        or arm_config["solver"] != manifest["solver"]
    ):
        raise ValueError("IK region arm placement/solver configuration differs")
    for path in (
        manifest["config_path"],
        scanned_config["input"],
        arm_config["workcell"],
        arm_config["alignment"],
        arm_config["arm_model"],
        arm_config["usd"],
        config["model"],
        config["object_geometry"],
    ):
        if resolve_demo_path(path, root) not in {resolve_demo_path(key, root) for key in hashes}:
            raise ValueError(f"IK region does not cover this input: {path}")
    geometry = read_config(directory / "collision_model.json")
    for path, expected in geometry["layers"].items():
        if not resolve_demo_path(path, root).is_file() or digest(resolve_demo_path(path, root)) != expected:
            raise ValueError(f"Stale IK region collision layer: {path}")
    hand = HandModel.load(root / config["model"])
    source = ReferenceMotion(
        resolve_demo_path(scanned_config["input"], root),
        hand,
        root / config["object_geometry"],
        config["world_frame"],
    )
    if reference is None:
        reference = ReferenceMotion(
            root / config["reference"],
            hand,
            root / config["object_geometry"],
            config["world_frame"],
        )
        if "reference_timing" in config:
            reference.configure_timing(config)
        else:
            reference.retime_for_control_horizon(
                config["episode_length_s"], config["physics_dt"] * config["control_decimation"]
            )
        reference = reference.for_playback(
            speed, config["physics_dt"] * config["control_decimation"]
        )
    if (
        reference.metadata["frame_ids"] != manifest["source_frames"]
        or reference.object.shape != source.object.shape
        or reference.wrist.shape != source.wrist.shape
        or reference.q.shape != source.q.shape
        or reference.times.shape != source.times.shape
        or not np.allclose(reference.object, source.object, atol=1e-9, rtol=0)
        or not np.allclose(reference.wrist, source.wrist, atol=1e-9, rtol=0)
        or not np.allclose(reference.q, source.q, atol=1e-9, rtol=0)
        or not np.allclose(reference.times, source.times / speed, atol=1e-8, rtol=0)
        or not np.isclose(reference.duration, manifest["reference_duration_s"], atol=1e-8, rtol=0)
    ):
        raise ValueError(
            "Policy reference motion or timing differs from the verified IK region; regenerate its map"
        )
    if digest(root / config["object_asset"]) != reference.metadata["object_asset_sha256"]:
        raise ValueError("Policy can asset differs from the verified geometry")
    arm = ArmModel.load(root / arm_config["arm_model"])
    if model_fingerprint(arm) != manifest["model_fingerprint"]:
        raise ValueError("IK region arm model fingerprint differs")
    solver = ArmIK(arm, IKOptions(**arm_config["solver"]))
    workcell = Workcell.load(root / arm_config["workcell"])
    nominal = np.asarray(
        workcell.resolve_alignment(read_config(root / arm_config["alignment"]))["world_from_source"]
    )
    grid = read_config(directory / "grid.json")
    candidates = []
    for row in grid:
        if not (
            row["initial_can_supported"]
            and row["ik_success"]
            and row.get("collision_free")
            and not row["near_singular_samples"]
            and not row["near_singular_transition_samples"]
            and not row["near_limit_samples"]
        ):
            continue
        xy = np.array([row["x_m"], row["y_m"]])
        world = nominal.copy()
        world[:2, 3] += xy - (world @ reference.object[0])[:2, 3]
        base = inverse(workcell.world_from_base) @ world
        with np.load(
            directory / "points" / f"{row['ix']:02}_{row['iy']:02}.npz", allow_pickle=False
        ) as saved:
            checked_pose(saved["world_from_source"])
            if (
                not saved["success"].all()
                or not saved["tested"].all()
                or not saved["transition_valid"][1:].all()
                or not np.allclose(world, saved["world_from_source"], atol=1e-9, rtol=0)
            ):
                raise ValueError("IK region row and saved trajectory disagree")
            initial_q = saved["q_arm"][0].copy()
        # Recheck the actual policy start through current FK/IK, not just flags.
        result = solver.solve(base @ reference.wrist[0], initial_q)
        if not result.success or result.near_singular or result.near_joint_limit:
            raise ValueError(f"Cached initial IK no longer passes at {xy.tolist()}")
        if not np.allclose(result.q, initial_q, atol=1e-6, rtol=0):
            raise ValueError("Initial IK branch changed relative to the collision-checked path")
        if abs(collision_bottom(world @ reference.object[0], reference.collision_shapes)) > 1e-6:
            raise ValueError("Random placement moved the can away from the tabletop")
        candidates.append(Placement((row["ix"], row["iy"]), xy, world, base, result.q.copy()))
    return RandomCanPlacement(
        candidates,
        dict(
            demo="2",
            region=str(directory),
            region_manifest_sha256=digest(directory / "manifest.json"),
            region_grid_sha256=digest(directory / "grid.json"),
            grid_step_m=manifest["grid_step_m"],
            playback_speed=speed,
            policy_reference_sha256=reference.metadata["reference_sha256"],
            reference_start_checked=True,
            full_reference_checked=True,
            interpretation="Retarget IK/collision/singularity-checked grid and identical policy start; residual policy motion and physical grasp are not certified",
        ),
        seed,
    )
