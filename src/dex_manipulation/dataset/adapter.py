"""Dataset schema adapter, preserving the certified legacy input loader unchanged."""

import json
from pathlib import Path

import numpy as np

from ..data import frame_mask, load_sequence as load_legacy_sequence, resolve_demo_path


def load_sequence(path, model, config, *, geometric_debug=False):
    settings = (
        json.loads(resolve_demo_path(config).read_text())
        if isinstance(config, (str, Path))
        else dict(config)
    )
    with np.load(path, allow_pickle=False) as archive:
        frame = (
            json.loads(str(archive["frame_metadata_json"]))
            if "frame_metadata_json" in archive
            else {}
        )
        if frame.get("dataset_schema") != 1 and settings.get("dataset_schema") != 1:
            return load_legacy_sequence(path, model, settings, geometric_debug=geometric_debug)
        ids = archive["frame_ids"].copy()
        saved_valid = archive["valid"].copy()
        saved_times = archive["timestamps_s"].copy()
    if saved_valid.dtype != np.bool_ or saved_valid.shape != ids.shape:
        raise ValueError("Saved valid mask must be boolean and match frame_ids")
    if (
        saved_times.shape != ids.shape
        or not np.isfinite(saved_times).all()
        or np.any(np.diff(saved_times) <= 0)
    ):
        raise ValueError("Saved timestamp count/order invalid")
    if frame.get("geometry_status") in ("monocular_estimate", "test_assumptions") or settings.get(
        "geometry_status"
    ) in ("monocular_estimate", "test_assumptions"):
        if not geometric_debug:
            raise ValueError(
                "Monocular or assumed test geometry requires metric alignment or --geometric-debug"
            )
    if settings.get("timestamps_s") is None and settings.get("retimed_fps") is None:
        settings["timestamps_s"] = saved_times.tolist()
    sequence = load_legacy_sequence(path, model, settings, geometric_debug=geometric_debug)
    sequence.valid &= saved_valid[frame_mask(ids, settings.get("frame_range"))]
    return sequence
