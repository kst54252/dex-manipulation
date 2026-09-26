"""Preserve decoded frames and their presentation timestamps; never infer 30 Hz."""

from pathlib import Path
import subprocess
import time
import json

import numpy as np

from .schema import fingerprint, new_directory, timeline, write_json


def video_timestamps(video):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return np.array(
        [
            float(frame["best_effort_timestamp_time"])
            for frame in json.loads(result.stdout)["frames"]
        ]
    )


def import_video(video, output, timestamps=None):
    import cv2

    video = Path(video).resolve()
    times = video_timestamps(video) if timestamps is None else np.asarray(timestamps, dtype=float)
    timeline(np.arange(len(times)), times)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot decode video: {video}")
    output = new_directory(output)
    frames = []
    size = None
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            index = len(frames)
            if index >= len(times):
                raise ValueError("Decoded frames exceed timestamp count")
            if size is None:
                size = [image.shape[1], image.shape[0]]
            if size != [image.shape[1], image.shape[0]]:
                raise ValueError("Image size changed inside a sequence")
            name = f"{index:06d}.png"
            if not cv2.imwrite(str(output / name), image):
                raise OSError(f"Failed to save {name}")
            frames.append(name)
    finally:
        cap.release()
    if len(frames) != len(times):
        raise ValueError(
            "Decoded frame count differs from timestamps; incomplete import has no capture.json"
        )
    record = dict(
        schema=1,
        source=str(video),
        source_sha256=fingerprint(video),
        time_basis="video_presentation_timestamps" if timestamps is None else "supplied_timestamps",
        frame_ids=list(range(len(frames))),
        timestamps_s=times.tolist(),
        images=frames,
        image_size=size,
    )
    write_json(output / "capture.json", record)
    return record


def record_camera(camera, seconds, output):
    """Host receive times are explicit, not mislabeled as camera exposure times."""
    import cv2

    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Capture duration must be positive")
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise ValueError(f"Cannot open camera {camera}")
    output = new_directory(output)
    images, times, size = [], [], None
    start = time.monotonic()
    try:
        while time.monotonic() - start < seconds:
            ok, frame = cap.read()
            timestamp = time.monotonic() - start
            if not ok:
                raise OSError("Camera capture failed")
            if size is None:
                size = [frame.shape[1], frame.shape[0]]
            if size != [frame.shape[1], frame.shape[0]]:
                raise ValueError("Image size changed during recording")
            name = f"{len(images):06d}.png"
            if not cv2.imwrite(str(output / name), frame):
                raise OSError(f"Failed to save {name}")
            images.append(name)
            times.append(timestamp)
    finally:
        cap.release()
    timeline(np.arange(len(times)), times)
    record = dict(
        schema=1,
        source=f"camera:{camera}",
        time_basis="host_monotonic_receive",
        frame_ids=list(range(len(images))),
        timestamps_s=times,
        image_size=size,
        images=images,
    )
    write_json(output / "capture.json", record)
    return record


def read_capture(path):
    from .schema import read_json

    path = Path(path).resolve()
    data = read_json(path)
    ids, times = timeline(np.asarray(data["frame_ids"]), data["timestamps_s"])
    if data.get("schema") != 1 or len(data["images"]) != len(ids) or not data.get("time_basis"):
        raise ValueError("Invalid capture manifest")
    images = [(path.parent / name).resolve() for name in data["images"]]
    if any(not p.is_relative_to(path.parent) or not p.is_file() for p in images):
        raise ValueError("Capture images must exist inside the capture directory")
    return data, ids, times, images
