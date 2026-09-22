"""Dispatch task rewards while preserving historical checkpoint contracts."""

from pathlib import Path
from ..tasks import config_task_id, load_task


def attach_grasp_reward(env, metadata, training_reference):
    # Compatibility name retained for the shared runner and existing clients.
    root = Path(__file__).resolve().parents[3]
    task = load_task(root, config_task_id(env.cfg))
    return task.entrypoint("reward")(env, metadata, training_reference)
