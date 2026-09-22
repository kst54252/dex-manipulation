"""Task discovery and adapters around shared FK, IK, retargeting and PPO."""

from .registry import DEFAULT_TASK, TaskSpec, config_task_id, list_tasks, load_task

__all__ = ["DEFAULT_TASK", "TaskSpec", "config_task_id", "list_tasks", "load_task"]
