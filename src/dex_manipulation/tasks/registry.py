"""File-based task registration; listing tasks never imports a simulator."""

from dataclasses import dataclass
import importlib
from pathlib import Path
import re

from ..configuration import read_config

DEFAULT_TASK = "can_pick"


def config_task_id(config):
    # Existing checkpoint bytes and hashes must remain intact.
    value = config.get("task_id", DEFAULT_TASK)
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", value):
        raise ValueError("task_id must be a lowercase name such as can_pick or drilling")
    return value


@dataclass(frozen=True)
class TaskSpec:
    root: Path
    path: Path
    definition: dict

    @property
    def id(self):
        return self.definition["id"]

    @property
    def label(self):
        return self.definition["label"]

    def catalog(self):
        catalog = read_config(self.definition["catalog"], self.root)
        for key in ("demos", "policies"):
            if not isinstance(catalog.get(key), dict):
                raise ValueError(f"{self.id}: catalog.{key} must be a mapping")
        return catalog

    def entrypoint(self, operation):
        target = self.definition.get("entrypoints", {}).get(operation)
        if not target:
            raise ValueError(
                f"{self.id}: {operation} 실행 어댑터와 입력 설정이 필요합니다. "
                f"./run.sh tasks {self.id} 로 필요한 입력을 확인하세요."
            )
        module, separator, name = target.partition(":")
        if not separator or not module.startswith("dex_manipulation.") or not name.isidentifier():
            raise ValueError(f"Invalid task entrypoint: {target}")
        try:
            handler = getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError) as error:
            raise ValueError(f"Cannot load task entrypoint {target}: {error}") from error
        if not callable(handler):
            raise ValueError(f"Task entrypoint is not callable: {target}")
        return handler


def load_task(root, task_id=None):
    root = Path(root).resolve()
    task_id = config_task_id({"task_id": DEFAULT_TASK if task_id is None else task_id})
    path = root / "config/tasks" / task_id / "task.json"
    if not path.is_file():
        raise ValueError(f"등록되지 않은 task: {task_id} (./run.sh tasks)")
    definition = read_config(path)
    if definition.get("schema") != 1 or definition.get("id") != task_id:
        raise ValueError(f"Task manifest schema/id mismatch: {path}")
    if not definition.get("label") or not isinstance(definition.get("entrypoints"), dict):
        raise ValueError(f"Invalid task manifest: {path}")
    return TaskSpec(root, path, definition)


def list_tasks(root):
    return [
        load_task(root, p.parent.name)
        for p in sorted((Path(root) / "config/tasks").glob("*/task.json"))
    ]
