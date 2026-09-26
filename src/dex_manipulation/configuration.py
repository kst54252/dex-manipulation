"""Resolve historical paths and expand explicit, relative JSON configuration inheritance."""

from copy import deepcopy
import json
from pathlib import Path

from .data import resolve_demo_path


def _merge(base, overrides):
    result = deepcopy(base)
    for name, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(name), dict):
            result[name] = _merge(result[name], value)
        else:
            result[name] = deepcopy(value)
    return result


def read_config(path, root=None, *, _parents=()):
    # Values (including saved reference paths) are never rewritten. Existing
    # resolved checkpoint settings remain ordinary JSON without inheritance.
    path = Path(path).expanduser()
    if root is None and path.is_file():
        path = path.resolve()
    path = resolve_demo_path(path, root).resolve()
    if path in _parents:
        raise ValueError(f"Configuration inheritance cycle: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or "$extends" not in value:
        return value
    parent = value.pop("$extends")
    if not isinstance(parent, str) or not parent or Path(parent).is_absolute():
        raise ValueError("$extends must name one relative JSON file")
    base = read_config(path.parent / parent, root, _parents=(*_parents, path))
    if not isinstance(base, dict):
        raise ValueError("Inherited configuration must be an object")
    return _merge(base, value)
