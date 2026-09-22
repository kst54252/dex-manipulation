"""Read project settings while preserving historical checkpoint path strings."""

import json
from pathlib import Path

from .data import resolve_demo_path


def read_config(path, root=None):
    # Resolve the file location only. Rewriting values would change the saved
    # controller/reference contract and invalidate existing policy weights.
    path = Path(path).expanduser()
    if root is not None or not path.is_file():
        path = resolve_demo_path(path, root)
    return json.loads(path.read_text())
