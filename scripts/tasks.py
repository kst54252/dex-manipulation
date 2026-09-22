#!/usr/bin/env python3
"""Inspect task manifests without loading policies or starting Isaac Sim."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dex_manipulation.tasks import DEFAULT_TASK, list_tasks, load_task


def main(argv=None):
    parser = argparse.ArgumentParser(prog="./run.sh tasks", description=__doc__)
    parser.add_argument("task", nargs="?")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        tasks = [load_task(ROOT, args.task)] if args.task else list_tasks(ROOT)
        if args.json:
            print(json.dumps([task.definition for task in tasks], ensure_ascii=False, indent=2))
            return 0
        for task in tasks:
            default = " (기본)" if task.id == DEFAULT_TASK else ""
            print(f"{task.id}{default}: {task.label}")
            print(f"  설정: {task.path.relative_to(ROOT)}")
            print(f"  데모: {', '.join(task.catalog()['demos']) or '입력 등록 필요'}")
            print(f"  단계: {' → '.join(task.definition.get('phases', []))}")
            if args.task:
                for item in task.definition.get("inputs", []):
                    state = "있음" if (ROOT / item["path"]).is_file() else "등록 필요"
                    print(f"  {item['role']}: {item['path']} [{state}]")
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
