#!/usr/bin/env bash
# Isolated five-finger demo2 experiment; existing train.sh/run.sh are unchanged.
set -euo pipefail
DEX_DEMO2_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${DEX_PYTHON:-}" ]]; then
    for candidate in "$HOME/IsaacLab/.venv/bin/python" "$DEX_DEMO2_ROOT/.venv/bin/python"; do
        if [[ -x "$candidate" ]]; then DEX_PYTHON="$candidate"; break; fi
    done
fi
if [[ -z "${DEX_PYTHON:-}" || ! -x "$DEX_PYTHON" ]]; then
    echo 'DEX_PYTHON=/절대경로/python ./train_demo2.sh 로 Isaac Python을 지정하세요.' >&2
    exit 2
fi
export PYTHONPATH="$DEX_DEMO2_ROOT/src:$DEX_DEMO2_ROOT/.deps:$DEX_DEMO2_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "$DEX_PYTHON" "$DEX_DEMO2_ROOT/scripts/demo2_policy.py" "$@"
