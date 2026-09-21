#!/usr/bin/env bash
# One entry point from any directory; no shell activation or exports required.
set -euo pipefail
DEX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${DEX_PYTHON:-}" ]]; then
    for candidate in "$HOME/IsaacLab/.venv/bin/python" "$DEX_ROOT/.venv/bin/python"; do
        if [[ -x "$candidate" ]]; then
            DEX_PYTHON="$candidate"
            break
        fi
    done
fi
if [[ -z "${DEX_PYTHON:-}" || ! -x "$DEX_PYTHON" ]]; then
    echo 'Isaac Python을 찾지 못했습니다. DEX_PYTHON=/절대경로/python ./run.sh 로 지정하세요.' >&2
    exit 2
fi
export PYTHONPATH="$DEX_ROOT/src:$DEX_ROOT/.deps:$DEX_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "$DEX_PYTHON" "$DEX_ROOT/scripts/run.py" "$@"
