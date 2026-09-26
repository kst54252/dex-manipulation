#!/usr/bin/env bash
# One entry point from any directory; no shell activation or exports required.
set -euo pipefail
DEX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == setup ]]; then
    shift
    exec bash "$DEX_ROOT/scripts/setup_runtime.sh" "$@"
fi
if [[ "${1:-}" == ros || "${1:-}" == robot || ( "${1:-}" == vcb && "${2:-}" =~ ^(bridge|send|status|mirror)$ ) ]]; then
    DEX_ROS_SETUP="${DEX_ROS_SETUP:-/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash}"
    if [[ ! -f "$DEX_ROS_SETUP" ]]; then
        echo "ROS 설정을 찾지 못했습니다: $DEX_ROS_SETUP (DEX_ROS_SETUP으로 지정)" >&2
        exit 2
    fi
    set +u
    source "$DEX_ROS_SETUP"
    set -u
    export ROS_LOG_DIR="${ROS_LOG_DIR:-$DEX_ROOT/local/logs/ros}"
fi
if [[ -z "${DEX_PYTHON:-}" && "${1:-}" == robot && "${2:-}" != sim ]]; then
    if [[ -x "$DEX_ROOT/local/robot-venv/bin/python" ]]; then
        DEX_PYTHON="$DEX_ROOT/local/robot-venv/bin/python"
    elif [[ -x "$DEX_ROOT/local/hardware-venv/bin/python" ]] && "$DEX_ROOT/local/hardware-venv/bin/python" -c 'import pxr, yaml' >/dev/null 2>&1; then
        DEX_PYTHON="$DEX_ROOT/local/hardware-venv/bin/python"
    fi
fi
if [[ -z "${DEX_PYTHON:-}" && "${1:-}" == vcb && "${2:-}" != mirror && -x "$DEX_ROOT/local/vcb-venv/bin/python" ]]; then
    DEX_PYTHON="$DEX_ROOT/local/vcb-venv/bin/python"
fi
if [[ -z "${DEX_PYTHON:-}" && -x "$DEX_ROOT/local/hardware-venv/bin/python" ]]; then
    if [[ "${1:-}" == tactile || ( "${1:-}" == execute && ( "${2:-}" == hardware || "${2:-}" == probe || "${2:-}" == inspect || "${2:-}" == dry-run ) ) ]]; then
        DEX_PYTHON="$DEX_ROOT/local/hardware-venv/bin/python"
    fi
fi
if [[ -z "${DEX_PYTHON:-}" ]]; then
    for candidate in "$HOME/IsaacLab/.venv/bin/python" "$DEX_ROOT/.venv/bin/python"; do
        if [[ -x "$candidate" ]]; then
            DEX_PYTHON="$candidate"
            break
        fi
    done
fi
if [[ -z "${DEX_PYTHON:-}" && ( "${1:-}" == dataset || "${1:-}" == check ) ]]; then
    DEX_PYTHON="$(command -v python3 || true)"
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
