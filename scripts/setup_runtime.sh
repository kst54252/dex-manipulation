#!/usr/bin/env bash
# Install dependencies in an explicit environment; never start or command a robot.
set -euo pipefail
DEX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-}"
case "$mode" in
  hardware)
    host_python="${2:-python3}"
    "$host_python" -m venv --system-site-packages "$DEX_ROOT/local/hardware-venv"
    runtime_python="$DEX_ROOT/local/hardware-venv/bin/python"
    "$runtime_python" -m pip install -r "$DEX_ROOT/config/dependencies/hardware.txt"
    if [[ ! -e "$DEX_ROOT/local/hardware.json" ]]; then
      cp "$DEX_ROOT/config/hardware.example.json" "$DEX_ROOT/local/hardware.json"
    fi
    ;;
  robot)
    host_python="${2:-python3}"
    "$host_python" -m venv --system-site-packages "$DEX_ROOT/local/robot-venv"
    runtime_python="$DEX_ROOT/local/robot-venv/bin/python"
    "$runtime_python" -m pip install -e "$DEX_ROOT[robot,hardware]"
    if [[ ! -e "$DEX_ROOT/local/hardware.json" ]]; then
      cp "$DEX_ROOT/config/hardware.example.json" "$DEX_ROOT/local/hardware.json"
    fi
    ;;
  sim)
    runtime_python="${2:-${DEX_PYTHON:-$HOME/IsaacLab/.venv/bin/python}}"
    if [[ ! -x "$runtime_python" ]]; then
      echo '사용법: ./run.sh setup sim /Isaac환경/bin/python' >&2
      exit 2
    fi
    "$runtime_python" -c 'import importlib.util; assert importlib.util.find_spec("isaacsim"), "Install Isaac Sim in this Python environment first"'
    if ! "$runtime_python" -m pip --version >/dev/null 2>&1; then
      "$runtime_python" -m ensurepip --upgrade
    fi
    "$runtime_python" -m pip install -e "$DEX_ROOT[rl,validation]"
    ;;
  *)
    echo '사용법: ./run.sh setup {hardware|robot} [python3] 또는 ./run.sh setup sim /Isaac환경/bin/python' >&2
    exit 2
    ;;
esac
DEX_PYTHON="$runtime_python" "$DEX_ROOT/run.sh" check
if [[ "$mode" == hardware ]]; then
  echo '실물 IP·RS485 포트·실측 관절 보정은 local/hardware.json에 입력하세요. 연결이나 모터 명령은 수행하지 않았습니다.'
else
  echo "준비 완료. 다른 Python 경로를 사용했다면 DEX_PYTHON=$runtime_python ./run.sh 로 실행하세요."
fi
