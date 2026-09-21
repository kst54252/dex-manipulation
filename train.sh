#!/usr/bin/env bash
# Share Isaac Python discovery and environment setup with the playback launcher.
set -euo pipefail
DEX_TRAIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$DEX_TRAIN_ROOT/run.sh" train "$@"
