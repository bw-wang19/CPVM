#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/home/wbw/workspace
CONFIG_PATH=${1:-/home/wbw/workspace/CPVM/config/topk_merge.yaml}
PYTHON_BIN=${PYTHON_BIN:-python}

cd "$WORKSPACE"
"$PYTHON_BIN" -m CPVM.code.topk_merge --config "$CONFIG_PATH"
