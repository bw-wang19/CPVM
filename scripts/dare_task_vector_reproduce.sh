#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/home/wbw/workspace
CONFIG_PATH=${1:-/home/wbw/workspace/CPVM/config/dare_task_vector.yaml}
PYTHON_BIN=${PYTHON_BIN:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

cd "$WORKSPACE"
exec "$PYTHON_BIN" -m CPVM.code.dare_trait_seed_sweep --config "$CONFIG_PATH"
