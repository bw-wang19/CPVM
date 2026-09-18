#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/home/wbw/workspace
CONFIG_PATH=${1:-/home/wbw/workspace/CPVM/config/aime25_test.yaml}
PYTHON_BIN=${PYTHON_BIN:-python}

cd "$WORKSPACE"

command=(
    "$PYTHON_BIN"
    -m CPVM.code.test.aime25_test
    --config "$CONFIG_PATH"
)
if [[ -n "${AIME_CONDA_ENV:-}" ]]; then
    CONDA_BIN=${CONDA_BIN:-/home/wbw/anaconda3/bin/conda}
    command=(
        "$CONDA_BIN" run --no-capture-output -n "$AIME_CONDA_ENV"
        "${command[@]}"
    )
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" "${command[@]}"
