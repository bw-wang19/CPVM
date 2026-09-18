#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/home/wbw/workspace
CONFIG_PATH=${1:-/home/wbw/workspace/CPVM/config/gsm8k_test.yaml}
PYTHON_BIN=${PYTHON_BIN:-python}

cd "$WORKSPACE"

command=(
    "$PYTHON_BIN"
    -m CPVM.code.test.gsm8k_test
    --config "$CONFIG_PATH"
)
if [[ -n "${GSM8K_CONDA_ENV:-}" ]]; then
    CONDA_BIN=${CONDA_BIN:-/home/wbw/anaconda3/bin/conda}
    command=(
        "$CONDA_BIN" run --no-capture-output -n "$GSM8K_CONDA_ENV"
        "${command[@]}"
    )
fi

VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" "${command[@]}"
