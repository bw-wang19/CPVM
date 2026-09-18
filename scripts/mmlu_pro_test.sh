#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/home/wbw/workspace
CONFIG_PATH=${1:-/home/wbw/workspace/CPVM/config/mmlu_pro_test.yaml}
PYTHON_BIN=${PYTHON_BIN:-python}

cd "$WORKSPACE"

# 默认使用当前 shell 中的 Python；仅在显式设置 MMLU_CONDA_ENV 时切换环境。
command=(
    "$PYTHON_BIN"
    -m CPVM.code.test.mmlu_pro_test
    --config "$CONFIG_PATH"
)
if [[ -n "${MMLU_CONDA_ENV:-}" ]]; then
    CONDA_BIN=${CONDA_BIN:-/home/wbw/anaconda3/bin/conda}
    command=(
        "$CONDA_BIN" run --no-capture-output -n "$MMLU_CONDA_ENV"
        "${command[@]}"
    )
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" "${command[@]}"
