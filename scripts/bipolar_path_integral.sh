#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/home/wbw/workspace}"
CONFIG_PATH="${1:-$WORKSPACE/CPVM/config/bipolar_path_integral.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

cd "$WORKSPACE"
exec "$PYTHON_BIN" -m CPVM.code.path_integral --config "$CONFIG_PATH"
