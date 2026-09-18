#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-/home/wbw/workspace/CPVM/config/topk_overlap.yaml}

cd /home/wbw/workspace
python -m CPVM.code.topk_overlap --config "$CONFIG_PATH"
