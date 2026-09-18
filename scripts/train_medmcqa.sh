#!/usr/bin/env bash
set -euo pipefail

export NO_PROXY="127.0.0.1,localhost,api.wandb.ai,wandb.ai,.wandb.ai" 
export no_proxy="127.0.0.1,localhost,api.wandb.ai,wandb.ai,.wandb.ai"

export WANDB_PROJECT="${WANDB_PROJECT:-cpvm}"

WORKSPACE="/home/wbw/workspace"
CONFIG_YAML="${1:-/home/wbw/workspace/CPVM/config/train_medmcqa.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

cd "$WORKSPACE"

torchrun \
    --standalone \
    --nproc_per_node="$NPROC_PER_NODE" \
    --module CPVM.code.train_medmcqa \
    "$CONFIG_YAML"
