# pkill -9 python

export NO_PROXY="127.0.0.1,localhost,api.wandb.ai,wandb.ai,.wandb.ai" 
export no_proxy="127.0.0.1,localhost,api.wandb.ai,wandb.ai,.wandb.ai"
export WANDB_PROJECT='cpvm'

WORKSPACE=/home/wbw/workspace
CONFIG_YAML=/home/wbw/workspace/CPVM/config/train_big5chat.yaml

cd "$WORKSPACE" || exit 1

CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=29500 \
    --module CPVM.code.train_big5chat \
    "$CONFIG_YAML"
