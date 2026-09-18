export WANDB_PROJECT='cpvm'

WORKSPACE=/home/wbw/workspace
DPO_YAML=/home/wbw/workspace/CPVM/config/dpo.yaml

cd "$WORKSPACE" || exit 1

CUDA_VISIBLE_DEVICES=0,1 \
torchrun \
    --standalone \
    --nproc_per_node=2 \
    --module CPVM.code.dpo \
    "$DPO_YAML"
