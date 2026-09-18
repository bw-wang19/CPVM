pkill -9 python

WORKSPACE=/home/wbw/workspace
MERGE_CONFIG=/home/wbw/workspace/CPVM/config/merge.yaml

cd "$WORKSPACE" || exit 1

python -m CPVM.code.utils.merge --config "$MERGE_CONFIG"
