WORKSPACE=/home/wbw/workspace
PATH_INTEGRAL_CONFIG=/home/wbw/workspace/CPVM/config/path_integral.yaml

cd "$WORKSPACE" || exit 1

CUDA_VISIBLE_DEVICES=0,1 \
/home/wbw/anaconda3/bin/conda run --no-capture-output -n cpvm-cu132 \
python -m CPVM.code.path_integral \
    --config "$PATH_INTEGRAL_CONFIG"
