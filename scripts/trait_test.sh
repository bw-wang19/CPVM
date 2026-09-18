WORKSPACE=/home/wbw/workspace
TRAIT_TEST_CONFIG=/home/wbw/workspace/CPVM/config/trait_test.yaml

cd "$WORKSPACE" || exit 1

CUDA_VISIBLE_DEVICES=0,1 \
python -m CPVM.code.test.trait_test \
    --config "$TRAIT_TEST_CONFIG"
