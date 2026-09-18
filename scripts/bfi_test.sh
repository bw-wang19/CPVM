WORKSPACE=/home/wbw/workspace
BFI_TEST_CONFIG=/home/wbw/workspace/CPVM/config/bfi_test.yaml

cd "$WORKSPACE" || exit 1

python -m CPVM.code.test.bfi_test \
    --config "$BFI_TEST_CONFIG"
