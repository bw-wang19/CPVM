import torch

from CPVM.code.test.trait_test import TRAITTester


class IndexTokenizer:
    """Encode each prompt as its position so output order is easy to verify."""

    def __call__(self, prompts, **kwargs):
        count = len(prompts)
        return {
            "input_ids": torch.arange(count).reshape(count, 1),
            "attention_mask": torch.ones(count, 1, dtype=torch.long),
        }


def test_data_parallel_shards_run_on_both_replicas_and_restore_order():
    tester = object.__new__(TRAITTester)
    tester.use_vllm = False
    tester.tokenizer = IndexTokenizer()
    tester.max_length = 32
    tester.models = ["gpu0-model", "gpu1-model"]
    tester.input_devices = [torch.device("cpu"), torch.device("cpu")]
    assignments = {}

    def fake_forward(model, device, inputs):
        positions = inputs["input_ids"][:, 0].tolist()
        for position in positions:
            assignments[position] = model
        return [[position] for position in positions]

    tester.get_hf_option_probabilities = fake_forward
    probabilities = tester.get_option_probabilities(["p0", "p1", "p2", "p3", "p4"])

    assert probabilities == [[0], [1], [2], [3], [4]]
    assert assignments == {
        0: "gpu0-model",
        1: "gpu1-model",
        2: "gpu0-model",
        3: "gpu1-model",
        4: "gpu0-model",
    }
