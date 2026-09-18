"""Focused numerical tests for CPVM path-integrated attribution."""

from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

from safetensors.torch import save_file
import torch

from CPVM.code.path_integral import (
    MatchedCompletionObjective,
    accumulate_contributions,
    build_parameter_path,
    integration_points,
    response_log_probabilities,
    set_path_parameters,
)


def test_endpoint_baseline_is_shared_by_both_interpolations():
    """K=1 must remain endpoint-gradient times the total displacement."""

    for interpolation in ("linear", "parabolic"):
        point = integration_points(1, interpolation)[0]
        assert point.position == 1.0
        assert point.tangent_scale == 1.0
        assert point.weight == 1.0


def test_linear_and_parabolic_quadrature_recover_same_line_integral():
    """Both schedules should integrate d(theta^3) from theta=0 to theta=2."""

    delta = 2.0
    for interpolation in ("linear", "parabolic"):
        attribution = 0.0
        for point in integration_points(64, interpolation):
            theta = point.position * delta
            gradient = 3.0 * theta * theta
            attribution += (
                point.weight
                * gradient
                * point.tangent_scale
                * delta
            )
        assert abs(attribution - 8.0) < 1e-8


def test_response_log_probability_normalizes_each_answer_length():
    """Each response should be token-averaged before the dataset average."""

    logits = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 2.0]],
            [[1.0, 0.0], [4.0, -1.0]],
        ]
    )
    labels = torch.tensor([[0, 1], [0, -100]])
    actual = response_log_probabilities(logits, labels)

    log_probs = logits.log_softmax(dim=-1)
    expected = torch.stack(
        [
            (log_probs[0, 0, 0] + log_probs[0, 1, 1]) / 2,
            log_probs[1, 0, 0],
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_parameter_block_keeps_its_shape_during_accumulation():
    """A parameter key must map to a contribution tensor of identical shape."""

    model = torch.nn.Linear(2, 2, bias=False)
    base = {"weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    delta = {"weight": torch.tensor([[2.0, -1.0], [0.5, 3.0]])}
    contributions = {"weight": torch.zeros_like(base["weight"])}

    set_path_parameters(model, base, delta, position=0.25)
    torch.testing.assert_close(
        model.weight,
        base["weight"] + 0.25 * delta["weight"],
    )

    model.weight.grad = torch.ones_like(model.weight)
    accumulate_contributions(model, delta, contributions, scale=0.5)
    assert contributions["weight"].shape == model.weight.shape
    torch.testing.assert_close(contributions["weight"], 0.5 * delta["weight"])


def test_parameter_path_loads_endpoint_state_dict_directly():
    """Endpoint safetensors should produce the exact parameter-wise delta."""

    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    endpoint_weight = torch.tensor([[2.5, 1.0], [3.5, 8.0]])

    with TemporaryDirectory() as directory:
        save_file(
            {"weight": endpoint_weight},
            str(Path(directory) / "model.safetensors"),
        )
        base, delta = build_parameter_path(model, directory)

    torch.testing.assert_close(base["weight"], model.weight)
    torch.testing.assert_close(delta["weight"], endpoint_weight - model.weight)


def test_objective_uses_causal_answer_positions_and_negates_l_star():
    """Optimized logits must use t-1 positions and negate counter likelihood."""

    class TinyCausalLM(torch.nn.Module):
        """Expose position-specific logits as a tiny differentiable language model."""

        def __init__(self):
            """Create one trainable two-token logit vector per sequence position."""

            super().__init__()
            self.scores = torch.nn.Parameter(torch.tensor(
                [[0.1, -0.1], [0.2, -0.2], [-0.3, 0.3], [0.0, 0.0]]
            ))
            self.last_keep = None

        def forward(self, input_ids, attention_mask, logits_to_keep=0):
            """Return only the requested hidden positions as vocabulary logits."""

            self.last_keep = logits_to_keep
            indices = list(range(input_ids.shape[1])) if logits_to_keep == 0 else logits_to_keep
            logits = self.scores[indices].unsqueeze(0).expand(input_ids.shape[0], -1, -1)
            return SimpleNamespace(logits=logits)

    def collate(examples):
        """Stack the already padded toy examples."""

        return {
            key: torch.tensor([example[key] for example in examples])
            for key in examples[0]
        }

    example = {
        "input_ids": [1, 1, 0, 1],
        "attention_mask": [1, 1, 1, 1],
        "labels": [-100, -100, 0, 1],
    }
    objective = MatchedCompletionObjective.__new__(MatchedCompletionObjective)
    objective.num_examples = 1
    objective.encoded = {"L_plus": [example], "L_star": [example]}
    objective.collator = collate
    model = TinyCausalLM()

    objective.backward(model, "L_plus", 1, torch.device("cpu"), True)
    plus_gradient = model.scores.grad.detach().clone()
    assert model.last_keep == [1, 2]

    model.zero_grad(set_to_none=True)
    objective.backward(model, "L_star", 1, torch.device("cpu"), True)
    torch.testing.assert_close(model.scores.grad, -plus_gradient)
