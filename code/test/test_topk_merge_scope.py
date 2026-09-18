"""Regression coverage for Top-k selection inside layer/module scopes."""

import math
from unittest.mock import patch

import numpy as np
import pytest
from safetensors.torch import save_file
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from CPVM.code.topk_merge import (
    RandomSelectionMask,
    TopKSelection,
    find_topk_selection,
    merge_topk_parameters,
    run_topk_merge,
)


def test_contribution_topk_ranks_only_the_requested_parameter_names(tmp_path):
    path = tmp_path / "contributions.safetensors"
    save_file(
        {
            "inside_a": torch.tensor([-4.0, 7.0, 3.0, 2.0]),
            "inside_b": torch.tensor([6.0, 5.0, 1.0, 0.0]),
            "outside": torch.tensor([1e9, 1e8]),
        },
        path,
    )
    selection = find_topk_selection(
        path,
        k_percent=25,
        chunk_numel=3,
        selection_mode="positive",
        parameter_names=["inside_a", "inside_b"],
    )

    assert selection.total_numel == 8
    assert selection.selected_numel == 2
    assert selection.threshold_score == 6.0
    assert selection.threshold_ties_to_keep == 1

    implicit_global = find_topk_selection(path, 20, chunk_numel=3)
    explicit_global = find_topk_selection(
        path, 20, chunk_numel=3, parameter_names=None
    )
    assert explicit_global == implicit_global
    assert explicit_global.total_numel == 10
    assert explicit_global.threshold_score == 1e8


def test_scoped_runner_records_scope_and_whole_model_percentages(tmp_path):
    selection = TopKSelection(200, 10, None, None, 0, candidate_numel=50)
    with patch(
        "CPVM.code.topk_merge.merge_topk_parameters", return_value=selection
    ) as merge_mock:
        summary = run_topk_merge(
            base_model_path="base",
            finetuned_model_path="finetuned",
            l_plus_contribution_path=None,
            k_percent=20,
            alpha=1,
            output_dir=str(tmp_path),
            experiment_name="experiment",
            ranking_method="random",
            use_weight_rescale=True,
            seed=17,
            layers=[3, 1, 3],
            modules=["mlp.gate_proj"],
        )

    assert "layers-1-3" in summary["run_name"]
    assert "modules-mlp-gate_proj" in summary["run_name"]
    assert "seed17" in summary["run_name"]
    assert summary["selected_percent"] == pytest.approx(5)
    assert summary["candidate_percent"] == pytest.approx(25)
    assert summary["rescale_factor"] == 5
    assert summary["selection"]["total_numel"] == 200
    assert summary["selection"]["candidate_numel"] == 50
    assert set(merge_mock.call_args.kwargs["layers"]) == {1, 3}
    assert merge_mock.call_args.kwargs["modules"] == ["mlp.gate_proj"]
    assert merge_mock.call_args.kwargs["seed"] == 17


def test_none_scopes_preserve_the_existing_global_run_name(tmp_path):
    selection = TopKSelection(100, 20, 1, 1.0, 1, candidate_numel=100)
    with patch(
        "CPVM.code.topk_merge.merge_topk_parameters", return_value=selection
    ):
        summary = run_topk_merge(
            base_model_path="base",
            finetuned_model_path="finetuned",
            l_plus_contribution_path="contributions.safetensors",
            k_percent=20,
            alpha=1,
            output_dir=str(tmp_path),
            experiment_name="experiment",
            layers=None,
            modules=None,
        )

    assert summary["run_name"] == "experiment-L_plus-top-positive20pct-alpha1"
    assert summary["selected_percent"] == pytest.approx(20)
    assert summary["candidate_percent"] == pytest.approx(100)


@pytest.fixture
def tiny_layer_checkpoints(tmp_path):
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        tie_word_embeddings=False,
    )
    with torch.random.fork_rng():
        torch.manual_seed(123)
        base = LlamaForCausalLM(config)
        finetuned = LlamaForCausalLM(config)
    finetuned.load_state_dict(base.state_dict())
    original = {
        name: parameter.detach().clone() for name, parameter in base.named_parameters()
    }
    offset = 1
    contributions = {}
    with torch.no_grad():
        for name, parameter in sorted(finetuned.named_parameters()):
            ranks = torch.arange(offset, offset + parameter.numel()).reshape_as(parameter)
            signs = (ranks % 2) * 2 - 1
            parameter.add_(signs * ranks / 4096)
            contributions[name] = ranks.float()
            offset += parameter.numel()
    endpoint = {
        name: parameter.detach().clone()
        for name, parameter in finetuned.named_parameters()
    }

    base_path = tmp_path / "base"
    finetuned_path = tmp_path / "finetuned"
    contribution_path = tmp_path / "contributions.safetensors"
    base.save_pretrained(base_path, safe_serialization=True)
    finetuned.save_pretrained(finetuned_path, safe_serialization=True)
    save_file(contributions, contribution_path)

    class DummyTokenizer:
        def save_pretrained(self, output_dir):
            return output_dir

    with patch(
        "CPVM.code.topk_merge.AutoTokenizer.from_pretrained",
        return_value=DummyTokenizer(),
    ):
        yield {
            "base_path": str(base_path),
            "finetuned_path": str(finetuned_path),
            "contribution_path": str(contribution_path),
            "original": original,
            "endpoint": endpoint,
            "contributions": contributions,
        }


@pytest.mark.parametrize(
    "ranking_method,layers,modules,expected_scope",
    [
        ("contribution", [1], None, "layer"),
        ("delta_magnitude", None, ["self_attn.q_proj"], "module"),
        ("finetuned_magnitude", [1], ["mlp"], "intersection"),
        ("random", [1], ["mlp"], "intersection"),
    ],
)
def test_topk_merges_only_inside_the_requested_scope(
    tmp_path, tiny_layer_checkpoints, capsys,
    ranking_method, layers, modules, expected_scope,
):
    data = tiny_layer_checkpoints
    original = data["original"]
    endpoint = data["endpoint"]
    if expected_scope == "layer":
        names = sorted(name for name in original if name.startswith("model.layers.1."))
    elif expected_scope == "module":
        names = sorted(name for name in original if name.endswith(".self_attn.q_proj.weight"))
    else:
        names = sorted(name for name in original if name.startswith("model.layers.1.mlp."))
    candidate_numel = sum(original[name].numel() for name in names)
    total_numel = sum(value.numel() for value in original.values())
    k_percent = 25
    selected_numel = math.ceil(candidate_numel * k_percent / 100)

    # Scores outside the layer cannot consume its requested budget.
    if ranking_method == "contribution":
        scoped_contributions = {
            name: values if name in names else torch.full_like(values, 1e9)
            for name, values in data["contributions"].items()
        }
        save_file(scoped_contributions, data["contribution_path"])

    output_path = tmp_path / "scoped"
    selection = merge_topk_parameters(
        base_model_path=data["base_path"],
        finetuned_model_path=data["finetuned_path"],
        l_plus_contribution_path=(
            data["contribution_path"] if ranking_method == "contribution" else None
        ),
        merged_model_output_path=str(output_path),
        k_percent=k_percent,
        alpha=0.375,
        ranking_method=ranking_method,
        use_weight_rescale=True,
        seed=7,
        layers=layers,
        modules=modules,
        merge_dtype="float32",
        chunk_numel=13,
    )
    printed = capsys.readouterr().out
    assert f"Selected {selected_numel:,}/{total_numel:,}" in printed
    assert selection.total_numel == total_numel
    assert selection.candidate_numel == candidate_numel
    assert selection.selected_numel == selected_numel

    if ranking_method == "random":
        expected_mask = RandomSelectionMask(
            candidate_numel, k_percent, seed=7, chunk_numel=17
        ).mask(candidate_numel)
    else:
        if ranking_method == "contribution":
            scores = [data["contributions"][name].reshape(-1) for name in names]
        elif ranking_method == "delta_magnitude":
            scores = [(endpoint[name] - original[name]).abs().reshape(-1) for name in names]
        else:
            scores = [endpoint[name].abs().reshape(-1) for name in names]
        flat_scores = torch.cat(scores).numpy()
        expected_mask = np.zeros(candidate_numel, dtype=bool)
        expected_mask[np.argsort(-flat_scores, kind="stable")[:selected_numel]] = True

    expected = {name: value.clone() for name, value in original.items()}
    offset = 0
    for name in names:
        count = original[name].numel()
        mask = torch.from_numpy(expected_mask[offset:offset + count]).reshape_as(original[name])
        expected[name] += mask * (0.375 / 0.25) * (endpoint[name] - original[name])
        offset += count
    merged = AutoModelForCausalLM.from_pretrained(output_path)
    changed_numel = 0
    for name, parameter in merged.named_parameters():
        torch.testing.assert_close(parameter, expected[name])
        changed_numel += int(torch.count_nonzero(parameter != original[name]))
    assert changed_numel == selected_numel


def test_scope_at_one_hundred_percent_interpolates_every_selected_parameter(
    tmp_path, tiny_layer_checkpoints,
):
    data = tiny_layer_checkpoints
    original = data["original"]
    endpoint = data["endpoint"]
    selected_names = {
        name for name in original if name.startswith("model.layers.1.mlp.")
    }
    total_numel = sum(value.numel() for value in original.values())
    candidate_numel = sum(original[name].numel() for name in selected_names)
    alpha = 0.375
    output_path = tmp_path / "topk"
    selection = merge_topk_parameters(
        base_model_path=data["base_path"],
        finetuned_model_path=data["finetuned_path"],
        l_plus_contribution_path=None,
        merged_model_output_path=str(output_path),
        k_percent=100,
        alpha=alpha,
        ranking_method="contribution",
        selection_mode="positive",
        use_weight_rescale=True,
        layers=[1],
        modules=["mlp"],
        merge_dtype="float32",
        chunk_numel=13,
    )

    assert selection.total_numel == total_numel
    assert selection.candidate_numel == candidate_numel
    assert selection.selected_numel == candidate_numel
    assert 0 < candidate_numel < total_numel
    assert selection.select_all
    merged = AutoModelForCausalLM.from_pretrained(output_path)
    merged_parameters = dict(merged.named_parameters())
    assert merged_parameters.keys() == original.keys()
    changed_numel = 0
    for name, parameter in merged_parameters.items():
        if name in selected_names:
            expected = original[name] + alpha * (endpoint[name] - original[name])
            torch.testing.assert_close(parameter, expected)
        else:
            assert torch.equal(parameter, original[name])
        changed_numel += int(torch.count_nonzero(parameter != original[name]))
    assert changed_numel == candidate_numel


def test_unmatched_scope_does_not_write_a_checkpoint(tmp_path, tiny_layer_checkpoints):
    data = tiny_layer_checkpoints
    output_path = tmp_path / "unmatched"
    with pytest.raises(ValueError, match="No model parameters matched"):
        merge_topk_parameters(
            base_model_path=data["base_path"],
            finetuned_model_path=data["finetuned_path"],
            l_plus_contribution_path=None,
            merged_model_output_path=str(output_path),
            k_percent=100,
            alpha=1,
            layers=[999],
        )
    assert not output_path.exists()
