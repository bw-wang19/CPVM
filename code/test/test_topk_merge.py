import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import pytest
import torch
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel
from unittest.mock import patch

from CPVM.code.topk_merge import (
    RandomSelectionMask,
    TopKSelection,
    build_run_name,
    contribution_chunks,
    contribution_score_bits,
    find_topk_selection,
    find_topk_selections,
    merge_topk_parameters,
    merge_selected_values,
    run_topk_merge,
    sample_random_indices,
    selection_mask,
)


def test_run_name_tracks_selection_mode_k_and_alpha():
    assert (
        build_run_name("experiment", 1, 1, "positive")
        == "experiment-L_plus-top-positive1pct-alpha1"
    )
    assert (
        build_run_name("experiment", 20.0, 1, "absolute", "L_contrast")
        == "experiment-L_contrast-top-absolute20pct-alpha1"
    )


def test_runner_builds_output_path_and_saves_model(tmp_path):
    selection = TopKSelection(100, 20, 1, 1.0, 1)

    with patch(
        "CPVM.code.topk_merge.merge_topk_parameters",
        return_value=selection,
    ) as merge_mock:
        summary = run_topk_merge(
            base_model_path="base",
            finetuned_model_path="finetuned",
            l_plus_contribution_path="contributions.safetensors",
            k_percent=20.0,
            alpha=1.0,
            output_dir=str(tmp_path / "models"),
            experiment_name="extraversion-high",
        )

    expected_name = "extraversion-high-L_plus-top-positive20pct-alpha1"
    expected_model_path = str(tmp_path / "models" / expected_name)
    assert summary["run_name"] == expected_name
    assert summary["output_path"] == expected_model_path
    assert merge_mock.call_args.kwargs["merged_model_output_path"] == expected_model_path


def collect_selected_values(
    path,
    selection,
    chunk_numel,
    selection_mode="absolute",
):
    """Apply the streaming mask in production order for a small test file."""

    selected_values = []
    ties_remaining = selection.threshold_ties_to_keep
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for name in sorted(handle.keys()):
            for _, chunk in contribution_chunks(handle, name, chunk_numel):
                mask, ties_remaining = selection_mask(
                    contribution_score_bits(chunk, selection_mode),
                    selection,
                    ties_remaining,
                )
                selected_values.extend(chunk.numpy()[mask].tolist())
    return selected_values, ties_remaining


def test_radix_selection_is_global_exact_and_deterministic(tmp_path):
    contribution_path = tmp_path / "contributions.safetensors"
    save_file(
        {
            "a": torch.tensor([[-5.0, 1.0], [2.0, 0.0]]),
            "b": torch.tensor([3.0, 2.0, 0.0, 1.0]),
        },
        contribution_path,
    )

    selection = find_topk_selection(contribution_path, k_percent=37.5, chunk_numel=2)
    selected_values, ties_remaining = collect_selected_values(
        contribution_path, selection, chunk_numel=2
    )

    assert selection.total_numel == 8
    assert selection.selected_numel == 3
    assert selection.threshold_score == 2.0
    assert selection.threshold_ties_to_keep == 1
    assert selected_values == [-5.0, 2.0, 3.0]
    assert ties_remaining == 0



@pytest.mark.parametrize(
    "selection_mode,k_percents",
    [
        ("absolute", [0, 12.5, 37.5, 62.5, 100]),
        ("positive", [0, 12.5, 37.5, 62.5, 100]),
    ],
)
def test_batched_radix_selections_match_independent_searches(
    tmp_path, selection_mode, k_percents
):
    contribution_path = tmp_path / "contributions.safetensors"
    save_file(
        {
            "a": torch.tensor([9.0, 8.0, 8.0, 1.0]),
            "b": torch.tensor([10.0, 8.0, 0.0, -3.0]),
        },
        contribution_path,
    )

    batched = find_topk_selections(
        contribution_path,
        k_percents,
        chunk_numel=3,
        selection_mode=selection_mode,
    )
    independent = {
        float(k_percent): find_topk_selection(
            contribution_path,
            k_percent,
            chunk_numel=2,
            selection_mode=selection_mode,
        )
        for k_percent in k_percents
    }

    assert batched == independent


def test_batched_lcontrast_selections_match_independent_searches(tmp_path):
    plus_path = tmp_path / "L_plus.safetensors"
    star_path = tmp_path / "L_star.safetensors"
    save_file(
        {"weight": torch.tensor([9.0, 8.0, 8.0, 1.0, -3.0])},
        plus_path,
    )
    save_file(
        {"weight": torch.tensor([1.0, 0.0, 0.0, 4.0, 3.0])},
        star_path,
    )
    options = {
        "chunk_numel": 2,
        "selection_mode": "positive",
        "l_star_contribution_path": star_path,
        "contribution_channel": "L_contrast",
    }
    k_percents = [0, 20, 40, 60, 80, 100]

    batched = find_topk_selections(plus_path, k_percents, **options)
    independent = {
        float(k_percent): find_topk_selection(
            plus_path, k_percent, **options
        )
        for k_percent in k_percents
    }

    assert batched == independent

def test_positive_selection_ranks_raw_contributions(tmp_path):
    contribution_path = tmp_path / "contributions.safetensors"
    save_file(
        {
            "a": torch.tensor([[-5.0, 1.0], [2.0, 0.0]]),
            "b": torch.tensor([3.0, 2.0, 0.0, 1.0]),
        },
        contribution_path,
    )

    selection = find_topk_selection(
        contribution_path,
        k_percent=25.0,
        chunk_numel=2,
        selection_mode="positive",
    )
    selected_values, ties_remaining = collect_selected_values(
        contribution_path,
        selection,
        chunk_numel=2,
        selection_mode="positive",
    )

    assert selection.total_numel == 8
    assert selection.selected_numel == 2
    assert selection.threshold_score == 2.0
    assert selection.threshold_ties_to_keep == 1
    assert selected_values == [2.0, 3.0]
    assert ties_remaining == 0


def test_selection_handles_zero_and_one_hundred_percent(tmp_path):
    contribution_path = tmp_path / "contributions.safetensors"
    save_file({"weight": torch.tensor([3.0, 0.0, -1.0])}, contribution_path)

    empty = find_topk_selection(contribution_path, k_percent=0)
    full = find_topk_selection(contribution_path, k_percent=100)
    assert collect_selected_values(contribution_path, empty, 2)[0] == []
    assert collect_selected_values(contribution_path, full, 2)[0] == [3.0, 0.0, -1.0]


def test_positive_selection_keeps_fixed_budget_across_zero(tmp_path):
    contribution_path = tmp_path / "contributions.safetensors"
    save_file({"weight": torch.tensor([3.0, 0.0, -1.0])}, contribution_path)

    full = find_topk_selection(
        contribution_path,
        k_percent=100,
        selection_mode="positive",
    )
    assert collect_selected_values(
        contribution_path,
        full,
        2,
        selection_mode="positive",
    )[0] == [3.0, 0.0, -1.0]
    assert full.select_all is True
    assert full.threshold_score is None


def test_lstar_and_lcontrast_channels_rank_selected_objective(tmp_path):
    plus_path = tmp_path / "L_plus.safetensors"
    star_path = tmp_path / "L_star.safetensors"
    save_file({"weight": torch.tensor([5.0, 1.0, -10.0, 0.0])}, plus_path)
    save_file({"weight": torch.tensor([-5.0, 9.0, 20.0, 0.0])}, star_path)

    options = {
        "k_percent": 25,
        "chunk_numel": 2,
        "selection_mode": "positive",
        "l_star_contribution_path": star_path,
    }
    star = find_topk_selection(
        plus_path, contribution_channel="L_star", **options
    )
    contrast = find_topk_selection(
        plus_path, contribution_channel="L_contrast", **options
    )

    assert star.threshold_score == 20.0
    assert contrast.threshold_score == 10.0


def test_merge_selected_values_uses_base_to_finetuned_interpolation():
    base = torch.tensor([0.0, 10.0, 20.0])
    finetuned = torch.tensor([10.0, 20.0, 0.0])
    selected_indices = torch.tensor([0, 2])

    merge_selected_values(base, finetuned, selected_indices, alpha=0.5)

    torch.testing.assert_close(base, torch.tensor([5.0, 10.0, 10.0]))


def test_tiny_sharded_model_merge_matches_full_reference(tmp_path):
    """Exercise sharded checkpoint reads, sparse merge, save, and reload."""

    config = GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_embd=8,
        n_layer=1,
        n_head=2,
        tie_word_embeddings=True,
    )
    base = GPT2LMHeadModel(config)
    finetuned = GPT2LMHeadModel(config)
    finetuned.load_state_dict(base.state_dict())
    base_values = {
        name: parameter.detach().clone() for name, parameter in base.named_parameters()
    }
    with torch.no_grad():
        for parameter in finetuned.parameters():
            parameter.add_(1.0)

    contribution_values = {}
    offset = 1
    for name, parameter in base.named_parameters():
        contribution_values[name] = torch.arange(
            offset,
            offset + parameter.numel(),
            dtype=torch.float32,
        ).reshape_as(parameter)
        offset += parameter.numel()
    first_name = sorted(contribution_values)[0]
    contribution_values[first_name].view(-1)[0] = -1e9

    base_path = tmp_path / "base"
    finetuned_path = tmp_path / "finetuned"
    output_path = tmp_path / "merged"
    contribution_path = tmp_path / "contributions.safetensors"
    base.save_pretrained(base_path, safe_serialization=True, max_shard_size="2KB")
    finetuned.save_pretrained(
        finetuned_path, safe_serialization=True, max_shard_size="2KB"
    )
    save_file(contribution_values, contribution_path)
    assert (finetuned_path / "model.safetensors.index.json").exists()

    class DummyTokenizer:
        def save_pretrained(self, output_dir):
            return output_dir

    with patch(
        "CPVM.code.topk_merge.AutoTokenizer.from_pretrained",
        return_value=DummyTokenizer(),
    ):
        selection = merge_topk_parameters(
            base_model_path=str(base_path),
            finetuned_model_path=str(finetuned_path),
            l_plus_contribution_path=str(contribution_path),
            merged_model_output_path=str(output_path),
            k_percent=12.5,
            alpha=0.25,
            merge_dtype="float32",
            chunk_numel=7,
            max_shard_size="2KB",
        )

    all_scores = torch.cat(
        [contribution_values[name].reshape(-1) for name in sorted(contribution_values)]
    )
    threshold = torch.topk(all_scores, selection.selected_numel).values[-1]
    merged = AutoModelForCausalLM.from_pretrained(output_path)
    for name, parameter in merged.named_parameters():
        expected = base_values[name] + 0.25 * (
            contribution_values[name] >= threshold
        )
        torch.testing.assert_close(parameter, expected)


@pytest.mark.parametrize("seed, initial_count", [(0, 7), (1, 8), (4, 4)])
def test_random_selection_corrects_to_exact_global_budget(seed, initial_count):
    """Exercise unchanged, excess, and deficient Bernoulli masks."""

    mask_seed = np.random.SeedSequence(seed).spawn(2)[0]
    initial_mask = np.random.default_rng(mask_seed).random(23) < 0.3
    assert int(initial_mask.sum()) == initial_count

    sampler = RandomSelectionMask(23, 30, seed=seed, chunk_numel=5)
    actual = np.concatenate([sampler.mask(size) for size in (2, 0, 6, 1, 14)])

    assert sampler.selection.total_numel == 23
    assert sampler.selection.selected_numel == 7
    assert int(actual.sum()) == 7
    assert sampler.remaining_numel == 0
    assert int(np.count_nonzero(actual != initial_mask)) == abs(initial_count - 7)
    if initial_count > 7:
        assert np.all(~actual | initial_mask)
    else:
        assert np.all(~initial_mask | actual)
    with pytest.raises(ValueError, match="remaining parameter count"):
        sampler.mask(1)


def test_random_selection_is_reproducible_across_chunk_sizes():
    reference = RandomSelectionMask(137, 23, seed=42, chunk_numel=137).mask(137)
    streamed = RandomSelectionMask(137, 23, seed=42, chunk_numel=3)
    actual = np.concatenate([streamed.mask(size) for size in (1, 16, 0, 37, 83)])
    other_seed = RandomSelectionMask(137, 23, seed=43, chunk_numel=11).mask(137)

    np.testing.assert_array_equal(actual, reference)
    assert int(actual.sum()) == 32
    assert int(other_seed.sum()) == 32
    assert not np.array_equal(reference, other_seed)


@pytest.mark.parametrize("total, k_percent, expected", [(9, 0, 0), (9, 100, 9), (3, 99, 3), (0, 50, 0)])
def test_random_selection_handles_empty_full_and_rounded_budgets(total, k_percent, expected):
    sampler = RandomSelectionMask(total, k_percent, seed=7, chunk_numel=2)
    actual = sampler.mask(total)

    assert actual.dtype == np.bool_
    assert actual.size == total
    assert int(actual.sum()) == expected
    assert sampler.selection.selected_numel == expected
    assert sampler.mask(0).size == 0


def test_random_index_sampling_handles_a_multibillion_population():
    population_size = 4_000_000_000
    first = sample_random_indices(population_size, 17, np.random.default_rng(8))
    repeated = sample_random_indices(population_size, 17, np.random.default_rng(8))

    np.testing.assert_array_equal(first, repeated)
    assert first.dtype == np.int64
    assert first.size == 17
    assert np.all(np.diff(first) > 0)
    assert 0 <= first[0] < first[-1] < population_size
    assert first[-1] > 1_000_000_000
    assert sample_random_indices(population_size, 0, np.random.default_rng(8)).size == 0
    np.testing.assert_array_equal(
        sample_random_indices(6, 6, np.random.default_rng(8)), np.arange(6)
    )


def test_random_runner_tracks_seed_without_a_contribution_file(tmp_path):
    selection = TopKSelection(100, 20, None, None, 0)
    with patch(
        "CPVM.code.topk_merge.merge_topk_parameters", return_value=selection
    ) as merge_mock:
        summary = run_topk_merge(
            base_model_path="base",
            finetuned_model_path="models/finetuned",
            l_plus_contribution_path=None,
            k_percent=20,
            alpha=1,
            output_dir=str(tmp_path),
            ranking_method="random",
            use_weight_rescale=True,
            seed=17,
        )

    expected_name = "finetuned-random-keep-20pct-alpha1-dare-rescaled-seed17"
    assert summary["run_name"] == expected_name
    assert summary["output_path"] == str(tmp_path / expected_name)
    assert summary["seed"] == 17
    assert summary["random_sampling"] == "uniform_without_replacement"
    assert summary["rescale_factor"] == 5
    assert summary["l_plus_contribution_path"] is None
    assert merge_mock.call_args.kwargs["seed"] == 17
    assert merge_mock.call_args.kwargs["ranking_method"] == "random"
    assert merge_mock.call_args.kwargs["use_weight_rescale"] is True
    assert merge_mock.call_args.kwargs["l_plus_contribution_path"] is None
    assert merge_mock.call_args.kwargs["merged_model_output_path"] == summary["output_path"]


def test_random_rescaling_rejects_zero_retention_before_loading_models(tmp_path):
    with patch("CPVM.code.topk_merge.AutoModelForCausalLM.from_pretrained") as load_mock:
        with pytest.raises(ValueError, match="undefined when k_percent is 0"):
            merge_topk_parameters(
                base_model_path="base",
                finetuned_model_path="finetuned",
                l_plus_contribution_path=None,
                merged_model_output_path=str(tmp_path / "merged"),
                k_percent=0,
                alpha=1,
                ranking_method="random",
                use_weight_rescale=True,
            )
    load_mock.assert_not_called()


@pytest.mark.parametrize("use_weight_rescale", [False, True])
def test_tiny_sharded_random_merge_matches_rescaled_delta(tmp_path, use_weight_rescale):
    """Verify sparse delta arithmetic and tied weights after save/reload."""

    config = GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_embd=8,
        n_layer=1,
        n_head=2,
        tie_word_embeddings=True,
    )
    base = GPT2LMHeadModel(config)
    finetuned = GPT2LMHeadModel(config)
    finetuned.load_state_dict(base.state_dict())
    base_values = {
        name: parameter.detach().clone() for name, parameter in base.named_parameters()
    }
    with torch.no_grad():
        for number, parameter in enumerate(finetuned.parameters(), start=1):
            signs = (torch.arange(parameter.numel()) % 2 * 2 - 1).reshape_as(parameter)
            parameter.add_(signs * (number / 64))
    finetuned_values = {
        name: parameter.detach().clone() for name, parameter in finetuned.named_parameters()
    }

    base_path = tmp_path / "base"
    finetuned_path = tmp_path / "finetuned"
    output_path = tmp_path / "merged"
    base.save_pretrained(base_path, safe_serialization=True, max_shard_size="2KB")
    finetuned.save_pretrained(finetuned_path, safe_serialization=True, max_shard_size="2KB")
    assert (finetuned_path / "model.safetensors.index.json").exists()

    class DummyTokenizer:
        def save_pretrained(self, output_dir):
            return output_dir

    with patch(
        "CPVM.code.topk_merge.AutoTokenizer.from_pretrained",
        return_value=DummyTokenizer(),
    ):
        selection = merge_topk_parameters(
            base_model_path=str(base_path),
            finetuned_model_path=str(finetuned_path),
            l_plus_contribution_path=None,
            merged_model_output_path=str(output_path),
            k_percent=12.5,
            alpha=0.375,
            ranking_method="random",
            seed=73,
            use_weight_rescale=use_weight_rescale,
            merge_dtype="float32",
            chunk_numel=7,
            max_shard_size="2KB",
        )

    total_numel = sum(value.numel() for value in base_values.values())
    masks = RandomSelectionMask(total_numel, 12.5, seed=73, chunk_numel=13)
    effective_alpha = 0.375 / 0.125 if use_weight_rescale else 0.375
    merged = AutoModelForCausalLM.from_pretrained(output_path)
    merged_parameters = dict(merged.named_parameters())
    changed_numel = 0
    for name in sorted(base_values):
        original = base_values[name]
        mask = torch.from_numpy(masks.mask(original.numel())).reshape_as(original)
        expected = original + mask * effective_alpha * (finetuned_values[name] - original)
        torch.testing.assert_close(merged_parameters[name], expected)
        changed_numel += int(torch.count_nonzero(merged_parameters[name] != original))

    assert selection.total_numel == total_numel
    assert changed_numel == selection.selected_numel == masks.selection.selected_numel
    assert 0 < changed_numel < total_numel
    assert merged.transformer.wte.weight.data_ptr() == merged.lm_head.weight.data_ptr()
