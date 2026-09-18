"""Regression coverage for pairwise Top-k overlap analysis."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from safetensors.torch import save_file
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from CPVM.code.topk_merge import RandomSelectionMask
from CPVM.code.topk_overlap import compare_topk_overlap
from CPVM.code.utils.parameter_selection import select_parameters


METRICS = {
    "intersection_count",
    "random_expected_intersection",
    "overlap_coefficient",
    "jaccard",
    "overlap_lift",
    "chance_adjusted_overlap",
}


@pytest.fixture
def tiny_base(tmp_path):
    """Create a real but tiny checkpoint whose names exercise layer/module scopes."""

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
        model = LlamaForCausalLM(config)
    base_path = tmp_path / "base"
    model.save_pretrained(base_path, safe_serialization=True)

    parameters = dict(model.named_parameters())
    names = sorted(parameters)
    shapes = {name: tuple(parameters[name].shape) for name in names}
    total_numel = sum(parameters[name].numel() for name in names)
    return {
        "path": str(base_path),
        "model": model,
        "names": names,
        "shapes": shapes,
        "total_numel": total_numel,
    }


@pytest.fixture
def skip_heatmaps(monkeypatch):
    """Keep semantic tests fast; one dedicated test exercises real rendering."""

    monkeypatch.setattr(
        "CPVM.code.topk_overlap._save_heatmap",
        lambda *args, **kwargs: None,
    )


def _split_flat(values: np.ndarray, tiny_base) -> dict[str, torch.Tensor]:
    values = np.asarray(values, dtype=np.float32)
    assert values.size == tiny_base["total_numel"]
    tensors = {}
    offset = 0
    for name in tiny_base["names"]:
        count = math.prod(tiny_base["shapes"][name])
        tensors[name] = torch.from_numpy(
            values[offset : offset + count].copy()
        ).reshape(tiny_base["shapes"][name])
        offset += count
    assert offset == values.size
    return tensors


def _write_contribution_dir(
    root: Path,
    label: str,
    tiny_base,
    plus: np.ndarray,
    star: np.ndarray | None = None,
) -> str:
    directory = root / label
    directory.mkdir(parents=True)
    if star is None:
        star = np.zeros_like(plus)
    save_file(
        _split_flat(plus, tiny_base),
        directory / f"{label}-L_plus.safetensors",
    )
    save_file(
        _split_flat(star, tiny_base),
        directory / f"{label}-L_star.safetensors",
    )
    return str(directory)


def _selected_mask(order: np.ndarray, selected_numel: int) -> np.ndarray:
    mask = np.zeros(order.size, dtype=bool)
    mask[order[:selected_numel]] = True
    return mask


def _reference_contribution_mask(
    scores: np.ndarray,
    selection_mode: str,
    k_percent: float,
) -> np.ndarray:
    selected_numel = math.ceil(scores.size * k_percent / 100)
    if selection_mode == "positive":
        ranking_scores = scores
    elif selection_mode == "absolute":
        ranking_scores = np.abs(scores)
    else:
        raise AssertionError(selection_mode)
    order = np.argsort(-ranking_scores, kind="stable")
    return _selected_mask(order, selected_numel)


def _scores_with_order(
    order: np.ndarray,
    selected_numel: int,
    selection_mode: str,
) -> np.ndarray:
    """Create unique scores whose exact requested ranking follows order."""

    total_numel = order.size
    if selection_mode == "positive":
        scores = np.empty(total_numel, dtype=np.float32)
        scores[order] = (
            np.arange(total_numel, 0, -1, dtype=np.float32)
            - total_numel / 2
        )
        return scores

    magnitudes = np.arange(total_numel, 0, -1, dtype=np.float32)
    scores = np.empty(total_numel, dtype=np.float32)
    if selection_mode == "absolute":
        signs = np.where(np.arange(total_numel) % 2, -1.0, 1.0)
        scores[order] = magnitudes * signs
    else:
        raise AssertionError(f"Unexpected selection mode: {selection_mode}")
    return scores


def _metric_arrays(masks: list[np.ndarray], candidate_numel: int):
    selected_counts = np.array([int(mask.sum()) for mask in masks], dtype=np.int64)
    packed = np.stack(masks).astype(np.int64)
    intersections = packed @ packed.T
    expected = np.outer(selected_counts, selected_counts) / candidate_numel
    minimum = np.minimum.outer(selected_counts, selected_counts)
    union = selected_counts[:, None] + selected_counts[None, :] - intersections
    overlap = intersections / minimum
    jaccard = intersections / union
    lift = intersections / expected
    chance_denominator = minimum - expected
    chance_adjusted = np.divide(
        intersections - expected,
        chance_denominator,
        out=np.ones_like(expected, dtype=np.float64),
        where=chance_denominator != 0,
    )
    return {
        "intersection_count": intersections,
        "random_expected_intersection": expected,
        "overlap_coefficient": overlap,
        "jaccard": jaccard,
        "overlap_lift": lift,
        "chance_adjusted_overlap": chance_adjusted,
    }


def _assert_metric_frames(actual, expected, labels):
    assert set(actual) == METRICS
    for metric, expected_values in expected.items():
        frame = actual[metric]
        assert isinstance(frame, pd.DataFrame)
        assert frame.index.tolist() == labels
        assert frame.columns.tolist() == labels
        if metric == "intersection_count":
            np.testing.assert_array_equal(frame.to_numpy(), expected_values)
        else:
            np.testing.assert_allclose(
                frame.to_numpy(dtype=np.float64),
                expected_values,
                rtol=1e-12,
                atol=1e-12,
            )


def test_three_model_contribution_modes_and_metric_formulas(
    tiny_base, tmp_path, skip_heatmaps
):
    """One N-model run supports multiple modes and arbitrary decimal k values."""

    total_numel = tiny_base["total_numel"]
    k_percents = [5.5, 12.34]
    selected_numel = math.ceil(total_numel * max(k_percents) / 100)
    orders = [
        np.arange(total_numel),
        np.roll(np.arange(total_numel), -(selected_numel // 2)),
        np.arange(total_numel - 1, -1, -1),
    ]
    labels = ["model-a", "model-b", "model-c"]
    scores_by_model = [
        _scores_with_order(order, selected_numel, "absolute")
        for order in orders
    ]
    models = []
    for label, scores in zip(labels, scores_by_model):
        directory = _write_contribution_dir(
            tmp_path / "contribution", label, tiny_base, scores
        )
        models.append({"label": label, "contribution_dir": directory})

    strategies = [
        {
            "name": selection_mode,
            "ranking_method": "contribution",
            "contribution_channel": "L_plus",
            "selection_mode": selection_mode,
            "k_percents": k_percents,
        }
        for selection_mode in ("positive", "absolute")
    ]
    results = compare_topk_overlap(
        base_model_path=tiny_base["path"],
        models=models,
        strategies=strategies,
        output_dir=str(tmp_path / "multi-strategy-output"),
        chunk_numel=37,
    )
    assert set(results) == {
        (selection_mode, k_percent)
        for selection_mode in ("positive", "absolute")
        for k_percent in k_percents
    }
    for selection_mode in ("positive", "absolute"):
        for k_percent in k_percents:
            masks = [
                _reference_contribution_mask(
                    scores, selection_mode, k_percent
                )
                for scores in scores_by_model
            ]
            expected = _metric_arrays(masks, total_numel)
            _assert_metric_frames(
                results[(selection_mode, k_percent)], expected, labels
            )

def test_lcontrast_ranks_the_fp32_sum_of_both_channels(
    tiny_base, tmp_path, skip_heatmaps
):
    total_numel = tiny_base["total_numel"]
    k_percent = 9.75
    selected_numel = math.ceil(total_numel * k_percent / 100)
    orders = [
        np.arange(total_numel),
        np.roll(np.arange(total_numel), -selected_numel),
        np.arange(total_numel - 1, -1, -1),
    ]
    labels = ["first", "second", "third"]
    models = []
    for label, order in zip(labels, orders):
        combined = _scores_with_order(order, selected_numel, "positive")
        plus = np.arange(total_numel, dtype=np.float32)
        star = combined - plus
        directory = _write_contribution_dir(
            tmp_path / "contrast", label, tiny_base, plus, star
        )
        models.append({"label": label, "contribution_dir": directory})

    results = compare_topk_overlap(
        base_model_path=tiny_base["path"],
        models=models,
        strategies=[
            {
                "name": "contrast",
                "ranking_method": "contribution",
                "contribution_channel": "L_contrast",
                "selection_mode": "positive",
                "k_percents": [k_percent],
            }
        ],
        output_dir=str(tmp_path / "contrast-output"),
        chunk_numel=29,
    )
    expected = _metric_arrays(
        [_selected_mask(order, selected_numel) for order in orders],
        total_numel,
    )
    _assert_metric_frames(results[("contrast", k_percent)], expected, labels)


def test_random_uses_per_model_seeds_and_scoped_denominator(
    tiny_base, tmp_path, skip_heatmaps
):
    parameters = dict(tiny_base["model"].named_parameters())
    candidate_names, scope = select_parameters(
        parameters, layers=[0], modules=["mlp"]
    )
    assert candidate_names
    candidate_numel = scope.selected_numel
    k_percent = 23.45
    seeds = [3, 7, 11]
    labels = [f"seed-{seed}" for seed in seeds]
    masks = [
        RandomSelectionMask(
            candidate_numel, k_percent, seed=seed, chunk_numel=17
        ).mask(candidate_numel)
        for seed in seeds
    ]

    results = compare_topk_overlap(
        base_model_path=tiny_base["path"],
        models=[
            {"label": label, "seed": seed}
            for label, seed in zip(labels, seeds)
        ],
        strategies=[
            {
                "name": "scoped-random",
                "ranking_method": "random",
                "layers": [0],
                "modules": ["mlp"],
                "k_percents": [k_percent],
            }
        ],
        output_dir=str(tmp_path / "random-output"),
        chunk_numel=31,
    )
    expected = _metric_arrays(masks, candidate_numel)
    _assert_metric_frames(
        results[("scoped-random", k_percent)], expected, labels
    )
    assert not np.all(
        expected["intersection_count"]
        == expected["intersection_count"][0, 0]
    )


def test_contribution_scope_excludes_larger_scores_outside_it(
    tiny_base, tmp_path, skip_heatmaps
):
    parameters = dict(tiny_base["model"].named_parameters())
    candidate_names, scope = select_parameters(
        parameters, layers=[1], modules=["self_attn.q_proj"]
    )
    candidate_numel = scope.selected_numel
    k_percent = 31.25
    selected_numel = math.ceil(candidate_numel * k_percent / 100)
    candidate_global_indices = []
    offset = 0
    for name in tiny_base["names"]:
        count = math.prod(tiny_base["shapes"][name])
        if name in candidate_names:
            candidate_global_indices.extend(range(offset, offset + count))
        offset += count
    candidate_global_indices = np.asarray(candidate_global_indices)

    orders = [
        np.arange(candidate_numel),
        np.roll(np.arange(candidate_numel), -max(1, selected_numel // 2)),
    ]
    labels = ["scope-a", "scope-b"]
    models = []
    for label, order in zip(labels, orders):
        global_scores = np.full(tiny_base["total_numel"], 1e9, dtype=np.float32)
        scoped_scores = _scores_with_order(order, selected_numel, "positive")
        global_scores[candidate_global_indices] = scoped_scores
        directory = _write_contribution_dir(
            tmp_path / "scope", label, tiny_base, global_scores
        )
        models.append({"label": label, "contribution_dir": directory})

    results = compare_topk_overlap(
        base_model_path=tiny_base["path"],
        models=models,
        strategies=[
            {
                "name": "scoped-contribution",
                "ranking_method": "contribution",
                "contribution_channel": "L_plus",
                "selection_mode": "positive",
                "layers": [1],
                "modules": ["self_attn.q_proj"],
                "k_percents": [k_percent],
            }
        ],
        output_dir=str(tmp_path / "scope-output"),
        chunk_numel=13,
    )
    expected = _metric_arrays(
        [_selected_mask(order, selected_numel) for order in orders],
        candidate_numel,
    )
    _assert_metric_frames(
        results[("scoped-contribution", k_percent)], expected, labels
    )


def _write_ranked_sft(
    path: Path,
    tiny_base,
    ranking_method: str,
    scores: np.ndarray,
) -> str:
    model = LlamaForCausalLM.from_pretrained(tiny_base["path"])
    score_tensors = _split_flat(scores, tiny_base)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            values = score_tensors[name].to(parameter)
            if ranking_method == "delta_magnitude":
                parameter.add_(values)
            elif ranking_method == "finetuned_magnitude":
                parameter.copy_(values)
            else:
                raise AssertionError(ranking_method)
    model.save_pretrained(path, safe_serialization=True)
    return str(path)


@pytest.mark.parametrize("ranking_method", ["delta_magnitude", "finetuned_magnitude"])
def test_model_weight_ranking_methods_use_each_sft_checkpoint(
    ranking_method, tiny_base, tmp_path, skip_heatmaps
):
    total_numel = tiny_base["total_numel"]
    k_percent = 8.765
    selected_numel = math.ceil(total_numel * k_percent / 100)
    orders = [
        np.arange(total_numel),
        np.roll(np.arange(total_numel), -selected_numel),
        np.arange(total_numel - 1, -1, -1),
    ]
    labels = ["sft-a", "sft-b", "sft-c"]
    models = []
    for label, order in zip(labels, orders):
        scores = _scores_with_order(order, selected_numel, "absolute")
        sft_path = _write_ranked_sft(
            tmp_path / f"{ranking_method}-{label}",
            tiny_base,
            ranking_method,
            scores,
        )
        models.append({"label": label, "finetuned_model_path": sft_path})

    results = compare_topk_overlap(
        base_model_path=tiny_base["path"],
        models=models,
        strategies=[
            {
                "name": ranking_method,
                "ranking_method": ranking_method,
                "k_percents": [k_percent],
            }
        ],
        output_dir=str(tmp_path / f"{ranking_method}-output"),
        chunk_numel=23,
    )
    expected = _metric_arrays(
        [_selected_mask(order, selected_numel) for order in orders],
        total_numel,
    )
    _assert_metric_frames(results[(ranking_method, k_percent)], expected, labels)


def test_compare_writes_metric_csv_and_heatmap_png_svg(tiny_base, tmp_path):
    total_numel = tiny_base["total_numel"]
    k_percent = 17.125
    selected_numel = math.ceil(total_numel * k_percent / 100)
    labels = ["left", "right"]
    orders = [
        np.arange(total_numel),
        np.roll(np.arange(total_numel), -selected_numel // 2),
    ]
    models = []
    for label, order in zip(labels, orders):
        scores = _scores_with_order(order, selected_numel, "absolute")
        directory = _write_contribution_dir(
            tmp_path / "artifacts", label, tiny_base, scores
        )
        models.append({"label": label, "contribution_dir": directory})

    output_dir = tmp_path / "rendered"
    results = compare_topk_overlap(
        base_model_path=tiny_base["path"],
        models=models,
        strategies=[
            {
                "name": "artifact-test",
                "ranking_method": "contribution",
                "contribution_channel": "L_plus",
                "selection_mode": "absolute",
                "k_percents": [k_percent],
            }
        ],
        output_dir=str(output_dir),
        chunk_numel=19,
    )
    assert ("artifact-test", k_percent) in results

    csv_paths = sorted(output_dir.rglob("*.csv"))
    png_paths = sorted(output_dir.rglob("*.png"))
    svg_paths = sorted(output_dir.rglob("*.svg"))
    assert len(csv_paths) == len(METRICS)
    assert png_paths
    assert len(png_paths) == len(svg_paths)
    assert {path.stem for path in csv_paths} == METRICS
    for path in csv_paths:
        frame = pd.read_csv(path, index_col=0)
        assert frame.index.tolist() == labels
        assert frame.columns.tolist() == labels
    assert all(path.stat().st_size > 0 for path in png_paths + svg_paths)
