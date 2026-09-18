"""Compare exact Top-k parameter selections across any number of models.

Each configured strategy reproduces the selection semantics used by
:mod:`CPVM.code.topk_merge`: contribution, delta magnitude, fine-tuned
magnitude, or exact fixed-count random selection; contribution channel/mode and
layer/module scopes are preserved.  Every strategy and k value produces
pairwise matrices plus annotated heatmaps without materializing full-model
boolean masks.
"""

from __future__ import annotations

from contextlib import ExitStack
import gc
import json
import math
from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from safetensors import safe_open
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from CPVM.code.topk_merge import (
    CONTRIBUTION_CHANNELS,
    RANKING_METHODS,
    ModelTensorReader,
    RandomSelectionMask,
    TopKSelection,
    absolute_float_bits,
    chunk_indices,
    combined_contribution_chunks,
    contribution_score_bits,
    discover_contribution_files,
    find_model_topk_selection,
    find_topk_selection,
    find_topk_selections,
    model_ranking_chunks,
    resolve_contribution_paths,
    resolve_dtype,
    selection_mask,
)
from CPVM.code.utils.arguments import parse_args_yaml
from CPVM.code.utils.parameter_selection import select_parameters


SELECTION_MODES = ("positive", "absolute")
HEATMAP_METRICS = (
    "overlap_coefficient",
    "jaccard",
    "chance_adjusted_overlap",
)
METRIC_TITLES = {
    "overlap_coefficient": "Top-k overlap coefficient",
    "jaccard": "Top-k Jaccard similarity",
    "chance_adjusted_overlap": "Chance-adjusted Top-k overlap",
}


def _safe_component(value: str) -> str:
    """Return a stable filename component."""

    component = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    if not component:
        raise ValueError(f"Name has no filename-safe characters: {value!r}")
    return component


def _k_component(k_percent: float) -> str:
    return f"{k_percent:.12g}".replace(".", "p")


def _chunk_numel(shape: tuple[int, ...], index: object) -> int:
    if index is None:
        return math.prod(shape)
    rows = index[0].stop - index[0].start
    return rows * math.prod(shape[1:])


def _normalise_k_percents(strategy: dict[str, Any]) -> list[float]:
    raw = strategy.get("k_percents")
    if raw is None and "k_percent" in strategy:
        raw = [strategy["k_percent"]]
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        raw = [raw]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(
            f"Strategy {strategy.get('name')!r} must define nonempty k_percents"
        )

    values = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Every k_percents value must be an integer or float")
        value = float(value)
        if not math.isfinite(value) or not 1 <= value <= 100:
            raise ValueError("Every k_percents value must be within [1, 100]")
        if value not in values:
            values.append(value)
    return values


def _validate_models(models: list[dict[str, Any]]) -> list[str]:
    if not isinstance(models, list) or len(models) < 2:
        raise ValueError("models must contain at least two model definitions")
    if any(not isinstance(model, dict) for model in models):
        raise ValueError("Every models entry must be a mapping")
    labels = [model.get("label") for model in models]
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("Every model must have a nonempty label")
    if len(set(labels)) != len(labels):
        raise ValueError("Model labels must be unique")
    return labels


def _validate_strategy(strategy: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(strategy, dict):
        raise ValueError("Every strategies entry must be a mapping")
    name = strategy.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Every strategy must have a nonempty name")
    ranking_method = strategy.get("ranking_method")
    if ranking_method not in RANKING_METHODS:
        raise ValueError(
            f"Strategy {name!r}: ranking_method must be one of {RANKING_METHODS}"
        )

    layers = strategy.get("layers")
    modules = strategy.get("modules")
    if layers is not None and not isinstance(layers, list):
        raise ValueError(f"Strategy {name!r}: layers must be a list or null")
    if modules is not None and not isinstance(modules, list):
        raise ValueError(f"Strategy {name!r}: modules must be a list or null")

    if ranking_method == "contribution":
        channel = strategy.get("contribution_channel")
        mode = strategy.get("selection_mode")
        if channel not in CONTRIBUTION_CHANNELS:
            raise ValueError(
                f"Strategy {name!r}: contribution_channel must be one of "
                f"{CONTRIBUTION_CHANNELS}"
            )
        if mode not in SELECTION_MODES:
            raise ValueError(
                f"Strategy {name!r}: selection_mode must be one of "
                f"{SELECTION_MODES}"
            )

    merge_dtype = strategy.get("merge_dtype", "auto")
    if merge_dtype not in {"auto", "bfloat16", "float16", "float32"}:
        raise ValueError(
            f"Strategy {name!r}: merge_dtype must be auto, bfloat16, "
            "float16, or float32"
        )

    validated = dict(strategy)
    validated["k_percents"] = _normalise_k_percents(strategy)
    validated["layers"] = layers
    validated["modules"] = modules
    validated["merge_dtype"] = merge_dtype
    return validated


def _model_contribution_paths(
    model: dict[str, Any],
    contribution_channel: str,
) -> list[Path]:
    directory = model.get("contribution_dir")
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError(
            f"Model {model.get('label')!r} requires contribution_dir for "
            "contribution ranking"
        )
    files = discover_contribution_files(directory)
    return resolve_contribution_paths(
        files["L_plus"],
        files["L_star"],
        contribution_channel,
    )


def _load_base_parameters(
    base_model_path: str | None,
    merge_dtype: str,
):
    if not isinstance(base_model_path, str) or not base_model_path.strip():
        raise ValueError(
            "base_model_path is required for model-based ranking or "
            "layer/module-scoped selection"
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        dtype=resolve_dtype(merge_dtype),
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    return model, dict(model.named_parameters())


def _contribution_universe(
    contribution_paths: list[list[Path]],
) -> tuple[list[str], dict[str, tuple[int, ...]]]:
    """Validate contribution structures and return their common universe."""

    names = None
    shapes = {}
    with ExitStack() as stack:
        all_handles = []
        for paths in contribution_paths:
            handles = [
                stack.enter_context(
                    safe_open(str(path), framework="pt", device="cpu")
                )
                for path in paths
            ]
            all_handles.append(handles)
            for handle in handles[1:]:
                if set(handle.keys()) != set(handles[0].keys()):
                    raise ValueError(
                        "L_plus and L_star parameter names do not match"
                    )

        names = sorted(all_handles[0][0].keys())
        expected_names = set(names)
        for handles in all_handles:
            if any(set(handle.keys()) != expected_names for handle in handles):
                raise ValueError(
                    "All contribution inputs must contain exactly the same "
                    "parameter names"
                )

        for name in names:
            shape = tuple(all_handles[0][0].get_slice(name).get_shape())
            for handles in all_handles:
                if any(
                    tuple(handle.get_slice(name).get_shape()) != shape
                    for handle in handles
                ):
                    raise ValueError(
                        f"Contribution parameter shape mismatch: {name}"
                    )
            shapes[name] = shape
    return names, shapes


def _prepare_selections(
    strategy: dict[str, Any],
    models: list[dict[str, Any]],
    candidate_parameters: dict[str, torch.nn.Parameter] | None,
    candidate_names: list[str],
    candidate_numel: int,
    contribution_paths: list[list[Path]] | None,
    chunk_numel: int,
) -> list[dict[float, TopKSelection]]:
    """Prepare exact selections for every model and requested k."""

    ranking_method = strategy["ranking_method"]
    k_percents = strategy["k_percents"]
    has_scope = bool(strategy["layers"] or strategy["modules"])
    selections_by_model = []

    for model_index, model in enumerate(models):
        selections: dict[float, TopKSelection] = {}
        ranked_k = [
            k for k in k_percents if not (has_scope and k == 100)
        ]

        if ranking_method == "contribution":
            if ranked_k:
                paths = contribution_paths[model_index]
                batch = find_topk_selections(
                    paths[0],
                    ranked_k,
                    chunk_numel=chunk_numel,
                    selection_mode=strategy["selection_mode"],
                    l_star_contribution_path=(
                        paths[1] if len(paths) == 2 else None
                    ),
                    contribution_channel=strategy["contribution_channel"],
                    parameter_names=candidate_names if has_scope else None,
                )
                selections.update(batch)
        elif ranking_method in {"delta_magnitude", "finetuned_magnitude"}:
            finetuned_path = model.get("finetuned_model_path")
            if not isinstance(finetuned_path, str) or not finetuned_path.strip():
                raise ValueError(
                    f"Model {model['label']!r} requires finetuned_model_path "
                    f"for {ranking_method}"
                )
            for k_percent in ranked_k:
                selections[k_percent] = find_model_topk_selection(
                    candidate_parameters,
                    finetuned_path,
                    k_percent,
                    ranking_method,
                    chunk_numel,
                )
        else:
            if ranked_k:
                seed = model.get("seed")
                if isinstance(seed, bool) or not isinstance(seed, int):
                    raise ValueError(
                        f"Model {model['label']!r} must define an integer seed "
                        "for random ranking"
                    )
            for k_percent in ranked_k:
                selected_numel = math.ceil(
                    candidate_numel * k_percent / 100
                )
                selections[k_percent] = TopKSelection(
                    candidate_numel,
                    selected_numel,
                    None,
                    None,
                    0,
                    select_all=selected_numel == candidate_numel,
                )

        if has_scope and 100.0 in k_percents:
            selections[100.0] = TopKSelection(
                candidate_numel,
                candidate_numel,
                None,
                None,
                0,
                select_all=True,
            )

        if set(selections) != set(k_percents):
            missing = sorted(set(k_percents) - set(selections))
            raise RuntimeError(
                f"Failed to prepare selections for {model['label']}: {missing}"
            )
        if any(
            selection.total_numel != candidate_numel
            for selection in selections.values()
        ):
            raise ValueError(
                f"Model {model['label']!r} has a different candidate size"
            )
        selections_by_model.append(selections)

    return selections_by_model


def _accumulate_intersections(
    strategy: dict[str, Any],
    models: list[dict[str, Any]],
    selections: list[TopKSelection],
    candidate_parameters: dict[str, torch.nn.Parameter] | None,
    candidate_names: list[str],
    candidate_shapes: dict[str, tuple[int, ...]],
    contribution_paths: list[list[Path]] | None,
    chunk_numel: int,
    k_percent: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream aligned masks and accumulate pairwise intersections."""

    ranking_method = strategy["ranking_method"]
    model_count = len(models)
    intersections = np.zeros((model_count, model_count), dtype=np.int64)
    selected_counts = np.zeros(model_count, dtype=np.int64)
    ties_remaining = [
        selection.threshold_ties_to_keep for selection in selections
    ]

    with ExitStack() as stack:
        contribution_handles = None
        finetuned_readers = None
        random_masks = None

        if (
            ranking_method == "contribution"
            and not all(selection.select_all for selection in selections)
        ):
            contribution_handles = [
                [
                    stack.enter_context(
                        safe_open(str(path), framework="pt", device="cpu")
                    )
                    for path in paths
                ]
                for paths in contribution_paths
            ]
            expected_names = set(candidate_shapes)
            for handles in contribution_handles:
                if any(
                    not expected_names.issubset(handle.keys())
                    for handle in handles
                ):
                    raise ValueError(
                        "A contribution input is missing candidate parameters"
                    )
                for name, shape in candidate_shapes.items():
                    if any(
                        tuple(handle.get_slice(name).get_shape()) != shape
                        for handle in handles
                    ):
                        raise ValueError(
                            f"Contribution parameter shape mismatch: {name}"
                        )
        elif (
            ranking_method in {"delta_magnitude", "finetuned_magnitude"}
            and not all(selection.select_all for selection in selections)
        ):
            finetuned_readers = [
                stack.enter_context(
                    ModelTensorReader(model["finetuned_model_path"])
                )
                for model in models
            ]
            for model, reader in zip(models, finetuned_readers):
                if not set(candidate_names).issubset(reader.keys):
                    raise ValueError(
                        f"Fine-tuned checkpoint {model['label']!r} is missing "
                        "candidate parameters"
                    )
                for name, shape in candidate_shapes.items():
                    if reader.shape(name) != shape:
                        raise ValueError(
                            f"Fine-tuned parameter shape mismatch for "
                            f"{model['label']!r}: {name}"
                        )
        elif ranking_method == "random":
            if all(selection.select_all for selection in selections):
                random_masks = [None] * model_count
            else:
                random_masks = [
                    RandomSelectionMask(
                        selections[index].total_numel,
                        k_percent,
                        seed=models[index]["seed"],
                        chunk_numel=chunk_numel,
                    )
                    for index in range(model_count)
                ]
                selections = [mask.selection for mask in random_masks]

        for name in tqdm(
            candidate_names,
            desc=f"{strategy['name']} top-{k_percent:g}% overlap",
            unit="tensor",
        ):
            shape = candidate_shapes[name]
            iterators = []
            for model_index, selection in enumerate(selections):
                if selection.select_all or ranking_method == "random":
                    iterators.append(None)
                elif ranking_method == "contribution":
                    iterators.append(
                        iter(
                            combined_contribution_chunks(
                                contribution_handles[model_index],
                                name,
                                chunk_numel,
                            )
                        )
                    )
                else:
                    iterators.append(
                        iter(
                            model_ranking_chunks(
                                candidate_parameters,
                                finetuned_readers[model_index],
                                name,
                                chunk_numel,
                                ranking_method,
                            )
                        )
                    )

            for expected_index in chunk_indices(shape, chunk_numel):
                masks = []
                chunk_size = None
                for model_index, selection in enumerate(selections):
                    iterator = iterators[model_index]
                    if iterator is None:
                        size = _chunk_numel(shape, expected_index)
                        if (
                            ranking_method == "random"
                            and random_masks[model_index] is not None
                        ):
                            mask = random_masks[model_index].mask(size)
                        else:
                            mask = np.ones(size, dtype=bool)
                    else:
                        actual_index, scores = next(iterator)
                        if actual_index != expected_index:
                            raise RuntimeError(
                                f"Chunk traversal mismatch for parameter {name}"
                            )
                        size = scores.numel()
                        score_bits = (
                            contribution_score_bits(
                                scores, strategy["selection_mode"]
                            )
                            if ranking_method == "contribution"
                            else absolute_float_bits(scores)
                        )
                        mask, ties_remaining[model_index] = selection_mask(
                            score_bits,
                            selection,
                            ties_remaining[model_index],
                        )

                    if chunk_size is None:
                        chunk_size = size
                    elif size != chunk_size:
                        raise RuntimeError(
                            f"Chunk size mismatch for parameter {name}"
                        )
                    masks.append(mask)
                    selected_counts[model_index] += int(
                        np.count_nonzero(mask)
                    )

                for row in range(model_count):
                    diagonal = int(np.count_nonzero(masks[row]))
                    intersections[row, row] += diagonal
                    for column in range(row):
                        count = int(
                            np.count_nonzero(masks[row] & masks[column])
                        )
                        intersections[row, column] += count
                        intersections[column, row] += count

    expected_selected = np.array(
        [selection.selected_numel for selection in selections],
        dtype=np.int64,
    )
    if np.any(selected_counts != expected_selected) or any(ties_remaining):
        raise RuntimeError(
            "Top-k mask traversal did not match the exact selection counts"
        )
    return intersections, selected_counts


def pairwise_overlap_metrics(
    intersections: np.ndarray,
    selected_counts: np.ndarray,
    universe_numel: int,
) -> dict[str, np.ndarray]:
    """Derive raw, set-similarity, and chance-corrected matrices."""

    intersections = np.asarray(intersections, dtype=np.int64)
    selected_counts = np.asarray(selected_counts, dtype=np.int64)
    expected_shape = (len(selected_counts), len(selected_counts))
    if intersections.shape != expected_shape:
        raise ValueError("intersections shape does not match selected_counts")
    if universe_numel <= 0:
        raise ValueError("universe_numel must be positive")

    observed = intersections.astype(np.float64)
    sizes = selected_counts.astype(np.float64)
    minimum = np.minimum.outer(sizes, sizes)
    union = sizes[:, None] + sizes[None, :] - observed
    expected = np.outer(sizes, sizes) / universe_numel

    with np.errstate(divide="ignore", invalid="ignore"):
        overlap = np.divide(
            observed,
            minimum,
            out=np.full(expected_shape, np.nan),
            where=minimum > 0,
        )
        jaccard = np.divide(
            observed,
            union,
            out=np.full(expected_shape, np.nan),
            where=union > 0,
        )
        lift = np.divide(
            observed,
            expected,
            out=np.full(expected_shape, np.nan),
            where=expected > 0,
        )
        adjusted_denominator = minimum - expected
        chance_adjusted = np.divide(
            observed - expected,
            adjusted_denominator,
            out=np.full(expected_shape, np.nan),
            where=np.abs(adjusted_denominator) > np.finfo(np.float64).eps,
        )

    return {
        "intersection_count": intersections,
        "random_expected_intersection": expected,
        "overlap_coefficient": overlap,
        "jaccard": jaccard,
        "overlap_lift": lift,
        "chance_adjusted_overlap": chance_adjusted,
    }


def _save_heatmap(
    frame: pd.DataFrame,
    metric: str,
    strategy_name: str,
    k_percent: float,
    destination: Path,
) -> None:
    values = frame.to_numpy(dtype=np.float64)
    count = len(frame)
    figure_size = (
        max(7.0, 0.85 * count + 4.0),
        max(6.0, 0.72 * count + 3.0),
    )
    fig, ax = plt.subplots(figsize=figure_size, dpi=160)

    if metric == "chance_adjusted_overlap":
        image = ax.imshow(values, cmap="coolwarm", vmin=-1, vmax=1)
        colorbar_label = "Chance-adjusted overlap"
    else:
        image = ax.imshow(values, cmap="YlOrRd", vmin=0, vmax=1)
        colorbar_label = "Overlap"

    ax.set_xticks(np.arange(count), labels=frame.columns)
    ax.set_yticks(np.arange(count), labels=frame.index)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Model")
    ax.set_ylabel("Model")
    ax.set_title(
        f"{METRIC_TITLES[metric]}\n"
        f"strategy={strategy_name}; top {k_percent:g}%"
    )

    for row in range(count):
        for column in range(count):
            value = values[row, column]
            label = "N/A" if not np.isfinite(value) else f"{100 * value:.1f}%"
            if not np.isfinite(value):
                color = "black"
            elif metric == "chance_adjusted_overlap":
                color = "white" if abs(value) >= 0.55 else "black"
            else:
                color = "white" if value >= 0.62 else "black"
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _save_metric_outputs(
    metrics: dict[str, np.ndarray],
    labels: list[str],
    strategy_name: str,
    k_percent: float,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    frames = {}
    for metric, values in metrics.items():
        frame = pd.DataFrame(values, index=labels, columns=labels)
        frames[metric] = frame
        path = output_dir / f"{metric}.csv"
        if metric == "intersection_count":
            frame.to_csv(path)
        else:
            frame.to_csv(path, float_format="%.10g")
        if metric in HEATMAP_METRICS:
            _save_heatmap(
                frame,
                metric,
                strategy_name,
                k_percent,
                output_dir / metric,
            )
    return frames


def compute_topk_overlap(
    contribution_paths: list[str],
    k_percent: float,
    chunk_numel: int = 5_000_000,
    selection_mode: str = "absolute",
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility helper for direct single-channel contribution files."""

    if len(contribution_paths) < 2:
        raise ValueError("contribution_paths must contain at least two files")
    if selection_mode not in SELECTION_MODES:
        raise ValueError(f"selection_mode must be one of {SELECTION_MODES}")

    selections = [
        find_topk_selection(
            path,
            k_percent,
            chunk_numel,
            selection_mode=selection_mode,
        )
        for path in contribution_paths
    ]
    selected_numels = {selection.selected_numel for selection in selections}
    total_numels = {selection.total_numel for selection in selections}
    if len(total_numels) != 1 or len(selected_numels) != 1:
        raise ValueError("All contribution files must describe the same model size")

    selected_numel = selections[0].selected_numel
    if selected_numel == 0:
        raise ValueError("k_percent must select at least one parameter")

    model_count = len(contribution_paths)
    intersections = np.zeros((model_count, model_count), dtype=np.int64)
    selected_counts = np.zeros(model_count, dtype=np.int64)
    ties_remaining = [
        selection.threshold_ties_to_keep for selection in selections
    ]

    with ExitStack() as stack:
        handles = [
            stack.enter_context(
                safe_open(str(path), framework="pt", device="cpu")
            )
            for path in contribution_paths
        ]
        names = sorted(handles[0].keys())
        expected_names = set(names)
        if any(set(handle.keys()) != expected_names for handle in handles):
            raise ValueError(
                "All contribution files must contain exactly the same parameters"
            )

        for name in tqdm(names, desc="Comparing Top-k overlap", unit="tensor"):
            tensor_slices = [handle.get_slice(name) for handle in handles]
            shapes = [
                tuple(tensor_slice.get_shape()) for tensor_slice in tensor_slices
            ]
            if any(shape != shapes[0] for shape in shapes[1:]):
                raise ValueError(f"Contribution shape mismatch: {name}")

            for index in chunk_indices(shapes[0], chunk_numel):
                masks = []
                for model_index, (handle, tensor_slice, selection) in enumerate(
                    zip(handles, tensor_slices, selections)
                ):
                    values = (
                        handle.get_tensor(name)
                        if index is None
                        else tensor_slice[index]
                    )
                    mask, ties_remaining[model_index] = selection_mask(
                        contribution_score_bits(
                            values.reshape(-1), selection_mode
                        ),
                        selection,
                        ties_remaining[model_index],
                    )
                    masks.append(mask)
                    selected_counts[model_index] += int(
                        np.count_nonzero(mask)
                    )

                for row in range(model_count):
                    intersections[row, row] += int(
                        np.count_nonzero(masks[row])
                    )
                    for column in range(row):
                        count = int(
                            np.count_nonzero(masks[row] & masks[column])
                        )
                        intersections[row, column] += count
                        intersections[column, row] += count

    if np.any(selected_counts != selected_numel) or any(ties_remaining):
        raise RuntimeError("Top-k selection count does not match the requested k")

    overlap = intersections.astype(np.float64) / selected_numel
    return overlap, intersections


def compare_topk_overlap(
    models: list[dict[str, Any]],
    strategies: list[dict[str, Any]],
    output_dir: str,
    base_model_path: str | None = None,
    chunk_numel: int = 5_000_000,
) -> dict[tuple[str, float], dict[str, pd.DataFrame]]:
    """Run every configured strategy/k and save all pairwise outputs."""

    if not isinstance(chunk_numel, int) or isinstance(chunk_numel, bool):
        raise ValueError("chunk_numel must be a positive integer")
    if chunk_numel <= 0:
        raise ValueError("chunk_numel must be a positive integer")

    labels = _validate_models(models)
    if not isinstance(strategies, list) or not strategies:
        raise ValueError("strategies must contain at least one strategy")
    strategies = [_validate_strategy(strategy) for strategy in strategies]
    strategy_names = [strategy["name"] for strategy in strategies]
    if len(set(strategy_names)) != len(strategy_names):
        raise ValueError("Strategy names must be unique")
    safe_names = [_safe_component(name) for name in strategy_names]
    if len(set(safe_names)) != len(safe_names):
        raise ValueError("Strategy names must also be unique after filename sanitization")

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for strategy, safe_name in zip(strategies, safe_names):
        ranking_method = strategy["ranking_method"]
        has_scope = bool(strategy["layers"] or strategy["modules"])
        needs_base = ranking_method != "contribution" or has_scope
        _base_model = None
        parameters = None
        candidate_parameters = None
        contribution_paths = None

        try:
            ranking_needed = any(
                not (has_scope and k_percent == 100)
                for k_percent in strategy["k_percents"]
            )
            if ranking_method == "contribution" and ranking_needed:
                contribution_paths = [
                    _model_contribution_paths(
                        model, strategy["contribution_channel"]
                    )
                    for model in models
                ]

            if needs_base:
                _base_model, parameters = _load_base_parameters(
                    base_model_path,
                    strategy["merge_dtype"],
                )
                candidate_names, scope = select_parameters(
                    parameters,
                    strategy["layers"],
                    strategy["modules"],
                )
                candidate_names = sorted(candidate_names)
                candidate_parameters = {
                    name: parameters[name] for name in candidate_names
                }
                candidate_shapes = {
                    name: tuple(parameters[name].shape)
                    for name in candidate_names
                }
                candidate_numel = scope.selected_numel

                if ranking_method == "contribution" and ranking_needed:
                    expected_names = set(parameters)
                    with ExitStack() as stack:
                        for paths in contribution_paths:
                            handles = [
                                stack.enter_context(
                                    safe_open(
                                        str(path),
                                        framework="pt",
                                        device="cpu",
                                    )
                                )
                                for path in paths
                            ]
                            if any(
                                set(handle.keys()) != expected_names
                                for handle in handles
                            ):
                                raise ValueError(
                                    "Contribution parameters must exactly match "
                                    "the shared base model"
                                )
                            for name in candidate_names:
                                if any(
                                    tuple(handle.get_slice(name).get_shape())
                                    != candidate_shapes[name]
                                    for handle in handles
                                ):
                                    raise ValueError(
                                        f"Contribution parameter shape mismatch: "
                                        f"{name}"
                                    )
            else:
                candidate_names, candidate_shapes = _contribution_universe(
                    contribution_paths
                )
                candidate_numel = sum(
                    math.prod(shape) for shape in candidate_shapes.values()
                )

            print(
                f"Strategy {strategy['name']!r}: method={ranking_method}; "
                f"models={len(models)}; candidates={candidate_numel:,}; "
                f"k={strategy['k_percents']}",
                flush=True,
            )
            selections_by_model = _prepare_selections(
                strategy,
                models,
                candidate_parameters,
                candidate_names,
                candidate_numel,
                contribution_paths,
                chunk_numel,
            )

            for k_percent in strategy["k_percents"]:
                selections = [
                    model_selections[k_percent]
                    for model_selections in selections_by_model
                ]
                intersections, selected_counts = _accumulate_intersections(
                    strategy,
                    models,
                    selections,
                    candidate_parameters,
                    candidate_names,
                    candidate_shapes,
                    contribution_paths,
                    chunk_numel,
                    k_percent,
                )
                metrics = pairwise_overlap_metrics(
                    intersections,
                    selected_counts,
                    candidate_numel,
                )
                destination = (
                    root
                    / safe_name
                    / f"top-{_k_component(k_percent)}pct"
                )
                destination.mkdir(parents=True, exist_ok=True)
                metadata = {
                    "strategy": strategy,
                    "k_percent": k_percent,
                    "candidate_numel": candidate_numel,
                    "models": models,
                    "selected_numel": {
                        label: int(count)
                        for label, count in zip(labels, selected_counts)
                    },
                    "metric_definitions": {
                        "intersection_count": "|A intersect B|",
                        "random_expected_intersection": "|A|*|B|/candidate_numel",
                        "overlap_coefficient": "|A intersect B|/min(|A|,|B|)",
                        "jaccard": "|A intersect B|/|A union B|",
                        "overlap_lift": (
                            "observed intersection/random expected intersection"
                        ),
                        "chance_adjusted_overlap": (
                            "(observed-expected)/(min(|A|,|B|)-expected); "
                            "undefined when k=100%"
                        ),
                    },
                }
                (destination / "metadata.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                frames = _save_metric_outputs(
                    metrics,
                    labels,
                    strategy["name"],
                    k_percent,
                    destination,
                )
                all_results[(strategy["name"], k_percent)] = frames
                print(
                    f"Saved {strategy['name']} top-{k_percent:g}% overlap "
                    f"outputs to {destination}",
                    flush=True,
                )
        finally:
            candidate_parameters = parameters = None
            _base_model = None
            gc.collect()

    return all_results


def main() -> None:
    """Load the YAML interface and run all pairwise comparisons."""

    config = parse_args_yaml("Compare exact Top-k selections across N models")
    compare_topk_overlap(**config)


if __name__ == "__main__":
    main()
