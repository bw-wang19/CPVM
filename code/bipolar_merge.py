"""Bipolar CPVM merging with a shared coordinate t in [-1, 1].

Piecewise-linear merging combines the two segment attributions by coordinate.
Endpoint-linear and quadratic merging use attribution on the complete dense
low-to-high path. Each method selects one common coordinate mask, forms the
sparse endpoints, then interpolates them:

  d_high = alpha_high * rescale * M * (theta_plus - theta_base)
  d_low  = alpha_low  * rescale * M * (theta_star - theta_base)
  theta(t) = theta_base + c_high(t) * d_high + c_low(t) * d_low

The two piecewise branches use opposite target labels and path directions.
Corresponding channels therefore share the same low-to-high score direction
after swapping low's L_plus/L_star labels. Their coordinate sum ranks one mask
shared by both endpoint-base displacements.

The coefficients come from utils.bipolar_path and are shared with attribution.
The quadratic is a three-point interpolating polynomial, not an alpha^2
reparameterization of a line. The endpoint-linear midpoint is the mean of
the two resulting endpoints, while the other two paths pass through base.
Attribution describes the dense source path; sparsification/scaling constructs
a new path and is explicitly recorded rather than claimed to preserve it.

All merging runs on CPU. Checkpoint shards are read in chunks, masks are
selected once per path, and all t values reuse them.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, replace
import gc
import hashlib
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
from safetensors import safe_open
import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from CPVM.code.topk_merge import (
    CONTRIBUTION_CHANNELS,
    RANKING_METHODS,
    ModelTensorReader,
    RandomSelectionMask,
    RADIX_SHIFTS,
    TopKSelection,
    absolute_float_bits,
    chunk_indices,
    combined_contribution_chunks,
    contribution_score_bits,
    discover_contribution_files,
    find_topk_selection,
    ordered_bits_to_float,
    resolve_contribution_paths,
    resolve_dtype,
    selection_mask,
)
from CPVM.code.utils.arguments import parse_args_yaml
from CPVM.code.utils.bipolar_path import path_coefficients, validate_path_method
from CPVM.code.utils.bipolar_merge_evaluation import (
    evaluate_curve_point,
    load_evaluations,
)
from CPVM.code.utils.bipolar_merge_plots import write_evaluation_plots
from CPVM.code.utils.parameter_selection import select_parameters


POLES = ("high", "low")
DEFAULT_SELECTION = {
    "ranking_method": "contribution",
    "contribution_channel": "L_contrast",
    "selection_mode": "positive",
    "k_percent": 5.0,
    "layers": None,
    "modules": None,
    "seed": 0,
    "alpha": 1.0,
    "use_weight_rescale": False,
}


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _normalise_selection(selection):
    """Preserve topk_merge's selectors for one pole or one common path."""

    if not isinstance(selection, dict):
        raise ValueError("selection and each pole override must be mappings")
    unknown = set(selection) - set(DEFAULT_SELECTION)
    if unknown:
        raise ValueError(f"Unknown selection options: {sorted(unknown)}")
    settings = {**DEFAULT_SELECTION, **selection}
    if settings["ranking_method"] not in RANKING_METHODS:
        raise ValueError(f"ranking_method must be one of {RANKING_METHODS}")
    if settings["contribution_channel"] not in CONTRIBUTION_CHANNELS:
        raise ValueError(f"contribution_channel must be one of {CONTRIBUTION_CHANNELS}")
    if settings["selection_mode"] not in {"positive", "absolute"}:
        raise ValueError("selection_mode must be positive or absolute")
    settings["k_percent"] = _finite_number(settings["k_percent"], "k_percent")
    if not 0 <= settings["k_percent"] <= 100:
        raise ValueError("k_percent must be within [0, 100]")
    settings["alpha"] = _finite_number(settings["alpha"], "alpha")
    if type(settings["use_weight_rescale"]) is not bool:
        raise ValueError("use_weight_rescale must be a boolean")
    if settings["use_weight_rescale"] and settings["k_percent"] == 0:
        raise ValueError("Weight rescaling is undefined for k_percent=0")
    if type(settings["seed"]) is not int or settings["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    layers, modules = settings["layers"], settings["modules"]
    if layers is not None:
        if not isinstance(layers, list) or any(
            type(layer) is not int or layer < 0 for layer in layers
        ):
            raise ValueError("layers must be null or a list of nonnegative integers")
        settings["layers"] = sorted(set(layers))
    if modules is not None:
        if not isinstance(modules, list) or any(
            not isinstance(module, str) or not module.strip(".").strip()
            for module in modules
        ):
            raise ValueError("modules must be null or a list of module names")
        settings["modules"] = list(dict.fromkeys(modules))
    return settings


def _read_attribution_metadata(directory, path_method, pole, source_paths, trait):
    """Reject attribution from a different geometric path or endpoint pair.

    Legacy base-to-endpoint files without a sidecar may be used explicitly for
    piecewise_linear only. New geometries always require path provenance.
    """

    metadata_path = Path(directory).expanduser() / "attribution_metadata.json"
    if not metadata_path.is_file():
        if path_method != "piecewise_linear":
            raise ValueError(
                f"{path_method} requires newly computed path attribution with "
                f"{metadata_path}; old base-to-endpoint matrices cannot be reused"
            )
        return {
            "legacy_without_metadata": True,
            "declared_path_method": "piecewise_linear",
            "declared_target_pole": pole,
        }
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("path_method") != path_method:
        raise ValueError(f"Attribution path_method mismatch in {metadata_path}")
    if metadata.get("target_pole") != pole or metadata.get("trait") != trait:
        raise ValueError(f"Attribution trait/target_pole mismatch in {metadata_path}")
    if path_method == "piecewise_linear":
        expected = {
            "base_model_path": source_paths["base_model_path"],
            "endpoint_model_path": source_paths[
                "plus_model_path" if pole == "high" else "star_model_path"
            ],
        }
    else:
        if pole != "common":
            raise ValueError(f"{path_method} requires one common attribution directory")
        expected = {
            "base_model_path": source_paths["base_model_path"],
            "plus_model_path": source_paths["plus_model_path"],
            "star_model_path": source_paths["star_model_path"],
        }
        if (
            metadata.get("attribution_kind") != "common_full_path"
            or metadata.get("start_t") != -1
            or metadata.get("end_t") != 1
            or metadata.get("path_direction") != "low_to_high"
        ):
            raise ValueError(f"Attribution full-path interval mismatch in {metadata_path}")
    for key, value in expected.items():
        recorded = metadata.get(key)
        if not isinstance(recorded, str) or Path(recorded).expanduser().resolve() != Path(value):
            raise ValueError(f"Attribution {key} mismatch in {metadata_path}")
    return metadata


def _common_model_ranking_chunks(
    plus_reader, star_reader, parameters, name, chunk_numel, ranking_method,
):
    """Yield one score stream for a continuous two-endpoint path."""

    shape = tuple(parameters[name].shape)
    if plus_reader.shape(name) != shape or star_reader.shape(name) != shape:
        raise ValueError(f"Continuous endpoint shape mismatch for parameter: {name}")
    for index in chunk_indices(shape, chunk_numel):
        plus = plus_reader.read(name, index).reshape(-1).float()
        star = star_reader.read(name, index).reshape(-1).float()
        if ranking_method == "delta_magnitude":
            scores = plus.sub(star)
        elif ranking_method == "finetuned_magnitude":
            scores = torch.maximum(plus.abs(), star.abs())
        else:
            raise ValueError(f"Unsupported continuous ranking method: {ranking_method}")
        yield index, scores


def _find_common_model_topk_selection(
    parameters, plus_model_path, star_model_path, k_percent,
    ranking_method, chunk_numel,
):
    """Find an exact threshold for endpoint-difference or joint endpoint magnitude."""

    names = sorted(parameters)
    total_numel = sum(parameter.numel() for parameter in parameters.values())
    selected_numel = math.ceil(total_numel * k_percent / 100)
    if selected_numel == 0:
        return TopKSelection(total_numel, 0, None, None, 0)
    if selected_numel == total_numel:
        return TopKSelection(total_numel, total_numel, None, None, 0, True)

    with (
        ModelTensorReader(plus_model_path) as plus_reader,
        ModelTensorReader(star_model_path) as star_reader,
    ):
        if not set(names).issubset(plus_reader.keys) or not set(names).issubset(star_reader.keys):
            raise ValueError("Base and continuous-path endpoint parameter names do not match")
        rank = selected_numel
        prefix = 0
        for step, shift in enumerate(RADIX_SHIFTS, start=1):
            counts = np.zeros(256, dtype=np.int64)
            description = (
                f"Selecting common {ranking_method} threshold "
                f"({step}/{len(RADIX_SHIFTS)})"
            )
            for name in tqdm(names, desc=description, unit="tensor"):
                for _, scores in _common_model_ranking_chunks(
                    plus_reader, star_reader, parameters, name,
                    chunk_numel, ranking_method,
                ):
                    if shift == RADIX_SHIFTS[0] and not torch.isfinite(scores).all():
                        raise ValueError(f"Endpoint tensor contains NaN or Inf: {name}")
                    bits = absolute_float_bits(scores)
                    if shift != RADIX_SHIFTS[0]:
                        bits = bits[(bits >> (shift + 8)) == prefix]
                    digits = (bits >> shift) & np.uint32(0xFF)
                    counts += np.bincount(digits, minlength=256)
            for digit in range(255, -1, -1):
                digit_count = int(counts[digit])
                if rank > digit_count:
                    rank -= digit_count
                    continue
                prefix = (prefix << 8) | digit
                break
            else:
                raise RuntimeError("Common Top-k rank exceeded its prefix count")
    threshold_score = np.array([prefix], dtype=np.uint32).view(np.float32)[0]
    return TopKSelection(
        total_numel=total_numel,
        selected_numel=selected_numel,
        threshold_bits=prefix,
        threshold_score=float(threshold_score),
        threshold_ties_to_keep=rank,
    )


def _validate_piecewise_shared_metadata(metadata_by_pole):
    """Ensure both branch contrasts refer to the same fixed score and examples."""

    high = metadata_by_pole["high"]
    low = metadata_by_pole["low"]
    if any(metadata.get("legacy_without_metadata") for metadata in (high, low)):
        raise ValueError(
            "Shared piecewise contribution ranking requires attribution_metadata.json "
            "for both high and low branches to verify their objective"
        )
    high_objective = high.get("objective")
    low_objective = low.get("objective")
    if not isinstance(high_objective, dict) or not isinstance(low_objective, dict):
        raise ValueError("Shared piecewise contribution ranking requires objective metadata")
    if high_objective != low_objective:
        raise ValueError("Piecewise high/low attribution objective settings differ")
    for key in ("k", "interpolation", "quadrature", "attribution_scope"):
        if high.get(key) is None or high.get(key) != low.get(key):
            raise ValueError(f"Piecewise high/low attribution {key} differs or is missing")
    if high_objective.get("use_original_prompt") is not False:
        raise ValueError(
            "Shared piecewise contribution ranking requires use_original_prompt=false "
            "so both branches evaluate the same fixed score"
        )
    for pole, metadata in metadata_by_pole.items():
        if (
            metadata.get("attribution_kind") != "branch"
            or metadata.get("objective_target_pole") != pole
            or metadata.get("start_t") != 0
            or metadata.get("end_t") != (-1 if pole == "low" else 1)
            or metadata.get("path_direction") != "center_to_endpoint"
        ):
            raise ValueError(f"Piecewise {pole} attribution branch direction mismatch")
    if (
        not isinstance(high.get("num_examples"), int)
        or high["num_examples"] <= 0
        or high["num_examples"] != low.get("num_examples")
    ):
        raise ValueError("Piecewise high/low attribution sample counts differ or are missing")


def _piecewise_contribution_paths(files_by_pole, contribution_channel):
    """Map low's reversed objective and direction back to high-score channels."""

    if contribution_channel == "L_plus":
        channels = (("high", "L_plus"), ("low", "L_star"))
    elif contribution_channel == "L_star":
        channels = (("high", "L_star"), ("low", "L_plus"))
    elif contribution_channel == "L_contrast":
        channels = (
            ("high", "L_plus"), ("high", "L_star"),
            ("low", "L_plus"), ("low", "L_star"),
        )
    else:
        raise ValueError(f"Unsupported piecewise contribution channel: {contribution_channel}")
    return [files_by_pole[pole][channel] for pole, channel in channels]


def _find_piecewise_contribution_topk_selection(
    parameters, files_by_pole, k_percent, selection_mode,
    contribution_channel, chunk_numel,
):
    """Select exactly k% from the corresponding two-branch score channel."""

    names = sorted(parameters)
    total_numel = sum(parameter.numel() for parameter in parameters.values())
    selected_numel = math.ceil(total_numel * k_percent / 100)
    if selected_numel == 0:
        return TopKSelection(total_numel, 0, None, None, 0)
    if selected_numel == total_numel:
        return TopKSelection(total_numel, total_numel, None, None, 0, True)

    with ExitStack() as stack:
        handles = [
            stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
            for path in _piecewise_contribution_paths(files_by_pole, contribution_channel)
        ]
        for name in names:
            shape = tuple(parameters[name].shape)
            if any(name not in handle.keys() for handle in handles):
                raise ValueError(f"Piecewise contribution is missing parameter: {name}")
            if any(tuple(handle.get_slice(name).get_shape()) != shape for handle in handles):
                raise ValueError(f"Piecewise contribution shape mismatch for {name}")

        rank = selected_numel
        prefix = 0
        for step, shift in enumerate(RADIX_SHIFTS, start=1):
            counts = np.zeros(256, dtype=np.int64)
            description = (
                f"Selecting shared piecewise {contribution_channel} threshold "
                f"({step}/{len(RADIX_SHIFTS)})"
            )
            for name in tqdm(names, desc=description, unit="tensor"):
                for _, scores in combined_contribution_chunks(handles, name, chunk_numel):
                    if shift == RADIX_SHIFTS[0] and not torch.isfinite(scores).all():
                        raise ValueError(f"Piecewise contribution contains NaN or Inf: {name}")
                    bits = contribution_score_bits(scores, selection_mode)
                    if shift != RADIX_SHIFTS[0]:
                        bits = bits[(bits >> (shift + 8)) == prefix]
                    digits = (bits >> shift) & np.uint32(0xFF)
                    counts += np.bincount(digits, minlength=256)
            for digit in range(255, -1, -1):
                digit_count = int(counts[digit])
                if rank > digit_count:
                    rank -= digit_count
                    continue
                prefix = (prefix << 8) | digit
                break
            else:
                raise RuntimeError("Piecewise Top-k rank exceeded its prefix count")

    threshold_score = (
        ordered_bits_to_float(prefix)
        if selection_mode == "positive"
        else float(np.array([prefix], dtype=np.uint32).view(np.float32)[0])
    )
    return TopKSelection(
        total_numel=total_numel,
        selected_numel=selected_numel,
        threshold_bits=prefix,
        threshold_score=float(threshold_score),
        threshold_ties_to_keep=rank,
    )


def _prepare_piecewise_shared_updates(
    parameters, plus_model_path, star_model_path, settings, endpoint_alphas,
    contribution_dirs, source_paths, trait, sparsify, chunk_numel,
):
    """Share one coordinate mask across both piecewise endpoint displacements."""

    return _prepare_common_updates(
        parameters=parameters,
        plus_model_path=plus_model_path,
        star_model_path=star_model_path,
        settings=settings,
        endpoint_alphas=endpoint_alphas,
        contribution_dir=contribution_dirs,
        path_method="piecewise_linear",
        source_paths=source_paths,
        trait=trait,
        sparsify=sparsify,
        chunk_numel=chunk_numel,
    )


def _prepare_common_updates(
    parameters, plus_model_path, star_model_path, settings, endpoint_alphas,
    contribution_dir, path_method, source_paths, trait, sparsify, chunk_numel,
):
    """Select one coordinate mask and apply it to both endpoint-base deltas."""

    is_piecewise = path_method == "piecewise_linear"
    if sparsify:
        names, scope = select_parameters(
            parameters, settings["layers"], settings["modules"]
        )
    else:
        names, scope = select_parameters(parameters, None, None)
    names = sorted(names)
    candidate_parameters = {name: parameters[name] for name in names}
    k_percent = settings["k_percent"] if sparsify else 100.0
    full_scope = (
        sparsify and k_percent == 100
        and (is_piecewise or bool(settings["layers"] or settings["modules"]))
    )
    method = settings["ranking_method"]
    needs_contributions = (
        sparsify and not full_scope and k_percent > 0 and method == "contribution"
    )
    files = {}
    metadata = None
    if needs_contributions:
        if is_piecewise:
            if not isinstance(contribution_dir, dict):
                raise ValueError("piecewise_linear requires contribution_dirs.high/low")
            metadata = {}
            for pole in POLES:
                directory = contribution_dir.get(pole)
                if not directory:
                    raise ValueError(f"contribution_dirs.{pole} is required")
                files[pole] = discover_contribution_files(directory)
                metadata[pole] = _read_attribution_metadata(
                    directory, path_method, pole, source_paths, trait
                )
            _validate_piecewise_shared_metadata(metadata)
        else:
            if not contribution_dir:
                raise ValueError("contribution_dirs.common is required")
            files = discover_contribution_files(contribution_dir)
            metadata = _read_attribution_metadata(
                contribution_dir, path_method, "common", source_paths, trait
            )

    random_masks = None
    if not sparsify or full_scope:
        chosen = TopKSelection(
            scope.selected_numel, scope.selected_numel, None, None, 0, True
        )
    elif k_percent == 0:
        chosen = TopKSelection(scope.selected_numel, 0, None, None, 0)
    elif method == "random":
        random_masks = RandomSelectionMask(
            scope.selected_numel, k_percent, settings["seed"], chunk_numel
        )
        chosen = random_masks.selection
    elif method == "contribution":
        if is_piecewise:
            chosen = _find_piecewise_contribution_topk_selection(
                candidate_parameters, files, k_percent,
                settings["selection_mode"], settings["contribution_channel"],
                chunk_numel,
            )
        else:
            chosen = find_topk_selection(
                str(files["L_plus"]),
                k_percent,
                chunk_numel,
                selection_mode=settings["selection_mode"],
                l_star_contribution_path=str(files["L_star"]),
                contribution_channel=settings["contribution_channel"],
                parameter_names=names,
            )
    else:
        chosen = _find_common_model_topk_selection(
            candidate_parameters,
            plus_model_path,
            star_model_path,
            k_percent,
            method,
            chunk_numel,
        )

    if chosen.total_numel != scope.selected_numel:
        raise ValueError("Common contribution and candidate counts do not match")
    chosen = replace(
        chosen, total_numel=scope.total_numel, candidate_numel=scope.selected_numel
    )
    retain_fraction = chosen.selected_numel / scope.selected_numel
    rescale = (
        1.0 / retain_fraction
        if sparsify and settings["use_weight_rescale"] else 1.0
    )
    effective_alphas = {
        pole: endpoint_alphas[pole] * rescale if sparsify else 1.0
        for pole in POLES
    }
    report = {
        "target_pole": "common",
        "sparsify": sparsify,
        "settings": settings,
        "endpoint_alphas": dict(endpoint_alphas),
        "effective_endpoint_alphas": effective_alphas,
        "selection": asdict(chosen),
        "selected_percent_of_model": 100 * chosen.selected_numel / scope.total_numel,
        "selected_percent_of_candidate_scope": 100 * retain_fraction,
        "rescale_factor": rescale,
        "rescale_denominator": "actual selected scalars / candidate-scope scalars",
        "mask_application": "same coordinate mask on both endpoint-base deltas",
        **({
            "contribution_dirs": (
                {
                    pole: str(Path(contribution_dir[pole]).expanduser().resolve())
                    for pole in POLES
                }
                if needs_contributions else None
            )
        } if is_piecewise else {
            "contribution_dir": (
                str(Path(contribution_dir).expanduser().resolve())
                if needs_contributions else None
            )
        }),
        "contribution_files": (
            {
                pole: {channel: str(path) for channel, path in files[pole].items()}
                for pole in POLES
            }
            if is_piecewise and needs_contributions
            else {key: str(value) for key, value in files.items()}
        ),
        "attribution_metadata": metadata,
    }
    print(
        f"common: selected {chosen.selected_numel:,}/{scope.selected_numel:,} "
        f"candidate scalars; effective endpoint alphas={effective_alphas}",
        flush=True,
    )
    updates = {pole: {} for pole in POLES}
    if chosen.selected_numel == 0:
        return updates, report

    contribution_paths = []
    if needs_contributions:
        contribution_paths = (
            _piecewise_contribution_paths(files, settings["contribution_channel"])
            if is_piecewise
            else resolve_contribution_paths(
                files["L_plus"], files["L_star"], settings["contribution_channel"]
            )
        )
    ties_remaining = chosen.threshold_ties_to_keep
    selected_count = 0
    with ExitStack() as stack:
        plus_reader = stack.enter_context(ModelTensorReader(plus_model_path))
        star_reader = stack.enter_context(ModelTensorReader(star_model_path))
        readers = {"high": plus_reader, "low": star_reader}
        handles = [
            stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
            for path in contribution_paths
        ]
        if any(not set(names).issubset(reader.keys) for reader in readers.values()):
            raise ValueError("Continuous endpoints are missing candidate parameter names")
        if handles and any(not set(names).issubset(handle.keys()) for handle in handles):
            raise ValueError("Common contribution files are missing candidate tensors")

        for name in tqdm(names, desc="Preparing common endpoint mask", unit="tensor"):
            parameter = parameters[name]
            shape = tuple(parameter.shape)
            if any(reader.shape(name) != shape for reader in readers.values()):
                raise ValueError(f"Continuous endpoint shape mismatch for {name}")
            if any(tuple(handle.get_slice(name).get_shape()) != shape for handle in handles):
                raise ValueError(f"Common contribution shape mismatch for {name}")
            active_updates = {
                pole: torch.zeros(shape, dtype=torch.float32, device="cpu")
                for pole in POLES if effective_alphas[pole] != 0
            }
            if chosen.select_all or random_masks is not None:
                chunks = ((index, None) for index in chunk_indices(shape, chunk_numel))
            elif needs_contributions:
                chunks = combined_contribution_chunks(handles, name, chunk_numel)
            else:
                chunks = _common_model_ranking_chunks(
                    plus_reader, star_reader, candidate_parameters,
                    name, chunk_numel, method,
                )
            for index, scores in chunks:
                base_chunk = parameter.detach() if index is None else parameter.detach()[index]
                if random_masks is not None:
                    mask = random_masks.mask(base_chunk.numel())
                elif chosen.select_all:
                    mask = np.ones(base_chunk.numel(), dtype=bool)
                else:
                    bits = (
                        contribution_score_bits(scores, settings["selection_mode"])
                        if needs_contributions else absolute_float_bits(scores)
                    )
                    mask, ties_remaining = selection_mask(bits, chosen, ties_remaining)
                selected_count += int(mask.sum())
                if not mask.any():
                    continue
                mask_tensor = torch.from_numpy(mask)
                for pole, destination_tensor in active_updates.items():
                    delta = readers[pole].read(name, index).float().sub(base_chunk.float())
                    delta.reshape(-1).masked_fill_(~mask_tensor, 0.0)
                    delta.mul_(effective_alphas[pole])
                    destination = (
                        destination_tensor if index is None else destination_tensor[index]
                    )
                    destination.copy_(delta)
            for pole, update in active_updates.items():
                updates[pole][name] = update

    if selected_count != chosen.selected_numel or ties_remaining != 0:
        raise RuntimeError("Common selected mask count did not match its threshold")
    if random_masks is not None and random_masks.remaining_numel != 0:
        raise RuntimeError("Common random selection stream was not fully consumed")
    return updates, report


@torch.no_grad()
def _write_curve_point(model, base_parameters, high_updates, low_updates, t, path_method, chunk_numel):
    """Always reconstruct from frozen base/updates, avoiding cumulative drift."""

    high_coefficient, low_coefficient = path_coefficients(t, path_method)
    for name, parameter in tqdm(model.named_parameters(), desc=f"Writing t={t:g}", unit="tensor"):
        for index in chunk_indices(tuple(parameter.shape), chunk_numel):
            base = base_parameters[name] if index is None else base_parameters[name][index]
            value = base.float().clone()
            for coefficient, updates in (
                (high_coefficient, high_updates), (low_coefficient, low_updates)
            ):
                if coefficient and name in updates:
                    update = updates[name] if index is None else updates[name][index]
                    value.add_(update, alpha=coefficient)
            destination = parameter if index is None else parameter[index]
            destination.copy_(value.to(destination.dtype))


def _safe_name(name):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
    if not value or value in {".", ".."}:
        raise ValueError("experiment_name must contain filename-safe characters")
    return value


def run_bipolar_merge(
    base_model_path: str,
    plus_model_path: str,
    star_model_path: str,
    output_dir: str,
    trait: str,
    path_method: str,
    t_values: list[float] | None = None,
    t: float | None = None,
    sparsify: bool = True,
    selection: dict | None = None,
    pole_overrides: dict | None = None,
    contribution_dirs: dict | None = None,
    experiment_name: str | None = None,
    merge_dtype: str = "auto",
    chunk_numel: int = 5_000_000,
    max_shard_size: str = "5GB",
    save_models: bool = True,
    evaluations: dict | None = None,
) -> dict:
    """Build all requested points on one dense or sparse bipolar curve."""

    validate_path_method(path_method)
    if type(sparsify) is not bool:
        raise ValueError("sparsify must be a boolean")
    if type(save_models) is not bool:
        raise ValueError("save_models must be a boolean")
    evaluation_specs = load_evaluations(evaluations)
    if not save_models and not evaluation_specs:
        raise ValueError("save_models=false requires at least one evaluation")
    if not isinstance(trait, str) or not trait.strip():
        raise ValueError("trait must be a nonempty string")
    if type(chunk_numel) is not int or chunk_numel <= 0:
        raise ValueError("chunk_numel must be a positive integer")
    if t_values is not None and t is not None:
        raise ValueError("Set t_values or t, not both")
    raw_coordinates = t_values if t_values is not None else [0.0 if t is None else t]
    if not isinstance(raw_coordinates, (list, tuple)) or not raw_coordinates:
        raise ValueError("t_values must be a nonempty list")
    coordinates = list(dict.fromkeys(
        _finite_number(value, "t") for value in raw_coordinates
    ))
    for coordinate in coordinates:
        path_coefficients(coordinate, path_method)

    common = _normalise_selection(selection if selection is not None else {})
    overrides = {} if pole_overrides is None else pole_overrides
    directories = {} if contribution_dirs is None else contribution_dirs
    if not isinstance(overrides, dict) or set(overrides) - set(POLES):
        raise ValueError("pole_overrides must be a mapping containing only high/low")
    if not isinstance(directories, dict):
        raise ValueError("contribution_dirs must be a mapping")

    is_piecewise = path_method == "piecewise_linear"
    if is_piecewise:
        if set(directories) - set(POLES):
            raise ValueError("piecewise_linear contribution_dirs accepts only high/low")
    elif set(directories) - {"common"}:
        raise ValueError(
            f"{path_method} contribution_dirs accepts only the common key"
        )
    endpoint_alphas = {}
    for pole in POLES:
        override = overrides.get(pole, {})
        if not isinstance(override, dict):
            raise ValueError(f"pole_overrides.{pole} must be a mapping")
        unknown = set(override) - {"alpha"}
        if unknown:
            raise ValueError(
                f"{path_method} uses one common selection; "
                f"pole_overrides.{pole} may override only alpha, got {sorted(unknown)}"
            )
        endpoint_alphas[pole] = _finite_number(
            override.get("alpha", common["alpha"]), f"pole_overrides.{pole}.alpha"
        )
    selection_recipe = common
    sparsification_description = (
        "one common piecewise mask applied to both endpoint-base deltas"
        if is_piecewise
        else "one common full-path mask applied to both endpoint-base deltas"
    )

    sources = {
        "base_model_path": str(Path(base_model_path).expanduser().resolve()),
        "plus_model_path": str(Path(plus_model_path).expanduser().resolve()),
        "star_model_path": str(Path(star_model_path).expanduser().resolve()),
    }
    recipe = {
        "path_method": path_method,
        "source_models": sources,
        "trait": trait,
        "sparsify": sparsify,
        "sparsification": sparsification_description,
        "selection": selection_recipe if sparsify else None,
        "endpoint_alphas": endpoint_alphas if sparsify else None,
        "contribution_dirs": directories if sparsify else None,
        "merge_dtype": merge_dtype,
    }
    identity = hashlib.sha256(
        json.dumps(recipe, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    run_name = f"{_safe_name(experiment_name or trait)}-{'sparse' if sparsify else 'dense'}-{identity}"
    run_directory = Path(output_dir).expanduser().resolve() / path_method / run_name
    for source in map(Path, sources.values()):
        if run_directory == source or run_directory in source.parents or source in run_directory.parents:
            raise ValueError("Merge output must not contain, overwrite, or be inside a source model")

    summary_path = run_directory / "merge_summary.json"
    previous_report = None
    previous_points = []
    if summary_path.is_file():
        previous_report = json.loads(summary_path.read_text(encoding="utf-8"))
        if any(previous_report.get(key) != value for key, value in recipe.items()):
            raise ValueError(f"Existing merge summary has a different recipe: {summary_path}")
        previous_points = previous_report.get("points", [])

    # A completed evaluate-only sweep should not reload and rewrite a 4B model.
    if previous_report is not None and not save_models and evaluation_specs:
        def has_current_results(coordinate):
            point = next(
                (item for item in previous_points if item.get("t") == coordinate), None
            )
            if point is None:
                return False
            existing = point.get("evaluations") or {}
            return all(
                name in existing
                and existing[name].get("config_hash") == specification["config_hash"]
                and Path(existing[name].get("result_path", "")).is_file()
                for name, specification in evaluation_specs.items()
            )

        if all(has_current_results(coordinate) for coordinate in coordinates):
            report = {
                **previous_report,
                "requested_t_values": coordinates,
                "save_models_requested": False,
                "evaluation_protocols": {
                    name: {
                        "config_path": specification["config_path"],
                        "config_hash": specification["config_hash"],
                    }
                    for name, specification in evaluation_specs.items()
                },
            }
            report["plots"] = write_evaluation_plots(report, run_directory)
            temporary_summary = summary_path.with_suffix(".json.tmp")
            temporary_summary.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary_summary.replace(summary_path)
            print(f"Reused completed bipolar sweep: {summary_path}", flush=True)
            return report

    model = AutoModelForCausalLM.from_pretrained(
        sources["base_model_path"],
        trust_remote_code=True,
        dtype=resolve_dtype(merge_dtype),
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.requires_grad_(False)
    parameters = dict(model.named_parameters())
    updates = {}
    selection_reports = {}
    if is_piecewise:
        updates, common_report = _prepare_piecewise_shared_updates(
            parameters=parameters,
            plus_model_path=sources["plus_model_path"],
            star_model_path=sources["star_model_path"],
            settings=common,
            endpoint_alphas=endpoint_alphas,
            contribution_dirs=directories,
            source_paths=sources,
            trait=trait,
            sparsify=sparsify,
            chunk_numel=chunk_numel,
        )
        selection_reports["common"] = common_report
    else:
        updates, common_report = _prepare_common_updates(
            parameters=parameters,
            plus_model_path=sources["plus_model_path"],
            star_model_path=sources["star_model_path"],
            settings=common,
            endpoint_alphas=endpoint_alphas,
            contribution_dir=directories.get("common"),
            path_method=path_method,
            source_paths=sources,
            trait=trait,
            sparsify=sparsify,
            chunk_numel=chunk_numel,
        )
        selection_reports["common"] = common_report
    # The live model is reused to save each point; preserve a frozen base once.
    base_parameters = {
        name: parameter.detach().clone() for name, parameter in parameters.items()
    }
    tokenizer = AutoTokenizer.from_pretrained(
        sources["base_model_path"], trust_remote_code=True, use_fast=False
    )
    report = {
        **recipe,
        "run_name": run_name,
        "run_directory": str(run_directory),
        "coordinate": {"star": -1, "center": 0, "plus": 1},
        "center_model": (
            "mean of the prepared high/low endpoints"
            if path_method == "endpoint_linear" else "base"
        ),
        "attribution_path": (
            "dense low-to-high path assembled from both source branches"
            if is_piecewise else "one dense low-to-high full path"
        ),
        "prepared_endpoints": selection_reports,
        "requested_t_values": coordinates,
        "save_models_requested": save_models,
        "evaluation_protocols": {
            name: {
                "config_path": specification["config_path"],
                "config_hash": specification["config_hash"],
            }
            for name, specification in evaluation_specs.items()
        },
        "points": list(previous_points),
    }
    run_directory.mkdir(parents=True, exist_ok=True)

    def save_progress(point, metadata_path):
        """Persist each completed step so a long evaluation can be resumed."""
        report["points"] = sorted(
            [previous for previous in report["points"] if previous["t"] != point["t"]]
            + [point],
            key=lambda item: item["t"],
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        point_report = {**report, "point": point}
        point_report.pop("points")
        metadata_path.write_text(
            json.dumps(point_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_summary = summary_path.with_suffix(".json.tmp")
        temporary_summary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_summary.replace(summary_path)

    for coordinate in coordinates:
        coordinate_label = repr(coordinate).replace("-", "m").replace(".", "p")
        point_name = f"t-{coordinate_label}"
        previous_point = next(
            (item for item in report["points"] if item["t"] == coordinate), None
        )
        previous_evaluations = (
            dict(previous_point.get("evaluations") or {}) if previous_point else {}
        )
        pending = [
            name for name, specification in evaluation_specs.items()
            if (
                name not in previous_evaluations
                or previous_evaluations[name].get("config_hash")
                != specification["config_hash"]
                or not Path(previous_evaluations[name].get("result_path", "")).is_file()
            )
        ]
        if not save_models and not pending:
            print(f"Skipping t={coordinate:g}: all selected evaluations already exist", flush=True)
            continue

        temporary_directory = None
        if save_models:
            model_path = run_directory / point_name
            metadata_path = model_path / "bipolar_merge.json"
        else:
            # Existing vLLM evaluators require a model_path. Keep one temporary
            # checkpoint at a time, then delete it after all selected tests.
            temporary_directory = TemporaryDirectory(
                prefix=".temporary-models-", dir=run_directory
            )
            model_path = Path(temporary_directory.name) / point_name
            metadata_path = run_directory / "points" / f"{point_name}.json"

        coefficients = path_coefficients(coordinate, path_method)
        point = {
            "t": coordinate,
            "high_delta_coefficient": coefficients[0],
            "low_delta_coefficient": coefficients[1],
            "output_path": str(model_path) if save_models else None,
            "model_saved": save_models,
            "evaluations": previous_evaluations,
        }
        try:
            _write_curve_point(
                model, base_parameters, updates["high"], updates["low"],
                coordinate, path_method, chunk_numel,
            )
            print(
                f"Writing {path_method}, t={coordinate:g} to "
                f"{'saved' if save_models else 'temporary'} model {model_path}",
                flush=True,
            )
            model_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(
                model_path, safe_serialization=True, max_shard_size=max_shard_size
            )
            tokenizer.save_pretrained(model_path)
            save_progress(point, metadata_path)

            for name, specification in evaluation_specs.items():
                if name not in pending:
                    print(f"Reusing {name} result for t={coordinate:g}", flush=True)
                    continue
                point["evaluations"][name] = evaluate_curve_point(
                    name=name,
                    specification=specification,
                    model_path=model_path,
                    run_directory=run_directory,
                    run_name=run_name,
                    path_method=path_method,
                    trait=trait,
                    coordinate_label=coordinate_label,
                )
                save_progress(point, metadata_path)
        finally:
            if temporary_directory is not None:
                temporary_directory.cleanup()

    if evaluation_specs:
        report["plots"] = write_evaluation_plots(report, run_directory)
        temporary_summary = summary_path.with_suffix(".json.tmp")
        temporary_summary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_summary.replace(summary_path)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    del parameters, base_parameters, updates, model, tokenizer
    gc.collect()
    return report


def _preflight_path_attributions(config, methods):
    """Find missing contribution directories before starting a long sweep."""
    if not config.get("sparsify", True):
        return
    selection = config.get("selection") or {}
    base_settings = _normalise_selection(selection)
    overrides = config.get("pole_overrides") or {}
    if not isinstance(overrides, dict) or set(overrides) - set(POLES):
        raise ValueError("pole_overrides must be a mapping containing only high/low")
    for pole, override in overrides.items():
        if not isinstance(override, dict):
            raise ValueError(f"pole_overrides.{pole} must be a mapping")
        unknown = set(override) - {"alpha"}
        if unknown:
            raise ValueError(
                f"pole_overrides.{pole} may override only alpha, got {sorted(unknown)}"
            )
        if "alpha" in override:
            _finite_number(override["alpha"], f"pole_overrides.{pole}.alpha")
    source_paths = {
        key: str(Path(config[key]).expanduser().resolve())
        for key in ("base_model_path", "plus_model_path", "star_model_path")
    }
    missing = []
    for method, directories in methods.items():
        roles = POLES if method == "piecewise_linear" else ("common",)
        metadata_by_pole = {}
        full_scope = (
            base_settings["k_percent"] == 100
            and (
                method == "piecewise_linear"
                or bool(base_settings["layers"] or base_settings["modules"])
            )
        )
        needs_contributions = (
            base_settings["ranking_method"] == "contribution"
            and base_settings["k_percent"] > 0
            and not full_scope
        )
        if not needs_contributions:
            continue
        for role in roles:
            directory = directories.get(role)
            if not directory or not Path(directory).expanduser().is_dir():
                missing.append(f"{method}.{role}: {directory or '<unset>'}")
            else:
                try:
                    discover_contribution_files(directory)
                    metadata = _read_attribution_metadata(
                        directory, method, role, source_paths, config["trait"]
                    )
                    if method == "piecewise_linear":
                        metadata_by_pole[role] = metadata
                except (ValueError, FileNotFoundError, NotADirectoryError) as error:
                    missing.append(f"{method}.{role}: {error}")
        if method == "piecewise_linear" and len(metadata_by_pole) == len(POLES):
            try:
                _validate_piecewise_shared_metadata(metadata_by_pole)
            except ValueError as error:
                missing.append(f"{method}: {error}")
    if missing:
        raise FileNotFoundError(
            "Missing bipolar path attribution; compute it before the merge sweep:\n"
            + "\n".join(missing)
        )


def _comparison_settings_tag(reports):
    """Name a comparison using only effective K and rescale settings."""
    percentages = set()
    rescale_flags = set()
    for report in reports:
        if not report["sparsify"]:
            percentages.add(100.0)
            rescale_flags.add(False)
            continue
        selected = report["selection"]
        percentages.add(float(selected["k_percent"]))
        rescale_flags.add(bool(selected["use_weight_rescale"]))
    k_tag = "-".join(f"{value:g}" for value in sorted(percentages))
    rescale_tag = "-".join(str(int(value)) for value in sorted(rescale_flags))
    return f"k{k_tag}-rescale{rescale_tag}"


def _comparison_recipe(reports):
    """Retain the full effective recipe for provenance and collision checks."""
    paths = [
        {
            "path_method": report["path_method"],
            "source_models": report["source_models"],
            "trait": report["trait"],
            "sparsify": report["sparsify"],
            "selection": report["selection"],
            "endpoint_alphas": report["endpoint_alphas"],
            "contribution_dirs": report["contribution_dirs"],
            "merge_dtype": report["merge_dtype"],
            "requested_t_values": report["requested_t_values"],
            "evaluation_protocols": report["evaluation_protocols"],
        }
        for report in reports
    ]
    paths.sort(key=lambda item: item["path_method"])
    return {"paths": paths}


def _comparison_directory(parent, label, recipe, config_text):
    """Reuse an identical run or add run2/run3 when its short label collides."""
    serial = 1
    while True:
        name = label if serial == 1 else f"{label}-run{serial}"
        candidate = parent / name
        if not candidate.exists():
            return candidate
        manifest_path = candidate / "comparison_manifest.json"
        config_path = candidate / "bipolar_merge_config.yaml"
        if manifest_path.is_file() and config_path.is_file():
            try:
                previous = json.loads(manifest_path.read_text(encoding="utf-8"))
                same_recipe = previous.get("comparison_recipe") == recipe
                same_config = config_path.read_text(encoding="utf-8") == config_text
                if same_recipe and same_config:
                    return candidate
            except (OSError, ValueError):
                pass
        serial += 1


def main():
    config = parse_args_yaml("Bipolar CPVM merge: piecewise, endpoint-linear, or quadratic")
    complete_config = dict(config)
    methods = config.pop("path_methods", None)
    if methods is None:
        run_bipolar_merge(**config)
        return
    if "path_method" in config or "contribution_dirs" in config:
        raise ValueError("Use path_methods or path_method/contribution_dirs, not both")
    if not isinstance(methods, dict) or not methods:
        raise ValueError("path_methods must map path names to contribution directories")
    if len(methods) != len(set(methods)):
        raise ValueError("path_methods contains duplicate paths")
    for method, directories in methods.items():
        validate_path_method(method)
        if not isinstance(directories, dict):
            raise ValueError(f"path_methods.{method} must be a directory mapping")
    _preflight_path_attributions(config, methods)

    reports = []
    for method, directories in methods.items():
        print(f"Starting bipolar path: {method}", flush=True)
        reports.append(run_bipolar_merge(
            **config, path_method=method, contribution_dirs=directories
        ))
    if config.get("evaluations"):
        from CPVM.code.utils.bipolar_merge_plots import (
            write_pareto_tradeoff_plot,
            write_path_comparison_plots,
        )

        comparison_recipe = _comparison_recipe(reports)
        config_text = yaml.safe_dump(
            complete_config, allow_unicode=True, sort_keys=False
        )
        comparison_parent = (
            Path(config["output_dir"]).expanduser().resolve()
            / "comparison" / _safe_name(config.get("experiment_name") or config["trait"])
        )
        comparison_root = _comparison_directory(
            comparison_parent,
            _comparison_settings_tag(reports),
            comparison_recipe,
            config_text,
        )
        comparison_root.mkdir(parents=True, exist_ok=True)
        config_snapshot = comparison_root / "bipolar_merge_config.yaml"
        config_snapshot.write_text(config_text, encoding="utf-8")
        comparison_name = comparison_root.name
        manifest_path = comparison_root / "comparison_manifest.json"
        if not manifest_path.exists():
            manifest_path.write_text(
                json.dumps(
                    {"comparison_name": comparison_name,
                     "comparison_recipe": comparison_recipe,
                     "config_snapshot": str(config_snapshot),
                     "status": "plotting"},
                    ensure_ascii=False, indent=2,
                ),
                encoding="utf-8",
            )
        comparison = write_path_comparison_plots(reports, comparison_root)
        pareto = write_pareto_tradeoff_plot(reports, comparison_root)
        manifest = {
            "comparison_name": comparison_name,
            "comparison_directory": str(comparison_root),
            "config_snapshot": str(config_snapshot),
            "comparison_recipe": comparison_recipe,
            "path_runs": [
                {
                    "path_method": report["path_method"],
                    "run_name": report["run_name"],
                    "run_directory": report["run_directory"],
                    "merge_summary": str(Path(report["run_directory"]) / "merge_summary.json"),
                    "plots": report.get("plots", {}),
                }
                for report in reports
            ],
            "path_comparison": comparison,
            "trait_mmlu_pro_pareto": pareto,
        }
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_manifest.replace(manifest_path)
        print(
            json.dumps(
                {"comparison_manifest": str(manifest_path),
                 "path_comparison": comparison,
                 "trait_mmlu_pro_pareto": pareto},
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

