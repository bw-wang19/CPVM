"""Compose independently selected bipolar trait updates around one common base.

For each trait r, reuse bipolar_merge's selection and attribution checks.
Each path selects one coordinate mask for both endpoint-base deltas. Apply
that trait's path coefficients, then combine:

    theta_final = theta_base + sum_r weight_r * (
        c_high(t_r) * delta_r_high + c_low(t_r) * delta_r_low
    )

Trait weights are explicit and are not normalized. Overlapping coordinates
add algebraically; there is no second Top-k selection or conflict filter.
All selectors see the original, unmodified base. Selected updates are consumed
into an FP32 accumulator, and the live model is written only after every trait
has been processed. No intermediate trait checkpoints are saved.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from CPVM.code.bipolar_merge import (
    POLES,
    _finite_number,
    _normalise_selection,
    _prepare_common_updates,
    _prepare_piecewise_shared_updates,
    _safe_name,
)
from CPVM.code.topk_merge import chunk_indices, resolve_dtype
from CPVM.code.utils.arguments import parse_args_yaml
from CPVM.code.utils.bipolar_path import path_coefficients, validate_path_method


TRAIT_FIELDS = {
    "trait", "weight", "path_method", "t",
    "plus_model_path", "star_model_path",
    "selection", "pole_overrides", "contribution_dirs", "sparsify",
}
REQUIRED_TRAIT_FIELDS = {
    "trait", "weight", "path_method", "t",
    "plus_model_path", "star_model_path",
}


def _model_path(value, label: str) -> str:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{label} must be a nonempty model path")
    return str(Path(value).expanduser().resolve())


def _pole_mapping(value, label: str) -> dict:
    mapping = {} if value is None else value
    if not isinstance(mapping, dict) or set(mapping) - set(POLES):
        raise ValueError(f"{label} must be a mapping containing only high/low")
    return mapping


def _normalise_traits(traits, common_selection: dict, base_model_path: str) -> list[dict]:
    """Resolve global -> trait -> pole settings before reading any weights."""

    if not isinstance(traits, list) or not traits:
        raise ValueError("traits must be a nonempty list of trait configurations")
    normalised = []
    seen = set()
    for index, entry in enumerate(traits):
        label = f"traits[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{label} must be a mapping")
        unknown = set(entry) - TRAIT_FIELDS
        missing = REQUIRED_TRAIT_FIELDS - set(entry)
        if unknown:
            raise ValueError(f"{label} has unknown options: {sorted(unknown)}")
        if missing:
            raise ValueError(f"{label} requires explicit options: {sorted(missing)}")
        trait = entry["trait"]
        if not isinstance(trait, str) or not trait.strip():
            raise ValueError(f"{label}.trait must be a nonempty string")
        trait = trait.strip()
        if trait.casefold() in seen:
            raise ValueError(f"Configure each trait once; duplicate trait: {trait}")
        seen.add(trait.casefold())

        weight = _finite_number(entry["weight"], f"{label}.weight")
        method = validate_path_method(entry["path_method"])
        coordinate = _finite_number(entry["t"], f"{label}.t")
        coefficients = path_coefficients(coordinate, method)
        sparsify = entry.get("sparsify", True)
        if type(sparsify) is not bool:
            raise ValueError(f"{label}.sparsify must be a boolean")
        trait_selection = entry.get("selection")
        if trait_selection is None:
            trait_selection = {}
        if not isinstance(trait_selection, dict):
            raise ValueError(f"{label}.selection must be a mapping")
        settings = _normalise_selection({**common_selection, **trait_selection})
        overrides = _pole_mapping(entry.get("pole_overrides"), f"{label}.pole_overrides")
        raw_directories = entry.get("contribution_dirs") or {}
        if not isinstance(raw_directories, dict):
            raise ValueError(f"{label}.contribution_dirs must be a mapping")
        is_piecewise = method == "piecewise_linear"
        endpoint_alphas = None
        directories = {}
        if is_piecewise:
            if set(raw_directories) - set(POLES):
                raise ValueError(
                    f"{label}.contribution_dirs accepts only high/low for piecewise_linear"
                )
            endpoint_alphas = {}
            for pole in POLES:
                override = overrides.get(pole, {})
                if not isinstance(override, dict):
                    raise ValueError(f"{label}.pole_overrides.{pole} must be a mapping")
                unknown = set(override) - {"alpha"}
                if unknown:
                    raise ValueError(
                        f"{method} uses one common selection; {label}.pole_overrides.{pole} "
                        f"may override only alpha, got {sorted(unknown)}"
                    )
                endpoint_alphas[pole] = _finite_number(
                    override.get("alpha", settings["alpha"]),
                    f"{label}.pole_overrides.{pole}.alpha",
                )
                directory = raw_directories.get(pole)
                if directory is not None:
                    directory = _model_path(directory, f"{label}.contribution_dirs.{pole}")
                directories[pole] = directory
            resolved_selection = settings
            selection_kind = "common_piecewise_path"
        else:
            if set(raw_directories) - {"common"}:
                raise ValueError(
                    f"{label}.contribution_dirs accepts only common for {method}"
                )
            endpoint_alphas = {}
            for pole in POLES:
                override = overrides.get(pole, {})
                if not isinstance(override, dict):
                    raise ValueError(f"{label}.pole_overrides.{pole} must be a mapping")
                unknown = set(override) - {"alpha"}
                if unknown:
                    raise ValueError(
                        f"{method} uses one common selection; {label}.pole_overrides.{pole} "
                        f"may override only alpha, got {sorted(unknown)}"
                    )
                endpoint_alphas[pole] = _finite_number(
                    override.get("alpha", settings["alpha"]),
                    f"{label}.pole_overrides.{pole}.alpha",
                )
            directory = raw_directories.get("common")
            if directory is not None:
                directory = _model_path(directory, f"{label}.contribution_dirs.common")
            directories["common"] = directory
            resolved_selection = settings
            selection_kind = "common_full_path"

        source_paths = {
            "base_model_path": base_model_path,
            "plus_model_path": _model_path(entry["plus_model_path"], f"{label}.plus_model_path"),
            "star_model_path": _model_path(entry["star_model_path"], f"{label}.star_model_path"),
        }
        normalised.append({
            "trait": trait,
            "weight": weight,
            "path_method": method,
            "t": coordinate,
            "sparsify": sparsify,
            "source_models": source_paths,
            "selection": resolved_selection,
            "selection_kind": selection_kind,
            "endpoint_alphas": endpoint_alphas,
            "contribution_dirs": directories,
            "path_coefficients": dict(zip(POLES, coefficients)),
        })
    return normalised


@torch.no_grad()
def _consume_pole_update(accumulator: dict, updates: dict, coefficient: float) -> None:
    """Consume one already selected pole update; retain no per-trait copies."""

    # Pop tensors as they are consumed. An absent accumulator tensor can take
    # ownership of the update buffer instead of allocating another full copy.
    for name in list(updates):
        update = updates.pop(name)
        if name in accumulator:
            accumulator[name].add_(update, alpha=coefficient)
        else:
            accumulator[name] = update.mul_(coefficient)


@torch.no_grad()
def _apply_accumulated_delta(model, accumulator: dict, chunk_numel: int) -> None:
    """Add the combined FP32 delta to the base once, then cast for saving."""

    for name, parameter in tqdm(
        model.named_parameters(), desc="Writing multi-trait model", unit="tensor"
    ):
        update = accumulator.pop(name, None)
        if update is None:
            continue
        for index in chunk_indices(tuple(parameter.shape), chunk_numel):
            destination = parameter if index is None else parameter[index]
            delta = update if index is None else update[index]
            merged = destination.detach().float().add(delta)
            destination.copy_(merged.to(dtype=destination.dtype))
    if accumulator:
        raise RuntimeError("Accumulated updates contain parameters absent from the base")


def run_multi_trait_merge(
    base_model_path: str,
    traits: list[dict],
    output_dir: str,
    experiment_name: str = "multi-trait",
    selection: dict | None = None,
    merge_dtype: str = "auto",
    chunk_numel: int = 5_000_000,
    max_shard_size: str = "5GB",
) -> dict:
    """Produce one model from independently weighted bipolar trait deltas.

    Each traits entry explicitly supplies trait, weight, path_method, t,
    plus_model_path and star_model_path. Selection options follow the same
    names and semantics as bipolar_merge, with optional shared defaults here.
    t is a single coordinate per trait; trait weights are never normalized.
    """

    if type(chunk_numel) is not int or chunk_numel <= 0:
        raise ValueError("chunk_numel must be a positive integer")
    base_path = _model_path(base_model_path, "base_model_path")
    common_selection = _normalise_selection({} if selection is None else selection)
    trait_configs = _normalise_traits(traits, common_selection, base_path)
    recipe = {
        "base_model_path": base_path,
        "traits": trait_configs,
        "weight_normalization": "none",
        "overlap_rule": "sum weighted deltas at each coordinate",
        "selection_reference": (
            "unchanged common base; each trait selects one mask from its dense path"
        ),
        "merge_formula": "base + sum_trait weight * (c_high * d_high + c_low * d_low)",
        "merge_dtype": merge_dtype,
    }
    identity = hashlib.sha256(
        json.dumps(recipe, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    run_name = f"{_safe_name(experiment_name)}-{identity}"
    output_path = Path(output_dir).expanduser().resolve() / run_name
    source_paths = {base_path}
    for config in trait_configs:
        source_paths.update(config["source_models"].values())
    for source in map(Path, source_paths):
        if output_path == source or source in output_path.parents or output_path in source.parents:
            raise ValueError("Merge output must not contain, overwrite, or be inside a source model")

    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        trust_remote_code=True,
        dtype=resolve_dtype(merge_dtype),
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.requires_grad_(False)
    # These parameters stay unchanged until ALL selections have completed.
    parameters = dict(model.named_parameters())
    accumulator = {}
    trait_reports = []
    for config in trait_configs:
        trait = config["trait"]
        weight = config["weight"]
        method = config["path_method"]
        print(
            f"Trait {trait}: weight={weight:g}, path={method}, "
            f"t={config['t']:g}", flush=True,
        )
        pole_reports = {}
        common_selection_report = None
        coefficients = config["path_coefficients"]
        endpoint_alphas = config["endpoint_alphas"]
        active_poles = [
            pole for pole in POLES
            if coefficients[pole] != 0
            and (not config["sparsify"] or endpoint_alphas[pole] != 0)
        ]
        reason = None
        if weight == 0:
            reason = "zero_trait_weight"
        elif not active_poles:
            reason = "zero_effective_path_update"
        if reason is not None:
            common_selection_report = {
                "status": "skipped",
                "reason": reason,
                "selection_performed": False,
            }
            print(f"  {trait}/common: skipped ({reason})", flush=True)
            for pole in POLES:
                pole_reports[pole] = {
                    "status": "skipped",
                    "reason": reason,
                    "path_coefficient": coefficients[pole],
                    "trait_weight": weight,
                    "weighted_path_coefficient": weight * coefficients[pole],
                }
        else:
            if method == "piecewise_linear":
                updates_by_pole, selection_report = _prepare_piecewise_shared_updates(
                    parameters=parameters,
                    plus_model_path=config["source_models"]["plus_model_path"],
                    star_model_path=config["source_models"]["star_model_path"],
                    settings=config["selection"],
                    endpoint_alphas=endpoint_alphas,
                    contribution_dirs=config["contribution_dirs"],
                    source_paths=config["source_models"],
                    trait=trait,
                    sparsify=config["sparsify"],
                    chunk_numel=chunk_numel,
                )
            else:
                updates_by_pole, selection_report = _prepare_common_updates(
                    parameters=parameters,
                    plus_model_path=config["source_models"]["plus_model_path"],
                    star_model_path=config["source_models"]["star_model_path"],
                    settings=config["selection"],
                    endpoint_alphas=endpoint_alphas,
                    contribution_dir=config["contribution_dirs"]["common"],
                    path_method=method,
                    source_paths=config["source_models"],
                    trait=trait,
                    sparsify=config["sparsify"],
                    chunk_numel=chunk_numel,
                )
            common_selection_report = {
                "status": "prepared",
                "selection_performed": True,
                "path_selection": selection_report,
            }
            for pole in POLES:
                path_coefficient = coefficients[pole]
                weighted_coefficient = weight * path_coefficient
                update = updates_by_pole[pole]
                pole_reason = None
                if path_coefficient == 0:
                    pole_reason = "zero_path_coefficient"
                elif config["sparsify"] and endpoint_alphas[pole] == 0:
                    pole_reason = "zero_endpoint_alpha"
                if pole_reason is None:
                    _consume_pole_update(accumulator, update, weighted_coefficient)
                    status = "prepared"
                else:
                    status = "skipped"
                pole_reports[pole] = {
                    "status": status,
                    "reason": pole_reason,
                    "path_coefficient": path_coefficient,
                    "trait_weight": weight,
                    "weighted_path_coefficient": weighted_coefficient,
                    "shared_common_mask": True,
                }
            del updates_by_pole
            gc.collect()
        trait_reports.append({
            "trait": trait,
            "weight": weight,
            "effective_weight": weight,
            "path_method": method,
            "t": config["t"],
            "sparsify": config["sparsify"],
            "selection_kind": config["selection_kind"],
            "common_selection": common_selection_report,
            "poles": pole_reports,
        })

    _apply_accumulated_delta(model, accumulator, chunk_numel)
    tokenizer = AutoTokenizer.from_pretrained(
        base_path, trust_remote_code=True, use_fast=False
    )
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving multi-trait model to {output_path}", flush=True)
    model.save_pretrained(
        output_path, safe_serialization=True, max_shard_size=max_shard_size
    )
    tokenizer.save_pretrained(output_path)
    report = {
        **recipe,
        "run_name": run_name,
        "output_path": str(output_path),
        "trait_count": len(trait_configs),
        "trait_reports": trait_reports,
        "attribution_scope": (
            "independent dense bipolar path attribution for each trait; "
            "sparse weighted composition is not a joint multi-trait attribution"
        ),
        "selection_implementation": {
            "piecewise_linear": "CPVM.code.bipolar_merge._prepare_piecewise_shared_updates",
            "continuous": "CPVM.code.bipolar_merge._prepare_common_updates",
        },
        "accumulation_dtype": "float32",
        "execution_device": "cpu",
    }
    metadata_path = output_path / "multi_trait_merge.json"
    temporary_path = metadata_path.with_name(f".{metadata_path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary_path.replace(metadata_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Saved merge metadata to {metadata_path}", flush=True)
    del parameters, accumulator, model, tokenizer
    gc.collect()
    return report


def main() -> None:
    config = parse_args_yaml("Multi-trait composition of bipolar CPVM updates")
    run_multi_trait_merge(**config)


if __name__ == "__main__":
    main()

