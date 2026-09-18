"""Memory-safe parameter-space analysis for CPVM bipolar asymmetry.

The implementation reads safetensors directly from disk, one chunk at a time.
It therefore does not need three multi-billion-parameter models resident in RAM
or VRAM at the same time.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping
import warnings

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open


RAW_FIELDS = (
    "dot",
    "high_sq",
    "low_sq",
    "numel",
    "high_active",
    "low_active",
    "overlap",
    "opposite",
    "positive_dot",
    "abs_dot",
)

_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _new_stats() -> dict[str, float]:
    return {field: 0.0 for field in RAW_FIELDS}


def _add_stats(target: dict[str, float], source: Mapping[str, float]) -> None:
    for field in RAW_FIELDS:
        target[field] += float(source[field])


class _SafeTensorDirectory:
    """Open every safetensors shard in a Transformers model directory."""

    def __init__(self, model_dir: str | Path, stack: ExitStack):
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {self.model_dir}")

        self._handles = {}
        index_path = self.model_dir / "model.safetensors.index.json"

        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as file:
                self.weight_map = json.load(file)["weight_map"]
            filenames = sorted(set(self.weight_map.values()))
            for filename in filenames:
                tensor_path = self.model_dir / filename
                self._handles[filename] = stack.enter_context(
                    safe_open(str(tensor_path), framework="pt", device="cpu")
                )
        else:
            tensor_files = sorted(self.model_dir.glob("*.safetensors"))
            if len(tensor_files) != 1:
                raise FileNotFoundError(
                    f"Expected one model.safetensors file or an index in {self.model_dir}; "
                    f"found {len(tensor_files)} safetensors files."
                )
            tensor_path = tensor_files[0]
            handle = stack.enter_context(
                safe_open(str(tensor_path), framework="pt", device="cpu")
            )
            self._handles[tensor_path.name] = handle
            self.weight_map = {name: tensor_path.name for name in handle.keys()}

    @property
    def keys(self) -> set[str]:
        return set(self.weight_map)

    def _handle(self, name: str):
        return self._handles[self.weight_map[name]]

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._handle(name).get_slice(name).get_shape())

    def read(self, name: str, index):
        handle = self._handle(name)
        if index is None:
            return handle.get_tensor(name)
        return handle.get_slice(name)[index]


def _chunk_indices(shape: tuple[int, ...], max_numel: int) -> Iterable[object]:
    if max_numel <= 0:
        raise ValueError("max_numel must be positive")
    if len(shape) == 0:
        yield None
        return

    tail_numel = math.prod(shape[1:]) if len(shape) > 1 else 1
    rows_per_chunk = max(1, max_numel // max(tail_numel, 1))
    trailing = (slice(None),) * (len(shape) - 1)
    for start in range(0, shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, shape[0])
        yield (slice(start, stop),) + trailing


def _tensors_are_equal(
    reader: _SafeTensorDirectory,
    left_name: str,
    right_name: str,
    max_numel: int,
) -> bool:
    left_shape = reader.shape(left_name)
    if left_shape != reader.shape(right_name):
        return False
    for index in _chunk_indices(left_shape, max_numel):
        if not torch.equal(
            reader.read(left_name, index),
            reader.read(right_name, index),
        ):
            return False
    return True


def _layer_group(name: str) -> str:
    match = _LAYER_PATTERN.search(name)
    if match:
        return f"layer_{int(match.group(1)):02d}"
    if "embed_tokens" in name:
        return "embedding"
    if name.startswith("lm_head"):
        return "lm_head"
    if name.endswith("model.norm.weight") or name == "model.norm.weight":
        return "final_norm"
    return name.rsplit(".", 1)[0]


def _module_group(name: str) -> str:
    if "embed_tokens" in name:
        return "embedding"
    if ".self_attn." in name:
        module = name.split(".self_attn.", 1)[1].split(".", 1)[0]
        return f"attention.{module}"
    if ".mlp." in name:
        module = name.split(".mlp.", 1)[1].split(".", 1)[0]
        return f"mlp.{module}"
    if "norm" in name:
        return "normalization"
    if name.startswith("lm_head"):
        return "lm_head"
    return name.rsplit(".", 1)[0]


def _architecture_signature(model_dir: str | Path) -> dict[str, object]:
    config_path = Path(model_dir) / "config.json"
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    fields = (
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "vocab_size",
        "tie_word_embeddings",
    )
    return {field: config.get(field) for field in fields}


def derive_metrics(stats: Mapping[str, float]) -> dict[str, float]:
    """Turn additive sufficient statistics into interpretable asymmetry metrics."""

    dot = float(stats["dot"])
    high_sq = float(stats["high_sq"])
    low_sq = float(stats["low_sq"])
    if high_sq < 0.0 or low_sq < 0.0:
        raise ValueError("Squared update norms cannot be negative.")

    high_norm = math.sqrt(high_sq)
    low_norm = math.sqrt(low_sq)
    overlap = float(stats["overlap"])
    union = float(stats["high_active"] + stats["low_active"] - overlap)
    abs_dot = float(stats["abs_dot"])
    support_metrics = {
        "support_jaccard": overlap / union if union else float("nan"),
        "high_only_fraction": (
            (float(stats["high_active"]) - overlap) / union if union else float("nan")
        ),
        "low_only_fraction": (
            (float(stats["low_active"]) - overlap) / union if union else float("nan")
        ),
        "opposite_sign_rate": (
            float(stats["opposite"]) / overlap if overlap else float("nan")
        ),
        "positive_dot_mass_ratio": (
            float(stats["positive_dot"]) / abs_dot if abs_dot else float("nan")
        ),
    }
    if high_sq == 0.0 or low_sq == 0.0:
        undefined = float("nan")
        return {
            "cosine": undefined,
            "opposition_gap": undefined,
            "anti_angle_deg": undefined,
            "direction_asymmetry": undefined,
            "optimal_anti_scale": undefined,
            "optimal_anti_residual": undefined,
            "high_low_norm_ratio": high_norm / low_norm if low_norm else undefined,
            "midpoint_error": undefined,
            **support_metrics,
            "high_update_norm": high_norm,
            "low_update_norm": low_norm,
        }

    cosine = float(np.clip(dot / (high_norm * low_norm), -1.0, 1.0))

    # Find min_{scale >= 0} ||delta_high + scale * delta_low|| / ||delta_high||.
    optimal_scale = max(0.0, -dot / low_sq)
    residual_sq = max(
        0.0,
        high_sq + 2.0 * optimal_scale * dot + optimal_scale**2 * low_sq,
    )
    midpoint_sq = max(0.0, high_sq + low_sq + 2.0 * dot)

    return {
        "cosine": cosine,
        "opposition_gap": 1.0 + cosine,
        "anti_angle_deg": math.degrees(math.acos(-cosine)),
        "direction_asymmetry": math.sqrt(max(0.0, (1.0 + cosine) / 2.0)),
        "optimal_anti_scale": optimal_scale,
        "optimal_anti_residual": math.sqrt(residual_sq / high_sq),
        "high_low_norm_ratio": high_norm / low_norm,
        "midpoint_error": math.sqrt(midpoint_sq) / (high_norm + low_norm),
        **support_metrics,
        "high_update_norm": high_norm,
        "low_update_norm": low_norm,
    }


def _chunk_stats(
    base_chunk: torch.Tensor,
    high_chunk: torch.Tensor,
    low_chunk: torch.Tensor,
    active_epsilon: float,
) -> dict[str, float]:
    base = base_chunk.to(dtype=torch.float32)
    delta_high = high_chunk.to(dtype=torch.float32, copy=True).sub_(base)
    delta_low = low_chunk.to(dtype=torch.float32, copy=True).sub_(base)

    product = delta_high * delta_low
    high_active = delta_high.abs().gt(active_epsilon)
    low_active = delta_low.abs().gt(active_epsilon)
    overlap = high_active & low_active
    opposite = overlap & (
        ((delta_high > 0) & (delta_low < 0))
        | ((delta_high < 0) & (delta_low > 0))
    )
    support_product = product[overlap]

    return {
        "dot": product.sum(dtype=torch.float64).item(),
        "high_sq": delta_high.square().sum(dtype=torch.float64).item(),
        "low_sq": delta_low.square().sum(dtype=torch.float64).item(),
        "numel": float(delta_high.numel()),
        "high_active": float(high_active.sum(dtype=torch.int64).item()),
        "low_active": float(low_active.sum(dtype=torch.int64).item()),
        "overlap": float(overlap.sum(dtype=torch.int64).item()),
        "opposite": float(opposite.sum(dtype=torch.int64).item()),
        "positive_dot": support_product.clamp_min(0).sum(dtype=torch.float64).item(),
        "abs_dot": support_product.abs().sum(dtype=torch.float64).item(),
    }


def analyze_bipolar_asymmetry(
    base_model_path: str | Path,
    high_model_path: str | Path,
    low_model_path: str | Path,
    *,
    chunk_numel: int = 2_000_000,
    active_epsilon: float = 0.0,
    show_progress: bool = True,
) -> dict[str, object]:
    """Compare high/base/low task vectors without loading model objects.

    ``active_epsilon`` only affects support/sign metrics. Direction, norm and
    residual metrics always use every parameter value.
    """

    if active_epsilon < 0:
        raise ValueError("active_epsilon must be non-negative")

    paths = {
        "base": Path(base_model_path),
        "high": Path(high_model_path),
        "low": Path(low_model_path),
    }
    signatures = {name: _architecture_signature(path) for name, path in paths.items()}
    if not (signatures["base"] == signatures["high"] == signatures["low"]):
        raise ValueError(f"Model architectures do not match: {signatures}")

    global_stats = _new_stats()
    layer_stats: defaultdict[str, dict[str, float]] = defaultdict(_new_stats)
    module_stats: defaultdict[str, dict[str, float]] = defaultdict(_new_stats)
    tensor_stats: dict[str, dict[str, float]] = {}

    with ExitStack() as stack:
        readers = {
            name: _SafeTensorDirectory(path, stack) for name, path in paths.items()
        }
        common_keys = set.intersection(*(reader.keys for reader in readers.values()))
        all_keys = set.union(*(reader.keys for reader in readers.values()))
        tied_embeddings = bool(signatures["base"]["tie_word_embeddings"])
        if "lm_head.weight" in all_keys:
            if not tied_embeddings:
                raise ValueError(
                    "lm_head.weight is not shared by every checkpoint and the model "
                    "configuration does not declare tied word embeddings."
                )
            for reader_name, reader in readers.items():
                if "lm_head.weight" not in reader.keys:
                    continue
                if "model.embed_tokens.weight" not in reader.keys:
                    raise ValueError(
                        f"{reader_name} stores lm_head.weight without embed_tokens.weight."
                    )
                if not _tensors_are_equal(
                    reader,
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    chunk_numel,
                ):
                    raise ValueError(
                        f"{reader_name} declares tied word embeddings, but its stored "
                        "lm_head.weight differs from model.embed_tokens.weight."
                    )
            # Both names represent one logical parameter; count it only once.
            common_keys.discard("lm_head.weight")
        ignored_keys = all_keys - common_keys
        allowed_ignored = {"lm_head.weight"} if tied_embeddings else set()
        unexpected = ignored_keys - allowed_ignored
        if unexpected:
            raise ValueError(
                "The checkpoints have non-tied weights that are not shared by all three "
                f"models: {sorted(unexpected)}"
            )
        if not common_keys:
            raise ValueError("The three checkpoints have no common parameter keys.")

        names: Iterable[str] = sorted(common_keys)
        if show_progress:
            try:
                from tqdm.auto import tqdm

                names = tqdm(names, desc="Comparing parameter tensors")
            except ImportError:
                pass

        for name in names:
            shapes = {key: reader.shape(name) for key, reader in readers.items()}
            if len(set(shapes.values())) != 1:
                raise ValueError(f"Shape mismatch for {name}: {shapes}")

            current = _new_stats()
            for index in _chunk_indices(shapes["base"], chunk_numel):
                part = _chunk_stats(
                    readers["base"].read(name, index),
                    readers["high"].read(name, index),
                    readers["low"].read(name, index),
                    active_epsilon,
                )
                _add_stats(current, part)

            tensor_stats[name] = current
            _add_stats(global_stats, current)
            _add_stats(layer_stats[_layer_group(name)], current)
            _add_stats(module_stats[_module_group(name)], current)

    if global_stats["high_sq"] <= 0.0 or global_stats["low_sq"] <= 0.0:
        raise ValueError("At least one checkpoint has a zero update from the base model.")

    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "architecture": signatures["base"],
        "active_epsilon": active_epsilon,
        "chunk_numel": chunk_numel,
        "compared_key_count": len(common_keys),
        "ignored_keys": sorted(ignored_keys),
        "global_raw": global_stats,
        "global_metrics": derive_metrics(global_stats),
        "layer_raw": dict(layer_stats),
        "module_raw": dict(module_stats),
        "tensor_raw": tensor_stats,
    }


def _group_sort_key(name: str):
    if name == "embedding":
        return (0, 0)
    match = re.fullmatch(r"layer_(\d+)", name)
    if match:
        return (1, int(match.group(1)))
    if name == "final_norm":
        return (2, 0)
    if name == "lm_head":
        return (3, 0)
    return (4, name)


def _stats_frame(stats_by_group: Mapping[str, Mapping[str, float]], label: str) -> pd.DataFrame:
    rows = []
    for name, raw in stats_by_group.items():
        rows.append({label: name, **raw, **derive_metrics(raw)})
    frame = pd.DataFrame(rows)
    return frame.sort_values(label, key=lambda col: col.map(_group_sort_key)).reset_index(
        drop=True
    )


def build_result_tables(result: Mapping[str, object]):
    """Create global, layer, module and tensor DataFrames for notebook display."""

    metrics = result["global_metrics"]
    summary = pd.DataFrame(
        [
            ("cosine(Δhigh, Δlow)", metrics["cosine"], "-1"),
            ("1 + cosine", metrics["opposition_gap"], "0"),
            ("与完全反向的夹角（度）", metrics["anti_angle_deg"], "0"),
            ("尺度无关方向不对称", metrics["direction_asymmetry"], "0"),
            ("最佳正反向缩放系数", metrics["optimal_anti_scale"], "> 0"),
            ("最佳缩放后残差", metrics["optimal_anti_residual"], "0"),
            ("||Δhigh|| / ||Δlow||", metrics["high_low_norm_ratio"], "方向反向不要求 1；中点对称要求 1"),
            ("base 中点误差", metrics["midpoint_error"], "0"),
            ("非零支持 Jaccard", metrics["support_jaccard"], "1"),
            ("共同支持中的异号率", metrics["opposite_sign_rate"], "1"),
            ("同向点积质量占比", metrics["positive_dot_mass_ratio"], "0"),
        ],
        columns=["指标", "观测值", "完全反向/对称时"],
    )
    layer = _stats_frame(result["layer_raw"], "group")
    module = _stats_frame(result["module_raw"], "group")
    tensor = _stats_frame(result["tensor_raw"], "parameter")
    return summary, layer, module, tensor


def bootstrap_transformer_layers(
    layer_stats: Mapping[str, Mapping[str, float]],
    *,
    n_bootstrap: int = 5_000,
    seed: int = 20260824,
) -> pd.DataFrame:
    """Bootstrap blocks while keeping non-block parameters fixed.

    The resulting point estimate and interval both target the global metric.
    This remains a structural/descriptive interval, not a causal or seed-level
    population inference.
    """

    ordered_groups = sorted(
        layer_stats.items(), key=lambda item: _group_sort_key(item[0])
    )
    blocks = [raw for name, raw in ordered_groups if re.fullmatch(r"layer_\d+", name)]
    fixed = [raw for name, raw in ordered_groups if not re.fullmatch(r"layer_\d+", name)]
    if len(blocks) < 2:
        raise ValueError("Need at least two transformer layers for layer bootstrap.")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")

    raw_matrix = np.asarray(
        [[float(block[field]) for field in RAW_FIELDS] for block in blocks],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(blocks), size=(n_bootstrap, len(blocks)))
    sampled_sums = raw_matrix[indices].sum(axis=1)
    if fixed:
        fixed_vector = np.asarray(
            [sum(float(raw[field]) for raw in fixed) for field in RAW_FIELDS],
            dtype=np.float64,
        )
        sampled_sums += fixed_vector

    metric_names = (
        "cosine",
        "anti_angle_deg",
        "direction_asymmetry",
        "optimal_anti_residual",
        "high_low_norm_ratio",
        "midpoint_error",
        "support_jaccard",
        "opposite_sign_rate",
        "positive_dot_mass_ratio",
    )
    samples = {name: [] for name in metric_names}
    for row in sampled_sums:
        metrics = derive_metrics(dict(zip(RAW_FIELDS, row)))
        for name in metric_names:
            samples[name].append(metrics[name])

    point_raw = _new_stats()
    for _, raw in ordered_groups:
        _add_stats(point_raw, raw)
    point = derive_metrics(point_raw)

    rows = []
    for name in metric_names:
        low, high = np.quantile(np.asarray(samples[name]), [0.025, 0.975])
        rows.append(
            {
                "metric": name,
                "global_estimate": point[name],
                "ci_2.5%": float(low),
                "ci_97.5%": float(high),
            }
        )
    return pd.DataFrame(rows)


def training_progress_table(
    high_model_path: str | Path,
    low_model_path: str | Path,
) -> pd.DataFrame:
    """Read Trainer progress so a mismatched high/low budget is visible."""

    rows = []
    for polarity, model_path in (("high", high_model_path), ("low", low_model_path)):
        state_path = Path(model_path) / "trainer_state.json"
        if not state_path.exists():
            warnings.warn(f"No trainer_state.json found for {polarity}: {state_path}")
            rows.append({"polarity": polarity, "global_step": np.nan, "epoch": np.nan})
            continue
        with state_path.open("r", encoding="utf-8") as file:
            state = json.load(file)
        rows.append(
            {
                "polarity": polarity,
                "global_step": state.get("global_step"),
                "epoch": state.get("epoch"),
                "max_steps": state.get("max_steps"),
                "num_train_epochs": state.get("num_train_epochs"),
                "train_batch_size": state.get("train_batch_size"),
                "total_flos": state.get("total_flos"),
            }
        )
    return pd.DataFrame(rows)
