"""Merge globally top-ranked CPVM parameters into a base model checkpoint.

The main CPVM ranking mode orders raw contributions from largest to smallest
and keeps an exact Top-k budget; the selected range may cross zero.
Absolute-value ranking is an explicit alternative. Additional baselines rank
the magnitude of the SFT change or the magnitude of the SFT endpoint itself.
Random selection keeps an exact k% budget without ranking scores.
Optional layers/modules restrict the candidate scope before any selection.
With an active scope and k=100, all its parameters are merged without ranking.
The selected weights follow

    theta_out = theta_base + alpha * (theta_finetuned - theta_base).

DaRE-style rescaling divides each selected update by the retained fraction.
With random selection this is a fixed-count DARE variant: original DARE uses
independent Bernoulli masks and therefore retains k% only in expectation.

The global threshold is found exactly with four streaming radix passes over
the FP32 contribution file. This avoids materializing or sorting four billion
contribution values in memory.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
import gc
import json
import math
from pathlib import Path
import re
from typing import Iterator

import numpy as np
from safetensors import safe_open
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from CPVM.code.utils.arguments import parse_args_yaml
from CPVM.code.utils.parameter_selection import select_parameters


RADIX_SHIFTS = (24, 16, 8, 0)
CONTRIBUTION_CHANNELS = ("L_plus", "L_star", "L_contrast")
RANKING_METHODS = (
    "contribution",
    "delta_magnitude",
    "finetuned_magnitude",
    "random",
)


@dataclass(frozen=True)
class TopKSelection:
    """Exact Top-k threshold, counts, and deterministic threshold tie budget.

    Merge results report total_numel for the full model and candidate_numel for
    the search scope. Standalone threshold searches count only their inputs.
    """

    total_numel: int
    selected_numel: int
    threshold_bits: int | None
    threshold_score: float | None
    threshold_ties_to_keep: int
    select_all: bool = False
    candidate_numel: int | None = None


def sample_random_indices(
    population_size: int, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample sorted unique indices with O(count) storage (Floyd's algorithm)."""

    if not 0 <= count <= population_size:
        raise ValueError("count must be between 0 and population_size")
    chosen = set()
    for upper in range(population_size - count, population_size):
        candidate = int(rng.integers(upper + 1))
        chosen.add(upper if candidate in chosen else candidate)
    return np.array(sorted(chosen), dtype=np.int64)


class RandomSelectionMask:
    """Stream a uniform fixed-size subset, reproducible across chunk sizes.

    Count a Bernoulli mask in one RNG-only pass, then replay it while uniformly
    adding/removing the surplus or deficit. Conditional on its count, each
    Bernoulli subset is uniform; uniformly correcting its size preserves that
    property. Only chunk-sized masks and the correction indices are stored.
    The expected correction count is O(sqrt(N*q*(1-q))).
    """

    def __init__(
        self,
        total_numel: int,
        k_percent: float,
        seed: int = 0,
        chunk_numel: int = 5_000_000,
    ):
        if not 0 <= k_percent <= 100:
            raise ValueError("k_percent must be between 0 and 100")
        if total_numel < 0 or chunk_numel <= 0:
            raise ValueError("total_numel must be nonnegative and chunk_numel positive")
        selected_numel = math.ceil(total_numel * k_percent / 100)
        self.selection = TopKSelection(
            total_numel, selected_numel, None, None, 0,
            select_all=selected_numel == total_numel,
        )
        self.retain_rate = k_percent / 100
        self.remaining_numel = total_numel
        self.candidate_offset = 0
        self.remove_selected = False
        self.corrections = np.empty(0, dtype=np.int64)
        mask_seed, correction_seed = np.random.SeedSequence(seed).spawn(2)
        self.rng = np.random.default_rng(mask_seed)
        if selected_numel in (0, total_numel):
            return

        initial_count = 0
        for start in tqdm(
            range(0, total_numel, chunk_numel),
            desc="Counting random selection", unit="chunk",
        ):
            size = min(chunk_numel, total_numel - start)
            initial_count += int(np.count_nonzero(
                self.rng.random(size) < self.retain_rate
            ))
        self.remove_selected = initial_count > selected_numel
        candidate_count = (
            initial_count if self.remove_selected else total_numel - initial_count
        )
        self.corrections = sample_random_indices(
            candidate_count, abs(initial_count - selected_numel),
            np.random.default_rng(correction_seed),
        )
        self.rng = np.random.default_rng(mask_seed)

    def mask(self, numel: int) -> np.ndarray:
        """Return the next flattened mask in sorted parameter traversal order."""

        if not 0 <= numel <= self.remaining_numel:
            raise ValueError("Random mask chunk exceeds the remaining parameter count")
        self.remaining_numel -= numel
        if self.selection.selected_numel == 0:
            return np.zeros(numel, dtype=bool)
        if self.selection.select_all:
            return np.ones(numel, dtype=bool)

        mask = self.rng.random(numel) < self.retain_rate
        if self.corrections.size:
            candidates = mask if self.remove_selected else ~mask
            stop = self.candidate_offset + int(np.count_nonzero(candidates))
            left, right = np.searchsorted(
                self.corrections, [self.candidate_offset, stop]
            )
            if right > left:
                local_ranks = self.corrections[left:right] - self.candidate_offset
                indices = np.flatnonzero(candidates)[local_ranks]
                mask[indices] = not self.remove_selected
            self.candidate_offset = stop
        return mask


class ModelTensorReader:
    """Read a full-model safetensors checkpoint, including sharded models."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.stack = ExitStack()

    def __enter__(self):
        index_path = self.model_path / "model.safetensors.index.json"
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as file:
                self.weight_map = json.load(file)["weight_map"]
            filenames = sorted(set(self.weight_map.values()))
        else:
            filenames = ["model.safetensors"]
            with safe_open(
                str(self.model_path / filenames[0]), framework="pt", device="cpu"
            ) as handle:
                self.weight_map = {name: filenames[0] for name in handle.keys()}

        self.handles = {
            filename: self.stack.enter_context(
                safe_open(
                    str(self.model_path / filename), framework="pt", device="cpu"
                )
            )
            for filename in filenames
        }
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stack.close()

    @property
    def keys(self) -> set[str]:
        """Return all tensor names in the checkpoint."""

        return set(self.weight_map)

    def shape(self, name: str) -> tuple[int, ...]:
        """Return one tensor's shape without loading its values."""

        handle = self.handles[self.weight_map[name]]
        return tuple(handle.get_slice(name).get_shape())

    def read(self, name: str, index):
        """Read one tensor or one first-dimension slice from disk."""

        handle = self.handles[self.weight_map[name]]
        if index is None:
            return handle.get_tensor(name)
        return handle.get_slice(name)[index]


def chunk_indices(shape: tuple[int, ...], max_numel: int) -> Iterator[object]:
    """Yield first-dimension slices whose tensors contain at most max_numel values."""

    if not shape:
        yield None
        return

    row_numel = math.prod(shape[1:]) if len(shape) > 1 else 1
    rows_per_chunk = max(1, max_numel // row_numel)
    trailing = (slice(None),) * (len(shape) - 1)
    for start in range(0, shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, shape[0])
        yield (slice(start, stop),) + trailing


def contribution_chunks(handle, name: str, max_numel: int):
    """Yield indexed, flattened FP32 chunks from one contribution tensor."""

    tensor_slice = handle.get_slice(name)
    shape = tuple(tensor_slice.get_shape())
    for index in chunk_indices(shape, max_numel):
        tensor = handle.get_tensor(name) if index is None else tensor_slice[index]
        yield index, tensor.reshape(-1).float()


def discover_contribution_files(
    contribution_dir: str | Path,
) -> dict[str, Path]:
    """Find the unique matched L_plus/L_star pair in one result directory."""

    directory = Path(contribution_dir).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Contribution directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Contribution path is not a directory: {directory}")
    files = {}
    for channel in ("L_plus", "L_star"):
        matches = sorted(
            path for path in directory.glob(f"*-{channel}.safetensors")
            if path.is_file()
        )
        if len(matches) != 1:
            names = ", ".join(path.name for path in matches) or "none"
            raise ValueError(
                f"{directory} must contain exactly one *-{channel}.safetensors; "
                f"found {len(matches)}: {names}"
            )
        files[channel] = matches[0]
    plus_stem = files["L_plus"].name.removesuffix("-L_plus.safetensors")
    star_stem = files["L_star"].name.removesuffix("-L_star.safetensors")
    if plus_stem != star_stem:
        raise ValueError(
            "L_plus and L_star files are not a matched pair: "
            f"{files['L_plus'].name} vs {files['L_star'].name}"
        )
    return files


def resolve_contribution_paths(
    l_plus_contribution_path: str | Path | None,
    l_star_contribution_path: str | Path | None,
    contribution_channel: str,
) -> list[str | Path]:
    """Return the contribution files used by one objective channel."""

    if contribution_channel == "L_plus":
        if not l_plus_contribution_path:
            raise ValueError("L_plus requires l_plus_contribution_path")
        return [l_plus_contribution_path]
    if contribution_channel == "L_star":
        if not l_star_contribution_path:
            raise ValueError("L_star requires l_star_contribution_path")
        return [l_star_contribution_path]
    if contribution_channel == "L_contrast":
        if not l_plus_contribution_path:
            raise ValueError("L_contrast requires l_plus_contribution_path")
        if not l_star_contribution_path:
            raise ValueError("L_contrast requires l_star_contribution_path")
        return [l_plus_contribution_path, l_star_contribution_path]
    raise ValueError(
        f"contribution_channel must be one of {CONTRIBUTION_CHANNELS}"
    )


def combined_contribution_chunks(handles, name: str, max_numel: int):
    """Yield one channel or the FP32 per-chunk sum of aligned channels."""

    tensor_slices = [handle.get_slice(name) for handle in handles]
    shape = tuple(tensor_slices[0].get_shape())
    if any(
        tuple(tensor_slice.get_shape()) != shape
        for tensor_slice in tensor_slices[1:]
    ):
        raise ValueError(f"Contribution shape mismatch between channels: {name}")

    for index in chunk_indices(shape, max_numel):
        combined = None
        for handle, tensor_slice in zip(handles, tensor_slices):
            tensor = handle.get_tensor(name) if index is None else tensor_slice[index]
            values = tensor.reshape(-1).float()
            combined = values.clone() if combined is None else combined.add_(values)
        yield index, combined


def absolute_float_bits(values: torch.Tensor) -> np.ndarray:
    """Return monotonic IEEE uint32 keys for absolute FP32 contributions."""

    magnitudes = values.float().abs().contiguous().numpy()
    return magnitudes.view(np.uint32)


def ordered_float_bits(values: torch.Tensor) -> np.ndarray:
    """Encode FP32 values into uint32 keys that preserve numeric ordering."""

    values = values.float()
    # Canonicalize -0/+0 so zeros share one deterministic tie group.
    values = torch.where(values == 0, torch.zeros_like(values), values)
    bits = values.contiguous().numpy().view(np.uint32)
    sign = np.uint32(0x80000000)
    return np.where(bits & sign, ~bits, bits ^ sign)


def ordered_bits_to_float(bits: int) -> float:
    """Decode one monotonic uint32 key produced by :func:`ordered_float_bits`."""

    value_bits = (
        bits ^ 0x80000000 if bits & 0x80000000 else (~bits & 0xFFFFFFFF)
    )
    return float(np.array([value_bits], dtype=np.uint32).view(np.float32)[0])


def contribution_score_bits(
    values: torch.Tensor,
    selection_mode: str,
) -> np.ndarray:
    """Encode raw or absolute contribution scores for exact Top-k selection."""

    if selection_mode == "positive":
        return ordered_float_bits(values)
    if selection_mode == "absolute":
        return absolute_float_bits(values)
    raise ValueError("selection_mode must be 'positive' or 'absolute'")


def find_topk_selections(
    l_plus_contribution_path: str | Path | None,
    k_percents,
    chunk_numel: int = 5_000_000,
    selection_mode: str = "absolute",
    l_star_contribution_path: str | Path | None = None,
    contribution_channel: str = "L_plus",
    parameter_names: list[str] | None = None,
) -> dict[float, TopKSelection]:
    """Find several exact global Top-k thresholds in four shared radix scans.

    A separate :func:`find_topk_selection` call scans the contribution tensors
    four times for every requested percentage. This batched variant groups
    targets that currently share a radix prefix, so any number of curve points
    still reads each contribution chunk only once per radix byte. Threshold
    ties retain the same deterministic budget used by the single-k selector.
    """

    requested = [float(k_percent) for k_percent in k_percents]
    unique_percents = list(dict.fromkeys(requested))
    if any(not 0 <= k_percent <= 100 for k_percent in unique_percents):
        raise ValueError("k_percent must be between 0 and 100")
    if selection_mode not in {"positive", "absolute"}:
        raise ValueError("selection_mode must be 'positive' or 'absolute'")
    if not unique_percents:
        return {}

    contribution_paths = resolve_contribution_paths(
        l_plus_contribution_path,
        l_star_contribution_path,
        contribution_channel,
    )
    with ExitStack() as stack:
        handles = [
            stack.enter_context(
                safe_open(str(path), framework="pt", device="cpu")
            )
            for path in contribution_paths
        ]
        if parameter_names is None:
            names = sorted(handles[0].keys())
            if any(set(handle.keys()) != set(names) for handle in handles[1:]):
                raise ValueError("L_plus and L_star parameter names do not match")
        else:
            names = sorted(set(parameter_names))
            if not names:
                raise ValueError("No contribution parameters were selected")
            if any(not set(names).issubset(handle.keys()) for handle in handles):
                raise ValueError("Selected parameters are missing from contribution files")
        total_numel = sum(
            math.prod(handles[0].get_slice(name).get_shape()) for name in names
        )

        resolved = {}
        states = {}
        for k_percent in unique_percents:
            selected_numel = math.ceil(total_numel * k_percent / 100)
            if selected_numel == 0:
                resolved[k_percent] = TopKSelection(
                    total_numel, 0, None, None, 0
                )
            elif selected_numel == total_numel:
                resolved[k_percent] = TopKSelection(
                    total_numel, total_numel, None, None, 0, True
                )
            else:
                states[k_percent] = {
                    "selected_numel": selected_numel,
                    "rank": selected_numel,
                    "prefix": 0,
                }

        for step, shift in enumerate(RADIX_SHIFTS, start=1):
            if not states:
                break
            prefixes = sorted({state["prefix"] for state in states.values()})
            counts_by_prefix = {
                prefix: np.zeros(256, dtype=np.int64) for prefix in prefixes
            }
            description = (
                f"Selecting {len(states)} Top-k thresholds "
                f"({step}/{len(RADIX_SHIFTS)})"
            )
            for name in tqdm(names, desc=description, unit="tensor"):
                for _, chunk in combined_contribution_chunks(
                    handles, name, chunk_numel
                ):
                    if shift == RADIX_SHIFTS[0]:
                        if not torch.isfinite(chunk).all():
                            raise ValueError(
                                f"Contribution tensor contains NaN or Inf: {name}"
                            )
                    bits = contribution_score_bits(chunk, selection_mode)
                    if shift == RADIX_SHIFTS[0]:
                        digits = (bits >> shift) & np.uint32(0xFF)
                        counts_by_prefix[0] += np.bincount(digits, minlength=256)
                        continue

                    upper = bits >> (shift + 8)
                    for prefix in prefixes:
                        prefix_bits = bits[upper == prefix]
                        if prefix_bits.size:
                            digits = (prefix_bits >> shift) & np.uint32(0xFF)
                            counts_by_prefix[prefix] += np.bincount(
                                digits, minlength=256
                            )

            for state in states.values():
                counts = counts_by_prefix[state["prefix"]]
                rank = state["rank"]
                for digit in range(255, -1, -1):
                    digit_count = int(counts[digit])
                    if rank > digit_count:
                        rank -= digit_count
                        continue
                    state["prefix"] = (state["prefix"] << 8) | digit
                    state["rank"] = rank
                    break
                else:
                    raise RuntimeError("Radix selection rank exceeded its prefix count")

        for k_percent, state in states.items():
            prefix = state["prefix"]
            threshold_score = (
                ordered_bits_to_float(prefix)
                if selection_mode == "positive"
                else float(np.array([prefix], dtype=np.uint32).view(np.float32)[0])
            )
            resolved[k_percent] = TopKSelection(
                total_numel=total_numel,
                selected_numel=state["selected_numel"],
                threshold_bits=prefix,
                threshold_score=float(threshold_score),
                threshold_ties_to_keep=state["rank"],
            )

    return {k_percent: resolved[k_percent] for k_percent in unique_percents}


def find_topk_selection(
    l_plus_contribution_path: str | Path | None,
    k_percent: float,
    chunk_numel: int = 5_000_000,
    selection_mode: str = "absolute",
    l_star_contribution_path: str | Path | None = None,
    contribution_channel: str = "L_plus",
    parameter_names: list[str] | None = None,
) -> TopKSelection:
    """Find an exact global contribution threshold using four radix scans.

    k_percent uses the requested parameter_names as its denominator, or all
    contribution scalars when no names are provided. Positive mode ranks the
    raw contribution C from largest to smallest and always fills the requested
    budget, including zero or negative C when k crosses that boundary.
    L_contrast ranks the elementwise FP32 sum L_plus + L_star without writing
    a new file. Absolute mode ranks by |C|.
    """

    if not 0 <= k_percent <= 100:
        raise ValueError("k_percent must be between 0 and 100")
    if selection_mode not in {"positive", "absolute"}:
        raise ValueError("selection_mode must be 'positive' or 'absolute'")

    contribution_paths = resolve_contribution_paths(
        l_plus_contribution_path,
        l_star_contribution_path,
        contribution_channel,
    )
    with ExitStack() as stack:
        handles = [
            stack.enter_context(
                safe_open(str(path), framework="pt", device="cpu")
            )
            for path in contribution_paths
        ]
        if parameter_names is None:
            names = sorted(handles[0].keys())
            if any(set(handle.keys()) != set(names) for handle in handles[1:]):
                raise ValueError("L_plus and L_star parameter names do not match")
        else:
            names = sorted(set(parameter_names))
            if not names:
                raise ValueError("No contribution parameters were selected")
            if any(not set(names).issubset(handle.keys()) for handle in handles):
                raise ValueError("Selected parameters are missing from contribution files")
        total_numel = sum(
            math.prod(handles[0].get_slice(name).get_shape()) for name in names
        )
        selected_numel = math.ceil(total_numel * k_percent / 100)

        if selected_numel == 0:
            return TopKSelection(total_numel, 0, None, None, 0)
        if selected_numel == total_numel:
            return TopKSelection(total_numel, total_numel, None, None, 0, True)

        rank = selected_numel
        prefix = 0
        for step, shift in enumerate(RADIX_SHIFTS, start=1):
            counts = np.zeros(256, dtype=np.int64)
            description = f"Selecting Top-k threshold ({step}/{len(RADIX_SHIFTS)})"
            for name in tqdm(names, desc=description, unit="tensor"):
                for _, chunk in combined_contribution_chunks(
                    handles, name, chunk_numel
                ):
                    if shift == RADIX_SHIFTS[0]:
                        if not torch.isfinite(chunk).all():
                            raise ValueError(
                                f"Contribution tensor contains NaN or Inf: {name}"
                            )
                    bits = contribution_score_bits(chunk, selection_mode)
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


def model_ranking_chunks(
    parameters: dict[str, torch.nn.Parameter],
    finetuned: ModelTensorReader,
    name: str,
    max_numel: int,
    ranking_method: str,
):
    """Yield FP32 parameter scores for |delta| or |theta_finetuned| ranking."""

    parameter = parameters[name]
    shape = tuple(parameter.shape)
    if finetuned.shape(name) != shape:
        raise ValueError(f"Shape mismatch for parameter: {name}")

    for index in chunk_indices(shape, max_numel):
        base_chunk = parameter.data if index is None else parameter.data[index]
        finetuned_chunk = finetuned.read(name, index).reshape(-1)
        if ranking_method == "delta_magnitude":
            scores = finetuned_chunk.float().sub(base_chunk.reshape(-1).float())
        elif ranking_method == "finetuned_magnitude":
            scores = finetuned_chunk.float()
        else:
            raise ValueError(f"Unsupported model ranking method: {ranking_method}")
        yield index, scores


def find_model_topk_selection(
    parameters: dict[str, torch.nn.Parameter],
    finetuned_model_path: str | Path,
    k_percent: float,
    ranking_method: str,
    chunk_numel: int = 5_000_000,
) -> TopKSelection:
    """Find the exact global Top-k threshold for a model-weight ranking."""

    if ranking_method not in {"delta_magnitude", "finetuned_magnitude"}:
        raise ValueError("Model ranking must be delta_magnitude or finetuned_magnitude")
    if not 0 <= k_percent <= 100:
        raise ValueError("k_percent must be between 0 and 100")

    names = sorted(parameters)
    total_numel = sum(parameter.numel() for parameter in parameters.values())
    selected_numel = math.ceil(total_numel * k_percent / 100)
    if selected_numel == 0:
        return TopKSelection(total_numel, 0, None, None, 0)
    if selected_numel == total_numel:
        return TopKSelection(total_numel, total_numel, None, None, 0, True)

    with ModelTensorReader(finetuned_model_path) as finetuned:
        if not set(names).issubset(finetuned.keys):
            raise ValueError("Base and fine-tuned parameter names do not match")

        rank = selected_numel
        prefix = 0
        for step, shift in enumerate(RADIX_SHIFTS, start=1):
            counts = np.zeros(256, dtype=np.int64)
            description = (
                f"Selecting {ranking_method} threshold "
                f"({step}/{len(RADIX_SHIFTS)})"
            )
            for name in tqdm(names, desc=description, unit="tensor"):
                for _, scores in model_ranking_chunks(
                    parameters,
                    finetuned,
                    name,
                    chunk_numel,
                    ranking_method,
                ):
                    if shift == RADIX_SHIFTS[0] and not torch.isfinite(scores).all():
                        raise ValueError(f"Model tensor contains NaN or Inf: {name}")
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

    threshold_score = np.array([prefix], dtype=np.uint32).view(np.float32)[0]
    return TopKSelection(
        total_numel=total_numel,
        selected_numel=selected_numel,
        threshold_bits=prefix,
        threshold_score=float(threshold_score),
        threshold_ties_to_keep=rank,
    )


def selection_mask(
    score_bits: np.ndarray,
    selection: TopKSelection,
    ties_remaining: int,
) -> tuple[np.ndarray, int]:
    """Select one chunk while resolving threshold ties in deterministic order."""

    if selection.select_all:
        return np.ones(score_bits.shape, dtype=bool), ties_remaining
    if selection.threshold_bits is None:
        return np.zeros(score_bits.shape, dtype=bool), ties_remaining

    selected = score_bits > np.uint32(selection.threshold_bits)
    if ties_remaining:
        equal_indices = np.flatnonzero(score_bits == selection.threshold_bits)
        take = min(ties_remaining, len(equal_indices))
        selected[equal_indices[:take]] = True
        ties_remaining -= take
    return selected, ties_remaining


def merge_selected_values(
    base_values: torch.Tensor,
    finetuned_values: torch.Tensor,
    selected_indices: torch.Tensor,
    alpha: float,
) -> None:
    """Merge selected flattened values into the base tensor in place."""

    if alpha == 0 or selected_indices.numel() == 0:
        return

    finetuned_selected = finetuned_values.index_select(0, selected_indices)
    if alpha == 1:
        merged = finetuned_selected.to(base_values.dtype)
    else:
        base_selected = base_values.index_select(0, selected_indices).float()
        merged = base_selected.add_(
            finetuned_selected.float().sub_(base_selected), alpha=alpha
        ).to(base_values.dtype)
    base_values.index_copy_(0, selected_indices, merged)


def resolve_dtype(dtype: str):
    """Resolve a YAML dtype name for Transformers model loading."""

    if dtype == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]


def build_run_name(
    experiment_name: str,
    k_percent: float,
    alpha: float,
    selection_mode: str,
    contribution_channel: str = "L_plus",
    use_weight_rescale: bool = False,
    ranking_method: str = "contribution",
    seed: int = 0,
    layers: list[int] | None = None,
    modules: list[str] | None = None,
) -> str:
    """Build a name that records the ranking source, k, alpha, and scaling."""

    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "-", experiment_name).strip("-")
    selectors = []
    if layers:
        selectors.append("layers-" + "-".join(map(str, sorted(set(layers)))))
    if modules:
        module_label = "+".join(module.strip(".").replace(".", "-") for module in modules)
        selectors.append(f"modules-{module_label}")
    if selectors:
        prefix += "-" + "-".join(selectors)
    full_scope_merge = bool(selectors) and k_percent == 100
    rescale_tag = "-rescaled" if use_weight_rescale else ""
    if full_scope_merge:
        ranking_tag = "all-"
    elif ranking_method == "contribution":
        ranking_tag = f"{contribution_channel}-top-{selection_mode}"
    elif ranking_method == "random":
        ranking_tag = "random-keep-"
    else:
        ranking_tag = f"{ranking_method.replace('_', '-')}-top-"
    seed_tag = f"-seed{seed}" if ranking_method == "random" and not full_scope_merge else ""
    return f"{prefix}-{ranking_tag}{k_percent:g}pct-alpha{alpha:g}{rescale_tag}{seed_tag}"


def merge_topk_parameters(
    base_model_path: str,
    finetuned_model_path: str,
    l_plus_contribution_path: str | None,
    merged_model_output_path: str,
    k_percent: float,
    alpha: float,
    selection_mode: str = "positive",
    l_star_contribution_path: str | None = None,
    contribution_channel: str = "L_plus",
    merge_dtype: str = "auto",
    chunk_numel: int = 5_000_000,
    max_shard_size: str = "5GB",
    use_weight_rescale: bool = False,
    ranking_method: str = "contribution",
    seed: int = 0,
    layers: list[int] | None = None,
    modules: list[str] | None = None,
) -> TopKSelection:
    """Select Top-k within an optional scope, then merge into the full model."""

    if ranking_method not in RANKING_METHODS:
        raise ValueError(f"ranking_method must be one of {RANKING_METHODS}")
    if not 0 <= k_percent <= 100:
        raise ValueError("k_percent must be between 0 and 100")
    if chunk_numel <= 0:
        raise ValueError("chunk_numel must be positive")
    if use_weight_rescale and k_percent == 0:
        raise ValueError("DaRE-style rescaling is undefined when k_percent is 0")

    output_path = Path(merged_model_output_path)
    source_paths = {
        Path(base_model_path).resolve(),
        Path(finetuned_model_path).resolve(),
    }
    if output_path.resolve() in source_paths:
        raise ValueError("merged_model_output_path must differ from both source models")

    print(f"Loading base model from {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        dtype=resolve_dtype(merge_dtype),
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    parameters = dict(model.named_parameters())

    candidate_names, scope = select_parameters(parameters, layers, modules)
    candidate_names = sorted(candidate_names)
    candidate_parameters = {name: parameters[name] for name in candidate_names}
    
    has_scope = bool(layers or modules)
    full_scope_merge = has_scope and k_percent == 100
    needs_contributions = ranking_method == "contribution" and not full_scope_merge
    print(
        f"Candidate scope: {scope.selected_numel:,}/{scope.total_numel:,} scalar "
        f"parameters ({scope.selected_percent:.8f}% of the model) across "
        f"{scope.selected_tensor_count} tensors."
    )

    random_masks = None
    if full_scope_merge:
        selection = TopKSelection(
            scope.selected_numel, scope.selected_numel, None, None, 0, True,
        )
        ranking_label = "all parameters in the selected layers/modules (ranking skipped)"
    elif ranking_method == "random":
        random_masks = RandomSelectionMask(
            scope.selected_numel, k_percent, seed, chunk_numel,
        )
        selection = random_masks.selection
        ranking_label = f"random/seed{seed}"
    elif ranking_method == "contribution":
        selection = find_topk_selection(
            l_plus_contribution_path,
            k_percent,
            chunk_numel,
            selection_mode=selection_mode,
            l_star_contribution_path=l_star_contribution_path,
            contribution_channel=contribution_channel,
            parameter_names=candidate_names if has_scope else None,
        )
        ranking_label = f"{contribution_channel}/{selection_mode}"
    else:
        selection = find_model_topk_selection(
            candidate_parameters,
            finetuned_model_path,
            k_percent,
            ranking_method,
            chunk_numel,
        )
        ranking_label = ranking_method

    if selection.total_numel != scope.selected_numel:
        raise ValueError("Contribution and model parameter counts do not match")
    selection = replace(
        selection, total_numel=scope.total_numel, candidate_numel=scope.selected_numel,
    )
    print(
        f"Selected {selection.selected_numel:,}/{selection.total_numel:,} scalar parameters "
        f"({100 * selection.selected_numel / selection.total_numel:.8f}% of the model; "
        f"{100 * selection.selected_numel / selection.candidate_numel:.8f}% of candidate scope) "
        f"with ranking_method={ranking_label}."
    )
    if selection.threshold_score is not None:
        print(f"Candidate scope {ranking_label} threshold: {selection.threshold_score:.8e}")
    
    selected_params_ratio = selection.selected_numel / selection.total_numel
    rescale_factor = 1 / selected_params_ratio if use_weight_rescale else 1.0
    effective_alpha = alpha * rescale_factor
    print(
        f"Weight rescale={'DaRE' if use_weight_rescale else 'none'}; "
        f"rescale_factor={rescale_factor:g}; effective_alpha={effective_alpha:g}",
        flush=True,
    )

    contribution_paths = (
        resolve_contribution_paths(
            l_plus_contribution_path,
            l_star_contribution_path,
            contribution_channel,
        )
        if needs_contributions
        else []
    )

    ties_remaining = selection.threshold_ties_to_keep
    merged_numel = 0
    with ExitStack() as stack:
        contributions = [
            stack.enter_context(
                safe_open(str(path), framework="pt", device="cpu")
            )
            for path in contribution_paths
        ]
        finetuned = stack.enter_context(ModelTensorReader(finetuned_model_path))
        stack.enter_context(torch.no_grad())

        names = candidate_names
        if not set(names).issubset(finetuned.keys):
            raise ValueError("Base and fine-tuned parameter names do not match")
        if needs_contributions:
            if any(not set(names).issubset(handle.keys()) for handle in contributions):
                raise ValueError("Selected parameters are missing from contribution files")
            if not has_scope and any(
                set(handle.keys()) != set(parameters) for handle in contributions
            ):
                raise ValueError(
                    "Contribution, base, and fine-tuned parameter names do not match"
                )

        for name in tqdm(names, desc="Merging Top-k parameters", unit="tensor"):
            parameter = parameters[name]
            if full_scope_merge or ranking_method == "random":
                shape = tuple(parameter.shape)
                score_chunks = (
                    (index, None) for index in chunk_indices(shape, chunk_numel)
                )
            elif ranking_method == "contribution":
                shape = tuple(contributions[0].get_slice(name).get_shape())
                score_chunks = combined_contribution_chunks(
                    contributions, name, chunk_numel
                )
            else:
                shape = tuple(parameter.shape)
                score_chunks = model_ranking_chunks(
                    parameters,
                    finetuned,
                    name,
                    chunk_numel,
                    ranking_method,
                )

            if tuple(parameter.shape) != shape or finetuned.shape(name) != shape:
                raise ValueError(f"Shape mismatch for parameter: {name}")

            for index, scores in score_chunks:
                base_chunk = parameter.data if index is None else parameter.data[index]
                if full_scope_merge:
                    mask = np.ones(base_chunk.numel(), dtype=bool)
                elif random_masks is not None:
                    mask = random_masks.mask(base_chunk.numel())
                else:
                    score_bits = (
                        contribution_score_bits(scores, selection_mode)
                        if ranking_method == "contribution"
                        else absolute_float_bits(scores)
                    )
                    mask, ties_remaining = selection_mask(
                        score_bits, selection, ties_remaining,
                    )
                selected_indices = torch.from_numpy(np.flatnonzero(mask))
                merged_numel += selected_indices.numel()
                if effective_alpha == 0 or selected_indices.numel() == 0:
                    continue

                finetuned_chunk = finetuned.read(name, index).reshape(-1)
                merge_selected_values(
                    base_chunk.reshape(-1),
                    finetuned_chunk,
                    selected_indices,
                    effective_alpha,
                )

    if merged_numel != selection.selected_numel or ties_remaining != 0:
        raise RuntimeError("Top-k merge count did not match the requested parameter count")

    tokenizer = AutoTokenizer.from_pretrained(
        finetuned_model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    print(f"Saving merged model to {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    tokenizer.save_pretrained(output_path)

    del parameters, candidate_parameters
    del tokenizer, model
    gc.collect()
    return selection


def run_topk_merge(
    base_model_path: str,
    finetuned_model_path: str,
    contribution_dir: str | None,
    k_percent: float,
    alpha: float,
    output_dir: str,
    experiment_name: str | None = None,
    selection_mode: str = "positive",
    contribution_channel: str = "L_plus",
    merge_dtype: str = "auto",
    chunk_numel: int = 5_000_000,
    max_shard_size: str = "5GB",
    use_weight_rescale: bool = False,
    ranking_method: str = "contribution",
    seed: int = 0,
    layers: list[int] | None = None,
    modules: list[str] | None = None,
) -> dict:
    """Resolve one contribution directory, merge Top-k, and save the model."""
    full_scope_merge = bool(layers or modules) and k_percent == 100
    needs_contribution_files = ranking_method == "contribution" and not full_scope_merge
    if needs_contribution_files:
        if not contribution_dir:
            raise ValueError("contribution_dir is required for contribution ranking")
        contribution_files = discover_contribution_files(contribution_dir)
        contribution_dir = str(contribution_files["L_plus"].parent)
        l_plus_contribution_path = str(contribution_files["L_plus"])
        l_star_contribution_path = str(contribution_files["L_star"])
    else:
        l_plus_contribution_path = None
        l_star_contribution_path = None

    default_name = (
        Path(finetuned_model_path).name
        if ranking_method == "random" or l_plus_contribution_path is None
        else Path(l_plus_contribution_path).stem.removesuffix("-L_plus")
    )
    run_name = build_run_name(
        experiment_name or default_name,
        k_percent,
        alpha,
        selection_mode,
        contribution_channel,
        use_weight_rescale,
        ranking_method,
        seed,
        layers,
        modules,
    )
    output_path = str(Path(output_dir) / run_name)
    selection = merge_topk_parameters(
        base_model_path=base_model_path,
        finetuned_model_path=finetuned_model_path,
        l_plus_contribution_path=l_plus_contribution_path,
        merged_model_output_path=output_path,
        k_percent=k_percent,
        alpha=alpha,
        selection_mode=selection_mode,
        l_star_contribution_path=l_star_contribution_path,
        contribution_channel=contribution_channel,
        use_weight_rescale=use_weight_rescale,
        ranking_method=ranking_method,
        seed=seed,
        layers=layers,
        modules=modules,
        merge_dtype=merge_dtype,
        chunk_numel=chunk_numel,
        max_shard_size=max_shard_size,
    )

    candidate_numel = (
        selection.total_numel if selection.candidate_numel is None else selection.candidate_numel
    )
    
    
    summary = {
        "run_name": run_name,
        "output_path": output_path,
        "k_percent": k_percent,
        "alpha": alpha,
        "ranking_method": ranking_method,
        "layers": sorted(set(layers or [])),
        "modules": modules or [],
        "full_scope_merge": bool(layers or modules) and k_percent == 100,
        "selected_percent": 100 * selection.selected_numel / selection.total_numel,
        "candidate_percent": 100 * candidate_numel / selection.total_numel,
        "use_weight_rescale": use_weight_rescale,
        "rescale_factor": 100 / k_percent if use_weight_rescale else 1.0,
        "contribution_channel": contribution_channel,
        "contribution_dir": contribution_dir,
        "l_plus_contribution_path": l_plus_contribution_path,
        "l_star_contribution_path": l_star_contribution_path,
        "selection_mode": selection_mode,
        "selection": asdict(selection),
    }
    if ranking_method == "random":
        summary.update(seed=seed, random_sampling="uniform_without_replacement")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    """Load the YAML interface, merge Top-k parameters, and save the model."""

    config = parse_args_yaml("Global Top-k parameter merge")
    run_topk_merge(**config)


if __name__ == "__main__":
    main()
