"""Parameter-wise path-integrated attribution for CPVM.

The module follows single-endpoint or bipolar parameter paths and assigns
each scalar parameter its own contribution. Two channels are computed
separately:

* L_plus is the target-pole response log-probability.
* L_star is the negative counter-pole response log-probability, so a positive
  contribution means that the parameter update suppresses the counter pole.

Final parameter-name -> contribution-tensor mappings and path/model/objective
metadata are written. Bipolar runs also retain scalar endpoint completeness
statistics in metadata. Gradients, parameter deltas, quadrature nodes, and
other temporary tensors remain in memory and are discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
from typing import Mapping, Sequence

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

from CPVM.code.topk_merge import ModelTensorReader, discover_contribution_files
from CPVM.code.utils.arguments import Big5ChatArguments, parse_args_yaml
from CPVM.code.utils.bipolar_path import (
    branch_points,
    full_path_points,
    path_coefficients,
    validate_path_method,
)
from CPVM.code.utils.process import Big5Chat


SUPPORTED_K = (1, 2, 4, 8, 16, 32, 64)
CHANNELS = ("L_plus", "L_star")


@dataclass(frozen=True)
class PathPoint:
    """One numerical integration point along the parameter path."""

    alpha: float
    position: float
    tangent_scale: float
    weight: float


def interpolation_schedule(alpha: float, interpolation: str) -> tuple[float, float]:
    """Return path position and derivative for a normalized path coordinate.

    linear uses f(alpha)=alpha. parabolic uses f(alpha)=alpha^2.
    With only Base and Endpoint available, the latter is a non-uniform
    parameterization of the same line segment, not a geometrically different
    curve. Its derivative 2*alpha must be included in the line integral.
    """

    if interpolation == "linear":
        return alpha, 1.0
    if interpolation == "parabolic":
        return alpha * alpha, 2.0 * alpha
    raise ValueError("interpolation must be 'linear' or 'parabolic'")


def integration_points(k: int, interpolation: str) -> list[PathPoint]:
    """Build the requested K-point path quadrature.

    K=1 is the endpoint-gradient baseline from the research design. For
    K>1, Gauss-Legendre nodes and weights are mapped from [-1, 1] to [0, 1].
    """

    if k not in SUPPORTED_K:
        raise ValueError(f"k must be one of {SUPPORTED_K}")
    if interpolation not in {"linear", "parabolic"}:
        raise ValueError("interpolation must be 'linear' or 'parabolic'")
    if k == 1:
        return [PathPoint(alpha=1.0, position=1.0, tangent_scale=1.0, weight=1.0)]

    nodes, weights = np.polynomial.legendre.leggauss(k)
    points = []
    for node, weight in zip(nodes, weights):
        alpha = float((node + 1.0) / 2.0)
        position, tangent_scale = interpolation_schedule(alpha, interpolation)
        points.append(
            PathPoint(
                alpha=alpha,
                position=position,
                tangent_scale=tangent_scale,
                weight=float(weight / 2.0),
            )
        )
    return points


def response_log_probabilities(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return each response's token-length-normalized log-probability.

    logits and labels are already causally aligned. Label value -100 marks
    prompt or padding positions, which do not contribute to the target.
    """

    response_mask = labels.ne(-100)
    token_log_probabilities = -F.cross_entropy(
        logits[response_mask].float(),
        labels[response_mask],
        reduction="none",
    )
    row_ids = (
        torch.arange(labels.shape[0], device=labels.device)
        .unsqueeze(1)
        .expand_as(labels)[response_mask]
    )
    sums = torch.zeros(
        labels.shape[0],
        dtype=token_log_probabilities.dtype,
        device=labels.device,
    )
    sums.scatter_add_(0, row_ids, token_log_probabilities)
    return sums / response_mask.sum(dim=1)


class MatchedCompletionObjective:
    """Prepare matched target/counter BIG5-CHAT completions for attribution."""

    def __init__(
        self,
        tokenizer,
        dataset_path: str,
        trait: str,
        target_pole: str,
        attribution_split: str,
        val_ratio: float,
        seed: int,
        max_samples: int | None,
        max_length: int,
        filter_refusals: bool,
        use_original_prompt: bool,
        dialogue_prompt: str | None,
    ):
        """Load, pair, shuffle, and tokenize the two response channels once."""

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.big5chat_args = Big5ChatArguments(
            trait=trait,
            level=target_pole,
            use_original_prompt=use_original_prompt,
            dialogue_prompt=dialogue_prompt,
        )
        pairs = self._load_pairs(
            dataset_path=dataset_path,
            trait=trait,
            target_pole=target_pole,
            attribution_split=attribution_split,
            val_ratio=val_ratio,
            seed=seed,
            max_samples=max_samples,
            filter_refusals=filter_refusals,
        )
        self.num_examples = len(pairs)
        self.encoded = {
            "L_plus": [
                self._encode_completion(pair, "chosen") for pair in pairs
            ],
            "L_star": [
                self._encode_completion(pair, "rejected") for pair in pairs
            ],
        }
        self.collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8,
            label_pad_token_id=-100,
            return_tensors="pt",
        )

    def _load_pairs(
        self,
        dataset_path: str,
        trait: str,
        target_pole: str,
        attribution_split: str,
        val_ratio: float,
        seed: int,
        max_samples: int | None,
        filter_refusals: bool,
    ):
        """Create held-out prompt pairs with target and counter completions."""

        dataset = Big5Chat(dataset_path)
        split_dataset = Big5Chat.split_by_original_index(
            dataset,
            val_size=val_ratio,
            seed=seed,
        )
        trait_rows = split_dataset[attribution_split].filter(
            Big5Chat.filter_func,
            fn_kwargs={
                "trait_filter": {"trait": trait},
                "big5chat_args": self.big5chat_args,
            },
        )
        trait_rows = Big5Chat.refusal_filter(trait_rows, filter_refusals)
        pairs = Big5Chat.build_dpo_dataset(
            trait_rows,
            chosen_level=target_pole,
            big5chat_args=self.big5chat_args,
            filter_refusals=filter_refusals,
        ).shuffle(seed=seed)
        if max_samples is not None:
            pairs = pairs.select(range(min(max_samples, len(pairs))))
        print(f"Loaded {len(pairs)} matched {trait} attribution examples")
        return pairs

    def _encode_completion(self, pair: Mapping, completion_key: str) -> dict:
        """Tokenize one prompt-completion pair and mask every prompt token."""

        prompt_text = self.tokenizer.apply_chat_template(
            pair["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        completion_text = pair[completion_key][0]["content"]
        prompt_ids = self.tokenizer(
            prompt_text,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        completion_ids = self.tokenizer(
            completion_text,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        response_ids = completion_ids + [self.tokenizer.eos_token_id]
        input_ids = (prompt_ids + response_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + response_ids)[: self.max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    def backward(
        self,
        model,
        channel: str,
        batch_size: int,
        input_device: torch.device,
        keep_answer_logits: bool,
    ) -> None:
        """Accumulate the full-dataset gradient for one differentiable channel."""

        examples = self.encoded[channel]
        channel_sign = 1.0 if channel == "L_plus" else -1.0
        batch_starts = range(0, self.num_examples, batch_size)

        for start in tqdm(
            batch_starts,
            desc=f"{channel} batches",
            unit="batch",
            leave=False,
        ):
            batch = self.collator(examples[start : start + batch_size])
            labels = batch.pop("labels").to(input_device)
            model_inputs = {
                name: tensor.to(input_device) for name, tensor in batch.items()
            }
            shifted_labels = labels[:, 1:]

            if keep_answer_logits:
                kept_positions = (
                    shifted_labels.ne(-100)
                    .any(dim=0)
                    .nonzero(as_tuple=False)
                    .flatten()
                )
                outputs = model(
                    **model_inputs,
                    # A Python list stays device-neutral when Accelerate moves
                    # the final hidden states to another model-parallel shard.
                    logits_to_keep=kept_positions.tolist(),
                )
                aligned_labels = shifted_labels.index_select(
                    1,
                    kept_positions,
                ).to(outputs.logits.device)
                aligned_logits = outputs.logits
            else:
                outputs = model(**model_inputs)
                aligned_logits = outputs.logits[:, :-1, :]
                aligned_labels = shifted_labels.to(outputs.logits.device)

            per_example = response_log_probabilities(
                aligned_logits,
                aligned_labels,
            )
            objective = channel_sign * per_example.sum() / self.num_examples
            objective.backward()

    @torch.no_grad()
    def value(
        self,
        model,
        channel: str,
        batch_size: int,
        input_device: torch.device,
        keep_answer_logits: bool,
    ) -> float:
        """Evaluate the same scalar objective used by :meth:`backward`."""

        examples = self.encoded[channel]
        channel_sign = 1.0 if channel == "L_plus" else -1.0
        total = 0.0
        batch_starts = range(0, self.num_examples, batch_size)

        for start in tqdm(
            batch_starts,
            desc=f"Evaluating {channel} objective",
            unit="batch",
            leave=False,
        ):
            batch = self.collator(examples[start : start + batch_size])
            labels = batch.pop("labels").to(input_device)
            model_inputs = {
                name: tensor.to(input_device) for name, tensor in batch.items()
            }
            shifted_labels = labels[:, 1:]

            if keep_answer_logits:
                kept_positions = (
                    shifted_labels.ne(-100)
                    .any(dim=0)
                    .nonzero(as_tuple=False)
                    .flatten()
                )
                outputs = model(
                    **model_inputs,
                    logits_to_keep=kept_positions.tolist(),
                )
                aligned_labels = shifted_labels.index_select(
                    1,
                    kept_positions,
                ).to(outputs.logits.device)
                aligned_logits = outputs.logits
            else:
                outputs = model(**model_inputs)
                aligned_logits = outputs.logits[:, :-1, :]
                aligned_labels = shifted_labels.to(outputs.logits.device)

            per_example = response_log_probabilities(
                aligned_logits,
                aligned_labels,
            )
            total += channel_sign * per_example.double().sum().item()

        return total / self.num_examples


def resolve_dtype(dtype: str):
    """Translate a YAML dtype name into the value expected by Transformers."""

    if dtype == "auto":
        return "auto"
    return getattr(torch, dtype)


def checkpoint_name(model_path: str | Path) -> str:
    """Build a compact output name that disambiguates checkpoint directories."""

    path = Path(model_path)
    if path.name.startswith("checkpoint-"):
        return f"{path.parent.name}-{path.name}"
    return path.name


def build_parameter_path(
    model,
    endpoint_model_path: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Read endpoint tensors and construct Endpoint-Base deltas on CPU."""

    base_parameters: dict[str, torch.Tensor] = {}
    deltas: dict[str, torch.Tensor] = {}

    with ModelTensorReader(endpoint_model_path) as endpoint:
        endpoint_names = endpoint.keys
        for name, parameter in tqdm(
            model.named_parameters(),
            desc="Building parameter path",
            unit="tensor",
        ):
            if name not in endpoint_names:
                raise ValueError(f"Endpoint checkpoint is missing parameter: {name}")
            endpoint_shape = endpoint.shape(name)
            if endpoint_shape != tuple(parameter.shape):
                raise ValueError(
                    f"Endpoint parameter shape mismatch for {name}: "
                    f"base {tuple(parameter.shape)}, endpoint {endpoint_shape}"
                )
            base_tensor = parameter.detach().to("cpu", copy=True)
            endpoint_tensor = endpoint.read(name, None).detach().to(
                device="cpu",
                dtype=torch.float32,
                copy=True,
            )
            base_parameters[name] = base_tensor
            deltas[name] = endpoint_tensor.sub_(base_tensor.float())

    gc.collect()
    return base_parameters, deltas


@torch.no_grad()
def set_path_parameters(
    model,
    base_parameters: Mapping[str, torch.Tensor],
    deltas: Mapping[str, torch.Tensor],
    position: float,
) -> None:
    """Write theta_0 + position * delta into the sharded live model."""

    for name, parameter in model.named_parameters():
        interpolated = torch.add(
            base_parameters[name],
            deltas[name],
            alpha=position,
        )
        parameter.copy_(
            interpolated.to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
        )


def accumulate_contributions(
    model,
    deltas: Mapping[str, torch.Tensor],
    contributions: dict[str, torch.Tensor],
    scale: float,
) -> None:
    """Add scale * gradient * delta to every same-shaped CPU matrix."""

    for name, parameter in model.named_parameters():
        weighted_gradient = parameter.grad.detach().to(
            device="cpu",
            dtype=torch.float32,
        )
        weighted_gradient.mul_(deltas[name])
        contributions[name].add_(weighted_gradient, alpha=scale)


def integrate_contributions(
    model,
    objective: MatchedCompletionObjective,
    base_parameters: Mapping[str, torch.Tensor],
    deltas: Mapping[str, torch.Tensor],
    channel: str,
    interpolation: str,
    k: int,
    batch_size: int,
    input_device: torch.device,
    keep_answer_logits: bool,
) -> dict[str, torch.Tensor]:
    """Approximate one channel's parameter-wise line integral."""

    contributions = {
        name: torch.zeros_like(tensor, dtype=torch.float32, device="cpu")
        for name, tensor in base_parameters.items()
    }
    points = integration_points(k, interpolation)

    for point in tqdm(points, desc=f"Integrating {channel}", unit="point"):
        set_path_parameters(
            model,
            base_parameters,
            deltas,
            position=point.position,
        )
        model.zero_grad(set_to_none=True)
        objective.backward(
            model=model,
            channel=channel,
            batch_size=batch_size,
            input_device=input_device,
            keep_answer_logits=keep_answer_logits,
        )
        accumulate_contributions(
            model,
            deltas,
            contributions,
            scale=point.weight * point.tangent_scale,
        )

    model.zero_grad(set_to_none=True)
    return contributions


def save_contributions(
    contributions: Mapping[str, torch.Tensor],
    output_path: Path,
) -> None:
    """Save only the final parameter-name -> contribution-matrix mapping."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        save_file(
            {name: tensor.contiguous() for name, tensor in contributions.items()},
            str(temporary_path),
        )
        # Replace the directory entry rather than mutating an existing hardlink.
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def run_path_integral(
    base_model_path: str,
    endpoint_model_path: str,
    dataset_path: str,
    output_dir: str,
    trait: str,
    target_pole: str,
    interpolation: str,
    k: int,
    dtype: str = "bfloat16",
    device_map: str | None = "balanced",
    batch_size: int = 1,
    max_length: int = 1024,
    attribution_split: str = "val",
    val_ratio: float = 0.1,
    seed: int = 42,
    max_samples: int | None = 200,
    filter_refusals: bool = True,
    use_original_prompt: bool = False,
    dialogue_prompt: str | None = None,
    output_name: str | None = None,
    path_method: str = "piecewise_linear",
) -> dict[str, Path]:
    """Run the backward-compatible Base -> single-endpoint attribution."""

    validate_path_method(path_method)
    if path_method != "piecewise_linear":
        raise ValueError(
            "Single-endpoint attribution only supports piecewise_linear; "
            "provide plus_model_path and star_model_path for the other paths"
        )
    if target_pole not in {"high", "low"}:
        raise ValueError("target_pole must be 'high' or 'low'")
    integration_points(k, interpolation)
    run_name = output_name or checkpoint_name(endpoint_model_path)
    _validate_run_name(run_name)
    run_directory = Path(output_dir) / path_method / run_name
    output_stem = f"{trait}-{target_pole}-{interpolation}-K{k}"
    _check_output_pair(run_directory, output_stem)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    objective = MatchedCompletionObjective(
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        trait=trait,
        target_pole=target_pole,
        attribution_split=attribution_split,
        val_ratio=val_ratio,
        seed=seed,
        max_samples=max_samples,
        max_length=max_length,
        filter_refusals=filter_refusals,
        use_original_prompt=use_original_prompt,
        dialogue_prompt=dialogue_prompt,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        dtype=resolve_dtype(dtype),
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.eval()
    model.requires_grad_(True)
    input_device = model.get_input_embeddings().weight.device
    keep_answer_logits = (
        "logits_to_keep" in inspect.signature(model.forward).parameters
    )

    base_parameters, deltas = build_parameter_path(
        model,
        endpoint_model_path,
    )
    output_paths: dict[str, Path] = {}

    for channel in CHANNELS:
        contributions = integrate_contributions(
            model=model,
            objective=objective,
            base_parameters=base_parameters,
            deltas=deltas,
            channel=channel,
            interpolation=interpolation,
            k=k,
            batch_size=batch_size,
            input_device=input_device,
            keep_answer_logits=keep_answer_logits,
        )
        output_path = run_directory / f"{output_stem}-{channel}.safetensors"
        save_contributions(contributions, output_path)
        output_paths[channel] = output_path
        print(f"Saved {channel} contributions to {output_path}")
        del contributions
        gc.collect()

    _write_metadata(run_directory, {
        "path_method": path_method,
        "base_model_path": str(Path(base_model_path).expanduser().resolve()),
        "endpoint_model_path": str(Path(endpoint_model_path).expanduser().resolve()),
        "target_pole": target_pole,
        "trait": trait,
        "interpolation": interpolation,
        "k": k,
        "start_t": 0.0,
        "end_t": 1.0 if target_pole == "high" else -1.0,
        "path_direction": "center_to_endpoint",
        "center_description": "base_model",
        "attribution_scope": "dense_path_before_topk_selection",
        "objective": _objective_metadata(
            dataset_path, attribution_split, val_ratio, seed, max_samples,
            max_length, filter_refusals, use_original_prompt, dialogue_prompt,
        ),
        "num_examples": objective.num_examples,
        "channels": _channel_metadata(),
        "files": {channel: path.name for channel, path in output_paths.items()},
    })
    return output_paths



def _validate_run_name(run_name: str) -> None:
    if not run_name or run_name in {".", ".."} or Path(run_name).name != run_name:
        raise ValueError("output_name must be a single non-empty directory name")


def _check_output_pair(directory: Path, stem: str) -> None:
    """Keep each leaf unambiguous for existing Top-k contribution discovery."""
    expected = {f"{stem}-{channel}.safetensors" for channel in CHANNELS}
    unexpected = sorted(
        path.name for path in directory.glob("*.safetensors")
        if path.name not in expected
    )
    if unexpected:
        raise ValueError(
            f"{directory} already contains a different contribution run: "
            f"{unexpected}; choose a new output_name"
        )


def _write_metadata(directory: Path, metadata: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "attribution_metadata.json"
    temporary = directory / ".attribution_metadata.json.tmp"
    try:
        temporary.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _channel_metadata() -> dict[str, str]:
    return {
        "L_plus": "mean token-normalized log P(target-pole completion)",
        "L_star": "negative mean token-normalized log P(counter-pole completion)",
        "L_contrast": "L_plus + L_star; combined on demand by Top-k readers",
    }


def _objective_metadata(
    dataset_path: str,
    attribution_split: str,
    val_ratio: float,
    seed: int,
    max_samples: int | None,
    max_length: int,
    filter_refusals: bool,
    use_original_prompt: bool,
    dialogue_prompt: str | None,
) -> dict:
    return {
        "dataset_path": str(Path(dataset_path).expanduser().resolve()),
        "attribution_split": attribution_split,
        "val_ratio": val_ratio,
        "seed": seed,
        "max_samples": max_samples,
        "max_length": max_length,
        "filter_refusals": filter_refusals,
        "use_original_prompt": use_original_prompt,
        "dialogue_prompt": dialogue_prompt,
        "normalization": "response_token_mean_then_example_mean",
    }


def build_bipolar_parameter_path(
    model,
    plus_model_path: str,
    star_model_path: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Keep base and both full-parameter endpoint deltas in CPU memory."""
    base_parameters: dict[str, torch.Tensor] = {}
    plus_deltas: dict[str, torch.Tensor] = {}
    star_deltas: dict[str, torch.Tensor] = {}
    with (
        ModelTensorReader(plus_model_path) as plus_reader,
        ModelTensorReader(star_model_path) as star_reader,
    ):
        for name, parameter in tqdm(
            model.named_parameters(), desc="Building bipolar parameter path", unit="tensor"
        ):
            base_tensor = parameter.detach().to(device="cpu", copy=True)
            base_parameters[name] = base_tensor
            for pole, reader, destination in (
                ("high", plus_reader, plus_deltas),
                ("low", star_reader, star_deltas),
            ):
                if name not in reader.keys:
                    raise ValueError(f"{pole} endpoint is missing parameter {name}")
                shape = reader.shape(name)
                if shape != tuple(parameter.shape):
                    raise ValueError(
                        f"{pole} endpoint shape mismatch for {name}: "
                        f"base {tuple(parameter.shape)}, endpoint {shape}"
                    )
                endpoint = reader.read(name, None).detach().to(
                    device="cpu", dtype=torch.float32, copy=True,
                )
                destination[name] = endpoint.sub_(base_tensor.float())
    gc.collect()
    return base_parameters, plus_deltas, star_deltas


@torch.no_grad()
def set_bipolar_path_parameters(
    model,
    base_parameters: Mapping[str, torch.Tensor],
    plus_deltas: Mapping[str, torch.Tensor],
    star_deltas: Mapping[str, torch.Tensor],
    plus: float,
    star: float,
) -> None:
    """Set the model to the shared geometric path used by bipolar merge."""
    for name, parameter in model.named_parameters():
        interpolated = base_parameters[name].float().clone()
        if plus:
            interpolated.add_(plus_deltas[name], alpha=plus)
        if star:
            interpolated.add_(star_deltas[name], alpha=star)
        parameter.copy_(interpolated.to(device=parameter.device, dtype=parameter.dtype))


def integrate_bipolar_contributions(
    model,
    objective: MatchedCompletionObjective,
    base_parameters: Mapping[str, torch.Tensor],
    plus_deltas: Mapping[str, torch.Tensor],
    star_deltas: Mapping[str, torch.Tensor],
    channel: str,
    path_method: str,
    target_pole: str,
    interpolation: str,
    k: int,
    batch_size: int,
    input_device: torch.device,
    keep_answer_logits: bool,
) -> dict[str, torch.Tensor]:
    """Integrate a piecewise branch or one common continuous full path.

    Both signed delta components enter the tangent. K=1 retains the explicit
    terminal-gradient times net-displacement baseline; K>1 uses the actual
    path derivative, including the path orientation and schedule derivative.
    Continuous methods require target_pole='common' and a fixed high-pole
    objective throughout the entire low-to-high path.
    """
    contributions = {
        name: torch.zeros_like(tensor, dtype=torch.float32, device="cpu")
        for name, tensor in base_parameters.items()
    }
    if path_method == "piecewise_linear":
        points = branch_points(k, path_method, target_pole, interpolation)
    else:
        if target_pole != "common":
            raise ValueError("Continuous path attribution requires target_pole='common'")
        points = full_path_points(k, path_method, interpolation)
    for point in tqdm(
        points, desc=f"Integrating {path_method}/{target_pole}/{channel}", unit="point"
    ):
        set_bipolar_path_parameters(
            model, base_parameters, plus_deltas, star_deltas,
            plus=point.plus, star=point.star,
        )
        model.zero_grad(set_to_none=True)
        objective.backward(
            model=model,
            channel=channel,
            batch_size=batch_size,
            input_device=input_device,
            keep_answer_logits=keep_answer_logits,
        )
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            tangent = torch.mul(plus_deltas[name], point.tangent_plus)
            tangent.add_(star_deltas[name], alpha=point.tangent_star)
            gradient = parameter.grad.detach().to(device="cpu", dtype=torch.float32)
            contributions[name].addcmul_(gradient, tangent, value=point.weight)
        model.zero_grad(set_to_none=True)
    return contributions


def _summed_parameter_contribution(
    contributions: Mapping[str, torch.Tensor],
) -> float:
    """Sum every scalar attribution using float64 accumulation on CPU."""

    return sum(
        tensor.sum(dtype=torch.float64).item()
        for tensor in contributions.values()
    )


def _completeness_channel_record(
    start_objective: float,
    end_objective: float,
    summed_contribution: float,
) -> dict[str, float]:
    """Compare a line integral with the matching endpoint objective change."""

    endpoint_difference = end_objective - start_objective
    signed_error = summed_contribution - endpoint_difference
    absolute_error = abs(signed_error)
    relative_floor = 1e-12
    return {
        "start_objective": start_objective,
        "end_objective": end_objective,
        "endpoint_objective_difference": endpoint_difference,
        "summed_parameter_contribution": summed_contribution,
        "signed_error": signed_error,
        "absolute_error": absolute_error,
        "relative_error": absolute_error / max(abs(endpoint_difference), relative_floor),
    }


def measure_bipolar_completeness(
    model,
    objective: MatchedCompletionObjective,
    base_parameters: Mapping[str, torch.Tensor],
    plus_deltas: Mapping[str, torch.Tensor],
    star_deltas: Mapping[str, torch.Tensor],
    contributions: Mapping[str, torch.Tensor] | None,
    channel: str,
    path_method: str,
    target_pole: str,
    batch_size: int,
    input_device: torch.device,
    keep_answer_logits: bool,
    summed_contribution: float | None = None,
) -> dict[str, float]:
    """Numerically check sum(C_i) against F(path end)-F(path start)."""

    if (contributions is None) == (summed_contribution is None):
        raise ValueError(
            "Provide exactly one of contributions or summed_contribution"
        )
    if summed_contribution is None:
        summed_contribution = _summed_parameter_contribution(contributions)

    if path_method == "piecewise_linear":
        start_t = 0.0
        end_t = 1.0 if target_pole == "high" else -1.0
    else:
        start_t = -1.0
        end_t = 1.0

    start_plus, start_star = path_coefficients(start_t, path_method)
    end_plus, end_star = path_coefficients(end_t, path_method)
    set_bipolar_path_parameters(
        model,
        base_parameters,
        plus_deltas,
        star_deltas,
        plus=start_plus,
        star=start_star,
    )
    start_objective = objective.value(
        model=model,
        channel=channel,
        batch_size=batch_size,
        input_device=input_device,
        keep_answer_logits=keep_answer_logits,
    )
    set_bipolar_path_parameters(
        model,
        base_parameters,
        plus_deltas,
        star_deltas,
        plus=end_plus,
        star=end_star,
    )
    end_objective = objective.value(
        model=model,
        channel=channel,
        batch_size=batch_size,
        input_device=input_device,
        keep_answer_logits=keep_answer_logits,
    )
    return _completeness_channel_record(
        start_objective=start_objective,
        end_objective=end_objective,
        summed_contribution=summed_contribution,
    )


def _contrast_completeness_record(
    channels: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """Combine L_plus and L_star using L_contrast = L_plus + L_star."""

    return _completeness_channel_record(
        start_objective=(
            channels["L_plus"]["start_objective"]
            + channels["L_star"]["start_objective"]
        ),
        end_objective=(
            channels["L_plus"]["end_objective"]
            + channels["L_star"]["end_objective"]
        ),
        summed_contribution=(
            channels["L_plus"]["summed_parameter_contribution"]
            + channels["L_star"]["summed_parameter_contribution"]
        ),
    )


def _summed_saved_contributions(
    path: Path,
    chunk_numel: int = 5_000_000,
) -> float:
    """Sum a contribution file in bounded CPU-memory slices."""

    if not path.is_file():
        raise FileNotFoundError(f"Existing contribution file not found: {path}")
    total = 0.0
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        for name in reader.keys():
            tensor_slice = reader.get_slice(name)
            shape = tuple(tensor_slice.get_shape())
            if not shape:
                total += reader.get_tensor(name).double().item()
                continue
            row_numel = 1
            for dimension in shape[1:]:
                row_numel *= dimension
            rows_per_chunk = max(1, chunk_numel // max(1, row_numel))
            trailing = (slice(None),) * (len(shape) - 1)
            for start in range(0, shape[0], rows_per_chunk):
                index = (slice(start, min(start + rows_per_chunk, shape[0])), *trailing)
                chunk = tensor_slice[index]
                total += chunk.sum(dtype=torch.float64).item()
    return total


def _load_completeness_audit_metadata(
    directory: Path,
    expected: Mapping,
) -> dict:
    """Verify that existing tensors belong to the requested attribution run."""

    metadata_path = directory / "attribution_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"completeness_only requires existing metadata: {metadata_path}"
        )
    actual = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key in (
        "path_method", "target_pole", "trait", "interpolation", "k",
        "start_t", "end_t", "path_direction", "attribution_kind",
    ):
        if actual.get(key) != expected.get(key):
            raise ValueError(
                f"Completeness audit metadata mismatch for {key}: {metadata_path}"
            )
    for key in ("base_model_path", "plus_model_path", "star_model_path"):
        if key not in actual:
            raise ValueError(
                f"Completeness audit metadata is missing {key}: {metadata_path}"
            )
        actual_path = str(Path(actual[key]).expanduser().resolve())
        if actual_path != expected[key]:
            raise ValueError(
                f"Completeness audit model mismatch for {key}: {metadata_path}"
            )
    if "objective" not in actual:
        raise ValueError(
            f"Completeness audit metadata is missing objective settings: {metadata_path}"
        )
    actual_objective = dict(actual["objective"])
    if "dataset_path" in actual_objective:
        actual_objective["dataset_path"] = str(
            Path(actual_objective["dataset_path"]).expanduser().resolve()
        )
    if actual_objective != expected["objective"]:
        raise ValueError(
            f"Completeness audit objective/dataset mismatch: {metadata_path}"
        )
    if actual.get("files") != expected.get("files"):
        raise ValueError(
            f"Completeness audit contribution filenames mismatch: {metadata_path}"
        )
    return actual


@torch.no_grad()
def _set_model_from_checkpoint(model, checkpoint_path: str) -> None:
    """Replace live parameters from one checkpoint without retaining a delta."""

    with ModelTensorReader(checkpoint_path) as reader:
        for name, parameter in model.named_parameters():
            if name not in reader.keys:
                raise ValueError(
                    f"Completeness endpoint checkpoint is missing parameter: {name}"
                )
            if reader.shape(name) != tuple(parameter.shape):
                raise ValueError(
                    f"Completeness endpoint shape mismatch for {name}: "
                    f"model {tuple(parameter.shape)}, checkpoint {reader.shape(name)}"
                )
            tensor = reader.read(name, None)
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


def measure_existing_bipolar_completeness(
    model,
    objective: MatchedCompletionObjective,
    model_paths: Mapping[str, str],
    contribution_sums: Mapping[str, float],
    path_method: str,
    target_pole: str,
    batch_size: int,
    input_device: torch.device,
    keep_answer_logits: bool,
) -> dict[str, dict[str, float]]:
    """Audit saved contributions using only the two endpoint model states."""

    if path_method == "piecewise_linear":
        start_path = model_paths["base_model_path"]
        end_path = model_paths[
            "plus_model_path" if target_pole == "high" else "star_model_path"
        ]
    else:
        start_path = model_paths["star_model_path"]
        end_path = model_paths["plus_model_path"]

    _set_model_from_checkpoint(model, start_path)
    start_values = {
        channel: objective.value(
            model=model,
            channel=channel,
            batch_size=batch_size,
            input_device=input_device,
            keep_answer_logits=keep_answer_logits,
        )
        for channel in CHANNELS
    }
    _set_model_from_checkpoint(model, end_path)
    end_values = {
        channel: objective.value(
            model=model,
            channel=channel,
            batch_size=batch_size,
            input_device=input_device,
            keep_answer_logits=keep_answer_logits,
        )
        for channel in CHANNELS
    }
    channels = {
        channel: _completeness_channel_record(
            start_objective=start_values[channel],
            end_objective=end_values[channel],
            summed_contribution=contribution_sums[channel],
        )
        for channel in CHANNELS
    }
    channels["L_contrast"] = _contrast_completeness_record(channels)
    return channels


def _resolve_reuse(
    source_directory: str,
    destination: Path,
    expected_metadata: dict,
) -> tuple[dict[str, Path], dict]:
    """Reuse only an explicitly supplied matching piecewise branch."""
    source = Path(source_directory).expanduser().resolve()
    if source == destination.expanduser().resolve():
        raise ValueError("Reuse source and output directory must differ; sources are read-only")
    files = discover_contribution_files(source)
    metadata_file = source / "attribution_metadata.json"
    source_metadata = (
        json.loads(metadata_file.read_text(encoding="utf-8"))
        if metadata_file.exists() else {}
    )
    expected_stem = (
        f"{expected_metadata['trait']}-{expected_metadata['target_pole']}-"
        f"{expected_metadata['interpolation']}-K{expected_metadata['k']}"
    )
    actual_stem = files["L_plus"].name.removesuffix("-L_plus.safetensors")
    if actual_stem != expected_stem:
        raise ValueError(
            f"Reuse pair is {actual_stem!r}, expected {expected_stem!r}"
        )
    if not source_metadata:
        print(
            f"Reusing legacy contributions from {source}; model and dataset "
            "identity cannot be checked because attribution_metadata.json is absent"
        )
    else:
        for key in (
            "path_method", "target_pole", "trait", "interpolation", "k",
            "start_t", "end_t", "path_direction",
        ):
            if key in source_metadata and source_metadata[key] != expected_metadata[key]:
                raise ValueError(f"Reuse metadata mismatch for {key}: {source}")
        for key in ("base_model_path", "endpoint_model_path"):
            if key in source_metadata:
                actual = str(Path(source_metadata[key]).expanduser().resolve())
                if actual != expected_metadata[key]:
                    raise ValueError(f"Reuse model mismatch for {key}: {source}")
        if "objective" in source_metadata:
            actual_objective = dict(source_metadata["objective"])
            if "dataset_path" in actual_objective:
                actual_objective["dataset_path"] = str(
                    Path(actual_objective["dataset_path"]).expanduser().resolve()
                )
            if actual_objective != expected_metadata["objective"]:
                raise ValueError(f"Reuse objective/dataset configuration mismatch: {source}")
    return files, source_metadata


def _link_or_copy_contribution(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.reuse.tmp")
    temporary.unlink(missing_ok=True)
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def run_bipolar_path_integral(
    base_model_path: str,
    plus_model_path: str,
    star_model_path: str,
    dataset_path: str,
    output_dir: str,
    trait: str,
    path_method: str,
    k: int,
    target_poles: Sequence[str] = ("high", "low"),
    interpolation: str = "linear",
    dtype: str = "bfloat16",
    device_map: str | None = "balanced",
    batch_size: int = 1,
    max_length: int = 1024,
    attribution_split: str = "val",
    val_ratio: float = 0.1,
    seed: int = 42,
    max_samples: int | None = 200,
    filter_refusals: bool = True,
    use_original_prompt: bool = False,
    dialogue_prompt: str | None = None,
    output_name: str | None = None,
    reuse_contribution_dirs: Mapping[str, str] | None = None,
    completeness_only: bool = False,
) -> dict[str, dict[str, Path]]:
    """Attribute piecewise branches or one shared low-to-high full path.

    theta_plus is the high-pole full SFT endpoint; theta_star is the low-pole
    full SFT endpoint. Piecewise-linear keeps independent base-to-high and
    base-to-low attributions selected by target_poles. Endpoint-linear and
    quadratic ignore target_poles and write exactly one common attribution
    on [-1, 1], always targeting high completions against low completions.
    """
    validate_path_method(path_method)
    if not isinstance(completeness_only, bool):
        raise ValueError("completeness_only must be true or false")
    is_piecewise = path_method == "piecewise_linear"
    if is_piecewise:
        if isinstance(target_poles, str) or not target_poles:
            raise ValueError("target_poles must be a non-empty list of 'high'/'low'")
        poles = tuple(target_poles)
        if len(set(poles)) != len(poles) or any(p not in {"high", "low"} for p in poles):
            raise ValueError("target_poles must contain unique 'high'/'low' entries")
        for pole in poles:
            branch_points(k, path_method, pole, interpolation)
    else:
        poles = ("common",)
        full_path_points(k, path_method, interpolation)
        print(f"{path_method}: one common low-to-high attribution; target_poles is unused")
    reuse = dict(reuse_contribution_dirs or {})
    if completeness_only and reuse:
        raise ValueError(
            "completeness_only audits files in this run's output directory; "
            "reuse_contribution_dirs must be null"
        )
    if reuse and not is_piecewise:
        raise ValueError(
            "Only piecewise_linear can reuse old base-to-endpoint contributions; "
            "endpoint_linear and quadratic require fresh common full-path attribution"
        )
    if set(reuse) - set(poles):
        raise ValueError("reuse_contribution_dirs keys must be requested target_poles")
    if any(not isinstance(value, str) or not value for value in reuse.values()):
        raise ValueError("Each reuse_contribution_dirs value must be a non-empty directory path")
    model_paths = {
        "base_model_path": str(Path(base_model_path).expanduser().resolve()),
        "plus_model_path": str(Path(plus_model_path).expanduser().resolve()),
        "star_model_path": str(Path(star_model_path).expanduser().resolve()),
    }
    if model_paths["plus_model_path"] == model_paths["star_model_path"]:
        raise ValueError("plus_model_path and star_model_path must identify different endpoints")
    endpoint_identity = (
        model_paths["plus_model_path"] + "\0" + model_paths["star_model_path"]
    )
    pair_hash = hashlib.sha256(endpoint_identity.encode()).hexdigest()[:10]
    run_name = output_name or f"{trait}-{pair_hash}"
    _validate_run_name(run_name)
    run_directory = Path(output_dir).expanduser() / path_method / run_name
    objective_metadata = _objective_metadata(
        dataset_path, attribution_split, val_ratio, seed, max_samples,
        max_length, filter_refusals, use_original_prompt, dialogue_prompt,
    )
    all_metadata: dict[str, dict] = {}
    output_paths: dict[str, dict[str, Path]] = {}
    resolved_reuse: dict[str, tuple[dict[str, Path], dict]] = {}
    # Resolve every reuse before creating files or loading the model.
    for pole in poles:
        directory = run_directory / pole
        stem = f"{trait}-{pole}-{interpolation}-K{k}"
        _check_output_pair(directory, stem)
        output_paths[pole] = {
            channel: directory / f"{stem}-{channel}.safetensors"
            for channel in CHANNELS
        }
        metadata = {
            **model_paths,
            "endpoint_model_path": model_paths[
                "star_model_path" if pole == "low" else "plus_model_path"
            ],
            "path_method": path_method,
            "target_pole": pole,
            "objective_target_pole": pole if is_piecewise else "high",
            "attribution_kind": "branch" if is_piecewise else "common_full_path",
            "trait": trait,
            "interpolation": interpolation,
            "k": k,
            "quadrature": (
                "endpoint_gradient_times_net_displacement" if k == 1
                else "gauss_legendre_with_true_path_tangent"
            ),
            "start_t": 0.0 if is_piecewise else -1.0,
            "end_t": -1.0 if pole == "low" else 1.0,
            "path_direction": "center_to_endpoint" if is_piecewise else "low_to_high",
            "center_description": (
                "endpoint_mean: (theta_plus + theta_star) / 2"
                if path_method == "endpoint_linear" else "base_model"
            ),
            "attribution_scope": "dense_path_before_topk_selection",
            "objective": objective_metadata,
            "channels": (
                _channel_metadata() if is_piecewise else {
                    "L_plus": "mean token-normalized log P(high-pole completion)",
                    "L_star": "negative mean token-normalized log P(low-pole completion)",
                    "L_contrast": "L_plus + L_star; fixed high-minus-low full-path objective",
                }
            ),
            "files": {channel: path.name for channel, path in output_paths[pole].items()},
            "reused": pole in reuse,
        }
        if completeness_only:
            for contribution_path in output_paths[pole].values():
                if not contribution_path.is_file():
                    raise FileNotFoundError(
                        "completeness_only requires existing contribution file: "
                        f"{contribution_path}"
                    )
            metadata = _load_completeness_audit_metadata(directory, metadata)
        all_metadata[pole] = metadata
        if pole in reuse:
            resolved_reuse[pole] = _resolve_reuse(reuse[pole], directory, metadata)
    # No output leaf may alias ANY reuse source, including the opposite pole.
    reuse_sources = {Path(value).expanduser().resolve() for value in reuse.values()}
    if any((run_directory / pole).resolve() in reuse_sources for pole in poles):
        raise ValueError("An output leaf aliases a reuse source; choose another output_name")
    for pole, (source_files, source_metadata) in resolved_reuse.items():
        for channel in CHANNELS:
            _link_or_copy_contribution(source_files[channel], output_paths[pole][channel])
            print(f"Reused {pole}/{channel} -> {output_paths[pole][channel]}")
        metadata = all_metadata[pole]
        metadata.update({
            "reuse_source_directory": str(Path(reuse[pole]).expanduser().resolve()),
            "reuse_source_metadata_available": bool(source_metadata),
            "reuse_objective_metadata_verified": "objective" in source_metadata,
            "num_examples": source_metadata.get("num_examples"),
        })
        if "completeness" in source_metadata:
            metadata["completeness"] = source_metadata["completeness"]
            metadata["completeness_reused_from_source"] = True
        _write_metadata(run_directory / pole, metadata)
    pending_poles = [pole for pole in poles if pole not in resolved_reuse]
    if not pending_poles:
        print("All requested piecewise branches reused; no model or tokenizer was loaded")
        return output_paths
    tokenizer = AutoTokenizer.from_pretrained(
        model_paths["base_model_path"], trust_remote_code=True, use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_paths["base_model_path"],
        trust_remote_code=True,
        dtype=resolve_dtype(dtype),
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.eval()
    model.requires_grad_(not completeness_only)
    input_device = model.get_input_embeddings().weight.device
    keep_answer_logits = "logits_to_keep" in inspect.signature(model.forward).parameters
    try:
        if completeness_only:
            base_parameters = plus_deltas = star_deltas = None
        else:
            base_parameters, plus_deltas, star_deltas = build_bipolar_parameter_path(
                model, model_paths["plus_model_path"], model_paths["star_model_path"],
            )
        for pole in pending_poles:
            objective_pole = pole if is_piecewise else "high"
            objective = MatchedCompletionObjective(
                tokenizer=tokenizer,
                dataset_path=dataset_path,
                trait=trait,
                target_pole=objective_pole,
                attribution_split=attribution_split,
                val_ratio=val_ratio,
                seed=seed,
                max_samples=max_samples,
                max_length=max_length,
                filter_refusals=filter_refusals,
                use_original_prompt=use_original_prompt,
                dialogue_prompt=dialogue_prompt,
            )
            if objective.num_examples == 0:
                raise ValueError(f"No matched attribution examples available for {trait}/{pole}")
            if completeness_only:
                contribution_sums = {}
                for channel in CHANNELS:
                    output_path = output_paths[pole][channel]
                    contribution_sums[channel] = _summed_saved_contributions(output_path)
                    print(
                        f"Read existing {pole}/{channel} contribution sum from "
                        f"{output_path}"
                    )
                completeness_channels = measure_existing_bipolar_completeness(
                    model=model,
                    objective=objective,
                    model_paths=model_paths,
                    contribution_sums=contribution_sums,
                    path_method=path_method,
                    target_pole=pole,
                    batch_size=batch_size,
                    input_device=input_device,
                    keep_answer_logits=keep_answer_logits,
                )
            else:
                completeness_channels: dict[str, dict[str, float]] = {}
                for channel in CHANNELS:
                    contributions = integrate_bipolar_contributions(
                        model=model,
                        objective=objective,
                        base_parameters=base_parameters,
                        plus_deltas=plus_deltas,
                        star_deltas=star_deltas,
                        channel=channel,
                        path_method=path_method,
                        target_pole=pole,
                        interpolation=interpolation,
                        k=k,
                        batch_size=batch_size,
                        input_device=input_device,
                        keep_answer_logits=keep_answer_logits,
                    )
                    output_path = output_paths[pole][channel]
                    save_contributions(contributions, output_path)
                    print(f"Saved {pole}/{channel} contributions to {output_path}")
                    completeness_channels[channel] = measure_bipolar_completeness(
                        model=model,
                        objective=objective,
                        base_parameters=base_parameters,
                        plus_deltas=plus_deltas,
                        star_deltas=star_deltas,
                        contributions=contributions,
                        channel=channel,
                        path_method=path_method,
                        target_pole=pole,
                        batch_size=batch_size,
                        input_device=input_device,
                        keep_answer_logits=keep_answer_logits,
                    )
                    del contributions
                    gc.collect()
                completeness_channels["L_contrast"] = _contrast_completeness_record(
                    completeness_channels
                )
            for channel in (*CHANNELS, "L_contrast"):
                check = completeness_channels[channel]
                print(
                    f"Completeness {path_method}/{pole}/{channel}: "
                    f"delta_F={check['endpoint_objective_difference']:.8g}, "
                    f"sum_C={check['summed_parameter_contribution']:.8g}, "
                    f"abs_error={check['absolute_error']:.8g}, "
                    f"relative_error={check['relative_error']:.8g}"
                )
            all_metadata[pole]["num_examples"] = objective.num_examples
            all_metadata[pole]["completeness_only_audit"] = completeness_only
            all_metadata[pole]["completeness"] = {
                "definition": "sum_i C_i approximates F(path_end)-F(path_start)",
                "audit_mode": (
                    "existing_contribution_files_endpoint_forward_only"
                    if completeness_only else "computed_immediately_after_attribution"
                ),
                "sample_and_normalization_match": (
                    "Endpoint objectives use the same encoded examples, response-token "
                    "normalization, example averaging, and channel signs as attribution"
                ),
                "numerical_caveat": (
                    "Residual error includes finite-K quadrature error and model/gradient "
                    "precision; parameter contributions are summed in float64"
                ),
                "relative_error_denominator": "max(abs(endpoint_objective_difference), 1e-12)",
                **completeness_channels,
            }
            _write_metadata(run_directory / pole, all_metadata[pole])
            del objective
            gc.collect()
    finally:
        model.zero_grad(set_to_none=True)
        del model
        gc.collect()
    return output_paths


def main() -> None:
    """Load the YAML run interface and execute both path integrals."""

    config = parse_args_yaml("CPVM path-integrated attribution -> *.yaml")
    if "plus_model_path" in config or "star_model_path" in config:
        run_bipolar_path_integral(**config)
    else:
        run_path_integral(**config)


if __name__ == "__main__":
    main()
