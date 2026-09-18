"""Create MergeLM DaRE checkpoints, with optional legacy Task Vector output."""

from __future__ import annotations

import gc
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from CPVM.code.utils.arguments import parse_args_yaml


MERGELM_ROOT = Path(__file__).resolve().parents[1] / "ref" / "MergeLM"
sys.path.insert(0, str(MERGELM_ROOT))

from model_merging_methods.mask_weights_utils import mask_model_weights  # noqa: E402
from model_merging_methods.task_vector import TaskVector  # noqa: E402
from utils.utils import set_random_seed  # noqa: E402


DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def load_cpu_model(model_path: str, dtype: str):
    """Load one checkpoint on CPU in the arithmetic dtype used for merging."""

    return AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=DTYPES[dtype],
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )


def copy_parameters(model, parameters: dict[str, torch.Tensor]) -> None:
    """Copy a MergeLM parameter dictionary into an existing model."""

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(parameters[name])


def save_model(
    model, tokenizer, output_path: Path, save_dtype: str, max_shard_size: str
) -> None:
    """Cast one completed merge and save it as a standard HF checkpoint."""

    model.to(DTYPES[save_dtype])
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    tokenizer.save_pretrained(output_path)
    (output_path / ".merge_complete").touch()


def checkpoint_exists(output_path: Path) -> bool:
    """Return whether this script finished saving a checkpoint."""

    return (output_path / ".merge_complete").exists()


def dare_output_path(output_dir: str, retain_percent: float, seed: int) -> Path:
    """Build an output name that records nominal retention and random seed."""

    return Path(output_dir) / (
        f"extraversion-high-dare-retain{retain_percent:g}pct-seed{seed}"
    )


def task_vector_output_path(output_dir: str, alpha: float) -> Path:
    """Build the single-model Task Vector output name."""

    return Path(output_dir) / f"extraversion-high-task-vector-alpha{alpha:g}"


def merge_dare_models(config: dict, base_model, tokenizer) -> list[Path]:
    """Create all random Drop-And-REscale task-vector checkpoints."""

    outputs = []
    for retain_percent in config["retain_percentages"]:
        retain_rate = retain_percent / 100
        drop_rate = 1 - retain_rate
        output_path = dare_output_path(
            config["output_dir"], retain_percent, config["seed"]
        )
        if config.get("skip_existing", True) and checkpoint_exists(output_path):
            print(f"Skipping existing DaRE model: {output_path}", flush=True)
            outputs.append(output_path)
            continue

        print(
            f"\n=== DaRE retain={retain_percent:g}% drop={drop_rate:g} "
            f"rescale={1 / retain_rate:g} seed={config['seed']} ===",
            flush=True,
        )

        finetuned_model = load_cpu_model(
            config["finetuned_model_path"], config["merge_dtype"]
        )
        set_random_seed(config["seed"])
        merged_parameters = mask_model_weights(
            finetuned_model=finetuned_model,
            pretrained_model=base_model,
            exclude_param_names_regex=[],
            weight_format="delta_weight",
            weight_mask_rate=drop_rate,
            use_weight_rescale=True,
            mask_strategy="random",
        )
        copy_parameters(finetuned_model, merged_parameters)
        del merged_parameters
        save_model(
            finetuned_model,
            tokenizer,
            output_path,
            config["save_dtype"],
            config["max_shard_size"],
        )
        del finetuned_model
        gc.collect()
        print(f"Saved DaRE model to {output_path}", flush=True)
        outputs.append(output_path)
    return outputs


def merge_task_vector(config: dict, base_model, tokenizer) -> Path:
    """Create theta_base + alpha * (theta_finetuned - theta_base)."""

    alpha = float(config["task_vector_alpha"])
    output_path = task_vector_output_path(config["output_dir"], alpha)
    if config.get("skip_existing", True) and checkpoint_exists(output_path):
        print(f"Skipping existing Task Vector model: {output_path}", flush=True)
        return output_path

    print(f"\n=== Task Vector alpha={alpha:g} ===", flush=True)

    finetuned_model = load_cpu_model(
        config["finetuned_model_path"], config["merge_dtype"]
    )
    task_vector = TaskVector(
        pretrained_model=base_model,
        finetuned_model=finetuned_model,
        exclude_param_names_regex=[],
    )
    merged_parameters = task_vector.combine_with_pretrained_model(
        pretrained_model=base_model,
        scaling_coefficient=alpha,
    )
    copy_parameters(finetuned_model, merged_parameters)
    del merged_parameters, task_vector
    save_model(
        finetuned_model,
        tokenizer,
        output_path,
        config["save_dtype"],
        config["max_shard_size"],
    )
    del finetuned_model
    gc.collect()
    print(f"Saved Task Vector model to {output_path}", flush=True)
    return output_path


def run(config: dict) -> list[Path]:
    """Load the shared base once and generate the configured merge outputs."""

    print(f"MergeLM source: {MERGELM_ROOT}", flush=True)
    print(
        f"Merge arithmetic={config['merge_dtype']}; saved dtype={config['save_dtype']}",
        flush=True,
    )
    base_model = load_cpu_model(config["base_model_path"], config["merge_dtype"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["finetuned_model_path"], trust_remote_code=True, use_fast=False
    )
    outputs = merge_dare_models(config, base_model, tokenizer)
    if config.get("include_task_vector", False):
        outputs.append(merge_task_vector(config, base_model, tokenizer))
    print("\nGenerated models:", flush=True)
    for output_path in outputs:
        print(output_path, flush=True)
    return outputs


def main() -> None:
    """Load the YAML interface and run the configured merge."""

    run(parse_args_yaml("MergeLM DaRE reproduction"))


if __name__ == "__main__":
    main()
