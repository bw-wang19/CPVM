"""Evaluate Top-k models against the matching path-integral objective."""

import gc
import inspect
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from CPVM.code.path_integral import MatchedCompletionObjective, response_log_probabilities
from CPVM.code.test.trait_test import TRAITTester
from CPVM.code.topk_curve_merge import open_curve_merge
from CPVM.code.topk_merge import resolve_dtype


LOSS_DEFINITIONS = {
    "L_plus": "mean(target_response_nll)",
    "L_star": "-mean(counter_response_nll)",
    "L_contrast": "mean(target_response_nll - counter_response_nll)",
}


def _mean_validation_loss(model, tokenizer, config):
    """Return the negative attribution objective on matched validation pairs.

    Every response is token-normalized, then valid pairs receive equal weight.
    Target and counter statistics always use exactly the same valid pairs.
    """
    channel = config["contribution_channel"]
    if channel not in LOSS_DEFINITIONS:
        raise ValueError(f"Unsupported contribution_channel: {channel}")
    target_level = config.get("target_pole") or config.get("level", "high")
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    # Reuse attribution pairing, prompt choice, tokenization, EOS, and masking.
    # The validation curve always uses val, even if attribution used train.
    objective = MatchedCompletionObjective(
        tokenizer=tokenizer,
        dataset_path=config["dataset_path"],
        trait=config["trait"],
        target_pole=target_level,
        attribution_split="val",
        val_ratio=float(config.get("val_ratio", 0.1)),
        seed=int(config.get("seed", 42)),
        max_samples=config.get("max_samples"),
        max_length=int(config.get("max_length", 1024)),
        filter_refusals=config.get("filter_refusals", True),
        use_original_prompt=config.get("use_original_prompt", False),
        dialogue_prompt=config.get("dialogue_prompt"),
    )
    valid_indices = [
        index for index in range(objective.num_examples)
        if all(
            any(label != -100 for label in objective.encoded[side][index]["labels"][1:])
            for side in ("L_plus", "L_star")
        )
    ]
    if not valid_indices:
        raise ValueError("No matched validation pairs have response tokens on both sides after truncation")
    skipped_samples = objective.num_examples - len(valid_indices)
    print(
        f"Validation: {config['trait']}/{target_level}, channel={channel}, "
        f"matched pairs={len(valid_indices)}, skipped pairs={skipped_samples}"
    )
    input_device = model.get_input_embeddings().weight.device
    batch_size = int(config.get("batch_size", 8))
    if batch_size <= 0:
        raise ValueError("validation batch_size must be positive")
    keep_answer_logits = "logits_to_keep" in inspect.signature(model.forward).parameters
    nll_sums = {"L_plus": 0.0, "L_star": 0.0}
    token_counts = {"L_plus": 0, "L_star": 0}
    with torch.inference_mode():
        for start in tqdm(range(0, len(valid_indices), batch_size), desc="Matched validation loss", unit="batch"):
            indices = valid_indices[start:start + batch_size]
            for side in ("L_plus", "L_star"):
                features = [objective.encoded[side][index] for index in indices]
                batch = objective.collator(features)
                labels = batch.pop("labels")
                shifted_labels = labels[:, 1:]
                inputs = {key: value.to(input_device) for key, value in batch.items()}
                if keep_answer_logits:
                    kept_positions = shifted_labels.ne(-100).any(dim=0).nonzero(as_tuple=True)[0]
                    outputs = model(
                        **inputs, use_cache=False, logits_to_keep=kept_positions.tolist(),
                    )
                    aligned_labels = shifted_labels.index_select(1, kept_positions).to(outputs.logits.device)
                    aligned_logits = outputs.logits
                else:
                    outputs = model(**inputs, use_cache=False)
                    aligned_logits = outputs.logits[:, :-1, :]
                    aligned_labels = shifted_labels.to(outputs.logits.device)
                # This is the same per-response mean log probability used by
                # MatchedCompletionObjective.backward, with its sign reversed.
                per_example_nll = -response_log_probabilities(aligned_logits, aligned_labels)
                nll_sums[side] += per_example_nll.double().sum().item()
                token_counts[side] += aligned_labels.ne(-100).sum().item()
                del outputs, aligned_logits, aligned_labels, per_example_nll, labels, batch, inputs
    target_nll = nll_sums["L_plus"] / len(valid_indices)
    counter_nll = nll_sums["L_star"] / len(valid_indices)
    # Attribution maximizes logp(target), -logp(counter), or their sum.
    # Negating this loss in the plot recovers the corresponding objective.
    losses = {
        "L_plus": target_nll,
        "L_star": -counter_nll,
        "L_contrast": target_nll - counter_nll,
    }
    return {
        "validation_loss": losses[channel],
        "contribution_channel": channel,
        "target_level": target_level,
        "validation_loss_definition": LOSS_DEFINITIONS[channel],
        "target_nll": target_nll,
        "counter_nll": counter_nll,
        "validation_samples": len(valid_indices),
        "validation_response_tokens": token_counts["L_plus"] + token_counts["L_star"],
        "validation_target_response_tokens": token_counts["L_plus"],
        "validation_counter_response_tokens": token_counts["L_star"],
        "validation_skipped_samples": skipped_samples,
    }


def evaluate_curve_models(
    merge_config: dict,
    validation_config: dict,
    trait_config: dict,
    output_path: str,
) -> pd.DataFrame:
    """Merge and evaluate 0, 10, ..., 100% using one resident model.

    Only the original base/SFT checkpoints are read. Exact thresholds are
    prepared together; without rescaling, ascending k points add only newly
    selected SFT weights. No intermediate or endpoint model is saved.
    TRAIT receives only the main trait's questions, and scores stay in 0..100
    units. Validation follows the selected matched-completion objective.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    trait = validation_config["trait"]
    channel = validation_config["contribution_channel"]
    base_path = str(merge_config["base_model_path"])
    sft_path = str(merge_config["finetuned_model_path"])
    tester_config = dict(trait_config)
    tester_config.pop("model_paths", None)
    tester_config.pop("model_path", None)
    # Set before TRAITTester.load_questions: other traits are never sent to
    # inference, even if trait_test.yaml requests all personalities.
    tester_config.update(
        use_vllm=False,
        dtype="auto",
        device_map=None,
        data_parallel_size=1,
        tensor_parallel_size=1,
        adapter_path=None,
        personalities=[trait],
        output_path=str(destination.parent / "trait"),
    )
    model_dtype = merge_config.get("merge_dtype", "auto")
    if model_dtype == "auto":
        model_dtype = validation_config.get("dtype", "auto")
    records = []
    model = base_tokenizer = sft_tokenizer = tokenizer = tester = apply_merge = None
    try:
        base_tokenizer = AutoTokenizer.from_pretrained(
            base_path, trust_remote_code=True, use_fast=False,
        )
        sft_tokenizer = AutoTokenizer.from_pretrained(
            sft_path, trust_remote_code=True, use_fast=False,
        )
        for tokenizer in (base_tokenizer, sft_tokenizer):
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
        print(f"Loading one model for the in-memory Top-k sweep: {base_path}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_path, trust_remote_code=True,
            dtype=resolve_dtype(model_dtype),
            device_map=validation_config.get("device_map", "auto"),
            low_cpu_mem_usage=True,
            offload_state_dict=False,
        ).eval()
        model.requires_grad_(False)
        print(f"TRAIT question filter: {trait} only", flush=True)
        curve_percents = list(range(0, 101, 10))
        with open_curve_merge(
            model, merge_config, k_percents=curve_percents
        ) as apply_merge:
            for k_percent in curve_percents:
                print(f"\nMerging and evaluating k={k_percent}% in memory", flush=True)
                merge_metrics = apply_merge(k_percent)
                tokenizer = base_tokenizer if k_percent == 0 else sft_tokenizer
                loss_metrics = _mean_validation_loss(model, tokenizer, validation_config)
                model_name = f"topk-{k_percent}pct-{channel}-in-memory"
                settings = dict(tester_config)
                settings.update(
                    model_path=base_path if k_percent == 0 else sft_path,
                    preloaded_model=model,
                    preloaded_tokenizer=tokenizer,
                    result_model_name=model_name,
                    test_name=f"{channel} Top-k {k_percent}% / {trait}",
                )
                tester = TRAITTester(**settings)
                del settings
                trait_record = tester.run_test()
                record = {
                    "k_percent": k_percent,
                    "model_name": model_name,
                    "base_model_path": base_path,
                    "finetuned_model_path": sft_path,
                    "contribution_dir": merge_config.get("contribution_dir"),
                    "l_plus_contribution_path": merge_config.get("l_plus_contribution_path"),
                    "l_star_contribution_path": merge_config.get("l_star_contribution_path"),
                    "main_trait": trait,
                    "trait_score": float(trait_record["scores"][trait]),
                    "trait_num_questions": trait_record["num_questions"],
                    "use_weight_rescale": merge_config.get("use_weight_rescale", False),
                    **merge_metrics,
                    **loss_metrics,
                }
                records.append(record)
                pd.DataFrame(records).to_csv(destination, index=False)
                print(
                    f"k={k_percent}% | {trait}={record['trait_score']:.4f} | "
                    f"{channel} validation loss={record['validation_loss']:.6f} | "
                    f"TRAIT questions={record['trait_num_questions']} | "
                    f"validation pairs={record['validation_samples']}",
                    flush=True,
                )
                # TRAIT holds references to the same model; discard them before
                # the next merge, while keeping our single model resident.
                tester = None
    finally:
        tester = apply_merge = model = tokenizer = base_tokenizer = sft_tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    frame = pd.DataFrame(records)
    frame.to_csv(destination, index=False)
    print("\n" + frame.to_string(index=False))
    print(f"Curve metrics saved to {destination}")
    return frame
