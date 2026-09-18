#!/usr/bin/env python
"""Supervised fine-tuning on the official MedMCQA train/validation splits.

The optimization target is an assistant response whose optional explanation is
followed by a canonical final ``\\boxed{A}``--``\\boxed{D}`` line.  The system
and user prompt tokens are masked from the causal-language-model loss.

The local test parquet is deliberately never loaded: its ``cop`` labels are
all -1 and it must remain an evaluation-only split.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Optional

# Match the existing CPVM two-GPU training setup.  Users may override either
# value in the environment before launching torchrun.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from CPVM.code.utils.arguments import LoraArguments, ModelArguments
from CPVM.code.utils.medmcqa import answer_letter, boxed_answer, format_prompt
from CPVM.code.utils.model import load_model_tokenizer


logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = (
    "id",
    "question",
    "opa",
    "opb",
    "opc",
    "opd",
    "cop",
    "choice_type",
    "exp",
)


@dataclass
class MedMCQADataArguments:
    """Local MedMCQA preprocessing controls."""

    data_path: str = field(
        metadata={"help": "Local MedMCQA root containing data/*.parquet"}
    )
    max_length: int = field(default=1024)
    include_explanation: bool = field(
        default=True,
        metadata={
            "help": "Train on exp followed by a boxed letter; missing exp uses the letter only"
        },
    )
    drop_overlength: bool = field(
        default=True,
        metadata={
            "help": "Drop examples that exceed max_length instead of truncating the final answer"
        },
    )
    dataset_num_proc: int = field(default=8)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    preview_samples: int = field(default=2)


def build_assistant_target(row: dict[str, Any], include_explanation: bool) -> str:
    """Build a response that always ends in the canonical boxed gold letter."""

    final_answer = boxed_answer(answer_letter(row.get("cop")))
    explanation = row.get("exp")
    if include_explanation and isinstance(explanation, str) and explanation.strip():
        return f"{explanation.strip()}\n\n{final_answer}"
    return final_answer


def _as_token_ids(value: Any, *, source: str) -> list[int]:
    """Normalize the list-like result returned by apply_chat_template."""

    if hasattr(value, "keys") and "input_ids" in value:
        value = value["input_ids"]

    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        not isinstance(value, list)
        or (value and isinstance(value[0], list))
        or any(type(token_id) is not int for token_id in value)
    ):
        raise TypeError(f"Tokenizer returned invalid token IDs for {source}")
    return value


def tokenize_medmcqa_example(
    row: dict[str, Any],
    *,
    tokenizer,
    max_length: int,
    include_explanation: bool,
    drop_overlength: bool,
) -> dict[str, Any]:
    """Tokenize one example and supervise only the assistant turn.

    Prompt-only and complete conversations are rendered independently through
    the native chat template.  Their shared prefix determines the exact label
    boundary, including model-specific assistant-role control tokens.
    """

    if type(max_length) is not int or max_length <= 0:
        raise ValueError(f"max_length must be a positive integer, got {max_length!r}")

    prompt = format_prompt(row)
    target = build_assistant_target(row, include_explanation)
    prompt_messages = [{"role": "user", "content": prompt}]
    full_messages = prompt_messages + [{"role": "assistant", "content": target}]

    prompt_ids = _as_token_ids(
        tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        ),
        source="prompt",
    )
    full_ids = _as_token_ids(
        tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
        ),
        source="full conversation",
    )

    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "The tokenizer chat template does not give the prompt-only and "
            "full conversation a common assistant-prefix; assistant-only loss "
            "cannot be constructed safely."
        )
    supervised_ids = full_ids[len(prompt_ids) :]
    if not supervised_ids:
        raise ValueError("The chat template produced no assistant target tokens")

    sequence_length = len(full_ids)
    if sequence_length > max_length:
        if not drop_overlength:
            row_id = row.get("id", "<unknown>")
            raise ValueError(
                f"MedMCQA example {row_id!r} needs {sequence_length} tokens, "
                f"exceeding max_length={max_length}; enable drop_overlength or "
                "increase max_length."
            )
        # Never truncate: doing so could remove the final boxed answer while
        # leaving a seemingly valid explanation-only training target.
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "_keep": False,
            "_sequence_length": sequence_length,
        }

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * sequence_length,
        "labels": [-100] * len(prompt_ids) + supervised_ids,
        "_keep": True,
        "_sequence_length": sequence_length,
    }


def _split_parquet_files(data_path: str, split: str) -> list[str]:
    """Resolve one official local split without consulting a remote hub."""

    root = Path(data_path).expanduser().resolve()
    paths = sorted((root / "data").glob(f"{split}-*.parquet"))
    if not paths:
        paths = sorted(root.glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No {split}-*.parquet files found in {root / 'data'} or {root}"
        )
    return [str(path) for path in paths]


def load_official_splits(data_path: str) -> DatasetDict:
    """Load train and validation only; never resolve or load MedMCQA test."""

    data_files = {
        "train": _split_parquet_files(data_path, "train"),
        "validation": _split_parquet_files(data_path, "validation"),
    }
    dataset = load_dataset("parquet", data_files=data_files)

    for split in ("train", "validation"):
        missing = sorted(set(REQUIRED_COLUMNS) - set(dataset[split].column_names))
        if missing:
            raise ValueError(f"MedMCQA {split} split is missing columns: {missing}")

    train_ids = set(dataset["train"]["id"])
    validation_ids = set(dataset["validation"]["id"])
    overlap = train_ids & validation_ids
    if overlap:
        examples = sorted(str(value) for value in overlap)[:5]
        raise ValueError(
            "MedMCQA train/validation IDs overlap; refusing a leaky split. "
            f"Examples: {examples}"
        )
    return DatasetDict(
        train=dataset["train"],
        validation=dataset["validation"],
    )


def _limit_split(dataset: Dataset, limit: Optional[int], name: str) -> Dataset:
    if limit is None:
        return dataset
    if type(limit) is not int or limit <= 0:
        raise ValueError(f"max_{name}_samples must be a positive integer or null")
    return dataset.select(range(min(limit, len(dataset))))


def _tokenize_split(
    dataset: Dataset,
    *,
    split: str,
    tokenizer,
    data_args: MedMCQADataArguments,
) -> tuple[Dataset, dict[str, int]]:
    """Tokenize one split, drop explicitly marked long rows, and report counts."""

    mapped = dataset.map(
        tokenize_medmcqa_example,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": data_args.max_length,
            "include_explanation": data_args.include_explanation,
            "drop_overlength": data_args.drop_overlength,
        },
        remove_columns=dataset.column_names,
        num_proc=data_args.dataset_num_proc,
        desc=f"Tokenizing MedMCQA {split}",
    )
    keep_flags = mapped["_keep"]
    kept_positions = [index for index, keep in enumerate(keep_flags) if keep]
    dropped = len(mapped) - len(kept_positions)
    tokenized = mapped.select(kept_positions)
    sequence_lengths = tokenized["_sequence_length"]
    tokenized = tokenized.remove_columns(["_keep", "_sequence_length"])
    if not tokenized:
        raise ValueError(f"No MedMCQA {split} examples remain after preprocessing")

    summary = {
        "raw_examples": len(dataset),
        "kept_examples": len(tokenized),
        "dropped_overlength": dropped,
        "max_sequence_length": max(sequence_lengths),
    }
    logger.info(
        "MedMCQA %s: raw=%d, kept=%d, dropped_overlength=%d, max_tokens=%d",
        split,
        summary["raw_examples"],
        summary["kept_examples"],
        summary["dropped_overlength"],
        summary["max_sequence_length"],
    )
    return tokenized, summary


def prepare_medmcqa_datasets(
    data_args: MedMCQADataArguments,
    tokenizer,
) -> tuple[DatasetDict, dict[str, Any], list[dict[str, str]]]:
    """Load official splits and construct assistant-only training records."""

    if type(data_args.dataset_num_proc) is not int or data_args.dataset_num_proc <= 0:
        raise ValueError("dataset_num_proc must be a positive integer")
    if type(data_args.preview_samples) is not int or data_args.preview_samples < 0:
        raise ValueError("preview_samples must be a non-negative integer")

    raw = load_official_splits(data_args.data_path)
    raw_train = _limit_split(raw["train"], data_args.max_train_samples, "train")
    raw_validation = _limit_split(
        raw["validation"], data_args.max_eval_samples, "eval"
    )
    previews = [
        {
            "prompt": format_prompt(raw_train[index]),
            "assistant_target": build_assistant_target(
                raw_train[index], data_args.include_explanation
            ),
        }
        for index in range(min(data_args.preview_samples, len(raw_train)))
    ]
    train, train_summary = _tokenize_split(
        raw_train,
        split="train",
        tokenizer=tokenizer,
        data_args=data_args,
    )
    validation, validation_summary = _tokenize_split(
        raw_validation,
        split="validation",
        tokenizer=tokenizer,
        data_args=data_args,
    )
    summary: dict[str, Any] = {
        "data_path": str(Path(data_args.data_path).expanduser().resolve()),
        "loaded_splits": ["train", "validation"],
        "test_split_loaded": False,
        "include_explanation": data_args.include_explanation,
        "drop_overlength": data_args.drop_overlength,
        "max_length": data_args.max_length,
        "train": train_summary,
        "validation": validation_summary,
    }
    return DatasetDict(train=train, validation=validation), summary, previews


def _parse_arguments():
    parser = HfArgumentParser(
        (ModelArguments, MedMCQADataArguments, LoraArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith((".json", ".yaml", ".yml")):
        return parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    return parser.parse_args_into_dataclasses()


def _write_data_summary(output_dir: str, summary: dict[str, Any]) -> None:
    path = Path(output_dir) / "medmcqa_data_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    model_args, data_args, lora_args, training_args = _parse_arguments()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=(
            logging.INFO
            if training_args.local_rank in (-1, 0)
            else logging.WARNING
        ),
    )
    set_seed(training_args.seed)

    if training_args.resume_from_checkpoint and model_args.resume_wandb_run_id:
        os.environ["WANDB_RESUME"] = "must"
        os.environ["WANDB_RUN_ID"] = model_args.resume_wandb_run_id

    model_name = Path(model_args.model_name_or_path.rstrip("/")).name
    target_tag = "explanation-boxed" if data_args.include_explanation else "boxed"
    run_name = (
        f"{model_name}-sft-{model_args.finetune_type}-medmcqa-{target_tag}-"
        f"ep{training_args.num_train_epochs:g}"
    )
    output_subdir = "adapters" if model_args.finetune_type == "lora" else "full"
    training_args.run_name = run_name
    training_args.output_dir = str(
        Path(training_args.output_dir) / output_subdir / run_name
    )
    logger.info("Loading model and tokenizer from %s", model_args.model_name_or_path)
    model, tokenizer = load_model_tokenizer(model_args, lora_args)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    with training_args.main_process_first(desc="Processing MedMCQA"):
        dataset, data_summary, previews = prepare_medmcqa_datasets(data_args, tokenizer)

    if training_args.should_log:
        logger.info("Output directory: %s", training_args.output_dir)
        logger.info("MedMCQA data summary: %s", data_summary)
        for index, preview in enumerate(previews):
            logger.info(
                "MedMCQA preview %d prompt:\n%s\nAssistant target:\n%s",
                index,
                preview["prompt"],
                preview["assistant_target"],
            )
    if training_args.should_save:
        _write_data_summary(training_args.output_dir, data_summary)

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    else:
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        logger.info(
            "Full fine-tuning parameters: trainable=%d, total=%d (%.2f%%)",
            trainable,
            total,
            100.0 * trainable / total,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8,
        ),
    )

    if training_args.do_train:
        train_result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        train_metrics = train_result.metrics
        train_metrics["train_examples"] = len(dataset["train"])
        train_metrics["train_dropped_overlength"] = data_summary["train"][
            "dropped_overlength"
        ]
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()
        trainer.save_model()
        if training_args.should_save:
            tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        eval_metrics = trainer.evaluate()
        eval_metrics["eval_examples"] = len(dataset["validation"])
        eval_metrics["eval_dropped_overlength"] = data_summary["validation"][
            "dropped_overlength"
        ]
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    logger.info("MedMCQA SFT finished. Output: %s", training_args.output_dir)


if __name__ == "__main__":
    main()
