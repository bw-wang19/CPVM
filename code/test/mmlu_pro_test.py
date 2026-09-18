"""Minimal MMLU-Pro 5-shot CoT evaluation."""

from collections import defaultdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import re

from datasets import load_dataset
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from CPVM.code.utils.arguments import parse_args_yaml
from CPVM.code.utils.batch_evaluation import run_models


CHOICES = "ABCDEFGHIJ"
INSTRUCTION = (
    "The following are multiple choice questions (with answers) about {category}. "
    'Think step by step and then finish your answer with "the answer is (X)" '
    "where X is the correct letter choice."
)


def normalize_example(example):
    """Remove placeholder N/A options from one dataset row."""
    row = dict(example)
    row["options"] = [option for option in row["options"] if option != "N/A"]
    return row


def format_question(example, include_answer):
    """Format one demonstration or one unanswered question."""
    lines = ["Question:", example["question"], "Options:"]
    lines += [
        f"{CHOICES[index]}. {option}"
        for index, option in enumerate(example["options"])
    ]
    if include_answer:
        lines.append(
            example["cot_content"].replace(
                "A: Let's think step by step.",
                "Answer: Let's think step by step.",
                1,
            )
        )
    else:
        lines.append("Answer: Let's think step by step.")
    return "\n".join(lines)


def build_raw_prompt(example, demonstrations):
    """Build the official category-specific few-shot prompt."""
    prompt = INSTRUCTION.format(category=example["category"]) + "\n"
    for demonstration in demonstrations:
        prompt += format_question(demonstration, True) + "\n\n"
    return prompt + format_question(example, False)


def extract_answer(text):
    """Extract an option letter with the official three-stage parser."""
    match = re.search(r"answer is \(?([A-J])\)?", text)
    if match:
        return match.group(1), "answer_is"
    match = re.search(r".*[aA]nswer:\s*([A-J])", text, re.DOTALL)
    if match:
        return match.group(1), "answer_colon"
    match = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    if match:
        return match.group(0), "last_letter"
    return None, "unparsed"


def load_data(config):
    """Load validation demonstrations and selected test questions."""
    dataset = load_dataset(config["dataset_path"])
    validation = [normalize_example(row) for row in dataset["validation"]]
    questions = [normalize_example(row) for row in dataset["test"]]

    demonstrations = defaultdict(list)
    for row in validation:
        demonstrations[row["category"]].append(row)

    if config.get("categories"):
        questions = [
            row for row in questions if row["category"] in config["categories"]
        ]
    questions.sort(key=lambda row: row["category"])
    if config.get("max_samples") is not None:
        questions = questions[: config["max_samples"]]
    return questions, demonstrations


def build_fitting_prompt(example, demonstrations, tokenizer, config):
    """Drop demonstrations until prompt and output fit the context window."""
    demonstrations = demonstrations[example["category"]][: config["n_shot"]]
    while True:
        prompt = build_raw_prompt(example, demonstrations)
        if (
            len(tokenizer.encode(prompt))
            < config["max_length"] - config["max_new_tokens"]
            or not demonstrations
        ):
            return prompt, len(demonstrations)
        demonstrations = demonstrations[:-1]


def vllm_worker(rank, prompts, config, devices, shared_outputs):
    """Run one prompt shard on one independent GPU replica."""
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[rank]
    from vllm import LLM, SamplingParams

    model = LLM(
        model=config["model_path"],
        trust_remote_code=True,
        dtype=config["dtype"],
        tensor_parallel_size=1,
        gpu_memory_utilization=config["gpu_memory_utilization"],
        max_model_len=config["max_length"],
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=config["max_new_tokens"],
        stop=["Question:"],
    )
    responses = []
    starts = range(0, len(prompts), config["batch_size"])
    for start in tqdm(starts, desc=f"MMLU-Pro GPU {rank}", unit="batch"):
        outputs = model.generate(
            prompts[start : start + config["batch_size"]],
            sampling,
            use_tqdm=False,
        )
        responses += [output.outputs[0].text for output in outputs]
    shared_outputs[rank] = responses


def generate_vllm(prompts, config):
    """Split prompts over independent single-GPU vLLM processes."""
    size = config["data_parallel_size"]
    default_devices = ",".join(str(rank) for rank in range(size))
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", default_devices).split(",")
    context = mp.get_context("spawn")

    with context.Manager() as manager:
        shared_outputs = manager.dict()
        processes = []
        for rank in range(size):
            process = context.Process(
                target=vllm_worker,
                args=(
                    rank,
                    prompts[rank::size],
                    config,
                    devices,
                    shared_outputs,
                ),
            )
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        if any(process.exitcode for process in processes):
            raise RuntimeError("A vLLM data-parallel worker failed.")
        parts = dict(shared_outputs)

    responses = [None] * len(prompts)
    for rank in range(size):
        responses[rank::size] = parts[rank]
    return responses


def generate_transformers(prompts, tokenizer, config):
    """Generate with the ordinary Transformers backend."""
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        trust_remote_code=True,
        dtype=config["dtype"],
        device_map=config["device_map"],
    ).eval()
    device = model.get_input_embeddings().weight.device
    responses = []
    starts = range(0, len(prompts), config["batch_size"])
    for start in tqdm(starts, desc="MMLU-Pro", unit="batch"):
        batch = tokenizer(
            prompts[start : start + config["batch_size"]],
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            output_ids = model.generate(
                **batch,
                do_sample=False,
                max_new_tokens=config["max_new_tokens"],
                stop_strings=["Question:"],
                tokenizer=tokenizer,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_ids = output_ids[:, batch["input_ids"].shape[1] :]
        responses += tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
    return responses


def summarize_results(results):
    """Calculate overall and per-category scores."""
    grouped = defaultdict(list)
    for result in results:
        grouped[result["category"]].append(result)
    category_scores = {}
    for category, rows in sorted(grouped.items()):
        correct = sum(row["correct"] for row in rows)
        category_scores[category] = {
            "total": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows),
        }
    total = len(results)
    summary = {
        "num_questions": total,
        "correct": sum(row["correct"] for row in results),
        "accuracy": sum(row["correct"] for row in results) / total,
        "strict_accuracy": sum(row["strict_correct"] for row in results) / total,
        "parse_rate": sum(not row["parse_failed"] for row in results) / total,
    }
    return summary, category_scores


def result_path(config):
    """Build the JSON result path."""
    source = Path(config["model_path"])
    model_name = source.name
    if model_name.startswith("checkpoint-"):
        model_name = f"{source.parent.name}-{model_name}"
    return Path(config["output_path"]) / (
        f"mmlu-pro-{model_name}-{config['test_name']}.json"
    )


def run(config):
    """Run inference, score predictions and save one report."""
    questions, demonstrations = load_data(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"],
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    prompt_data = [
        build_fitting_prompt(row, demonstrations, tokenizer, config)
        for row in tqdm(questions, desc="Building prompts")
    ]
    prompts = [prompt for prompt, _ in prompt_data]
    if config["use_vllm"]:
        responses = generate_vllm(prompts, config)
    else:
        responses = generate_transformers(prompts, tokenizer, config)

    rng = random.Random(config["seed"])
    results = []
    for row, (_, shots), response in zip(questions, prompt_data, responses):
        prediction, method = extract_answer(response)
        scored_prediction = prediction or rng.choice(CHOICES[: len(row["options"])])
        result = {
            "question_id": row["question_id"],
            "category": row["category"],
            "answer": row["answer"],
            "prediction": prediction,
            "scored_prediction": scored_prediction,
            "correct": scored_prediction == row["answer"],
            "strict_correct": prediction == row["answer"],
            "parse_failed": prediction is None,
            "extraction_method": method,
            "num_shots": shots,
        }
        if config["save_responses"]:
            result["model_response"] = response
        results.append(result)

    summary, category_scores = summarize_results(results)
    report = {
        "benchmark": "MMLU-Pro",
        "model_path": config["model_path"],
        "settings": config,
        "summary": summary,
        "category_scores": category_scores,
        "predictions": results,
    }
    output_path = result_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(f"Accuracy: {summary['accuracy']:.2%}")
    print(f"Strict accuracy: {summary['strict_accuracy']:.2%}")
    print(f"Parse rate: {summary['parse_rate']:.2%}")
    print(f"Saved to: {output_path}")
    return report


def main():
    """Load YAML and start the evaluation."""
    config = parse_args_yaml("MMLU-Pro evaluation config -> *.yaml")
    run_models(config, run, "MMLU-Pro")


if __name__ == "__main__":
    main()
