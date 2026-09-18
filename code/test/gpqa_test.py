"""Qwen-compatible GPQA-Diamond avg@N evaluation with vLLM data parallelism."""

import asyncio
import csv
from collections import defaultdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import random
import re
import statistics
import traceback

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from CPVM.code.utils.arguments import parse_args_yaml
from CPVM.code.utils.batch_evaluation import run_models


LETTERS = "ABCD"
BOXED_ANSWER = re.compile(r"\\boxed\s*\{\s*([A-Da-d])\s*\}")
PROMPT = """{question}
Answer Choices: (A) {choice_a} (B) {choice_b} (C) {choice_c} (D) {choice_d}
On the last line of your response, place your final answer letter within \\boxed{{}} (e.g., \\boxed{{A}}), ensuring only the letter is inside."""


def get_samples_per_question(config):
    """Return the configured positive number of samples drawn per question."""

    value = config.get("samples_per_question")
    if type(value) is not int or value <= 0:
        raise ValueError("samples_per_question must be a positive integer")
    return value


def load_questions(dataset_path, seed):
    """Load GPQA-Diamond and apply the GPQA repository's seeded choice shuffle."""

    csv_path = Path(dataset_path) / "gpqa_diamond.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    rng = random.Random(seed)
    questions = []
    for row in rows:
        choices = [
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
            row["Correct Answer"],
        ]
        rng.shuffle(choices)
        questions.append(
            {
                "record_id": row["Record ID"],
                "domain": row["High-level domain"],
                "question": row["Question"].strip(),
                "choices": [choice.strip() for choice in choices],
                "answer": LETTERS[choices.index(row["Correct Answer"])],
            }
        )
    return questions


def build_prompts(questions, tokenizer):
    """Format the Qwen GPQA prompt and apply the model's native chat template."""

    prompts = []
    for row in questions:
        content = PROMPT.format(
            question=row["question"],
            choice_a=row["choices"][0],
            choice_b=row["choices"][1],
            choice_c=row["choices"][2],
            choice_d=row["choices"][3],
        )
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def extract_answer(text):
    """Return the final boxed A--D letter, or None when the format is invalid."""

    matches = BOXED_ANSWER.findall(text)
    return matches[-1].upper() if matches else None


def _serialize_completion_outputs(outputs, save_responses):
    """Convert final vLLM completion outputs to the existing sample schema."""

    samples = []
    for sample in outputs:
        item = {
            "prediction": extract_answer(sample.text),
            "finish_reason": sample.finish_reason,
            "output_tokens": len(sample.token_ids),
        }
        if save_responses:
            item["response"] = sample.text
        samples.append(item)
    return samples


async def _generate_one(engine, sampling, rank, index, prompt, save_responses):
    """Submit one question and return its final avg@N samples."""

    final_output = None
    request_id = f"gpqa-{rank}-{index}"
    async for output in engine.generate(prompt, sampling, request_id):
        final_output = output
    if final_output is None or not final_output.finished:
        raise RuntimeError(f"GPQA request {request_id} ended without a final output")
    outputs = sorted(final_output.outputs, key=lambda sample: sample.index)
    output_indices = [sample.index for sample in outputs]
    if output_indices != list(range(sampling.n)):
        raise RuntimeError(
            f"GPQA request {request_id} returned completion indices "
            f"{output_indices}; expected 0..{sampling.n - 1}"
        )
    return index, _serialize_completion_outputs(outputs, save_responses)


async def _serve_async_requests(
    engine,
    sampling,
    rank,
    task_queue,
    result_queue,
    max_in_flight,
    save_responses,
):
    """Continuously refill one GPU engine as individual questions finish."""

    if type(max_in_flight) is not int or max_in_flight <= 0:
        raise ValueError("max_in_flight must be a positive integer")
    pending = set()
    exhausted = False
    try:
        while pending or not exhausted:
            while not exhausted and len(pending) < max_in_flight:
                try:
                    task = await asyncio.to_thread(task_queue.get, True, 1)
                except queue.Empty:
                    continue
                if task is None:
                    exhausted = True
                    break
                index, prompt = task
                pending.add(
                    asyncio.create_task(
                        _generate_one(
                            engine,
                            sampling,
                            rank,
                            index,
                            prompt,
                            save_responses,
                        )
                    )
                )

            if not pending:
                continue
            completed, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            outcomes = await asyncio.gather(*completed, return_exceptions=True)
            errors = [
                outcome
                for outcome in outcomes
                if isinstance(outcome, BaseException)
            ]
            if errors:
                raise errors[0]
            for index, samples in outcomes:
                result_queue.put(("result", [index], [samples]))
    finally:
        for request in pending:
            request.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


async def _run_async_vllm_worker(
    rank, devices, task_queue, result_queue, config
):
    """Create and serve one complete AsyncLLMEngine replica on one GPU."""

    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import RequestOutputKind

    print(f"GPQA worker {rank}: loading model on GPU {devices[rank]}", flush=True)
    engine_args = AsyncEngineArgs(
        model=config["model_path"],
        trust_remote_code=True,
        dtype=config["dtype"],
        tensor_parallel_size=1,
        gpu_memory_utilization=config["gpu_memory_utilization"],
        max_model_len=config["max_model_len"],
        seed=config["sampling_seed"],
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    try:
        sampling = SamplingParams(
            n=config["samples_per_question"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            top_k=config["top_k"],
            min_p=config["min_p"],
            presence_penalty=config["presence_penalty"],
            repetition_penalty=config["repetition_penalty"],
            max_tokens=config["max_new_tokens"],
            seed=config["sampling_seed"],
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        print(f"GPQA worker {rank}: ready", flush=True)
        await _serve_async_requests(
            engine,
            sampling,
            rank,
            task_queue,
            result_queue,
            config["work_size"],
            config["save_responses"],
        )
    finally:
        engine.shutdown()


def vllm_worker(rank, devices, task_queue, result_queue, config):
    """Run one continuously replenished single-GPU vLLM replica."""

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices[rank]
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        asyncio.run(
            _run_async_vllm_worker(
                rank, devices, task_queue, result_queue, config
            )
        )
    except Exception:
        result_queue.put(("error", rank, traceback.format_exc()))


def generate_vllm(prompts, config):
    """Continuously distribute prompts over independent async GPU replicas."""

    size = config["data_parallel_size"]
    default_devices = ",".join(str(rank) for rank in range(size))
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", default_devices).split(",")
    context = mp.get_context("spawn")
    task_queue = context.Queue()
    result_queue = context.Queue()

    work_size = config["work_size"]
    if type(work_size) is not int or work_size <= 0:
        raise ValueError("work_size must be a positive integer")
    tasks = list(enumerate(prompts))
    for task in tasks:
        task_queue.put(task)
    for _ in range(size):
        task_queue.put(None)

    processes = [
        context.Process(
            target=vllm_worker,
            args=(rank, devices, task_queue, result_queue, config),
        )
        for rank in range(size)
    ]
    responses = [None] * len(prompts)
    completed_tasks = 0
    sample_count = get_samples_per_question(config)
    started_processes = []
    try:
        for process in processes:
            process.start()
            started_processes.append(process)

        with tqdm(
            total=len(prompts),
            desc=f"GPQA-Diamond avg@{sample_count}",
            unit="question",
        ) as bar:
            while completed_tasks < len(tasks):
                try:
                    message = result_queue.get(timeout=30)
                except queue.Empty:
                    failed = [
                        process for process in processes if process.exitcode
                    ]
                    if failed:
                        raise RuntimeError(
                            "A vLLM data-parallel worker exited unexpectedly"
                        )
                    continue

                if message[0] == "error":
                    raise RuntimeError(
                        f"GPQA worker {message[1]} failed:\n{message[2]}"
                    )

                _, indices, batches = message
                for index, samples in zip(indices, batches):
                    responses[index] = samples
                completed_tasks += 1
                bar.update(len(indices))
    except BaseException:
        for process in started_processes:
            if process.is_alive():
                process.terminate()
        for process in started_processes:
            process.join()
        raise

    for process in processes:
        process.join()
    if any(process.exitcode for process in processes):
        raise RuntimeError("A vLLM data-parallel worker failed")
    return responses


def summarize(results, samples_per_question):
    """Calculate Qwen's average sample accuracy and useful audit statistics."""

    if type(samples_per_question) is not int or samples_per_question <= 0:
        raise ValueError("samples_per_question must be a positive integer")
    mismatched = [
        index
        for index, row in enumerate(results)
        if len(row["samples"]) != samples_per_question
    ]
    if mismatched:
        raise ValueError(
            "Every GPQA question must contain exactly "
            f"{samples_per_question} samples; mismatches at indices {mismatched[:5]}"
        )
    samples = [sample for row in results for sample in row["samples"]]
    output_tokens = [sample["output_tokens"] for sample in samples]
    grouped = defaultdict(list)
    for row in results:
        grouped[row["domain"]].extend(row["samples"])

    correct = sum(sample["correct"] for sample in samples)
    truncated_samples = [
        sample for sample in samples if sample["finish_reason"] == "length"
    ]
    completed_samples = [
        sample for sample in samples if sample["finish_reason"] != "length"
    ]
    length_truncated = len(truncated_samples)
    length_truncated_correct = sum(
        sample["correct"] for sample in truncated_samples
    )
    completed_correct = sum(sample["correct"] for sample in completed_samples)
    length_truncated_output_tokens = sum(
        sample["output_tokens"] for sample in truncated_samples
    )
    total_output_tokens = sum(output_tokens)
    return {
        "num_questions": len(results),
        "samples_per_question": samples_per_question,
        "total_samples": len(samples),
        "correct_samples": correct,
        "average_accuracy": correct / len(samples),
        "parse_rate": sum(sample["prediction"] is not None for sample in samples)
        / len(samples),
        "total_output_tokens": total_output_tokens,
        "average_output_tokens": statistics.fmean(output_tokens),
        "median_output_tokens": statistics.median(output_tokens),
        "p90_output_tokens": statistics.quantiles(
            output_tokens, n=10, method="inclusive"
        )[8],
        "p95_output_tokens": statistics.quantiles(
            output_tokens, n=20, method="inclusive"
        )[18],
        "max_output_tokens": max(output_tokens),
        "length_truncated": length_truncated,
        "length_truncated_rate": length_truncated / len(samples),
        "length_truncated_correct": length_truncated_correct,
        "length_truncated_accuracy": (
            length_truncated_correct / length_truncated
            if length_truncated else None
        ),
        "non_truncated_samples": len(completed_samples),
        "non_truncated_correct": completed_correct,
        "non_truncated_accuracy": (
            completed_correct / len(completed_samples)
            if completed_samples else None
        ),
        "strict_completion_accuracy": completed_correct / len(samples),
        "length_truncated_output_tokens": length_truncated_output_tokens,
        "length_truncated_token_rate": (
            length_truncated_output_tokens / total_output_tokens
            if total_output_tokens else 0.0
        ),
        "sample_index_accuracy": [
            sum(row["samples"][index]["correct"] for row in results) / len(results)
            for index in range(samples_per_question)
        ],
        "domain_accuracy": {
            domain: sum(sample["correct"] for sample in domain_samples)
            / len(domain_samples)
            for domain, domain_samples in sorted(grouped.items())
        },
    }


def result_path(config):
    """Build the JSON report path from model and test names."""

    sample_count = get_samples_per_question(config)
    source = Path(config["model_path"])
    model_name = source.name
    if model_name.startswith("checkpoint-"):
        model_name = f"{source.parent.name}-{model_name}"
    run_tag = f"avg{sample_count}-dp{config['data_parallel_size']}"
    test_name = config["test_name"]
    if not test_name.endswith(run_tag):
        test_name = f"{test_name}-{run_tag}"
    return Path(config["output_path"]) / f"gpqa-diamond-{model_name}-{test_name}.json"


def run(config):
    """Run the complete official-style GPQA-Diamond avg@N evaluation."""

    sample_count = get_samples_per_question(config)
    questions = load_questions(config["dataset_path"], config["choice_seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"], trust_remote_code=True, use_fast=False
    )
    prompts = build_prompts(questions, tokenizer)
    prompt_lengths = [
        len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts
    ]
    max_prompt_tokens = max(prompt_lengths)
    if max_prompt_tokens + config["max_new_tokens"] > config["max_model_len"]:
        raise ValueError("max_model_len is too small for the longest prompt and output")

    print(
        f"GPQA-Diamond: {len(questions)} questions x "
        f"{sample_count} samples; max prompt={max_prompt_tokens} tokens"
    )
    responses = generate_vllm(prompts, config)

    results = []
    for question, samples in zip(questions, responses):
        for sample in samples:
            sample["correct"] = sample["prediction"] == question["answer"]
        results.append(
            {
                "record_id": question["record_id"],
                "domain": question["domain"],
                "answer": question["answer"],
                "correct_samples": sum(sample["correct"] for sample in samples),
                "samples": samples,
            }
        )

    summary = summarize(results, sample_count)
    report = {
        "benchmark": "GPQA-Diamond",
        "metric": f"average_accuracy@{sample_count}",
        "model_path": config["model_path"],
        "settings": config,
        "summary": summary,
        "predictions": results,
    }
    path = result_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(
        f"GPQA-Diamond avg@{sample_count}: "
        f"{summary['correct_samples']}/{summary['total_samples']} = "
        f"{summary['average_accuracy']:.2%}"
    )
    print(f"Parse rate: {summary['parse_rate']:.2%}")
    print(
        "Output tokens: "
        f"mean={summary['average_output_tokens']:.1f}, "
        f"median={summary['median_output_tokens']:.1f}, "
        f"p90={summary['p90_output_tokens']:.1f}, "
        f"p95={summary['p95_output_tokens']:.1f}, "
        f"max={summary['max_output_tokens']}"
    )
    print(
        "Length-truncated samples: "
        f"{summary['length_truncated']}/{summary['total_samples']} = "
        f"{summary['length_truncated_rate']:.2%}"
    )
    truncated_accuracy = summary["length_truncated_accuracy"]
    non_truncated_accuracy = summary["non_truncated_accuracy"]
    print(
        "Length-truncated accuracy: "
        + (
            f"{summary['length_truncated_correct']}/"
            f"{summary['length_truncated']} = {truncated_accuracy:.2%}"
            if truncated_accuracy is not None else "N/A (no truncated samples)"
        )
    )
    print(
        "Non-truncated accuracy: "
        + (
            f"{summary['non_truncated_correct']}/"
            f"{summary['non_truncated_samples']} = {non_truncated_accuracy:.2%}"
            if non_truncated_accuracy is not None else "N/A (no completed samples)"
        )
    )
    print(
        "Strict completion accuracy: "
        f"{summary['non_truncated_correct']}/{summary['total_samples']} = "
        f"{summary['strict_completion_accuracy']:.2%}"
    )
    print(
        "Length-truncated token share: "
        f"{summary['length_truncated_output_tokens']}/"
        f"{summary['total_output_tokens']} = "
        f"{summary['length_truncated_token_rate']:.2%}"
    )
    print(f"Saved to: {path}")
    return report


def main():
    """Load YAML and start the GPQA evaluation."""

    config = parse_args_yaml("GPQA-Diamond official-style avg@N evaluation")
    run_models(config, run, "GPQA-Diamond")


if __name__ == "__main__":
    main()
