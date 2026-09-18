"""GSM8K zero-shot CoT exact-match evaluation with async vLLM replicas."""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import statistics
import traceback

import pyarrow.parquet as pq
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from CPVM.code.utils.arguments import parse_args_yaml
from CPVM.code.utils.batch_evaluation import run_models


PROTOCOL_NAME = "lm-eval gsm8k_cot_zeroshot v3"
PROMPT_TEMPLATE = "Q: {question}\n A: Let's think step by step."
GOLD_ANSWER_RE = re.compile(
    r"####\s*(-?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?)\s*$"
)
NUMBER_PATTERN = r"-?\$?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?"
STRICT_ANSWER_RE = re.compile(
    rf"The answer is\s+({NUMBER_PATTERN})(?=\s*[.]?(?:\s|$))",
    flags=re.IGNORECASE,
)
FLEXIBLE_ANSWER_RE = re.compile(NUMBER_PATTERN)


def get_samples_per_question(config):
    """Return the configured positive number of generations per question."""

    value = config.get("samples_per_question")
    if type(value) is not int or value <= 0:
        raise ValueError("samples_per_question must be a positive integer")
    return value


def _normalize_answer(value):
    """Apply the formatting normalization used for GSM8K exact match."""

    return value.replace("$", "").replace(",", "").strip()


def extract_gold_answer(solution):
    """Extract the official GSM8K final answer following the terminal ####."""

    match = GOLD_ANSWER_RE.search(solution)
    if match is None:
        raise ValueError("GSM8K gold solution has no terminal '#### number'")
    return _normalize_answer(match.group(1))


def extract_answer(text, strict=False):
    """Extract a normalized numeric answer using lm-eval strict/flexible rules."""

    if strict:
        matches = STRICT_ANSWER_RE.findall(text)
    else:
        matches = FLEXIBLE_ANSWER_RE.findall(text)
    return _normalize_answer(matches[-1]) if matches else None


def load_questions(dataset_path):
    """Load the complete local GSM8K main/test split from Parquet."""

    root = Path(dataset_path).expanduser().resolve()
    paths = sorted((root / "main").glob("test-*.parquet"))
    if not paths:
        paths = sorted(root.glob("test-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No main/test-*.parquet files found below {root}")

    questions = []
    for path in paths:
        table = pq.read_table(path, columns=["question", "answer"])
        for row in table.to_pylist():
            question = row.get("question")
            solution = row.get("answer")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Invalid GSM8K question in {path}")
            if not isinstance(solution, str) or not solution.strip():
                raise ValueError(f"Invalid GSM8K answer in {path}")
            questions.append(
                {
                    "id": len(questions),
                    "question": question.strip(),
                    "answer": extract_gold_answer(solution),
                }
            )
    return questions


def build_prompts(questions, tokenizer):
    """Apply the lm-eval zero-shot CoT prompt through the model chat template."""

    prompts = []
    for row in questions:
        content = PROMPT_TEMPLATE.format(question=row["question"])
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def _serialize_completion(output, sample_index, seed, save_responses):
    """Convert one final vLLM completion to the stable result schema."""

    item = {
        "sample_index": sample_index,
        "seed": seed,
        "prediction": extract_answer(output.text),
        "strict_prediction": extract_answer(output.text, strict=True),
        "finish_reason": output.finish_reason,
        "output_tokens": len(output.token_ids),
    }
    if save_responses:
        item["response"] = output.text
    return item


async def _generate_one(
    engine,
    sampling_factory,
    rank,
    task,
    base_seed,
    samples_per_question,
    save_responses,
):
    """Generate one sample with a task-stable seed independent of GPU rank."""

    question_index, sample_index, prompt = task
    seed = base_seed + question_index * samples_per_question + sample_index
    sampling = sampling_factory(seed)
    request_id = f"gsm8k-{rank}-{question_index}-{sample_index}"
    final_output = None
    async for output in engine.generate(prompt, sampling, request_id):
        final_output = output
    if final_output is None or not final_output.finished:
        raise RuntimeError(f"GSM8K request {request_id} ended without a final output")
    outputs = sorted(final_output.outputs, key=lambda sample: sample.index)
    if len(outputs) != 1 or outputs[0].index != 0:
        indices = [sample.index for sample in outputs]
        raise RuntimeError(
            f"GSM8K request {request_id} returned completion indices {indices}; "
            "expected exactly [0]"
        )
    return (
        question_index,
        sample_index,
        _serialize_completion(outputs[0], sample_index, seed, save_responses),
    )


async def _serve_async_requests(
    engine,
    sampling_factory,
    rank,
    task_queue,
    result_queue,
    max_in_flight,
    base_seed,
    samples_per_question,
    save_responses,
):
    """Keep one GPU full and refill it at individual-sample granularity."""

    if type(max_in_flight) is not int or max_in_flight <= 0:
        raise ValueError("work_size must be a positive integer")
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
                pending.add(
                    asyncio.create_task(
                        _generate_one(
                            engine,
                            sampling_factory,
                            rank,
                            task,
                            base_seed,
                            samples_per_question,
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
                outcome for outcome in outcomes if isinstance(outcome, BaseException)
            ]
            if errors:
                raise errors[0]
            for question_index, sample_index, sample in outcomes:
                result_queue.put(("result", question_index, sample_index, sample))
    finally:
        for request in pending:
            request.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


async def _run_async_vllm_worker(rank, devices, task_queue, result_queue, config):
    """Create one single-GPU AsyncLLMEngine and serve sample tasks."""

    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import RequestOutputKind

    print(f"GSM8K worker {rank}: loading model on GPU {devices[rank]}", flush=True)
    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=config["model_path"],
            trust_remote_code=True,
            dtype=config["dtype"],
            tensor_parallel_size=1,
            gpu_memory_utilization=config["gpu_memory_utilization"],
            max_model_len=config["max_model_len"],
            seed=config["sampling_seed"],
        )
    )

    def sampling_factory(seed):
        return SamplingParams(
            n=1,
            temperature=config["temperature"],
            top_p=config["top_p"],
            top_k=config["top_k"],
            min_p=config["min_p"],
            presence_penalty=config["presence_penalty"],
            repetition_penalty=config["repetition_penalty"],
            max_tokens=config["max_new_tokens"],
            seed=seed,
            stop=config.get("stop"),
            output_kind=RequestOutputKind.FINAL_ONLY,
        )

    try:
        print(f"GSM8K worker {rank}: ready", flush=True)
        await _serve_async_requests(
            engine,
            sampling_factory,
            rank,
            task_queue,
            result_queue,
            config["work_size"],
            config["sampling_seed"],
            get_samples_per_question(config),
            config["save_responses"],
        )
    finally:
        engine.shutdown()


def vllm_worker(rank, devices, task_queue, result_queue, config):
    """Run one complete asynchronous vLLM replica on one visible GPU."""

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices[rank]
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        asyncio.run(
            _run_async_vllm_worker(rank, devices, task_queue, result_queue, config)
        )
    except Exception:
        result_queue.put(("error", rank, traceback.format_exc()))


def generate_vllm(prompts, config):
    """Distribute individual generation samples dynamically across GPUs."""

    size = config["data_parallel_size"]
    if type(size) is not int or size <= 0:
        raise ValueError("data_parallel_size must be a positive integer")
    work_size = config["work_size"]
    if type(work_size) is not int or work_size <= 0:
        raise ValueError("work_size must be a positive integer")
    sample_count = get_samples_per_question(config)
    if config["temperature"] == 0 and sample_count != 1:
        raise ValueError(
            "Greedy GSM8K evaluation requires samples_per_question: 1; "
            "use temperature > 0 for Avg@N sampling"
        )

    default_devices = ",".join(str(rank) for rank in range(size))
    devices = [
        device.strip()
        for device in os.environ.get("CUDA_VISIBLE_DEVICES", default_devices).split(",")
        if device.strip()
    ]
    if len(devices) < size:
        raise ValueError(
            f"data_parallel_size={size}, but only {len(devices)} GPU IDs are visible"
        )

    context = mp.get_context("spawn")
    task_queue = context.Queue()
    result_queue = context.Queue()
    tasks = [
        (question_index, sample_index, prompt)
        for question_index, prompt in enumerate(prompts)
        for sample_index in range(sample_count)
    ]
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
    responses = [[None] * sample_count for _ in range(len(prompts))]
    completed_samples = 0
    started_processes = []
    try:
        for process in processes:
            process.start()
            started_processes.append(process)

        with tqdm(
            total=len(tasks),
            desc=f"GSM8K Avg@{sample_count}",
            unit="sample",
        ) as bar:
            while completed_samples < len(tasks):
                try:
                    message = result_queue.get(timeout=30)
                except queue.Empty:
                    failed = [
                        process
                        for process in processes
                        if process.exitcode not in (None, 0)
                    ]
                    if failed:
                        raise RuntimeError("A GSM8K vLLM worker exited unexpectedly")
                    if all(process.exitcode is not None for process in processes):
                        raise RuntimeError(
                            "All GSM8K workers exited before returning every sample"
                        )
                    continue

                if message[0] == "error":
                    raise RuntimeError(
                        f"GSM8K worker {message[1]} failed:\n{message[2]}"
                    )
                _, question_index, sample_index, sample = message
                if responses[question_index][sample_index] is not None:
                    raise RuntimeError(
                        "A GSM8K sample result was returned more than once"
                    )
                responses[question_index][sample_index] = sample
                completed_samples += 1
                bar.update()
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
        raise RuntimeError("A GSM8K vLLM worker failed")
    if any(sample is None for row in responses for sample in row):
        raise RuntimeError("GSM8K generation finished with missing samples")
    return responses


def _percentile(values, fraction):
    """Return an inclusive linearly interpolated percentile."""

    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(results, samples_per_question):
    """Calculate Avg@N, Test@N, parsing, and output-length diagnostics."""

    if type(samples_per_question) is not int or samples_per_question <= 0:
        raise ValueError("samples_per_question must be a positive integer")
    mismatched = [
        index
        for index, row in enumerate(results)
        if len(row["samples"]) != samples_per_question
    ]
    if mismatched:
        raise ValueError(
            "Every GSM8K question must contain exactly "
            f"{samples_per_question} samples; mismatches at {mismatched[:5]}"
        )
    samples = [sample for row in results for sample in row["samples"]]
    if not samples:
        raise ValueError("GSM8K results must contain at least one sample")

    output_tokens = [sample["output_tokens"] for sample in samples]
    correct = sum(sample["correct"] for sample in samples)
    strict_correct = sum(sample["strict_correct"] for sample in samples)
    truncated_samples = [
        sample for sample in samples if sample["finish_reason"] == "length"
    ]
    completed_samples = [
        sample for sample in samples if sample["finish_reason"] != "length"
    ]
    truncated_correct = sum(sample["correct"] for sample in truncated_samples)
    completed_correct = sum(sample["correct"] for sample in completed_samples)
    total_output_tokens = sum(output_tokens)
    truncated_output_tokens = sum(
        sample["output_tokens"] for sample in truncated_samples
    )
    questions_with_a_correct_sample = sum(
        any(sample["correct"] for sample in row["samples"]) for row in results
    )

    return {
        "num_questions": len(results),
        "samples_per_question": samples_per_question,
        "total_samples": len(samples),
        "correct_samples": correct,
        "average_accuracy": correct / len(samples),
        "test_at_n": questions_with_a_correct_sample / len(results),
        "strict_correct_samples": strict_correct,
        "strict_accuracy": strict_correct / len(samples),
        "parse_rate": sum(sample["prediction"] is not None for sample in samples)
        / len(samples),
        "strict_parse_rate": sum(
            sample["strict_prediction"] is not None for sample in samples
        )
        / len(samples),
        "total_output_tokens": total_output_tokens,
        "average_output_tokens": statistics.fmean(output_tokens),
        "median_output_tokens": statistics.median(output_tokens),
        "p90_output_tokens": _percentile(output_tokens, 0.90),
        "p95_output_tokens": _percentile(output_tokens, 0.95),
        "max_output_tokens": max(output_tokens),
        "length_truncated": len(truncated_samples),
        "length_truncated_rate": len(truncated_samples) / len(samples),
        "length_truncated_correct": truncated_correct,
        "length_truncated_accuracy": (
            truncated_correct / len(truncated_samples) if truncated_samples else None
        ),
        "non_truncated_samples": len(completed_samples),
        "non_truncated_correct": completed_correct,
        "non_truncated_accuracy": (
            completed_correct / len(completed_samples) if completed_samples else None
        ),
        "strict_completion_accuracy": completed_correct / len(samples),
        "length_truncated_output_tokens": truncated_output_tokens,
        "length_truncated_token_rate": (
            truncated_output_tokens / total_output_tokens
            if total_output_tokens
            else 0.0
        ),
        "sample_index_accuracy": [
            sum(row["samples"][index]["correct"] for row in results) / len(results)
            for index in range(samples_per_question)
        ],
    }


def result_path(config):
    """Build a result name containing protocol, Avg@N, and DP size."""

    sample_count = get_samples_per_question(config)
    source = Path(config["model_path"])
    model_name = source.name
    if model_name.startswith("checkpoint-"):
        model_name = f"{source.parent.name}-{model_name}"
    run_tag = f"avg{sample_count}-dp{config['data_parallel_size']}"
    test_name = config["test_name"]
    if not test_name.endswith(run_tag):
        test_name = f"{test_name}-{run_tag}"
    return Path(config["output_path"]) / f"gsm8k-{model_name}-{test_name}.json"


def run(config):
    """Run one model on GSM8K main/test with the recorded protocol."""

    sample_count = get_samples_per_question(config)
    if config["temperature"] == 0 and sample_count != 1:
        raise ValueError(
            "Greedy GSM8K evaluation requires samples_per_question: 1; "
            "use temperature > 0 for Avg@N sampling"
        )
    questions = load_questions(config["dataset_path"])
    expected_questions = config.get("expected_num_questions")
    if expected_questions is not None and len(questions) != expected_questions:
        raise ValueError(
            f"Expected {expected_questions} GSM8K questions, found {len(questions)}"
        )

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

    decoding = "greedy" if config["temperature"] == 0 else "sampling"
    print(
        f"GSM8K main/test: {len(questions)} questions x {sample_count} "
        f"samples; protocol={PROTOCOL_NAME}; decoding={decoding}; "
        f"max prompt={max_prompt_tokens} tokens",
        flush=True,
    )
    responses = generate_vllm(prompts, config)

    predictions = []
    for question, samples in zip(questions, responses):
        for sample in samples:
            sample["correct"] = sample["prediction"] == question["answer"]
            sample["strict_correct"] = sample["strict_prediction"] == question["answer"]
        predictions.append(
            {
                "id": question["id"],
                "question": question["question"],
                "answer": question["answer"],
                "correct_samples": sum(sample["correct"] for sample in samples),
                "samples": samples,
            }
        )

    summary = summarize(predictions, sample_count)
    report = {
        "benchmark": "GSM8K",
        "metric": f"average_exact_match@{sample_count}",
        "protocol": {
            "name": PROTOCOL_NAME,
            "dataset_config": "main",
            "split": "test",
            "num_fewshot": 0,
            "prompt_template": PROMPT_TEMPLATE,
            "apply_chat_template": True,
            "decoding": decoding,
            "answer_extraction": {
                "primary": "last numeric expression (lm-eval flexible-extract)",
                "strict": "last 'The answer is <number>' expression",
                "normalization": "remove commas and dollar signs, then string exact match",
            },
        },
        "model_path": config["model_path"],
        "settings": config,
        "summary": summary,
        "predictions": predictions,
    }
    path = result_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(
        f"GSM8K Avg@{sample_count} exact match: "
        f"{summary['correct_samples']}/{summary['total_samples']} = "
        f"{summary['average_accuracy']:.2%}"
    )
    print(
        f"GSM8K Test@{sample_count} (at least one correct): {summary['test_at_n']:.2%}"
    )
    print(
        f"Strict-format Avg@{sample_count}: "
        f"{summary['strict_correct_samples']}/{summary['total_samples']} = "
        f"{summary['strict_accuracy']:.2%}"
    )
    print(
        f"Parse rate: {summary['parse_rate']:.2%}; "
        f"strict parse rate: {summary['strict_parse_rate']:.2%}"
    )
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
    """Load YAML and run all configured models serially."""

    config = parse_args_yaml("GSM8K lm-eval zero-shot CoT async-vLLM evaluation")
    run_models(config, run, "GSM8K")


if __name__ == "__main__":
    main()
