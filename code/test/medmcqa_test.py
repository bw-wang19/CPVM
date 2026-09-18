"""MedMCQA validation evaluation with asynchronous vLLM replicas."""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import statistics
import time
import traceback

import pyarrow.parquet as pq
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from CPVM.code.utils.arguments import parse_args_yaml
from CPVM.code.utils.batch_evaluation import run_models
from CPVM.code.utils.medmcqa import (
    PROMPT_TEMPLATE,
    answer_letter,
    extract_boxed_answer,
    format_prompt,
)


PROTOCOL_NAME = "CPVM MedMCQA validation zero-shot generative MCQA v1"


def get_samples_per_question(config):
    """Return the configured positive number of generations per question."""

    value = config.get("samples_per_question")
    if type(value) is not int or value <= 0:
        raise ValueError("samples_per_question must be a positive integer")
    return value


def _sample_seed(base_seed, question_index, sample_index):
    """Map a question/sample pair to a unique, scheduling-independent seed."""

    values = (base_seed, question_index, sample_index)
    if any(type(value) is not int for value in values):
        raise ValueError("sampling_seed and sample coordinates must be integers")
    if any(value < 0 for value in values):
        raise ValueError("sampling_seed and sample coordinates must be nonnegative")
    diagonal = question_index + sample_index
    return base_seed + diagonal * (diagonal + 1) // 2 + sample_index


def load_questions(dataset_path, split="validation"):
    """Load a labelled local MedMCQA Parquet split.

    The public MedMCQA test Parquet has ``cop=-1`` for every row, so it cannot
    be scored locally.  The official lm-eval task also uses validation as its
    test split.
    """

    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a nonempty string")
    split = split.strip()
    root = Path(dataset_path).expanduser().resolve()
    if root.is_file():
        paths = [root]
    else:
        paths = sorted((root / "data").glob(f"{split}-*.parquet"))
        if not paths:
            paths = sorted(root.glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No {split}-*.parquet files found below {root}"
        )

    required = (
        "id",
        "question",
        "opa",
        "opb",
        "opc",
        "opd",
        "cop",
        "choice_type",
        "subject_name",
        "topic_name",
    )
    questions = []
    seen_ids = set()
    for path in paths:
        table = pq.read_table(path, columns=list(required))
        for row_index, row in enumerate(table.to_pylist()):
            location = f"{path} row {row_index}"
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError(f"Invalid MedMCQA id at {location}")
            identifier = identifier.strip()
            if identifier in seen_ids:
                raise ValueError(f"Duplicate MedMCQA id {identifier!r}")
            seen_ids.add(identifier)

            text_fields = ("question", "opa", "opb", "opc", "opd")
            for name in text_fields:
                if not isinstance(row.get(name), str) or not row[name].strip():
                    raise ValueError(f"Invalid MedMCQA {name!r} at {location}")

            correct_index = row.get("cop")
            if type(correct_index) is not int or not 0 <= correct_index < 4:
                hint = (
                    " The public test split is unlabelled (cop=-1); use "
                    "split: validation for local evaluation."
                    if correct_index == -1 else ""
                )
                raise ValueError(
                    f"Invalid or unavailable MedMCQA cop={correct_index!r} at "
                    f"{location}.{hint}"
                )

            questions.append(
                {
                    "id": identifier,
                    "question": row["question"].strip(),
                    "opa": row["opa"].strip(),
                    "opb": row["opb"].strip(),
                    "opc": row["opc"].strip(),
                    "opd": row["opd"].strip(),
                    "choices": [
                        row[name].strip()
                        for name in ("opa", "opb", "opc", "opd")
                    ],
                    "answer": answer_letter(correct_index),
                    "choice_type": str(row.get("choice_type") or "Unknown").strip(),
                    "subject": str(row.get("subject_name") or "Unknown").strip(),
                    "topic": (
                        row["topic_name"].strip()
                        if isinstance(row.get("topic_name"), str)
                        and row["topic_name"].strip()
                        else None
                    ),
                }
            )
    if not questions:
        raise ValueError(f"MedMCQA split {split!r} contains no questions")
    return questions


def build_prompts(questions, tokenizer):
    """Apply the lm-eval-style question layout through the chat template."""

    prompts = []
    for row in questions:
        content = format_prompt(row)
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def extract_answer(text):
    """Return the final strictly boxed A--D answer, or None if unparseable."""

    return extract_boxed_answer(text)


def _serialize_completion(output, sample_index, seed, save_responses):
    item = {
        "sample_index": sample_index,
        "seed": seed,
        "prediction": extract_answer(output.text),
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
    save_responses,
):
    """Generate one sample so work can be balanced at sample granularity."""

    question_index, sample_index, prompt = task
    seed = _sample_seed(base_seed, question_index, sample_index)
    sampling = sampling_factory(seed)
    request_id = f"medmcqa-{rank}-{question_index}-{sample_index}"
    final_output = None
    async for output in engine.generate(prompt, sampling, request_id):
        final_output = output
    if final_output is None or not final_output.finished:
        raise RuntimeError(f"MedMCQA request {request_id} ended without a final output")
    outputs = sorted(final_output.outputs, key=lambda sample: sample.index)
    if len(outputs) != 1 or outputs[0].index != 0:
        indices = [sample.index for sample in outputs]
        raise RuntimeError(
            f"MedMCQA request {request_id} returned completion indices {indices}; "
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
    save_responses,
):
    """Keep one engine full and refill it whenever an individual sample ends."""

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
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import RequestOutputKind

    print(f"MedMCQA worker {rank}: loading model on GPU {devices[rank]}", flush=True)
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
        print(f"MedMCQA worker {rank}: ready", flush=True)
        await _serve_async_requests(
            engine,
            sampling_factory,
            rank,
            task_queue,
            result_queue,
            config["work_size"],
            config["sampling_seed"],
            config["save_responses"],
        )
    finally:
        engine.shutdown()


def vllm_worker(rank, devices, task_queue, result_queue, config):
    """Run one independent single-GPU AsyncLLMEngine replica."""

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices[rank]
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        asyncio.run(
            _run_async_vllm_worker(rank, devices, task_queue, result_queue, config)
        )
    except Exception:
        result_queue.put(("error", rank, traceback.format_exc()))


def _terminate_processes(processes, grace_seconds=10):
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + grace_seconds
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
    for process in processes:
        process.join()


def generate_vllm(prompts, config):
    """Dynamically distribute individual generations over GPU replicas."""

    size = config["data_parallel_size"]
    if type(size) is not int or size <= 0:
        raise ValueError("data_parallel_size must be a positive integer")
    work_size = config["work_size"]
    if type(work_size) is not int or work_size <= 0:
        raise ValueError("work_size must be a positive integer")
    sample_count = get_samples_per_question(config)
    if config["temperature"] == 0 and sample_count != 1:
        raise ValueError(
            "Greedy MedMCQA evaluation requires samples_per_question: 1; "
            "use temperature > 0 for Avg@N sampling"
        )

    inactivity_timeout = config.get("inactivity_timeout_seconds", 900)
    shutdown_timeout = config.get("worker_shutdown_timeout_seconds", 120)
    for name, value in (
        ("inactivity_timeout_seconds", inactivity_timeout),
        ("worker_shutdown_timeout_seconds", shutdown_timeout),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{name} must be a positive number")

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
    responses = [[None] * sample_count for _ in prompts]
    completed_samples = 0
    started_processes = []
    last_progress = time.monotonic()
    try:
        for process in processes:
            process.start()
            started_processes.append(process)

        with tqdm(
            total=len(tasks),
            desc=f"MedMCQA Avg@{sample_count}",
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
                        raise RuntimeError("A MedMCQA vLLM worker exited unexpectedly")
                    if all(process.exitcode is not None for process in processes):
                        raise RuntimeError(
                            "All MedMCQA workers exited before returning every sample"
                        )
                    if time.monotonic() - last_progress >= inactivity_timeout:
                        raise RuntimeError(
                            "MedMCQA made no generation progress for "
                            f"{inactivity_timeout:g} seconds; terminating workers"
                        )
                    continue

                if message[0] == "error":
                    raise RuntimeError(
                        f"MedMCQA worker {message[1]} failed:\n{message[2]}"
                    )
                _, question_index, sample_index, sample = message
                if responses[question_index][sample_index] is not None:
                    raise RuntimeError("A MedMCQA sample result was returned more than once")
                responses[question_index][sample_index] = sample
                completed_samples += 1
                last_progress = time.monotonic()
                bar.update()
    except BaseException:
        _terminate_processes(started_processes)
        raise

    shutdown_deadline = time.monotonic() + shutdown_timeout
    for process in processes:
        process.join(timeout=max(0.0, shutdown_deadline - time.monotonic()))
    stuck_processes = [process for process in processes if process.is_alive()]
    if stuck_processes:
        _terminate_processes(stuck_processes)
        raise RuntimeError(
            "MedMCQA workers did not shut down within "
            f"{shutdown_timeout:g} seconds"
        )
    if any(process.exitcode for process in processes):
        raise RuntimeError("A MedMCQA vLLM worker failed")
    if any(sample is None for row in responses for sample in row):
        raise RuntimeError("MedMCQA generation finished with missing samples")
    return responses


def _percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _group_metrics(results, group_key, samples_per_question):
    grouped = {}
    for row in results:
        grouped.setdefault(str(row[group_key]), []).append(row)
    output = {}
    for label, rows in sorted(grouped.items()):
        samples = [sample for row in rows for sample in row["samples"]]
        output[label] = {
            "num_questions": len(rows),
            "total_samples": len(samples),
            "samples_per_question": samples_per_question,
            "average_accuracy": sum(sample["correct"] for sample in samples)
            / len(samples),
            "test_at_n": sum(
                any(sample["correct"] for sample in row["samples"]) for row in rows
            )
            / len(rows),
            "parse_rate": sum(
                sample["prediction"] is not None for sample in samples
            )
            / len(samples),
        }
    return output


def summarize(results, samples_per_question):
    """Calculate accuracy, parsing, output-length, and truncation diagnostics."""

    if type(samples_per_question) is not int or samples_per_question <= 0:
        raise ValueError("samples_per_question must be a positive integer")
    mismatched = [
        index
        for index, row in enumerate(results)
        if len(row["samples"]) != samples_per_question
    ]
    if mismatched:
        raise ValueError(
            "Every MedMCQA question must contain exactly "
            f"{samples_per_question} samples; mismatches at {mismatched[:5]}"
        )
    samples = [sample for row in results for sample in row["samples"]]
    if not samples:
        raise ValueError("MedMCQA results must contain at least one sample")

    output_tokens = [sample["output_tokens"] for sample in samples]
    correct = sum(sample["correct"] for sample in samples)
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
        "parse_rate": sum(sample["prediction"] is not None for sample in samples)
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
            truncated_output_tokens / total_output_tokens if total_output_tokens else 0.0
        ),
        "sample_index_accuracy": [
            sum(row["samples"][index]["correct"] for row in results) / len(results)
            for index in range(samples_per_question)
        ],
        "by_subject": _group_metrics(results, "subject", samples_per_question),
        "by_choice_type": _group_metrics(
            results, "choice_type", samples_per_question
        ),
    }


def result_path(config):
    sample_count = get_samples_per_question(config)
    source = Path(config["model_path"])
    model_name = source.name
    if model_name.startswith("checkpoint-"):
        model_name = f"{source.parent.name}-{model_name}"
    run_tag = f"avg{sample_count}-dp{config['data_parallel_size']}"
    test_name = config["test_name"]
    if not test_name.endswith(run_tag):
        test_name = f"{test_name}-{run_tag}"
    return Path(config["output_path"]) / f"medmcqa-{model_name}-{test_name}.json"


def run(config):
    """Evaluate one model on the labelled MedMCQA validation split."""

    sample_count = get_samples_per_question(config)
    if config["temperature"] == 0 and sample_count != 1:
        raise ValueError(
            "Greedy MedMCQA evaluation requires samples_per_question: 1; "
            "use temperature > 0 for Avg@N sampling"
        )
    split = config.get("split", "validation")
    questions = load_questions(config["dataset_path"], split)
    expected_questions = config.get("expected_num_questions")
    if expected_questions is not None and len(questions) != expected_questions:
        raise ValueError(
            f"Expected {expected_questions} MedMCQA questions, found {len(questions)}"
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
        f"MedMCQA {split}: {len(questions)} questions x {sample_count} samples; "
        f"protocol={PROTOCOL_NAME}; decoding={decoding}; "
        f"max prompt={max_prompt_tokens} tokens",
        flush=True,
    )
    responses = generate_vllm(prompts, config)

    predictions = []
    for question, samples in zip(questions, responses):
        for sample in samples:
            sample["correct"] = sample["prediction"] == question["answer"]
        predictions.append(
            {
                "id": question["id"],
                "question": question["question"],
                "choices": question["choices"],
                "answer": question["answer"],
                "choice_type": question["choice_type"],
                "subject": question["subject"],
                "topic": question["topic"],
                "correct_samples": sum(sample["correct"] for sample in samples),
                "samples": samples,
            }
        )

    summary = summarize(predictions, sample_count)
    report = {
        "benchmark": "MedMCQA",
        "metric": f"average_accuracy@{sample_count}",
        "protocol": {
            "name": PROTOCOL_NAME,
            "dataset": "openlifescienceai/medmcqa",
            "split": split,
            "num_fewshot": 0,
            "prompt_template": PROMPT_TEMPLATE,
            "apply_chat_template": True,
            "decoding": decoding,
            "reference_format": (
                "lm-eval MedMCQA Question/Choices/A-D/Answer layout"
            ),
            "evaluation_mode": (
                "generative boxed-letter exact match; unlike lm-eval's "
                "choice-loglikelihood acc/acc_norm"
            ),
            "label_mapping": "cop 0,1,2,3 -> A,B,C,D",
            "seed_assignment": (
                "sampling_seed + CantorPair(question_index, sample_index); "
                "unique, GPU-independent, and stable when Avg@N changes"
            ),
            "answer_extraction": "final \\boxed{A-D} expression",
            "aggregation": {
                f"Avg@{sample_count}": "mean correctness over all generations",
                f"Test@{sample_count}": (
                    "fraction of questions with at least one correct generation"
                ),
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
        f"MedMCQA Avg@{sample_count}: "
        f"{summary['correct_samples']}/{summary['total_samples']} = "
        f"{summary['average_accuracy']:.2%}"
    )
    print(f"MedMCQA Test@{sample_count}: {summary['test_at_n']:.2%}")
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
            if truncated_accuracy is not None
            else "N/A (no truncated samples)"
        )
    )
    print(
        "Non-truncated accuracy: "
        + (
            f"{summary['non_truncated_correct']}/"
            f"{summary['non_truncated_samples']} = {non_truncated_accuracy:.2%}"
            if non_truncated_accuracy is not None
            else "N/A (no completed samples)"
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
    config = parse_args_yaml("MedMCQA zero-shot generative async-vLLM evaluation")
    run_models(config, run, "MedMCQA")


if __name__ == "__main__":
    main()
