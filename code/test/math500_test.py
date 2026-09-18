"""MATH-500 zero-shot CoT evaluation with asynchronous vLLM replicas."""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import signal
import statistics
import threading
import time
import traceback

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from CPVM.code.utils.arguments import parse_args_yaml
from CPVM.code.utils.batch_evaluation import run_models
from CPVM.code.utils.prm800k_grader import grade_answer
from CPVM.code.utils.prm800k_math_normalize import normalize_answer


PROTOCOL_NAME = "CPVM MATH-500 zero-shot CoT + PRM800K grader v1"
PRM800K_GRADER_COMMIT = "7ecc794703b2877f63226f2477a49b34f9b25163"
PROMPT_TEMPLATE = r"""Solve the following problem step by step.
Put your final answer in \boxed{ANSWER}.

Problem:
{problem}"""
ANSWER_LINE_RE = re.compile(
    r"^\s*(?:(?:therefore|thus|hence),?\s*)?(?:the\s+)?"
    r"(?:final\s+)?answer\s*(?:is\s*)?(?::|=)?\s*(?P<answer>\S.*)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
INLINE_FINAL_ANSWER_RE = re.compile(
    r"\b(?:the\s+)?final\s+answer\s+is\s*:?\s*(?P<answer>[^\n]+)",
    flags=re.IGNORECASE,
)


def get_samples_per_question(config):
    """Return the configured positive number of generations per question."""

    value = config.get("samples_per_question")
    if type(value) is not int or value <= 0:
        raise ValueError("samples_per_question must be a positive integer")
    return value


def _sample_seed(base_seed, question_index, sample_index):
    """Map a question/sample pair to a unique seed independent of Avg@N."""

    values = (base_seed, question_index, sample_index)
    if any(type(value) is not int for value in values):
        raise ValueError("sampling_seed and sample coordinates must be integers")
    if any(value < 0 for value in values):
        raise ValueError("sampling_seed and sample coordinates must be nonnegative")
    diagonal = question_index + sample_index
    return base_seed + diagonal * (diagonal + 1) // 2 + sample_index


def _is_escaped(text, index):
    """Return whether the character at index has an odd backslash prefix."""

    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def extract_boxed_answer(text):
    """Extract the final balanced boxed or fbox expression."""

    commands = ("\\boxed", "\\fbox")
    candidates = []
    for command in commands:
        start = 0
        while (command_index := text.find(command, start)) >= 0:
            candidates.append((command_index, command))
            start = command_index + len(command)

    for command_index, command in sorted(candidates, reverse=True):
        cursor = command_index + len(command)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != "{":
            continue

        depth = 0
        answer_start = cursor + 1
        for index in range(cursor, len(text)):
            character = text[index]
            if character == "{" and not _is_escaped(text, index):
                depth += 1
            elif character == "}" and not _is_escaped(text, index):
                depth -= 1
                if depth == 0:
                    answer = text[answer_start:index].strip()
                    if answer:
                        return answer
                    break
                if depth < 0:
                    break
    return None


def _clean_flexible_answer(answer):
    """Remove presentation delimiters around an explicitly labelled answer."""

    value = answer.strip().strip("*" + chr(96)).strip()
    while value.endswith((".", ",", ";")):
        value = value[:-1].rstrip()
    wrappers = (("$", "$"), (r"\(", r"\)"), (r"\[", r"\]"))
    for left, right in wrappers:
        if value.startswith(left) and value.endswith(right):
            value = value[len(left) : -len(right)].strip()
            break
    return value or None


def extract_answer(text, strict=True):
    """Extract the last box, with an explicit-label fallback if requested."""

    boxed = extract_boxed_answer(text)
    if boxed is not None or strict:
        return boxed

    matches = [
        (match.start(), match.group("answer"))
        for pattern in (ANSWER_LINE_RE, INLINE_FINAL_ANSWER_RE)
        for match in pattern.finditer(text)
    ]
    if not matches:
        return None
    _, answer = max(matches, key=lambda item: item[0])
    nested_box = extract_boxed_answer(answer)
    return nested_box if nested_box is not None else _clean_flexible_answer(answer)


def load_questions(dataset_path):
    """Load and validate the local OpenAI MATH-500 test JSONL."""

    source_path = Path(dataset_path).expanduser().resolve()
    path = source_path if source_path.is_file() else source_path / "test.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"MATH-500 test.jsonl not found at {path}")

    required = ("problem", "solution", "answer", "subject", "level", "unique_id")
    questions = []
    seen_ids = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from error
            missing = [name for name in required if name not in row]
            if missing:
                raise ValueError(
                    f"MATH-500 line {line_number} is missing fields {missing}"
                )
            for name in ("problem", "solution", "answer", "subject", "unique_id"):
                if not isinstance(row[name], str) or not row[name].strip():
                    raise ValueError(
                        f"Invalid MATH-500 {name!r} at line {line_number}"
                    )
            if type(row["level"]) is not int or not 1 <= row["level"] <= 5:
                raise ValueError(f"Invalid MATH-500 level at line {line_number}")
            if row["unique_id"] in seen_ids:
                raise ValueError(f"Duplicate MATH-500 unique_id {row['unique_id']!r}")
            seen_ids.add(row["unique_id"])

            solution_answer = extract_boxed_answer(row["solution"])
            if normalize_answer(solution_answer) != normalize_answer(row["answer"]):
                raise ValueError(
                    "MATH-500 answer disagrees with the final boxed solution at "
                    f"line {line_number}"
                )
            questions.append(
                {
                    "id": row["unique_id"],
                    "problem": row["problem"].strip(),
                    "answer": row["answer"].strip(),
                    "subject": row["subject"].strip(),
                    "level": row["level"],
                }
            )
    if not questions:
        raise ValueError(f"MATH-500 has no questions in {path}")
    return questions


def build_prompts(questions, tokenizer):
    """Apply the fixed zero-shot CoT prompt through the model chat template."""

    prompts = []
    for row in questions:
        content = PROMPT_TEMPLATE.replace("{problem}", row["problem"])
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
        "prediction": extract_answer(output.text, strict=True),
        "flexible_prediction": extract_answer(output.text, strict=False),
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
    """Generate one sample with a task-stable seed independent of GPU rank."""

    question_index, sample_index, prompt = task
    seed = _sample_seed(base_seed, question_index, sample_index)
    sampling = sampling_factory(seed)
    request_id = f"math500-{rank}-{question_index}-{sample_index}"
    final_output = None
    async for output in engine.generate(prompt, sampling, request_id):
        final_output = output
    if final_output is None or not final_output.finished:
        raise RuntimeError(f"MATH-500 request {request_id} ended without a final output")
    outputs = sorted(final_output.outputs, key=lambda sample: sample.index)
    if len(outputs) != 1 or outputs[0].index != 0:
        indices = [sample.index for sample in outputs]
        raise RuntimeError(
            f"MATH-500 request {request_id} returned completion indices {indices}; "
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

    print(f"MATH-500 worker {rank}: loading model on GPU {devices[rank]}", flush=True)
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
        print(f"MATH-500 worker {rank}: ready", flush=True)
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
    """Run one complete asynchronous vLLM replica on one visible GPU."""

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices[rank]
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        asyncio.run(
            _run_async_vllm_worker(rank, devices, task_queue, result_queue, config)
        )
    except Exception:
        result_queue.put(("error", rank, traceback.format_exc()))


def _terminate_processes(processes, grace_seconds=10):
    """Terminate workers, escalating to SIGKILL if graceful termination stalls."""

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
    """Distribute individual generation samples dynamically across GPUs."""

    size = config["data_parallel_size"]
    if type(size) is not int or size <= 0:
        raise ValueError("data_parallel_size must be a positive integer")
    work_size = config["work_size"]
    if type(work_size) is not int or work_size <= 0:
        raise ValueError("work_size must be a positive integer")
    sample_count = get_samples_per_question(config)
    inactivity_timeout = config.get("inactivity_timeout_seconds", 900)
    shutdown_timeout = config.get("worker_shutdown_timeout_seconds", 120)
    for name, value in (
        ("inactivity_timeout_seconds", inactivity_timeout),
        ("worker_shutdown_timeout_seconds", shutdown_timeout),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive number")
    if config["temperature"] == 0 and sample_count != 1:
        raise ValueError(
            "Greedy MATH-500 evaluation requires samples_per_question: 1; "
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
    last_progress = time.monotonic()
    try:
        for process in processes:
            process.start()
            started_processes.append(process)

        with tqdm(
            total=len(tasks),
            desc=f"MATH-500 Avg@{sample_count}",
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
                        raise RuntimeError("A MATH-500 vLLM worker exited unexpectedly")
                    if all(process.exitcode is not None for process in processes):
                        raise RuntimeError(
                            "All MATH-500 workers exited before returning every sample"
                        )
                    if time.monotonic() - last_progress >= inactivity_timeout:
                        raise RuntimeError(
                            "MATH-500 made no generation progress for "
                            f"{inactivity_timeout:g} seconds; terminating workers"
                        )
                    continue

                if message[0] == "error":
                    raise RuntimeError(
                        f"MATH-500 worker {message[1]} failed:\n{message[2]}"
                    )
                _, question_index, sample_index, sample = message
                if responses[question_index][sample_index] is not None:
                    raise RuntimeError(
                        "A MATH-500 sample result was returned more than once"
                    )
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
            "MATH-500 workers did not shut down within "
            f"{shutdown_timeout:g} seconds"
        )
    if any(process.exitcode for process in processes):
        raise RuntimeError("A MATH-500 vLLM worker failed")
    if any(sample is None for row in responses for sample in row):
        raise RuntimeError("MATH-500 generation finished with missing samples")
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


class _GradingTimeout(BaseException):
    """Internal signal used to stop a pathological SymPy comparison."""


def _raise_grading_timeout(signum, frame):
    del signum, frame
    raise _GradingTimeout("MATH-500 answer grading timed out")


def _validate_grading_limits(timeout_seconds, max_answer_chars):
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or timeout_seconds <= 0
    ):
        raise ValueError("grading_timeout_seconds must be a positive number")
    if type(max_answer_chars) is not int or max_answer_chars <= 0:
        raise ValueError("max_answer_chars must be a positive integer")


def score_prediction(
    prediction,
    gold,
    timeout_seconds=2.0,
    max_answer_chars=2048,
):
    """Run the PRM800K grader with a length guard and a wall-clock timeout."""

    _validate_grading_limits(timeout_seconds, max_answer_chars)
    if prediction is None:
        return False, "unparsed"
    if len(prediction) > max_answer_chars:
        return False, "answer_too_long"

    can_use_alarm = (
        threading.current_thread() is threading.main_thread()
        and hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
    )
    previous_handler = None
    try:
        if can_use_alarm:
            previous_handler = signal.signal(signal.SIGALRM, _raise_grading_timeout)
            signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
        correct = bool(grade_answer(prediction, gold))
    except _GradingTimeout:
        return False, "timeout"
    except Exception:
        return False, "error"
    finally:
        if can_use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)
    return correct, "correct" if correct else "incorrect"


def hendrycks_equivalent(prediction, gold):
    """Apply the original MATH normalized-string equivalence diagnostic."""

    return (
        prediction is not None
        and normalize_answer(prediction) == normalize_answer(gold)
    )


def _group_metrics(results, group_key, samples_per_question):
    """Aggregate primary and diagnostic scores for each subject or level."""

    grouped = {}
    for row in results:
        grouped.setdefault(str(row[group_key]), []).append(row)

    output = {}
    for label, rows in sorted(grouped.items()):
        samples = [sample for row in rows for sample in row["samples"]]
        output[label] = {
            "num_questions": len(rows),
            "total_samples": len(samples),
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
            "flexible_accuracy": sum(
                sample["flexible_correct"] for sample in samples
            )
            / len(samples),
            "hendrycks_accuracy": sum(
                sample["hendrycks_correct"] for sample in samples
            )
            / len(samples),
            "samples_per_question": samples_per_question,
        }
    return output


def summarize(results, samples_per_question):
    """Calculate accuracy, parsing, grading, and output-length diagnostics."""

    if type(samples_per_question) is not int or samples_per_question <= 0:
        raise ValueError("samples_per_question must be a positive integer")
    mismatched = [
        index
        for index, row in enumerate(results)
        if len(row["samples"]) != samples_per_question
    ]
    if mismatched:
        raise ValueError(
            "Every MATH-500 question must contain exactly "
            f"{samples_per_question} samples; mismatches at {mismatched[:5]}"
        )
    samples = [sample for row in results for sample in row["samples"]]
    if not samples:
        raise ValueError("MATH-500 results must contain at least one sample")

    output_tokens = [sample["output_tokens"] for sample in samples]
    correct = sum(sample["correct"] for sample in samples)
    flexible_correct = sum(sample["flexible_correct"] for sample in samples)
    hendrycks_correct = sum(sample["hendrycks_correct"] for sample in samples)
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
    questions_with_a_flexible_correct_sample = sum(
        any(sample["flexible_correct"] for sample in row["samples"])
        for row in results
    )

    summary = {
        "num_questions": len(results),
        "samples_per_question": samples_per_question,
        "total_samples": len(samples),
        "correct_samples": correct,
        "average_accuracy": correct / len(samples),
        "test_at_n": questions_with_a_correct_sample / len(results),
        "flexible_correct_samples": flexible_correct,
        "flexible_accuracy": flexible_correct / len(samples),
        "flexible_test_at_n": (
            questions_with_a_flexible_correct_sample / len(results)
        ),
        "hendrycks_correct_samples": hendrycks_correct,
        "hendrycks_accuracy": hendrycks_correct / len(samples),
        "parse_rate": sum(sample["prediction"] is not None for sample in samples)
        / len(samples),
        "flexible_parse_rate": sum(
            sample["flexible_prediction"] is not None for sample in samples
        )
        / len(samples),
        "grader_timeout_count": sum(
            sample["grader_status"] == "timeout" for sample in samples
        ),
        "grader_error_count": sum(
            sample["grader_status"] == "error" for sample in samples
        ),
        "answer_too_long_count": sum(
            sample["grader_status"] == "answer_too_long" for sample in samples
        ),
        "flexible_grader_timeout_count": sum(
            sample["flexible_grader_status"] == "timeout" for sample in samples
        ),
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
    if all("subject" in row for row in results):
        summary["by_subject"] = _group_metrics(
            results, "subject", samples_per_question
        )
    if all("level" in row for row in results):
        summary["by_level"] = _group_metrics(
            results, "level", samples_per_question
        )
    return summary



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
    return Path(config["output_path"]) / f"math500-{model_name}-{test_name}.json"


def run(config):
    """Run one model on the local OpenAI MATH-500 test split."""

    sample_count = get_samples_per_question(config)
    if config["temperature"] == 0 and sample_count != 1:
        raise ValueError(
            "Greedy MATH-500 evaluation requires samples_per_question: 1; "
            "use temperature > 0 for Avg@N sampling"
        )
    grading_timeout = config.get("grading_timeout_seconds", 2.0)
    max_answer_chars = config.get("max_answer_chars", 2048)
    _validate_grading_limits(grading_timeout, max_answer_chars)

    questions = load_questions(config["dataset_path"])
    expected_questions = config.get("expected_num_questions")
    if expected_questions is not None and len(questions) != expected_questions:
        raise ValueError(
            f"Expected {expected_questions} MATH-500 questions, found {len(questions)}"
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
        f"MATH-500 test: {len(questions)} questions x {sample_count} "
        f"samples; protocol={PROTOCOL_NAME}; decoding={decoding}; "
        f"max prompt={max_prompt_tokens} tokens",
        flush=True,
    )
    responses = generate_vllm(prompts, config)

    predictions = []
    for question, samples in zip(questions, responses):
        for sample in samples:
            boxed_score = score_prediction(
                sample["prediction"],
                question["answer"],
                grading_timeout,
                max_answer_chars,
            )
            sample["correct"], sample["grader_status"] = boxed_score
            sample["hendrycks_correct"] = hendrycks_equivalent(
                sample["prediction"], question["answer"]
            )

            if sample["flexible_prediction"] == sample["prediction"]:
                flexible_score = boxed_score
            else:
                flexible_score = score_prediction(
                    sample["flexible_prediction"],
                    question["answer"],
                    grading_timeout,
                    max_answer_chars,
                )
            (
                sample["flexible_correct"],
                sample["flexible_grader_status"],
            ) = flexible_score

        predictions.append(
            {
                "id": question["id"],
                "problem": question["problem"],
                "answer": question["answer"],
                "subject": question["subject"],
                "level": question["level"],
                "correct_samples": sum(sample["correct"] for sample in samples),
                "flexible_correct_samples": sum(
                    sample["flexible_correct"] for sample in samples
                ),
                "samples": samples,
            }
        )

    summary = summarize(predictions, sample_count)
    report = {
        "benchmark": "MATH-500",
        "metric": f"average_prm800k_grade_answer@{sample_count}",
        "protocol": {
            "name": PROTOCOL_NAME,
            "dataset": "OpenAI PRM800K MATH-500",
            "dataset_config": "default",
            "split": "test",
            "num_fewshot": 0,
            "prompt_template": PROMPT_TEMPLATE,
            "apply_chat_template": True,
            "decoding": decoding,
            "seed_assignment": (
                "sampling_seed + CantorPair(question_index, sample_index); "
                "unique, GPU-independent, and stable when Avg@N changes"
            ),
            "answer_extraction": {
                "primary": (
                    "content of the final balanced \\\\boxed{...} or \\\\fbox{...}"
                ),
                "flexible_diagnostic": (
                    "primary extraction, then the final explicit "
                    "'Final answer:' or 'Answer:' line"
                ),
                "no_last_number_fallback": True,
            },
            "grading": {
                "primary": "OpenAI PRM800K grade_answer",
                "source_commit": PRM800K_GRADER_COMMIT,
                "method": (
                    "Hendrycks normalization followed by conservative "
                    "tuple-aware SymPy equivalence"
                ),
                "scope": (
                    "PRM800K-compatible conservative score, not complete "
                    "semantic equivalence for arbitrary mathematical objects"
                ),
                "diagnostic": "Hendrycks MATH normalized-string equivalence",
                "timeout_seconds_per_answer": grading_timeout,
                "max_answer_characters": max_answer_chars,
            },
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
        f"MATH-500 Avg@{sample_count} (boxed, PRM800K): "
        f"{summary['correct_samples']}/{summary['total_samples']} = "
        f"{summary['average_accuracy']:.2%}"
    )
    print(
        f"MATH-500 Test@{sample_count} (at least one boxed answer correct): "
        f"{summary['test_at_n']:.2%}"
    )
    print(
        f"Flexible-format Avg@{sample_count} diagnostic: "
        f"{summary['flexible_correct_samples']}/{summary['total_samples']} = "
        f"{summary['flexible_accuracy']:.2%}"
    )
    print(
        f"Hendrycks-normalized Avg@{sample_count} diagnostic: "
        f"{summary['hendrycks_correct_samples']}/{summary['total_samples']} = "
        f"{summary['hendrycks_accuracy']:.2%}"
    )
    print(
        f"Boxed parse rate: {summary['parse_rate']:.2%}; "
        f"flexible parse rate: {summary['flexible_parse_rate']:.2%}"
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
    print(
        "Primary grader diagnostics: "
        f"timeouts={summary['grader_timeout_count']}, "
        f"errors={summary['grader_error_count']}, "
        f"answers over limit={summary['answer_too_long_count']}"
    )
    print(f"Saved to: {path}")
    return report


def main():
    """Load YAML and run all configured models serially."""

    config = parse_args_yaml("MATH-500 zero-shot CoT async-vLLM evaluation")
    run_models(config, run, "MATH-500")


if __name__ == "__main__":
    main()

