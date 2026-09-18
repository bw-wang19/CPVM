"""Minimal AIME 2025 mean@N evaluation."""

import asyncio
import json
import multiprocessing as mp
import os
from pathlib import Path
import re

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from CPVM.code.utils.arguments import parse_args_yaml
from CPVM.code.utils.batch_evaluation import run_models


PROMPT = r"""Solve the following AIME problem. Show your reasoning clearly.
Put the final answer in \boxed{{N}}, where N is an integer from 000 to 999.

Problem:
{problem}"""


def get_samples_per_question(config):
    """Return the positive number of samples requested for every problem."""

    value = config.get("samples_per_question", config.get("repeats"))
    if type(value) is not int or value <= 0:
        raise ValueError("samples_per_question must be a positive integer")
    return value


def distribute_samples(sample_count, worker_count):
    """Split samples exactly across workers, including a possible remainder."""

    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    if type(worker_count) is not int or worker_count <= 0:
        raise ValueError("data_parallel_size must be a positive integer")
    base, remainder = divmod(sample_count, worker_count)
    return [base + (rank < remainder) for rank in range(worker_count)]


def sample_seed_offsets(sample_counts):
    """Return non-overlapping seed offsets for each worker's sample block."""

    offsets = []
    next_offset = 0
    for count in sample_counts:
        if type(count) is not int or count < 0:
            raise ValueError("sample counts must be non-negative integers")
        offsets.append(next_offset)
        next_offset += count
    return offsets


def load_questions(dataset_path):
    """Read the 30 problems from the downloaded JSONL file."""
    path = Path(dataset_path) / "test.jsonl"
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def build_prompts(questions, tokenizer):
    """Apply the model chat template to every problem."""
    prompts = []
    for question in questions:
        message = PROMPT.format(problem=question["problem"])
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": message}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def extract_answer(text):
    """Extract the last boxed or explicitly stated 0--999 integer."""
    boxed = re.findall(r"\\boxed\{\s*([0-9]{1,3})\s*\}", text)
    if boxed:
        return int(boxed[-1])
    final = re.findall(
        r"(?:final answer|answer is)[^0-9]{0,20}([0-9]{1,3})(?![0-9])",
        text,
        flags=re.IGNORECASE,
    )
    if final:
        return int(final[-1])
    numbers = re.findall(r"(?<![0-9])([0-9]{1,3})(?![0-9])", text)
    return int(numbers[-1]) if numbers else None


async def run_bounded(items, max_inflight, operation, consume):
    """Run a bounded async queue and refill it after each completion."""

    if type(max_inflight) is not int or max_inflight <= 0:
        raise ValueError("max_inflight must be a positive integer")
    iterator = iter(items)
    pending = {}

    def refill():
        while len(pending) < max_inflight:
            try:
                item = next(iterator)
            except StopIteration:
                return
            pending[asyncio.create_task(operation(item))] = item

    refill()
    try:
        while pending:
            done, _ = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                item = pending.pop(task)
                consume(item, task.result())
            refill()
    except BaseException:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise


async def _vllm_worker_async(
    rank,
    prompts,
    config,
    devices,
    samples_per_rank,
    sample_seed,
    shared_outputs,
):
    """Keep one GPU's vLLM engine supplied with independent prompt requests."""

    os.environ["CUDA_VISIBLE_DEVICES"] = devices[rank]
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=config["model_path"],
            trust_remote_code=True,
            dtype=config["dtype"],
            tensor_parallel_size=1,
            gpu_memory_utilization=config["gpu_memory_utilization"],
            max_model_len=config["max_length"],
        )
    )
    sampling = SamplingParams(
        n=samples_per_rank,
        temperature=config["temperature"],
        top_p=config["top_p"],
        top_k=config["top_k"],
        min_p=config["min_p"],
        max_tokens=config["max_new_tokens"],
        seed=sample_seed,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    responses = [None] * len(prompts)

    async def generate_one(index):
        final_output = None
        async for output in engine.generate(
            prompts[index], sampling, f"aime25-{rank}-{index}"
        ):
            final_output = output
        if final_output is None or not final_output.finished:
            raise RuntimeError(
                f"AIME GPU {rank} request {index} ended without a final output"
            )
        samples = sorted(final_output.outputs, key=lambda sample: sample.index)
        output_indices = [sample.index for sample in samples]
        if output_indices != list(range(samples_per_rank)):
            raise RuntimeError(
                f"AIME GPU {rank} request {index} returned completion indices "
                f"{output_indices}; expected 0..{samples_per_rank - 1}"
            )
        return [sample.text for sample in samples]

    try:
        with tqdm(
            total=len(prompts),
            desc=f"AIME25 GPU {rank} ({samples_per_rank} samples/question)",
            unit="question",
        ) as bar:

            def consume(index, samples):
                responses[index] = samples
                bar.update()

            await run_bounded(
                range(len(prompts)),
                config["batch_size"],
                generate_one,
                consume,
            )
        shared_outputs[rank] = responses
    finally:
        engine.shutdown()


def vllm_worker(
    rank,
    prompts,
    config,
    devices,
    samples_per_rank,
    sample_seed,
    shared_outputs,
):
    """Sample every prompt asynchronously on one independent GPU replica."""

    asyncio.run(
        _vllm_worker_async(
            rank,
            prompts,
            config,
            devices,
            samples_per_rank,
            sample_seed,
            shared_outputs,
        )
    )


def generate_vllm(prompts, config):
    """Combine exactly allocated sample groups from independent GPU replicas."""
    size = config["data_parallel_size"]
    sample_counts = distribute_samples(get_samples_per_question(config), size)
    seed_offsets = sample_seed_offsets(sample_counts)
    sample_seeds = [config["seed"] + offset for offset in seed_offsets]
    active_ranks = [rank for rank, count in enumerate(sample_counts) if count]
    allocations = ", ".join(
        f"GPU {rank}: n={sample_counts[rank]}, "
        f"seeds={sample_seeds[rank]}.."
        f"{sample_seeds[rank] + sample_counts[rank] - 1}"
        for rank in active_ranks
    )
    print(f"AIME25 sample allocation: {allocations}", flush=True)
    default_devices = ",".join(str(rank) for rank in range(size))
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", default_devices).split(",")
    context = mp.get_context("spawn")

    with context.Manager() as manager:
        shared_outputs = manager.dict()
        processes = [
            context.Process(
                target=vllm_worker,
                args=(
                    rank,
                    prompts,
                    config,
                    devices,
                    sample_counts[rank],
                    sample_seeds[rank],
                    shared_outputs,
                ),
            )
            for rank in active_ranks
        ]
        started_processes = []
        try:
            for process in processes:
                process.start()
                started_processes.append(process)
            while any(process.is_alive() for process in started_processes):
                for process in started_processes:
                    process.join(timeout=1)
                if any(
                    process.exitcode not in (None, 0)
                    for process in started_processes
                ):
                    raise RuntimeError("A vLLM data-parallel worker failed.")
        except BaseException:
            for process in started_processes:
                if process.is_alive():
                    process.terminate()
            for process in started_processes:
                process.join()
            raise
        if any(process.exitcode for process in processes):
            raise RuntimeError("A vLLM data-parallel worker failed.")
        parts = dict(shared_outputs)

    responses = []
    for index in range(len(prompts)):
        responses.append(
            [
                sample
                for rank in active_ranks
                for sample in parts[rank][index]
            ]
        )
    return responses


def generate_transformers(prompts, tokenizer, config):
    """Generate with the ordinary Transformers backend."""
    sample_count = get_samples_per_question(config)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        trust_remote_code=True,
        dtype=config["dtype"],
        device_map=config["device_map"],
    ).eval()
    device = model.get_input_embeddings().weight.device
    responses = []
    for start in tqdm(
        range(0, len(prompts), config["batch_size"]),
        desc="AIME25",
        unit="batch",
    ):
        inputs = tokenizer(
            prompts[start : start + config["batch_size"]],
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=True,
                temperature=config["temperature"],
                top_p=config["top_p"],
                top_k=config["top_k"],
                min_p=config["min_p"],
                num_return_sequences=sample_count,
                max_new_tokens=config["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        responses += [
            decoded[index : index + sample_count]
            for index in range(0, len(decoded), sample_count)
        ]
    return responses


def output_path(config):
    """Build a readable JSON result filename."""
    sample_count = get_samples_per_question(config)
    source = Path(config["model_path"])
    model_name = source.name
    if model_name.startswith("checkpoint-"):
        model_name = f"{source.parent.name}-{model_name}"
    run_tag = f"mean{sample_count}-dp{config['data_parallel_size']}"
    test_name = config["test_name"]
    if not test_name.endswith(run_tag):
        test_name = f"{test_name}-{run_tag}"
    return Path(config["output_path"]) / f"aime25-{model_name}-{test_name}.json"


def run(config):
    """Sample each problem repeatedly and report mean exact-match accuracy."""
    sample_count = get_samples_per_question(config)
    questions = load_questions(config["dataset_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"],
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    prompts = build_prompts(questions, tokenizer)

    if config["use_vllm"]:
        responses = generate_vllm(prompts, config)
    else:
        responses = generate_transformers(prompts, tokenizer, config)

    if len(responses) != len(questions):
        raise RuntimeError(
            f"Expected responses for {len(questions)} AIME problems, got {len(responses)}"
        )
    mismatched = [
        index for index, samples in enumerate(responses)
        if len(samples) != sample_count
    ]
    if mismatched:
        raise RuntimeError(
            "Every AIME problem must contain exactly "
            f"{sample_count} samples; mismatches at indices {mismatched[:5]}"
        )

    predictions = []
    for question, samples in zip(questions, responses):
        answer = int(question["answer"])
        sample_results = []
        for response in samples:
            prediction = extract_answer(response)
            sample = {
                "prediction": prediction,
                "correct": prediction == answer,
            }
            if config["save_responses"]:
                sample["response"] = response
            sample_results.append(sample)

        correct = sum(sample["correct"] for sample in sample_results)
        row = {
            "id": question["id"],
            "answer": answer,
            "correct_samples": correct,
            "num_samples": len(sample_results),
            "accuracy": correct / len(sample_results),
            "samples": sample_results,
        }
        predictions.append(row)

    total = sum(row["num_samples"] for row in predictions)
    correct = sum(row["correct_samples"] for row in predictions)
    accuracy = correct / total
    report = {
        "benchmark": "AIME 2025",
        "metric": f"mean_accuracy@{sample_count}",
        "model_path": config["model_path"],
        "settings": config,
        "summary": {
            "num_questions": len(predictions),
            "samples_per_question": sample_count,
            "total_samples": total,
            "correct_samples": correct,
            "mean_accuracy": accuracy,
        },
        "predictions": predictions,
    }
    path = output_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(
        f"Mean@{sample_count}: "
        f"{correct}/{total} = {accuracy:.2%}"
    )
    print(f"Saved to: {path}")
    return report


def main():
    """Load YAML and start evaluation."""
    config = parse_args_yaml("AIME25 evaluation config -> *.yaml")
    run_models(config, run, "AIME 2025")


if __name__ == "__main__":
    main()
