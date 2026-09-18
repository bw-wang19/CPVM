import asyncio
from queue import Queue
import random
from types import SimpleNamespace
from unittest.mock import patch

import CPVM.code.test.gpqa_test as gpqa
from CPVM.code.test.gpqa_test import (
    _serve_async_requests,
    extract_answer,
    generate_vllm,
    get_samples_per_question,
    result_path,
    summarize,
)


def test_generate_vllm_terminates_and_joins_workers_after_error():
    class FakeProcess:
        def __init__(self):
            self.alive = False
            self.terminated = False
            self.joined = False
            self.exitcode = None

        def start(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False

        def join(self):
            self.joined = True

    class FakeContext:
        def __init__(self):
            self.task_queue = Queue()
            self.result_queue = Queue()
            self.result_queue.put(("error", 0, "worker failed"))
            self.queue_count = 0
            self.processes = []

        def Queue(self):
            self.queue_count += 1
            return self.task_queue if self.queue_count == 1 else self.result_queue

        def Process(self, **kwargs):
            del kwargs
            process = FakeProcess()
            self.processes.append(process)
            return process

    class FakeBar:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, count):
            del count

    context = FakeContext()
    config = {
        "data_parallel_size": 2,
        "samples_per_question": 1,
        "work_size": 2,
    }
    with (
        patch.object(gpqa.mp, "get_context", return_value=context),
        patch.object(gpqa, "tqdm", return_value=FakeBar()),
    ):
        try:
            generate_vllm(["prompt"], config)
        except RuntimeError as error:
            assert "worker failed" in str(error)
        else:
            raise AssertionError("accepted a failed async worker")

    assert len(context.processes) == 2
    assert all(process.terminated for process in context.processes)
    assert all(process.joined for process in context.processes)


def test_async_worker_refills_one_question_at_a_time():
    class FakeEngine:
        def __init__(self):
            self.gates = {
                prompt: asyncio.Event() for prompt in ("first", "second", "third")
            }
            self.started = []
            self.active = 0
            self.max_active = 0

        async def generate(self, prompt, sampling, request_id):
            del sampling
            self.started.append((prompt, request_id))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await self.gates[prompt].wait()
                yield SimpleNamespace(
                    finished=True,
                    outputs=[
                        SimpleNamespace(
                            index=1,
                            text=r"answer \boxed{B}",
                            finish_reason="stop",
                            token_ids=[1, 2, 3],
                        ),
                        SimpleNamespace(
                            index=0,
                            text=r"answer \boxed{A}",
                            finish_reason="stop",
                            token_ids=[1, 2],
                        ),
                    ],
                )
            finally:
                self.active -= 1

    async def run_scenario():
        engine = FakeEngine()
        task_queue = Queue()
        result_queue = Queue()
        for task in enumerate(("first", "second", "third")):
            task_queue.put(task)
        task_queue.put(None)

        serving = asyncio.create_task(
            _serve_async_requests(
                engine,
                SimpleNamespace(n=2),
                rank=1,
                task_queue=task_queue,
                result_queue=result_queue,
                max_in_flight=2,
                save_responses=False,
            )
        )
        for _ in range(100):
            await asyncio.sleep(0.001)
            if len(engine.started) == 2:
                break
        assert [prompt for prompt, _ in engine.started] == ["first", "second"]

        engine.gates["first"].set()
        for _ in range(100):
            await asyncio.sleep(0.001)
            if len(engine.started) == 3:
                break
        assert [prompt for prompt, _ in engine.started] == [
            "first",
            "second",
            "third",
        ]
        assert not engine.gates["second"].is_set()

        engine.gates["second"].set()
        engine.gates["third"].set()
        await serving
        messages = []
        while not result_queue.empty():
            messages.append(result_queue.get_nowait())
        return engine, messages

    engine, messages = asyncio.run(run_scenario())
    assert engine.max_active == 2
    assert [request_id for _, request_id in engine.started] == [
        "gpqa-1-0",
        "gpqa-1-1",
        "gpqa-1-2",
    ]
    assert sorted(message[1][0] for message in messages) == [0, 1, 2]
    assert all(message[0] == "result" for message in messages)
    assert all(message[2][0][0]["output_tokens"] == 2 for message in messages)
    assert all(message[2][0][1]["output_tokens"] == 3 for message in messages)


def test_samples_per_question_is_a_positive_integer():
    assert get_samples_per_question({"samples_per_question": 3}) == 3
    for value in (None, 0, -1, 1.5, "3", True):
        try:
            get_samples_per_question({"samples_per_question": value})
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid sample count: {value!r}")


def test_result_path_records_the_configured_sample_count():
    config = {
        "model_path": "/models/example",
        "output_path": "/results",
        "test_name": "qwen-official",
        "samples_per_question": 3,
        "data_parallel_size": 2,
    }
    assert result_path(config).name == (
        "gpqa-diamond-example-qwen-official-avg3-dp2.json"
    )
    config["test_name"] = "qwen-official-avg3-dp2"
    assert result_path(config).name == (
        "gpqa-diamond-example-qwen-official-avg3-dp2.json"
    )


def test_extract_answer_uses_the_final_strict_boxed_letter():
    assert extract_answer(r"first \boxed{B}, finally \boxed{ c }") == "C"
    assert extract_answer("The answer is C") is None
    assert extract_answer(r"\boxed{C because reasons}") is None


def test_gpqa_choice_shuffle_matches_the_public_seeded_rule():
    choices = ["wrong-1", "wrong-2", "wrong-3", "correct"]
    random.Random(0).shuffle(choices)
    assert choices == ["wrong-3", "wrong-1", "wrong-2", "correct"]


def test_summary_averages_every_sample_without_voting():
    results = [
        {
            "domain": "Physics",
            "samples": [
                {
                    "prediction": "A",
                    "correct": True,
                    "finish_reason": "stop",
                    "output_tokens": 10,
                },
                {
                    "prediction": None,
                    "correct": False,
                    "finish_reason": "length",
                    "output_tokens": 40,
                },
            ],
        },
        {
            "domain": "Chemistry",
            "samples": [
                {
                    "prediction": "B",
                    "correct": False,
                    "finish_reason": "stop",
                    "output_tokens": 20,
                },
                {
                    "prediction": "C",
                    "correct": True,
                    "finish_reason": "stop",
                    "output_tokens": 30,
                },
            ],
        },
    ]
    summary = summarize(results, samples_per_question=2)
    assert summary["samples_per_question"] == 2
    assert summary["total_samples"] == 4
    assert summary["correct_samples"] == 2
    assert summary["average_accuracy"] == 0.5
    assert summary["parse_rate"] == 0.75
    assert summary["total_output_tokens"] == 100
    assert summary["average_output_tokens"] == 25
    assert summary["median_output_tokens"] == 25
    assert summary["p90_output_tokens"] == 37
    assert summary["p95_output_tokens"] == 38.5
    assert summary["max_output_tokens"] == 40
    assert summary["length_truncated"] == 1
    assert summary["length_truncated_rate"] == 0.25
    assert summary["sample_index_accuracy"] == [0.5, 0.5]

    try:
        summarize(results, samples_per_question=3)
    except ValueError as error:
        assert "exactly 3 samples" in str(error)
    else:
        raise AssertionError("accepted a configured/actual sample-count mismatch")
