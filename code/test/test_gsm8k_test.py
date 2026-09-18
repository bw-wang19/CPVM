"""Lightweight regression tests for the GSM8K evaluator."""

import asyncio
import json
from queue import Queue
from types import SimpleNamespace

import pandas as pd
import pytest

import CPVM.code.test.gsm8k_test as gsm8k
from CPVM.code.test.gsm8k_test import (
    PROMPT_TEMPLATE,
    _serve_async_requests,
    build_prompts,
    extract_answer,
    extract_gold_answer,
    get_samples_per_question,
    load_questions,
    result_path,
    summarize,
)


def test_load_questions_reads_main_test_parquet_and_normalizes_gold(tmp_path):
    dataset = tmp_path / "gsm8k"
    main = dataset / "main"
    main.mkdir(parents=True)
    pd.DataFrame(
        {
            "question": ["How many?", "What is the change?"],
            "answer": [
                "Reasoning with <<1000+234=1234>>.\n#### 1,234",
                "Reasoning with <<2-5=-3>>.\n#### -3",
            ],
        }
    ).to_parquet(main / "test-00000-of-00001.parquet", index=False)

    questions = load_questions(str(dataset))

    assert len(questions) == 2
    assert [row["id"] for row in questions] == [0, 1]
    assert [row["question"] for row in questions] == [
        "How many?",
        "What is the change?",
    ]
    assert [row["answer"] for row in questions] == ["1234", "-3"]


def test_build_prompts_uses_one_user_message_and_native_chat_template():
    class FakeTokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            self.calls.append((messages, tokenize, add_generation_prompt))
            return "CHAT:" + messages[0]["content"]

    tokenizer = FakeTokenizer()
    prompts = build_prompts([{"question": "Janet has 16 eggs."}], tokenizer)

    assert len(prompts) == 1
    messages, tokenize, add_generation_prompt = tokenizer.calls[0]
    assert messages == [
        {
            "role": "user",
            "content": ("Q: Janet has 16 eggs.\n A: Let's think step by step."),
        }
    ]
    assert messages[0]["content"] == PROMPT_TEMPLATE.format(
        question="Janet has 16 eggs."
    )
    assert tokenize is False
    assert add_generation_prompt is True
    assert prompts[0].startswith("CHAT:")


def test_samples_per_question_must_be_a_positive_integer():
    assert get_samples_per_question({"samples_per_question": 3}) == 3
    for value in (None, 0, -1, 1.5, "3", True):
        with pytest.raises(ValueError, match="positive integer"):
            get_samples_per_question({"samples_per_question": value})


def test_gold_answer_requires_the_terminal_hash_marker():
    assert extract_gold_answer("work\n#### 1,234") == "1234"
    assert extract_gold_answer("work\n#### -10") == "-10"
    with pytest.raises(ValueError, match="terminal"):
        extract_gold_answer("work but no final marker")


def test_answer_extraction_matches_lm_eval_strict_and_flexible_modes():
    assert extract_answer("work. The answer is 1,234.", strict=True) == "1234"
    assert extract_answer("work. The answer is -10", strict=True) == "-10"
    assert extract_answer("work\n#### 18", strict=True) is None

    assert extract_answer("First calculate 16 - 7 = 9. Final answer: 18") == "18"
    assert extract_answer("The values are 21 and finally -3") == "-3"
    assert extract_answer("The cost is finally $1,234") == "1234"
    assert extract_answer("No numeric answer was produced.") is None


def test_summary_reports_avg_test_parse_length_and_truncation_metrics():
    results = [
        {
            "samples": [
                {
                    "prediction": "18",
                    "strict_prediction": "18",
                    "correct": True,
                    "strict_correct": True,
                    "finish_reason": "stop",
                    "output_tokens": 10,
                },
                {
                    "prediction": "17",
                    "strict_prediction": None,
                    "correct": False,
                    "strict_correct": False,
                    "finish_reason": "stop",
                    "output_tokens": 20,
                },
            ]
        },
        {
            "samples": [
                {
                    "prediction": None,
                    "strict_prediction": None,
                    "correct": False,
                    "strict_correct": False,
                    "finish_reason": "length",
                    "output_tokens": 60,
                },
                {
                    "prediction": "7",
                    "strict_prediction": "7",
                    "correct": False,
                    "strict_correct": False,
                    "finish_reason": "stop",
                    "output_tokens": 30,
                },
            ]
        },
        {
            "samples": [
                {
                    "prediction": "5",
                    "strict_prediction": None,
                    "correct": True,
                    "strict_correct": False,
                    "finish_reason": "length",
                    "output_tokens": 50,
                },
                {
                    "prediction": "4",
                    "strict_prediction": "4",
                    "correct": False,
                    "strict_correct": False,
                    "finish_reason": "stop",
                    "output_tokens": 40,
                },
            ]
        },
    ]

    summary = summarize(results, 2)

    assert summary["num_questions"] == 3
    assert summary["samples_per_question"] == 2
    assert summary["total_samples"] == 6
    assert summary["correct_samples"] == 2
    assert summary["average_accuracy"] == pytest.approx(2 / 6)
    assert summary["test_at_n"] == pytest.approx(2 / 3)
    assert summary["strict_correct_samples"] == 1
    assert summary["strict_accuracy"] == pytest.approx(1 / 6)
    assert summary["parse_rate"] == pytest.approx(5 / 6)
    assert summary["strict_parse_rate"] == pytest.approx(3 / 6)
    assert summary["total_output_tokens"] == 210
    assert summary["average_output_tokens"] == 35
    assert summary["median_output_tokens"] == 35
    assert summary["p90_output_tokens"] == 55
    assert summary["p95_output_tokens"] == 57.5
    assert summary["max_output_tokens"] == 60
    assert summary["length_truncated"] == 2
    assert summary["length_truncated_rate"] == pytest.approx(2 / 6)
    assert summary["length_truncated_correct"] == 1
    assert summary["length_truncated_accuracy"] == 0.5
    assert summary["non_truncated_samples"] == 4
    assert summary["non_truncated_correct"] == 1
    assert summary["non_truncated_accuracy"] == 0.25
    assert summary["strict_completion_accuracy"] == pytest.approx(1 / 6)
    assert summary["length_truncated_output_tokens"] == 110
    assert summary["length_truncated_token_rate"] == pytest.approx(110 / 210)
    assert summary["sample_index_accuracy"] == [pytest.approx(2 / 3), 0.0]

    with pytest.raises(ValueError, match="exactly 3 samples"):
        summarize(results, 3)


def test_result_path_records_dynamic_avg_n_without_duplicate_suffix():
    config = {
        "model_path": "/models/example",
        "output_path": "/results",
        "test_name": "zero-shot",
        "samples_per_question": 3,
        "data_parallel_size": 2,
    }
    assert result_path(config).name == "gsm8k-example-zero-shot-avg3-dp2.json"
    config["test_name"] = "zero-shot-avg3-dp2"
    assert result_path(config).name == "gsm8k-example-zero-shot-avg3-dp2.json"


def test_async_worker_refills_at_sample_granularity_with_unique_seeds():
    """A finished sample must admit replacement while another sample is slow."""

    class FakeEngine:
        def __init__(self):
            self.gates = {key: asyncio.Event() for key in ((0, 0), (0, 1), (1, 0))}
            self.started = []
            self.active = 0
            self.max_active = 0

        async def generate(self, prompt, sampling, request_id):
            question_index, sample_index = map(int, request_id.rsplit("-", 2)[-2:])
            key = (question_index, sample_index)
            self.started.append((key, prompt, sampling.seed, request_id, sampling.n))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await self.gates[key].wait()
                yield SimpleNamespace(
                    finished=True,
                    outputs=[
                        SimpleNamespace(
                            index=0,
                            text=(
                                f"work. The answer is {question_index + sample_index}"
                            ),
                            finish_reason="stop",
                            token_ids=[1, 2, 3],
                        )
                    ],
                )
            finally:
                self.active -= 1

    async def scenario():
        engine = FakeEngine()
        task_queue = Queue()
        result_queue = Queue()
        task_queue.put((0, 0, "q0"))
        task_queue.put((0, 1, "q0"))
        task_queue.put((1, 0, "q1"))
        task_queue.put(None)

        def sampling_factory(seed):
            return SimpleNamespace(n=1, seed=seed)

        serving = asyncio.create_task(
            _serve_async_requests(
                engine,
                sampling_factory,
                rank=1,
                task_queue=task_queue,
                result_queue=result_queue,
                max_in_flight=2,
                base_seed=100,
                samples_per_question=2,
                save_responses=False,
            )
        )
        for _ in range(100):
            await asyncio.sleep(0.001)
            if len(engine.started) == 2:
                break
        assert {entry[0] for entry in engine.started} == {(0, 0), (0, 1)}

        engine.gates[(0, 0)].set()
        for _ in range(100):
            await asyncio.sleep(0.001)
            if len(engine.started) == 3:
                break
        assert {entry[0] for entry in engine.started} == {
            (0, 0),
            (0, 1),
            (1, 0),
        }
        assert not engine.gates[(0, 1)].is_set()

        engine.gates[(0, 1)].set()
        engine.gates[(1, 0)].set()
        await asyncio.wait_for(serving, timeout=1)
        messages = []
        while not result_queue.empty():
            messages.append(result_queue.get_nowait())
        return engine, messages

    engine, messages = asyncio.run(scenario())
    assert engine.max_active == 2
    assert {entry[0]: entry[2] for entry in engine.started} == {
        (0, 0): 100,
        (0, 1): 101,
        (1, 0): 102,
    }
    assert all(entry[4] == 1 for entry in engine.started)
    assert {
        (message[1], message[2]) for message in messages if message[0] == "result"
    } == {(0, 0), (0, 1), (1, 0)}


def test_async_worker_cancels_other_requests_when_one_fails():
    class FailingEngine:
        def __init__(self):
            self.slow_started = asyncio.Event()
            self.slow_cancelled = asyncio.Event()

        async def generate(self, prompt, sampling, request_id):
            del prompt, sampling
            question_index, sample_index = map(int, request_id.rsplit("-", 2)[-2:])
            if (question_index, sample_index) == (0, 0):
                await self.slow_started.wait()
                raise RuntimeError("generation failed")
            self.slow_started.set()
            try:
                await asyncio.Event().wait()
                yield  # pragma: no cover - keeps this an async generator
            finally:
                self.slow_cancelled.set()

    async def scenario():
        engine = FailingEngine()
        task_queue = Queue()
        result_queue = Queue()
        task_queue.put((0, 0, "fast failure"))
        task_queue.put((0, 1, "slow request"))
        task_queue.put(None)

        with pytest.raises(RuntimeError, match="generation failed"):
            await asyncio.wait_for(
                _serve_async_requests(
                    engine,
                    lambda seed: SimpleNamespace(n=1, seed=seed),
                    rank=0,
                    task_queue=task_queue,
                    result_queue=result_queue,
                    max_in_flight=2,
                    base_seed=7,
                    samples_per_question=2,
                    save_responses=False,
                ),
                timeout=1,
            )
        assert engine.slow_cancelled.is_set()
        assert result_queue.empty()

    asyncio.run(scenario())


def test_run_writes_dynamic_avg_n_report_without_loading_a_model(tmp_path, monkeypatch):
    dataset = tmp_path / "gsm8k" / "main"
    dataset.mkdir(parents=True)
    pd.DataFrame(
        {
            "question": ["What is 9 + 9?"],
            "answer": ["Add the values.\n#### 18"],
        }
    ).to_parquet(dataset / "test-00000-of-00001.parquet", index=False)

    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is True
            return f"<chat>{messages[0]['content']}"

        def encode(self, prompt, *, add_special_tokens):
            assert prompt.startswith("<chat>Q: What is 9 + 9?")
            assert add_special_tokens is False
            return [1, 2, 3]

    monkeypatch.setattr(
        gsm8k,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer()),
    )
    captured = {}

    def fake_generate(prompts, config):
        captured["prompts"] = prompts
        captured["config"] = config
        return [
            [
                {
                    "sample_index": 0,
                    "seed": 11,
                    "prediction": "18",
                    "strict_prediction": "18",
                    "finish_reason": "stop",
                    "output_tokens": 4,
                },
                {
                    "sample_index": 1,
                    "seed": 12,
                    "prediction": "17",
                    "strict_prediction": None,
                    "finish_reason": "length",
                    "output_tokens": 8,
                },
            ]
        ]

    monkeypatch.setattr(gsm8k, "generate_vllm", fake_generate)
    config = {
        "dataset_path": str(tmp_path / "gsm8k"),
        "expected_num_questions": 1,
        "model_path": str(tmp_path / "toy-model"),
        "output_path": str(tmp_path / "results"),
        "test_name": "zero-shot",
        "samples_per_question": 2,
        "data_parallel_size": 1,
        "temperature": 0.7,
        "max_new_tokens": 16,
        "max_model_len": 64,
    }

    report = gsm8k.run(config)
    path = tmp_path / "results" / "gsm8k-toy-model-zero-shot-avg2-dp1.json"

    assert captured["config"] is config
    assert captured["prompts"] == [
        "<chat>" + PROMPT_TEMPLATE.format(question="What is 9 + 9?")
    ]
    assert report["metric"] == "average_exact_match@2"
    assert report["protocol"]["name"] == gsm8k.PROTOCOL_NAME
    assert report["protocol"]["dataset_config"] == "main"
    assert report["protocol"]["split"] == "test"
    assert report["protocol"]["num_fewshot"] == 0
    assert report["protocol"]["prompt_template"] == PROMPT_TEMPLATE
    assert report["protocol"]["decoding"] == "sampling"
    assert report["summary"]["average_accuracy"] == 0.5
    assert report["summary"]["test_at_n"] == 1.0
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == report


def test_main_delegates_parsed_yaml_to_run_models(monkeypatch):
    config = {"models": ["one"]}
    seen = {}
    monkeypatch.setattr(gsm8k, "parse_args_yaml", lambda description: config)

    def fake_run_models(received_config, runner, benchmark):
        seen["call"] = (received_config, runner, benchmark)

    monkeypatch.setattr(gsm8k, "run_models", fake_run_models)

    gsm8k.main()

    assert seen["call"] == (config, gsm8k.run, "GSM8K")
