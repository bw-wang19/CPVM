"""Lightweight regression tests for the MATH-500 evaluator."""

import asyncio
import json
from queue import Queue
from types import SimpleNamespace

import pytest

import CPVM.code.test.math500_test as math500
from CPVM.code.test.math500_test import (
    PROMPT_TEMPLATE,
    _sample_seed,
    _serve_async_requests,
    build_prompts,
    extract_answer,
    extract_boxed_answer,
    get_samples_per_question,
    hendrycks_equivalent,
    load_questions,
    result_path,
    score_prediction,
    summarize,
)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _dataset_row(**overrides):
    row = {
        "problem": "Solve for x.",
        "solution": r"Working gives \boxed{\frac{1}{2}}.",
        "answer": r"\frac12",
        "subject": "Algebra",
        "level": 2,
        "unique_id": "test/algebra/1.json",
    }
    row.update(overrides)
    return row


def test_load_questions_reads_jsonl_validates_gold_and_preserves_metadata(tmp_path):
    dataset = tmp_path / "MATH-500"
    rows = [
        _dataset_row(problem="  Solve for x.  ", subject=" Algebra "),
        _dataset_row(
            problem="Find the ordered pair.",
            solution=r"An earlier \boxed{0} is superseded by \fbox{(1, \frac{2}{3})}.",
            answer=r"(1, \frac{2}{3})",
            subject="Geometry",
            level=5,
            unique_id="test/geometry/2.json",
        ),
    ]
    _write_jsonl(dataset / "test.jsonl", rows)

    questions = load_questions(str(dataset))

    assert questions == [
        {
            "id": "test/algebra/1.json",
            "problem": "Solve for x.",
            "answer": r"\frac12",
            "subject": "Algebra",
            "level": 2,
        },
        {
            "id": "test/geometry/2.json",
            "problem": "Find the ordered pair.",
            "answer": r"(1, \frac{2}{3})",
            "subject": "Geometry",
            "level": 5,
        },
    ]


def test_load_questions_rejects_a_gold_that_disagrees_with_final_box(tmp_path):
    path = tmp_path / "test.jsonl"
    _write_jsonl(path, [_dataset_row(answer="2")])

    with pytest.raises(ValueError, match="disagrees with the final boxed solution"):
        load_questions(path)


def test_build_prompts_uses_one_user_message_and_native_chat_template():
    class FakeTokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            self.calls.append((messages, tokenize, add_generation_prompt))
            return "CHAT:" + messages[0]["content"]

    tokenizer = FakeTokenizer()
    problem = r"Find {x} when x^2=1."
    prompts = build_prompts([{"problem": problem}], tokenizer)

    expected_content = PROMPT_TEMPLATE.replace("{problem}", problem)
    assert tokenizer.calls == [
        ([{"role": "user", "content": expected_content}], False, True)
    ]
    assert prompts == ["CHAT:" + expected_content]


def test_balanced_boxed_extraction_uses_the_final_box_and_nested_braces():
    text = (
        r"First \boxed{wrong}; finally \fbox  {\left(1, "
        r"\frac{2}{3}, \{4,5\}\right)} after the answer."
    )

    assert extract_boxed_answer(text) == (
        r"\left(1, \frac{2}{3}, \{4,5\}\right)"
    )
    assert extract_boxed_answer(r"unfinished \boxed{\frac{1}{2}") is None
    assert extract_boxed_answer(r"missing braces \boxed 2") is None
    assert extract_boxed_answer("no boxed answer") is None
    assert extract_boxed_answer(r"\boxed{valid} then \boxed{unfinished") == "valid"
    assert extract_boxed_answer(r"\fbox{valid} then \boxed no-braces") == "valid"


def test_flexible_extraction_requires_an_explicit_label_and_prefers_last_answer():
    text = (
        "A tempting answer is 7.\n"
        r"Answer: $\frac{1}{3}$" "\n"
        r"Therefore, the final answer is: \(\frac{2}{3}\)"
    )

    assert extract_answer(text, strict=True) is None
    assert extract_answer(text, strict=False) == r"\frac{2}{3}"
    assert extract_answer("Work ends with the number 42.", strict=False) is None
    assert extract_answer(r"Answer: \boxed{\sqrt{2}}", strict=False) == r"\sqrt{2}"
    assert extract_answer(r"Answer: 1,000;", strict=False) == "1,000"


def test_prm_scoring_is_symbolic_while_hendrycks_is_normalized_string_only():
    assert score_prediction("x+x", "2x", timeout_seconds=1) == (True, "correct")
    assert hendrycks_equivalent("x+x", "2x") is False
    assert hendrycks_equivalent(r"\frac12", r"\frac{1}{2}") is True
    assert score_prediction(None, "2") == (False, "unparsed")
    assert score_prediction("12345", "2", max_answer_chars=4) == (
        False,
        "answer_too_long",
    )


def test_summary_reports_scores_lengths_truncation_and_group_breakdowns():
    results = [
        {
            "subject": "Algebra",
            "level": 1,
            "samples": [
                {
                    "prediction": "2",
                    "flexible_prediction": "2",
                    "correct": True,
                    "flexible_correct": True,
                    "hendrycks_correct": True,
                    "grader_status": "correct",
                    "flexible_grader_status": "correct",
                    "finish_reason": "stop",
                    "output_tokens": 10,
                },
                {
                    "prediction": None,
                    "flexible_prediction": "2",
                    "correct": False,
                    "flexible_correct": True,
                    "hendrycks_correct": False,
                    "grader_status": "unparsed",
                    "flexible_grader_status": "correct",
                    "finish_reason": "length",
                    "output_tokens": 40,
                },
            ],
        },
        {
            "subject": "Geometry",
            "level": 5,
            "samples": [
                {
                    "prediction": "3",
                    "flexible_prediction": "3",
                    "correct": False,
                    "flexible_correct": False,
                    "hendrycks_correct": False,
                    "grader_status": "timeout",
                    "flexible_grader_status": "timeout",
                    "finish_reason": "stop",
                    "output_tokens": 20,
                },
                {
                    "prediction": "4",
                    "flexible_prediction": "4",
                    "correct": True,
                    "flexible_correct": True,
                    "hendrycks_correct": True,
                    "grader_status": "correct",
                    "flexible_grader_status": "correct",
                    "finish_reason": "length",
                    "output_tokens": 30,
                },
            ],
        },
    ]

    summary = summarize(results, 2)

    assert summary["num_questions"] == 2
    assert summary["samples_per_question"] == 2
    assert summary["total_samples"] == 4
    assert summary["correct_samples"] == 2
    assert summary["average_accuracy"] == 0.5
    assert summary["test_at_n"] == 1.0
    assert summary["flexible_correct_samples"] == 3
    assert summary["flexible_accuracy"] == 0.75
    assert summary["flexible_test_at_n"] == 1.0
    assert summary["hendrycks_correct_samples"] == 2
    assert summary["hendrycks_accuracy"] == 0.5
    assert summary["parse_rate"] == 0.75
    assert summary["flexible_parse_rate"] == 1.0
    assert summary["grader_timeout_count"] == 1
    assert summary["grader_error_count"] == 0
    assert summary["answer_too_long_count"] == 0
    assert summary["flexible_grader_timeout_count"] == 1
    assert summary["total_output_tokens"] == 100
    assert summary["average_output_tokens"] == 25
    assert summary["median_output_tokens"] == 25
    assert summary["p90_output_tokens"] == pytest.approx(37)
    assert summary["p95_output_tokens"] == pytest.approx(38.5)
    assert summary["max_output_tokens"] == 40
    assert summary["length_truncated"] == 2
    assert summary["length_truncated_rate"] == 0.5
    assert summary["length_truncated_correct"] == 1
    assert summary["length_truncated_accuracy"] == 0.5
    assert summary["non_truncated_samples"] == 2
    assert summary["non_truncated_correct"] == 1
    assert summary["non_truncated_accuracy"] == 0.5
    assert summary["strict_completion_accuracy"] == 0.25
    assert summary["length_truncated_output_tokens"] == 70
    assert summary["length_truncated_token_rate"] == 0.7
    assert summary["sample_index_accuracy"] == [0.5, 0.5]
    assert summary["by_subject"]["Algebra"]["average_accuracy"] == 0.5
    assert summary["by_subject"]["Algebra"]["flexible_accuracy"] == 1.0
    assert summary["by_level"]["5"]["hendrycks_accuracy"] == 0.5

    with pytest.raises(ValueError, match="exactly 3 samples"):
        summarize(results, 3)


def test_async_worker_refills_per_sample_and_assigns_task_stable_unique_seeds():
    """A fast sample admits replacement while another request remains active."""

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
                            text=rf"Work. \boxed{{{question_index + sample_index}}}",
                            finish_reason="stop",
                            token_ids=[1, 2, 3],
                        )
                    ],
                )
            finally:
                self.active -= 1

    async def wait_until(predicate):
        for _ in range(100):
            if predicate():
                return
            await asyncio.sleep(0.001)
        raise AssertionError("condition was not reached before timeout")

    async def scenario():
        engine = FakeEngine()
        task_queue = Queue()
        result_queue = Queue()
        task_queue.put((0, 0, "q0"))
        task_queue.put((0, 1, "q0"))
        task_queue.put((1, 0, "q1"))
        task_queue.put(None)

        serving = asyncio.create_task(
            _serve_async_requests(
                engine,
                lambda seed: SimpleNamespace(n=1, seed=seed),
                rank=1,
                task_queue=task_queue,
                result_queue=result_queue,
                max_in_flight=2,
                base_seed=100,
                save_responses=False,
            )
        )
        await wait_until(lambda: len(engine.started) == 2)
        assert {entry[0] for entry in engine.started} == {(0, 0), (0, 1)}

        engine.gates[(0, 0)].set()
        await wait_until(lambda: len(engine.started) == 3)
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
        (0, 1): 102,
        (1, 0): 101,
    }
    assert all(entry[4] == 1 for entry in engine.started)
    samples = {
        (message[1], message[2]): message[3]
        for message in messages
        if message[0] == "result"
    }
    assert set(samples) == {(0, 0), (0, 1), (1, 0)}
    assert samples[(1, 0)]["sample_index"] == 0
    assert samples[(1, 0)]["seed"] == 101
    assert samples[(1, 0)]["prediction"] == "1"
    assert "response" not in samples[(1, 0)]


def test_sample_seeds_are_unique_and_stable_when_avg_n_changes():
    avg1 = {_sample_seed(7, question, 0) for question in range(20)}
    avg10 = {
        _sample_seed(7, question, sample)
        for question in range(20)
        for sample in range(10)
    }

    assert avg1 <= avg10
    assert len(avg10) == 200
    assert _sample_seed(7, 12, 0) == _sample_seed(7, 12, 0)
    with pytest.raises(ValueError, match="nonnegative"):
        _sample_seed(7, -1, 0)


def test_result_path_records_dynamic_avg_n_without_duplicate_suffix():
    config = {
        "model_path": "/models/example",
        "output_path": "/results",
        "test_name": "zero-shot",
        "samples_per_question": 3,
        "data_parallel_size": 2,
    }

    assert result_path(config).name == "math500-example-zero-shot-avg3-dp2.json"
    config["test_name"] = "zero-shot-avg3-dp2"
    assert result_path(config).name == "math500-example-zero-shot-avg3-dp2.json"


def test_samples_per_question_must_be_a_positive_integer():
    assert get_samples_per_question({"samples_per_question": 3}) == 3
    for value in (None, 0, -1, 1.5, "3", True):
        with pytest.raises(ValueError, match="positive integer"):
            get_samples_per_question({"samples_per_question": value})


def test_run_writes_avg_n_report_without_loading_a_model(tmp_path, monkeypatch):
    dataset = tmp_path / "MATH-500"
    _write_jsonl(
        dataset / "test.jsonl",
        [
            _dataset_row(
                problem="What is 9 + 9?",
                solution=r"Add the values. \boxed{18}",
                answer="18",
            )
        ],
    )

    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is True
            return f"<chat>{messages[0]['content']}"

        def encode(self, prompt, *, add_special_tokens):
            assert prompt == "<chat>" + PROMPT_TEMPLATE.replace(
                "{problem}", "What is 9 + 9?"
            )
            assert add_special_tokens is False
            return [1, 2, 3]

    monkeypatch.setattr(
        math500,
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
                    "flexible_prediction": "18",
                    "finish_reason": "stop",
                    "output_tokens": 4,
                },
                {
                    "sample_index": 1,
                    "seed": 12,
                    "prediction": None,
                    "flexible_prediction": "18",
                    "finish_reason": "length",
                    "output_tokens": 8,
                },
            ]
        ]

    monkeypatch.setattr(math500, "generate_vllm", fake_generate)
    config = {
        "dataset_path": str(dataset),
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

    report = math500.run(config)
    path = tmp_path / "results" / "math500-toy-model-zero-shot-avg2-dp1.json"

    assert captured["config"] is config
    assert captured["prompts"] == [
        "<chat>" + PROMPT_TEMPLATE.replace("{problem}", "What is 9 + 9?")
    ]
    assert report["benchmark"] == "MATH-500"
    assert report["metric"] == "average_prm800k_grade_answer@2"
    assert report["protocol"]["name"] == math500.PROTOCOL_NAME
    assert report["protocol"]["split"] == "test"
    assert report["protocol"]["num_fewshot"] == 0
    assert report["protocol"]["prompt_template"] == PROMPT_TEMPLATE
    assert report["protocol"]["decoding"] == "sampling"
    assert report["protocol"]["grading"]["primary"] == (
        "OpenAI PRM800K grade_answer"
    )
    assert report["summary"]["average_accuracy"] == 0.5
    assert report["summary"]["test_at_n"] == 1.0
    assert report["summary"]["flexible_accuracy"] == 1.0
    assert report["summary"]["hendrycks_accuracy"] == 0.5
    assert report["predictions"][0]["samples"][1]["grader_status"] == "unparsed"
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == report


def test_main_delegates_parsed_yaml_to_run_models(monkeypatch):
    config = {"models": ["one"]}
    seen = {}
    monkeypatch.setattr(math500, "parse_args_yaml", lambda description: config)

    def fake_run_models(received_config, runner, benchmark):
        seen["call"] = (received_config, runner, benchmark)

    monkeypatch.setattr(math500, "run_models", fake_run_models)

    math500.main()

    assert seen["call"] == (config, math500.run, "MATH-500")
