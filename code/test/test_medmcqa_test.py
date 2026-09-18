"""Lightweight regression tests for the MedMCQA evaluator."""

import asyncio
import json
from queue import Queue
from types import SimpleNamespace

import pandas as pd
import pytest

import CPVM.code.test.medmcqa_test as medmcqa
from CPVM.code.test.medmcqa_test import (
    _sample_seed,
    _serve_async_requests,
    build_prompts,
    extract_answer,
    get_samples_per_question,
    load_questions,
    result_path,
    summarize,
)
from CPVM.code.utils.medmcqa import (
    PROMPT_TEMPLATE as SHARED_PROMPT_TEMPLATE,
    answer_letter,
    boxed_answer,
    format_prompt,
)


def _dataset_row(**overrides):
    row = {
        "id": "question-1",
        "question": "Which vitamin is found in animal foods?",
        "opa": "Vitamin C",
        "opb": "Vitamin B7",
        "opc": "Vitamin B12",
        "opd": "Vitamin K",
        "cop": 2,
        "choice_type": "single",
        "exp": "Vitamin B12 is the answer.",
        "subject_name": "Biochemistry",
        "topic_name": "Vitamins",
    }
    row.update(overrides)
    return row


def _write_split(root, split, rows):
    path = root / "data" / f"{split}-00000-of-00001.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_shared_protocol_validates_zero_based_labels_and_formats_answers():
    assert [answer_letter(index) for index in range(4)] == list("ABCD")
    assert boxed_answer("c") == r"\boxed{C}"
    for invalid in (-1, 4, True, "2"):
        with pytest.raises(ValueError):
            answer_letter(invalid)
    for invalid in ("", "AA", "E", 1):
        with pytest.raises(ValueError):
            boxed_answer(invalid)


def test_load_questions_reads_labelled_validation_and_preserves_metadata(tmp_path):
    dataset = tmp_path / "medmcqa"
    _write_split(
        dataset,
        "validation",
        [
            _dataset_row(),
            _dataset_row(
                id="question-2",
                question="  Second question? ",
                opa=" First ",
                opb=" Second ",
                opc=" Third ",
                opd=" Fourth ",
                cop=0,
                choice_type="multi",
                subject_name="Medicine",
                topic_name=None,
            ),
        ],
    )

    questions = load_questions(dataset, "validation")

    assert len(questions) == 2
    assert questions[0]["answer"] == "C"
    assert questions[0]["subject"] == "Biochemistry"
    assert questions[0]["topic"] == "Vitamins"
    assert questions[1]["answer"] == "A"
    assert questions[1]["question"] == "Second question?"
    assert questions[1]["choices"] == ["First", "Second", "Third", "Fourth"]
    assert questions[1]["choice_type"] == "multi"
    assert questions[1]["topic"] is None


def test_load_questions_rejects_unlabelled_public_test_split(tmp_path):
    dataset = tmp_path / "medmcqa"
    _write_split(dataset, "test", [_dataset_row(cop=-1)])

    with pytest.raises(ValueError, match="public test split is unlabelled"):
        load_questions(dataset, "test")


def test_prompt_is_shared_and_applied_as_one_chat_message():
    class FakeTokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            self.calls.append((messages, tokenize, add_generation_prompt))
            return "CHAT:" + messages[0]["content"]

    row = _dataset_row()
    question = {
        "question": row["question"],
        "opa": row["opa"],
        "opb": row["opb"],
        "opc": row["opc"],
        "opd": row["opd"],
        "choices": [row["opa"], row["opb"], row["opc"], row["opd"]],
    }
    tokenizer = FakeTokenizer()

    prompts = build_prompts([question], tokenizer)

    assert medmcqa.PROMPT_TEMPLATE == SHARED_PROMPT_TEMPLATE
    assert tokenizer.calls == [
        ([{"role": "user", "content": format_prompt(row)}], False, True)
    ]
    assert prompts == ["CHAT:" + format_prompt(row)]
    assert format_prompt(row).endswith("Answer:")


def test_answer_extraction_uses_the_final_strict_boxed_letter():
    assert extract_answer(r"Reasoning. \boxed{b}") == "B"
    assert extract_answer(r"First \boxed{A}; finally \boxed { d }") == "D"
    assert extract_answer("The answer is C") is None
    assert extract_answer(r"\boxed{AB}") is None


def test_samples_and_seed_mapping_are_valid_stable_and_unique():
    assert get_samples_per_question({"samples_per_question": 3}) == 3
    for invalid in (None, 0, -1, 1.5, "3", True):
        with pytest.raises(ValueError, match="positive integer"):
            get_samples_per_question({"samples_per_question": invalid})

    avg1 = {_sample_seed(7, question, 0) for question in range(20)}
    avg5 = {
        _sample_seed(7, question, sample)
        for question in range(20)
        for sample in range(5)
    }
    assert len(avg5) == 100
    assert avg1 <= avg5


def test_summary_reports_accuracy_length_truncation_and_groups():
    results = [
        {
            "subject": "Medicine",
            "choice_type": "single",
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
                    "output_tokens": 50,
                },
            ],
        },
        {
            "subject": "Medicine",
            "choice_type": "multi",
            "samples": [
                {
                    "prediction": "C",
                    "correct": False,
                    "finish_reason": "stop",
                    "output_tokens": 20,
                },
                {
                    "prediction": "D",
                    "correct": True,
                    "finish_reason": "length",
                    "output_tokens": 40,
                },
            ],
        },
        {
            "subject": "Surgery",
            "choice_type": "single",
            "samples": [
                {
                    "prediction": "B",
                    "correct": True,
                    "finish_reason": "stop",
                    "output_tokens": 30,
                },
                {
                    "prediction": "A",
                    "correct": False,
                    "finish_reason": "stop",
                    "output_tokens": 60,
                },
            ],
        },
    ]

    summary = summarize(results, 2)

    assert summary["num_questions"] == 3
    assert summary["total_samples"] == 6
    assert summary["correct_samples"] == 3
    assert summary["average_accuracy"] == 0.5
    assert summary["test_at_n"] == 1.0
    assert summary["parse_rate"] == pytest.approx(5 / 6)
    assert summary["average_output_tokens"] == 35
    assert summary["median_output_tokens"] == 35
    assert summary["p90_output_tokens"] == 55
    assert summary["p95_output_tokens"] == 57.5
    assert summary["max_output_tokens"] == 60
    assert summary["length_truncated_rate"] == pytest.approx(2 / 6)
    assert summary["length_truncated_accuracy"] == 0.5
    assert summary["non_truncated_accuracy"] == 0.5
    assert summary["strict_completion_accuracy"] == pytest.approx(2 / 6)
    assert summary["length_truncated_token_rate"] == pytest.approx(90 / 210)
    assert summary["sample_index_accuracy"] == [pytest.approx(2 / 3), pytest.approx(1 / 3)]
    assert summary["by_subject"]["Medicine"]["average_accuracy"] == 0.5
    assert summary["by_subject"]["Surgery"]["num_questions"] == 1
    assert summary["by_choice_type"]["multi"]["test_at_n"] == 1.0


def test_async_worker_refills_at_sample_granularity_with_unique_seeds():
    class FakeEngine:
        def __init__(self):
            self.gates = {key: asyncio.Event() for key in ((0, 0), (0, 1), (1, 0))}
            self.started = []

        async def generate(self, prompt, sampling, request_id):
            question_index, sample_index = map(int, request_id.rsplit("-", 2)[-2:])
            key = (question_index, sample_index)
            self.started.append((key, sampling.seed, sampling.n))
            await self.gates[key].wait()
            yield SimpleNamespace(
                finished=True,
                outputs=[
                    SimpleNamespace(
                        index=0,
                        text=r"\boxed{A}",
                        finish_reason="stop",
                        token_ids=[1, 2],
                    )
                ],
            )

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
        assert len(engine.started) == 3
        engine.gates[(0, 1)].set()
        engine.gates[(1, 0)].set()
        await asyncio.wait_for(serving, timeout=1)
        return engine

    engine = asyncio.run(scenario())
    assert len({entry[1] for entry in engine.started}) == 3
    assert all(entry[2] == 1 for entry in engine.started)


def test_result_path_and_run_report_dynamic_avg_n_without_model(tmp_path, monkeypatch):
    dataset = tmp_path / "medmcqa"
    _write_split(dataset, "validation", [_dataset_row()])

    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is True
            return "<chat>" + messages[0]["content"]

        def encode(self, prompt, *, add_special_tokens):
            assert prompt.startswith("<chat>Question:")
            assert add_special_tokens is False
            return [1, 2, 3]

    monkeypatch.setattr(
        medmcqa,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer()),
    )
    monkeypatch.setattr(
        medmcqa,
        "generate_vllm",
        lambda prompts, config: [
            [
                {
                    "sample_index": 0,
                    "seed": 11,
                    "prediction": "C",
                    "finish_reason": "stop",
                    "output_tokens": 4,
                },
                {
                    "sample_index": 1,
                    "seed": 12,
                    "prediction": "A",
                    "finish_reason": "length",
                    "output_tokens": 8,
                },
            ]
        ],
    )
    config = {
        "dataset_path": str(dataset),
        "split": "validation",
        "expected_num_questions": 1,
        "model_path": str(tmp_path / "toy-model"),
        "output_path": str(tmp_path / "results"),
        "test_name": "generative",
        "samples_per_question": 2,
        "data_parallel_size": 1,
        "temperature": 0.7,
        "max_new_tokens": 16,
        "max_model_len": 64,
    }

    report = medmcqa.run(config)
    path = tmp_path / "results" / "medmcqa-toy-model-generative-avg2-dp1.json"

    assert result_path(config) == path
    assert report["metric"] == "average_accuracy@2"
    assert report["protocol"]["split"] == "validation"
    assert report["protocol"]["prompt_template"] == SHARED_PROMPT_TEMPLATE
    assert report["summary"]["average_accuracy"] == 0.5
    assert report["summary"]["test_at_n"] == 1.0
    assert report["predictions"][0]["answer"] == "C"
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == report


def test_main_delegates_to_serial_multi_model_runner(monkeypatch):
    config = {"model_paths": ["one", "two"]}
    seen = {}
    monkeypatch.setattr(medmcqa, "parse_args_yaml", lambda description: config)

    def fake_run_models(received_config, runner, benchmark):
        seen["call"] = (received_config, runner, benchmark)

    monkeypatch.setattr(medmcqa, "run_models", fake_run_models)
    medmcqa.main()

    assert seen["call"] == (config, medmcqa.run, "MedMCQA")
