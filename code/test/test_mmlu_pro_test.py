from CPVM.code.test.mmlu_pro_test import (
    build_raw_prompt,
    extract_answer,
    summarize_results,
)


def example(question_id, answer="A", category="math"):
    return {
        "question_id": question_id,
        "question": f"Question {question_id}?",
        "options": ["one", "two", "three"],
        "answer": answer,
        "cot_content": (
            f"A: Let's think step by step. Reasoning {question_id}. "
            f"The answer is ({answer})."
        ),
        "category": category,
    }


def test_official_prompt_contains_demos_and_open_test_answer():
    demos = [example(index, answer="B") for index in range(5)]
    prompt = build_raw_prompt(example(99, answer="C"), demos)

    assert prompt.count("Question:\n") == 6
    assert prompt.count("The answer is (B).") == 5
    assert prompt.endswith("Answer: Let's think step by step.")
    assert "The answer is (C)." not in prompt


def test_answer_extraction_cascade():
    assert extract_answer("Therefore, the answer is (B).") == ("B", "answer_is")
    assert extract_answer("Reasoning\nAnswer: C") == ("C", "answer_colon")
    assert extract_answer("We compared A and finally B") == ("B", "last_letter")
    assert extract_answer("No final selection") == (None, "unparsed")


def test_summary_reports_accuracy_and_parse_rate():
    rows = [
        {
            "category": "a",
            "correct": True,
            "strict_correct": True,
            "parse_failed": False,
        },
        {
            "category": "a",
            "correct": False,
            "strict_correct": False,
            "parse_failed": True,
        },
        {
            "category": "b",
            "correct": True,
            "strict_correct": True,
            "parse_failed": False,
        },
    ]

    summary, categories = summarize_results(rows)
    assert summary["accuracy"] == 2 / 3
    assert summary["parse_rate"] == 2 / 3
    assert categories["a"]["accuracy"] == 0.5
