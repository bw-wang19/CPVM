"""Shared MedMCQA prompt and answer-formatting protocol."""

from __future__ import annotations

from collections.abc import Mapping
import re


LETTERS = "ABCD"
PROMPT_TEMPLATE = """Question: {question}
Choices:
A. {opa}
B. {opb}
C. {opc}
D. {opd}
Respond with the correct option letter. On the last line, put only the final answer letter inside \\boxed{{}} (for example, \\boxed{{A}}).
Answer:"""
BOXED_ANSWER_RE = re.compile(r"\\boxed\s*\{\s*([A-Da-d])\s*\}")


def answer_letter(cop):
    """Convert MedMCQA's zero-based ``cop`` label to A--D."""

    if type(cop) is not int or not 0 <= cop < len(LETTERS):
        raise ValueError(f"MedMCQA cop must be an integer from 0 to 3, got {cop!r}")
    return LETTERS[cop]


def boxed_answer(letter):
    """Return the canonical assistant-side final answer."""

    if not isinstance(letter, str) or len(letter) != 1 or letter.upper() not in LETTERS:
        raise ValueError(
            f"MedMCQA answer letter must be one of A, B, C, D, got {letter!r}"
        )
    return rf"\boxed{{{letter.upper()}}}"


def format_prompt(row: Mapping):
    """Format one raw MedMCQA row using the fixed generative protocol."""

    if not isinstance(row, Mapping):
        raise TypeError("MedMCQA row must be a mapping")
    fields = ("question", "opa", "opb", "opc", "opd")
    values = {}
    for name in fields:
        value = row.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"MedMCQA {name} must be a nonempty string")
        values[name] = value.strip()
    return PROMPT_TEMPLATE.format(**values)


def extract_boxed_answer(text):
    """Extract the final boxed A--D letter, or None when none is present."""

    matches = BOXED_ANSWER_RE.findall(text)
    return matches[-1].upper() if matches else None
