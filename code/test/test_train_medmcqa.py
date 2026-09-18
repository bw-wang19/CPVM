"""Lightweight tests for MedMCQA SFT preprocessing; no model is loaded."""

from __future__ import annotations

from datasets import Dataset, DatasetDict
import pytest

import CPVM.code.train_medmcqa as train_medmcqa
from CPVM.code.train_medmcqa import (
    build_assistant_target,
    load_official_splits,
    tokenize_medmcqa_example,
)
from CPVM.code.utils.medmcqa import format_prompt


def _row(**overrides):
    row = {
        "id": "question-1",
        "question": "Which vitamin deficiency causes scurvy?",
        "opa": "Vitamin A",
        "opb": "Vitamin B12",
        "opc": "Vitamin C",
        "opd": "Vitamin D",
        "cop": 2,
        "choice_type": "single",
        "exp": "Scurvy is caused by vitamin C deficiency.",
    }
    row.update(overrides)
    return row


class FakeChatTokenizer:
    """Small prefix-preserving chat tokenizer for preprocessing tests."""

    def __init__(self, target_tokens=3):
        self.target_tokens = target_tokens
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        self.calls.append((messages, add_generation_prompt))
        prefix = [10, 11, 12, 13]
        if add_generation_prompt:
            assert len(messages) == 1
            return prefix
        assert len(messages) == 2
        return prefix + list(range(20, 20 + self.target_tokens))


def test_target_uses_optional_explanation_and_always_ends_boxed_letter():
    assert build_assistant_target(_row(), True) == (
        "Scurvy is caused by vitamin C deficiency.\n\n\\boxed{C}"
    )
    assert build_assistant_target(_row(exp=None), True) == r"\boxed{C}"
    assert build_assistant_target(_row(), False) == r"\boxed{C}"
    with pytest.raises(ValueError, match="0 to 3"):
        build_assistant_target(_row(cop=-1), True)


def test_tokenization_reuses_shared_prompt_and_masks_every_prompt_token():
    tokenizer = FakeChatTokenizer(target_tokens=3)

    encoded = tokenize_medmcqa_example(
        _row(),
        tokenizer=tokenizer,
        max_length=16,
        include_explanation=True,
        drop_overlength=True,
    )

    assert tokenizer.calls[0] == (
        [{"role": "user", "content": format_prompt(_row())}],
        True,
    )
    assert tokenizer.calls[1][0][1] == {
        "role": "assistant",
        "content": (
            "Scurvy is caused by vitamin C deficiency.\n\n\\boxed{C}"
        ),
    }
    assert encoded["input_ids"] == [10, 11, 12, 13, 20, 21, 22]
    assert encoded["labels"] == [-100, -100, -100, -100, 20, 21, 22]
    assert encoded["attention_mask"] == [1] * 7
    assert encoded["_keep"] is True


def test_overlength_is_dropped_or_raises_but_is_never_truncated():
    tokenizer = FakeChatTokenizer(target_tokens=5)
    dropped = tokenize_medmcqa_example(
        _row(),
        tokenizer=tokenizer,
        max_length=8,
        include_explanation=True,
        drop_overlength=True,
    )
    assert dropped == {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "_keep": False,
        "_sequence_length": 9,
    }

    with pytest.raises(ValueError, match="needs 9 tokens"):
        tokenize_medmcqa_example(
            _row(),
            tokenizer=tokenizer,
            max_length=8,
            include_explanation=True,
            drop_overlength=False,
        )


def test_loader_passes_only_official_train_and_validation_files(
    tmp_path, monkeypatch
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train-00000-of-00001.parquet").touch()
    (data_dir / "validation-00000-of-00001.parquet").touch()
    (data_dir / "test-00000-of-00001.parquet").touch()

    train = Dataset.from_list([_row(id="train-id")])
    validation = Dataset.from_list([_row(id="validation-id")])
    captured = {}

    def fake_load_dataset(kind, *, data_files):
        captured["kind"] = kind
        captured["data_files"] = data_files
        return DatasetDict(train=train, validation=validation)

    monkeypatch.setattr(train_medmcqa, "load_dataset", fake_load_dataset)
    loaded = load_official_splits(str(tmp_path))

    assert captured["kind"] == "parquet"
    assert set(captured["data_files"]) == {"train", "validation"}
    assert all(
        "test-" not in path
        for paths in captured["data_files"].values()
        for path in paths
    )
    assert list(loaded) == ["train", "validation"]


def test_loader_rejects_train_validation_id_leakage(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train-00000-of-00001.parquet").touch()
    (data_dir / "validation-00000-of-00001.parquet").touch()
    same_row = Dataset.from_list([_row(id="duplicate-id")])
    monkeypatch.setattr(
        train_medmcqa,
        "load_dataset",
        lambda *args, **kwargs: DatasetDict(
            train=same_row,
            validation=same_row,
        ),
    )

    with pytest.raises(ValueError, match="IDs overlap"):
        load_official_splits(str(tmp_path))
