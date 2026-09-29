from __future__ import annotations

from typing import Any

import pytest
import torch

from main_code_relonation.config import RationaleConfig
from main_code_relonation.data import RationaleCollator, RationaleExample
from main_code_relonation.modeling import assistant_supervision_ids


class FakeTokenizer:
    chat_template = "fake-v1"
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **_: Any,
    ) -> list[int]:
        assert tokenize
        user_ids = [10, *[100 + ord(char) % 31 for char in messages[0]["content"]]]
        if add_generation_prompt:
            return [*user_ids, 20]
        target = messages[-1]["content"]
        return [*user_ids, 20, *[200 + ord(char) % 31 for char in target], 2]

    def pad(self, features: list[dict[str, list[int]]], **_: Any) -> dict[str, torch.Tensor]:
        width = max(len(item["input_ids"]) for item in features)
        ids = [item["input_ids"] + [0] * (width - len(item["input_ids"])) for item in features]
        masks = [item["attention_mask"] + [0] * (width - len(item["attention_mask"])) for item in features]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class BrokenPrefixTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> list[int]:
        value = super().apply_chat_template(messages, **kwargs)
        if not kwargs["add_generation_prompt"]:
            value[0] = 99
        return value


def config() -> RationaleConfig:
    return RationaleConfig(model_id="fake", max_length=10000)


def pseudo_row(identifier: str = "e1") -> dict[str, Any]:
    return {
        "id": identifier,
        "prompt": "논제",
        "essay": "에세이 원문",
        "source_split": "train",
        "conditioning_scores": {
            "content": 3.25,
            "organization": 4.0,
            "expression": 2.75,
        },
        "judge": {
            "content": {"score": 3.25, "rationale": "내용 근거"},
            "organization": {"score": 4.0, "rationale": "조직 근거"},
            "expression": {"score": 2.75, "rationale": "표현 근거"},
        },
    }


def test_only_assistant_json_and_template_terminal_are_supervised() -> None:
    tokenizer = FakeTokenizer()
    collator = RationaleCollator(tokenizer, config())
    ids, labels, boundary = collator.encode_row(pseudo_row())
    assert labels[:boundary] == [-100] * boundary
    assert labels[boundary:] == ids[boundary:]
    batch = collator([RationaleExample(pseudo_row())])
    assert batch["labels"].shape == batch["input_ids"].shape


def test_chat_template_prefix_mismatch_fails_loudly() -> None:
    with pytest.raises(ValueError, match="prefix"):
        assistant_supervision_ids(
            BrokenPrefixTokenizer(),
            [{"role": "user", "content": "x"}],
            "{}",
            config=config(),
        )


def test_conditioning_and_target_score_mismatch_fails() -> None:
    row = pseudo_row()
    row["judge"]["content"]["score"] = 4.0
    with pytest.raises(ValueError, match="conditioning score"):
        RationaleCollator(FakeTokenizer(), config()).encode_row(row)
