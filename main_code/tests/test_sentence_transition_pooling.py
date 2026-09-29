from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from main_code.config import TRAITS
from main_code.datasets import (
    EssayRegressionDataset,
    RegressionCollator,
    _sentence_end_offsets,
)
from main_code.models import RegressionScorer
from main_code.tests.config_helpers import legacy_config


class _CharacterTokenizer:
    """One character per token, including exact offsets, for structure tests."""

    model_max_length = 4096
    padding_side = "right"
    all_special_ids = [0]

    def __call__(self, text: str, **kwargs):
        length = min(len(text), int(kwargs.get("max_length", len(text))))
        encoded = {
            "input_ids": [1] * length,
            "attention_mask": [1] * length,
        }
        if kwargs.get("return_offsets_mapping"):
            encoded["offset_mapping"] = [
                (index, index + 1) for index in range(length)
            ]
        return encoded

    def pad(self, examples, *, padding, return_tensors):
        if padding is not True or return_tensors != "pt":
            raise AssertionError("unexpected padding arguments")
        width = max(len(example["input_ids"]) for example in examples)
        input_ids, attention_mask = [], []
        for example in examples:
            missing = width - len(example["input_ids"])
            if self.padding_side == "left":
                input_ids.append([0] * missing + example["input_ids"])
                attention_mask.append([0] * missing + example["attention_mask"])
            else:
                input_ids.append(example["input_ids"] + [0] * missing)
                attention_mask.append(example["attention_mask"] + [0] * missing)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class _PositionBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, input_ids: torch.Tensor, **_):
        positions = torch.arange(
            input_ids.shape[1], device=input_ids.device, dtype=torch.float32
        )
        hidden = torch.stack(
            (positions, positions.square(), torch.sin(positions)), dim=-1
        )
        hidden = hidden.unsqueeze(0).expand(input_ids.shape[0], -1, -1)
        return SimpleNamespace(last_hidden_state=hidden + self.anchor * 0)


def _row(item_id: str, essay: str) -> dict:
    return {
        "id": item_id,
        "prompt_num": "Q1",
        "prompt": "논제",
        "essay": essay,
        "score": {"content": 3.0, "organization": 3.0, "expression": 3.0},
    }


class SentenceTransitionCollatorTest(unittest.TestCase):
    def test_sentence_split_handles_quotes_and_final_fragment(self) -> None:
        essay = '첫 문장이다. "둘째 문장이다!" 마지막 조각'
        ends = _sentence_end_offsets(essay)
        self.assertEqual(len(ends), 3)
        self.assertEqual(ends[-1], len(essay))
        self.assertEqual(len(_sentence_end_offsets("첫 문장.\n\n둘째 문장.")), 2)

    def test_sentence_ids_cover_only_essay_and_padding(self) -> None:
        config = legacy_config(
            essay_surface="flat",
            organization_pooling="sentence_transition",
            max_length=4096,
        ).validate()
        rows = [
            _row("long", '첫 문장이다. "둘째 문장이다!" 마지막 조각'),
            _row("short", "한 문장이다."),
        ]
        dataset = EssayRegressionDataset(
            rows, config, split="validation", require_labels=True
        )
        batch = RegressionCollator(
            _CharacterTokenizer(),
            config,
            include_labels=True,
            include_metadata=False,
        )([dataset[0], dataset[1]])

        self.assertEqual(batch["sentence_ids"].shape, batch["input_ids"].shape)
        self.assertTrue(torch.all(batch["sentence_ids"][batch["essay_mask"] == 0] < 0))
        self.assertTrue(
            torch.all(batch["sentence_ids"][batch["attention_mask"] == 0] == -1)
        )
        first_ids = torch.unique_consecutive(
            batch["sentence_ids"][0][batch["essay_mask"][0].bool()]
        )
        second_ids = torch.unique_consecutive(
            batch["sentence_ids"][1][batch["essay_mask"][1].bool()]
        )
        self.assertEqual(first_ids.tolist(), [0, 1, 2])
        self.assertEqual(second_ids.tolist(), [0])

    def test_left_padding_uses_negative_sentence_id(self) -> None:
        config = legacy_config(
            essay_surface="flat",
            organization_pooling="sentence_transition",
            max_length=4096,
        ).validate()
        tokenizer = _CharacterTokenizer()
        tokenizer.padding_side = "left"
        rows = [_row("long", "첫 문장. 둘째 문장."), _row("short", "짧다.")]
        batch = RegressionCollator(
            tokenizer, config, include_labels=True, include_metadata=False
        )(rows)
        self.assertTrue(
            torch.all(batch["sentence_ids"][batch["attention_mask"] == 0] == -1)
        )
        self.assertTrue(torch.all(batch["sentence_ids"][batch["essay_mask"] == 0] < 0))


class SentenceTransitionModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.input_ids = torch.ones((1, 8), dtype=torch.long)
        self.attention_mask = torch.ones_like(self.input_ids)
        self.essay_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1]])
        self.sentence_ids = torch.tensor([[-1, 0, 0, 1, 1, 2, 2, 2]])

    def test_zero_init_is_shared_and_only_organization_can_change(self) -> None:
        scorer = RegressionScorer(
            _PositionBackbone(),
            3,
            legacy_config(
                pooling="mean", organization_pooling="sentence_transition"
            ),
        )
        initial = scorer.encode(
            self.input_ids,
            self.attention_mask,
            essay_mask=self.essay_mask,
            sentence_ids=self.sentence_ids,
        )
        self.assertEqual(tuple(initial.shape), (1, len(TRAITS), 3))
        torch.testing.assert_close(initial[:, 0], initial[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(initial[:, 1], initial[:, 2], rtol=0, atol=0)

        initial[0, 1, 0].backward()
        assert scorer.organization_transition_weight is not None
        self.assertIsNotNone(scorer.organization_transition_weight.grad)
        self.assertNotEqual(
            float(scorer.organization_transition_weight.grad.abs()), 0.0
        )

        with torch.no_grad():
            scorer.organization_transition_weight.fill_(1.0)
        changed = scorer.encode(
            self.input_ids,
            self.attention_mask,
            essay_mask=self.essay_mask,
            sentence_ids=self.sentence_ids,
        )
        torch.testing.assert_close(changed[:, 0], initial[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(changed[:, 2], initial[:, 2], rtol=0, atol=0)
        self.assertFalse(torch.equal(changed[:, 1], initial[:, 1]))
        self.assertIn(
            "organization_transition_weight", scorer.scoring_state_dict()
        )

    def test_one_sentence_falls_back_to_shared(self) -> None:
        scorer = RegressionScorer(
            _PositionBackbone(),
            3,
            legacy_config(
                pooling="mean", organization_pooling="sentence_transition"
            ),
        )
        assert scorer.organization_transition_weight is not None
        with torch.no_grad():
            scorer.organization_transition_weight.fill_(1.0)
        sentence_ids = torch.where(
            self.essay_mask.bool(),
            torch.zeros_like(self.sentence_ids),
            torch.full_like(self.sentence_ids, -1),
        )
        features = scorer.encode(
            self.input_ids,
            self.attention_mask,
            essay_mask=self.essay_mask,
            sentence_ids=sentence_ids,
        )
        self.assertTrue(torch.isfinite(features).all())
        torch.testing.assert_close(features[:, 0], features[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(features[:, 1], features[:, 2], rtol=0, atol=0)

    def test_sentence_ids_are_required(self) -> None:
        scorer = RegressionScorer(
            _PositionBackbone(),
            3,
            legacy_config(
                pooling="mean", organization_pooling="sentence_transition"
            ),
        )
        with self.assertRaisesRegex(ValueError, "sentence_ids"):
            scorer.encode(
                self.input_ids,
                self.attention_mask,
                essay_mask=self.essay_mask,
            )


if __name__ == "__main__":
    unittest.main()
