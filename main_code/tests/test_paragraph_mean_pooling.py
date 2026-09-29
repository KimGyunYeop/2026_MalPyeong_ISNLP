from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from main_code.config import TRAITS
from main_code.datasets import (
    EssayRegressionDataset,
    RegressionCollator,
    _paragraph_segment_ends,
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
        encoded = {"input_ids": [1] * length, "attention_mask": [1] * length}
        if kwargs.get("return_offsets_mapping"):
            encoded["offset_mapping"] = [(index, index + 1) for index in range(length)]
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


class ParagraphSegmentEndsTest(unittest.TestCase):
    def test_double_space_starts_a_new_paragraph(self) -> None:
        essay = "첫 문단이다.  둘째 문단이다.   셋째 문단이다."
        ends = _paragraph_segment_ends(essay)
        self.assertEqual(len(ends), 3)
        self.assertEqual(ends[-1], len(essay))

    def test_essay_without_the_cue_is_one_paragraph(self) -> None:
        essay = "한 칸 공백만 있는 글이다. 문단 경계가 없다."
        self.assertEqual(_paragraph_segment_ends(essay), [len(essay)])

    def test_single_space_is_not_a_paragraph_cue(self) -> None:
        self.assertEqual(len(_paragraph_segment_ends("가 나 다")), 1)


class ParagraphMeanCollatorTest(unittest.TestCase):
    def test_paragraph_ids_cover_only_essay_and_padding(self) -> None:
        config = legacy_config(
            essay_surface="official_raw",
            organization_pooling="paragraph_mean",
            max_length=4096,
        ).validate()
        rows = [
            _row("three", "첫 문단.  둘째 문단.  셋째 문단."),
            _row("one", "문단 경계가 없는 글."),
        ]
        dataset = EssayRegressionDataset(
            rows, config, split="validation", require_labels=True
        )
        batch = RegressionCollator(
            _CharacterTokenizer(), config, include_labels=True, include_metadata=False
        )([dataset[0], dataset[1]])

        self.assertEqual(batch["paragraph_ids"].shape, batch["input_ids"].shape)
        self.assertNotIn("sentence_ids", batch)
        self.assertTrue(torch.all(batch["paragraph_ids"][batch["essay_mask"] == 0] < 0))
        self.assertTrue(
            torch.all(batch["paragraph_ids"][batch["attention_mask"] == 0] == -1)
        )
        first = torch.unique(batch["paragraph_ids"][0][batch["essay_mask"][0].bool()])
        second = torch.unique(batch["paragraph_ids"][1][batch["essay_mask"][1].bool()])
        self.assertEqual(first.tolist(), [0, 1, 2])
        self.assertEqual(second.tolist(), [0])

    def test_left_padding_uses_negative_paragraph_id(self) -> None:
        config = legacy_config(
            essay_surface="official_raw",
            organization_pooling="paragraph_mean",
            max_length=4096,
        ).validate()
        tokenizer = _CharacterTokenizer()
        tokenizer.padding_side = "left"
        rows = [_row("long", "첫 문단.  둘째 문단."), _row("short", "짧다.")]
        batch = RegressionCollator(
            tokenizer, config, include_labels=True, include_metadata=False
        )(rows)
        self.assertTrue(
            torch.all(batch["paragraph_ids"][batch["attention_mask"] == 0] == -1)
        )
        self.assertTrue(torch.all(batch["paragraph_ids"][batch["essay_mask"] == 0] < 0))


class ParagraphMeanModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.input_ids = torch.ones((1, 8), dtype=torch.long)
        self.attention_mask = torch.ones_like(self.input_ids)
        self.essay_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1]])
        # 앞 문단은 token 2개, 뒤 문단은 5개다. 문단마다 같은 가중치를 주면
        # token 수로 평균하는 essay_mean과 값이 달라진다.
        self.paragraph_ids = torch.tensor([[-1, 0, 0, 1, 1, 1, 1, 1]])

    def _scorer(self) -> RegressionScorer:
        return RegressionScorer(
            _PositionBackbone(),
            3,
            legacy_config(
                pooling="mean",
                organization_pooling="paragraph_mean",
                essay_surface="official_raw",
            ),
        )

    def test_zero_init_is_shared_and_only_organization_can_change(self) -> None:
        scorer = self._scorer()
        initial = scorer.encode(
            self.input_ids,
            self.attention_mask,
            essay_mask=self.essay_mask,
            paragraph_ids=self.paragraph_ids,
        )
        self.assertEqual(tuple(initial.shape), (1, len(TRAITS), 3))
        torch.testing.assert_close(initial[:, 0], initial[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(initial[:, 1], initial[:, 2], rtol=0, atol=0)

        initial[0, 1, 0].backward()
        assert scorer.organization_paragraph_weight is not None
        self.assertIsNotNone(scorer.organization_paragraph_weight.grad)
        self.assertNotEqual(float(scorer.organization_paragraph_weight.grad.abs()), 0.0)

        with torch.no_grad():
            scorer.organization_paragraph_weight.fill_(1.0)
        changed = scorer.encode(
            self.input_ids,
            self.attention_mask,
            essay_mask=self.essay_mask,
            paragraph_ids=self.paragraph_ids,
        )
        torch.testing.assert_close(changed[:, 0], initial[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(changed[:, 2], initial[:, 2], rtol=0, atol=0)
        self.assertFalse(torch.equal(changed[:, 1], initial[:, 1]))
        self.assertIn("organization_paragraph_weight", scorer.scoring_state_dict())

    def test_one_paragraph_falls_back_to_shared(self) -> None:
        scorer = self._scorer()
        assert scorer.organization_paragraph_weight is not None
        with torch.no_grad():
            scorer.organization_paragraph_weight.fill_(1.0)
        single = torch.where(
            self.essay_mask.bool(),
            torch.zeros_like(self.paragraph_ids),
            torch.full_like(self.paragraph_ids, -1),
        )
        features = scorer.encode(
            self.input_ids,
            self.attention_mask,
            essay_mask=self.essay_mask,
            paragraph_ids=single,
        )
        self.assertTrue(torch.isfinite(features).all())
        torch.testing.assert_close(features[:, 0], features[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(features[:, 1], features[:, 2], rtol=0, atol=0)

    def test_paragraph_ids_are_required(self) -> None:
        scorer = self._scorer()
        with self.assertRaisesRegex(ValueError, "paragraph_ids"):
            scorer.encode(
                self.input_ids, self.attention_mask, essay_mask=self.essay_mask
            )


if __name__ == "__main__":
    unittest.main()
