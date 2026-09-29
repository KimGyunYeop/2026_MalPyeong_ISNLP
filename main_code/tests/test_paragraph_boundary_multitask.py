from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from main_code.config import RegressionConfig
from main_code.datasets import RegressionCollator, format_input_segments
from main_code.models import RegressionScorer
from main_code.tests.config_helpers import legacy_config


class _CharacterTokenizer:
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


def _row(item_id: str, *, validation: bool = False, suffix: str = "") -> dict:
    canonical = f"첫 문장.\n\n둘째 문장. 셋째 문장.{suffix}"
    official_raw = f"첫 문장.  둘째 문장. 셋째 문장.{suffix}"
    return {
        "id": item_id,
        "document_id": f"doc-{item_id}",
        "prompt_num": "Q1",
        "prompt": "논제",
        "essay": canonical,
        "essay_surfaces": {"official_raw": official_raw},
        "metadata": {"paragraph_count": 2},
        "source_dataset": "nikl_competition",
        "dataset_group": "competition",
        "source_split": "official_validation" if validation else "origin_pool_extra",
        "score": {"content": 3.0, "organization": 3.0, "expression": 3.0},
    }


def _config(**updates) -> RegressionConfig:
    values = {
        "training_mode": "lora_only",
        "essay_surface": "official_raw",
        "pooling": "mean",
        "organization_pooling": "shared",
        "paragraph_boundary_loss_weight": 0.1,
    }
    values.update(updates)
    return legacy_config(**values).validate()


class ParagraphBoundaryCollatorTest(unittest.TestCase):
    def test_next_sentence_tokens_receive_positive_then_negative_labels(self) -> None:
        config = _config()
        row = _row("train")
        tokenizer = _CharacterTokenizer()
        collator = RegressionCollator(
            tokenizer, config, include_labels=True, include_metadata=False
        )
        batch = collator([row])

        positions = torch.nonzero(batch["paragraph_boundary_mask"][0]).flatten()
        labels = batch["paragraph_boundary_labels"][0, positions]
        self.assertEqual(labels.tolist(), [1.0, 0.0])

        text = format_input_segments(row, config).text
        self.assertEqual([text[index] for index in positions.tolist()], ["둘", "셋"])
        summary = collator.paragraph_boundary_supervision_summary()
        assert summary is not None
        self.assertEqual(summary["unique_rows"], 1)
        self.assertEqual(summary["candidate_offsets"], 2)
        self.assertEqual(summary["candidate_positives"], 1)

    def test_left_padding_masks_padding_and_validation_has_no_gold_targets(self) -> None:
        config = _config()
        tokenizer = _CharacterTokenizer()
        tokenizer.padding_side = "left"
        collator = RegressionCollator(
            tokenizer, config, include_labels=True, include_metadata=False
        )
        batch = collator(
            [_row("train", suffix=" 긴 꼬리"), _row("validation", validation=True)]
        )
        padding = batch["attention_mask"] == 0
        self.assertFalse(torch.any(batch["paragraph_boundary_mask"][padding]))
        self.assertEqual(
            int(batch["paragraph_boundary_mask"][1].sum().item()),
            0,
        )
        summary = collator.paragraph_boundary_supervision_summary()
        assert summary is not None
        self.assertEqual(summary["unique_rows"], 1)


class _JointBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids: torch.Tensor, **_):
        positions = torch.arange(
            input_ids.shape[1], device=input_ids.device, dtype=torch.float32
        )
        hidden = torch.stack(
            (positions + 1, (positions + 1).square(), torch.ones_like(positions)),
            dim=-1,
        )
        hidden = hidden.unsqueeze(0).expand(input_ids.shape[0], -1, -1)
        return SimpleNamespace(last_hidden_state=hidden * self.lora_scale)


class ParagraphBoundaryModelTest(unittest.TestCase):
    def test_boundary_only_loss_updates_head_and_shared_lora(self) -> None:
        torch.manual_seed(7)
        config = _config(mse_loss_weight=0.0)
        scorer = RegressionScorer(_JointBackbone(), 3, config)
        scorer.set_training_stage("joint")
        scorer.train()
        input_ids = torch.ones((1, 5), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.full((1, 3), 3.0)
        boundary_labels = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
        boundary_mask = torch.tensor([[False, True, True, False, False]])

        output = scorer(
            input_ids,
            attention_mask,
            labels=labels,
            paragraph_boundary_labels=boundary_labels,
            paragraph_boundary_mask=boundary_mask,
        )
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        assert scorer.paragraph_boundary_head is not None
        self.assertGreater(
            float(scorer.paragraph_boundary_head.weight.grad.abs().sum()), 0.0
        )
        self.assertGreater(float(scorer.backbone.lora_scale.grad.abs()), 0.0)

    def test_disabled_config_has_no_boundary_module(self) -> None:
        scorer = RegressionScorer(_JointBackbone(), 3, legacy_config())
        self.assertIsNone(scorer.paragraph_boundary_head)

    def test_auxiliary_does_not_change_score_initialization_or_training_rng(self) -> None:
        torch.manual_seed(101)
        control = RegressionScorer(
            _JointBackbone(),
            3,
            _config(
                paragraph_boundary_loss_weight=0.0,
                detail_head_mode="scalar",
                detail_final_source="criterion",
                detail_expected_loss_weight=0.25,
            ),
        )
        control_next_random = torch.rand(4)

        torch.manual_seed(101)
        treatment = RegressionScorer(
            _JointBackbone(),
            3,
            _config(
                detail_head_mode="scalar",
                detail_final_source="criterion",
                detail_expected_loss_weight=0.25,
            ),
        )
        treatment_next_random = torch.rand(4)

        for name, control_tensor in control.state_dict().items():
            if name.startswith("paragraph_boundary_head."):
                continue
            self.assertTrue(
                torch.equal(control_tensor, treatment.state_dict()[name]),
                msg=name,
            )
        self.assertTrue(torch.equal(control_next_random, treatment_next_random))


class ParagraphBoundaryConfigTest(unittest.TestCase):
    def test_auxiliary_requires_exact_raw_joint_clean_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "official_raw"):
            legacy_config(
                training_mode="lora_only",
                paragraph_boundary_loss_weight=0.1,
            ).validate()
        with self.assertRaisesRegex(ValueError, "shared backbone"):
            legacy_config(
                training_mode="head_only",
                essay_surface="official_raw",
                paragraph_boundary_loss_weight=0.1,
            ).validate()


if __name__ == "__main__":
    unittest.main()
