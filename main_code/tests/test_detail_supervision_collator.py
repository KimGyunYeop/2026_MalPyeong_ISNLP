from __future__ import annotations

import unittest

import torch

from main_code.config import RegressionConfig
from main_code.datasets import (
    DETAIL_CRITERIA,
    DETAIL_MISSING_CLASS,
    DETAIL_PAD_RATER_ID,
    RegressionCollator,
    build_detail_rater_registry,
    detail_supervision_summary,
)
from main_code.tests.config_helpers import legacy_config


class DetailSupervisionCollatorTest(unittest.TestCase):
    class _Tokenizer:
        model_max_length = 64

        def __call__(self, texts, **_kwargs):
            batch_size = len(texts)
            return {
                "input_ids": torch.ones((batch_size, 2), dtype=torch.long),
                "attention_mask": torch.ones((batch_size, 2), dtype=torch.long),
            }

    @staticmethod
    def _row(
        essay_id: str,
        *,
        source: str,
        official_raters: list[str],
        rater_ids: dict[str, str],
        scores_by_rater: dict[str, float | None],
        official_score: float,
    ) -> dict:
        traits: dict[str, dict] = {}
        for trait in ("content", "organization", "expression"):
            criterion_names = [
                criterion
                for criterion in DETAIL_CRITERIA
                if criterion.startswith(f"{trait}_")
            ]
            traits[trait] = {
                "criteria_order": criterion_names,
                "criteria": {
                    criterion: {
                        "official_score": official_score,
                        "rater_scores": dict(scores_by_rater),
                    }
                    for criterion in criterion_names
                },
            }
        return {
            "id": essay_id,
            "source_dataset": source,
            "prompt_num": "Q1",
            "prompt": "prompt",
            "essay": "essay",
            "score": {
                "content": official_score,
                "organization": official_score,
                "expression": official_score,
            },
            "score_details": {
                "official_raters": official_raters,
                "rater_ids": rater_ids,
                "traits": traits,
            },
        }

    def _collator(self, config: RegressionConfig) -> RegressionCollator:
        return RegressionCollator(
            self._Tokenizer(),
            config,
            include_labels=True,
            include_metadata=False,
        )

    def test_default_collator_does_not_add_detail_inputs(self) -> None:
        row = self._row(
            "default",
            source="nikl_competition",
            official_raters=["evaluator1", "evaluator2"],
            rater_ids={"evaluator1": "100", "evaluator2": "101"},
            scores_by_rater={"evaluator1": 2.0, "evaluator2": 4.0},
            official_score=3.0,
        )

        batch = self._collator(legacy_config(max_length=64))([row])

        self.assertEqual(batch["labels"].shape, (1, 3))
        self.assertNotIn("criterion_scores", batch)
        self.assertNotIn("detail_rater_ids", batch)

    def test_official_distribution_and_all_rater_labels_are_distinct(self) -> None:
        # evaluator3 is useful for the train-only rater auxiliary, but is not
        # an official target rater and therefore must not alter the empirical
        # official distribution.
        row = self._row(
            "ordinary",
            source="nikl_competition",
            official_raters=["evaluator1", "evaluator2"],
            rater_ids={
                "evaluator1": "100",
                "evaluator2": "101",
                "evaluator3": "102",
            },
            scores_by_rater={
                "evaluator1": 1.0,
                "evaluator2": 5.0,
                "evaluator3": 2.0,
            },
            official_score=3.0,
        )
        registry = build_detail_rater_registry([row])
        config = legacy_config(
            max_length=64,
            detail_head_mode="categorical",
            detail_expected_loss_weight=0.25,
            detail_distribution_loss_weight=0.1,
            detail_rater_loss_weight=0.1,
            detail_rater_registry=registry,
        )

        batch = self._collator(config)([row])

        self.assertEqual(batch["criterion_scores"].shape, (1, 9))
        self.assertEqual(batch["criterion_distributions"].shape, (1, 9, 5))
        self.assertEqual(batch["criterion_mask"].shape, (1, 9))
        self.assertEqual(batch["criterion_distribution_mask"].shape, (1, 9))
        self.assertEqual(batch["detail_rater_ids"].shape, (1, 3))
        self.assertEqual(batch["detail_rater_labels"].shape, (1, 3, 9))
        self.assertEqual(batch["detail_rater_mask"].shape, (1, 3, 9))
        self.assertEqual(batch["criterion_scores"].dtype, torch.float32)
        self.assertEqual(batch["criterion_mask"].dtype, torch.bool)
        self.assertTrue(batch["criterion_mask"].all())
        self.assertTrue(batch["criterion_distribution_mask"].all())
        torch.testing.assert_close(
            batch["criterion_distributions"][0, 0],
            torch.tensor([0.5, 0.0, 0.0, 0.0, 0.5]),
        )
        # Hard labels are zero-based class indices 0..4.  All three actual
        # evaluators remain available to the auxiliary branch.
        self.assertEqual(batch["detail_rater_labels"][0, :, 0].tolist(), [0, 4, 1])
        self.assertTrue(batch["detail_rater_mask"].all())

    def test_rater_set_keeps_two_officials_and_does_not_duplicate_re_evaluator(
        self,
    ) -> None:
        ordinary = self._row(
            "ordinary-set",
            source="nikl_competition",
            official_raters=["evaluator1", "evaluator2"],
            rater_ids={"evaluator1": "100", "evaluator2": "101"},
            scores_by_rater={"evaluator1": 1.0, "evaluator2": 5.0},
            official_score=3.0,
        )
        re_evaluated = self._row(
            "single-official-set",
            source="nikl_competition",
            official_raters=["re-evaluator"],
            rater_ids={
                "evaluator1": "200",
                "evaluator2": "201",
                "re-evaluator": "299",
            },
            scores_by_rater={
                "evaluator1": 1.0,
                "evaluator2": 5.0,
                "re-evaluator": 4.0,
            },
            official_score=4.0,
        )
        config = legacy_config(
            max_length=64,
            detail_head_mode="rater_set",
            detail_final_source="criterion",
            detail_expected_loss_weight=0.25,
            detail_rater_set_loss_weight=0.1,
        ).validate()

        batch = self._collator(config)([ordinary, re_evaluated])

        self.assertEqual(batch["official_rater_labels"].shape, (2, 2, 9))
        self.assertEqual(batch["official_rater_mask"].shape, (2, 2, 9))
        self.assertEqual(batch["official_rater_labels"].dtype, torch.long)
        self.assertEqual(batch["official_rater_mask"].dtype, torch.bool)
        self.assertEqual(batch["official_rater_labels"][0, 0].tolist(), [0] * 9)
        self.assertEqual(batch["official_rater_labels"][0, 1].tolist(), [4] * 9)
        self.assertTrue(batch["official_rater_mask"][0].all())

        # Re-evaluation has one observed official.  Slot 2 stays explicitly
        # missing rather than copying the re-evaluator's class-3 label.
        self.assertEqual(batch["official_rater_labels"][1, 0].tolist(), [3] * 9)
        self.assertTrue(batch["official_rater_mask"][1, 0].all())
        self.assertEqual(
            batch["official_rater_labels"][1, 1].tolist(),
            [DETAIL_MISSING_CLASS] * 9,
        )
        self.assertFalse(batch["official_rater_mask"][1, 1].any())

        summary = detail_supervision_summary(
            [ordinary, re_evaluated],
            (),
            include_official_rater_set=True,
        )
        self.assertEqual(summary["official_rater_rating_valid"], 27)
        self.assertEqual(summary["rows_with_two_official_raters"], 1)
        self.assertEqual(summary["rows_with_one_official_rater"], 1)

    def test_re_evaluator_is_official_while_missing_original_block_is_masked(
        self,
    ) -> None:
        # This mirrors the 92 official re-evaluator rows, including the subset
        # where one original evaluator's complete nine-criterion block is null.
        row = self._row(
            "re-evaluated",
            source="nikl_competition",
            official_raters=["re-evaluator"],
            rater_ids={
                "evaluator3": "201",
                "re-evaluator": "299",
            },
            scores_by_rater={
                "evaluator1": None,
                "evaluator3": 5.0,
                "re-evaluator": 4.0,
            },
            official_score=4.0,
        )
        registry = build_detail_rater_registry([row])
        config = legacy_config(
            max_length=64,
            detail_head_mode="categorical",
            detail_expected_loss_weight=0.25,
            detail_distribution_loss_weight=0.1,
            detail_rater_loss_weight=0.1,
            detail_rater_registry=registry,
        )

        batch = self._collator(config)([row])

        torch.testing.assert_close(
            batch["criterion_distributions"][0, 0],
            torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
        )
        self.assertEqual(batch["criterion_scores"][0, 0].item(), 4.0)
        # The missing evaluator1 slot is discovered from rater_scores even
        # though its actual evaluator ID is absent from rater_ids.
        self.assertEqual(batch["detail_rater_ids"][0].tolist()[-1], -1)
        self.assertEqual(
            batch["detail_rater_labels"][0, -1].tolist(),
            [DETAIL_MISSING_CLASS] * 9,
        )
        self.assertFalse(batch["detail_rater_mask"][0, -1].any())
        self.assertEqual(batch["detail_rater_labels"][0, 0, 0].item(), 4)
        self.assertEqual(batch["detail_rater_labels"][0, 1, 0].item(), 3)

    def test_fractional_rater_score_is_masked_instead_of_rounded(self) -> None:
        row = self._row(
            "fractional-external",
            source="aihub_external",
            official_raters=["evaluator1"],
            rater_ids={"evaluator1": "7"},
            scores_by_rater={"evaluator1": 11.0 / 3.0},
            official_score=11.0 / 3.0,
        )
        config = legacy_config(
            max_length=64,
            detail_head_mode="categorical",
            detail_expected_loss_weight=0.25,
            detail_distribution_loss_weight=0.1,
        )

        batch = self._collator(config)([row])

        self.assertTrue(batch["criterion_mask"].all())
        self.assertFalse(batch["criterion_distribution_mask"].any())
        self.assertEqual(batch["criterion_distributions"].count_nonzero().item(), 0)
        self.assertAlmostEqual(
            batch["criterion_scores"][0, 0].item(), 11.0 / 3.0, places=6
        )

    def test_rater_padding_and_unknown_source_id_are_explicitly_masked(self) -> None:
        known = self._row(
            "known",
            source="source-a",
            official_raters=["evaluator1", "evaluator2"],
            rater_ids={"evaluator1": "7", "evaluator2": "8"},
            scores_by_rater={"evaluator1": 2.0, "evaluator2": 4.0},
            official_score=3.0,
        )
        # The raw number 7 is deliberately reused by another source.  It is a
        # different evaluator and is not in the train-built registry.
        unknown = self._row(
            "unknown",
            source="source-b",
            official_raters=["evaluator1"],
            rater_ids={"evaluator1": "7"},
            scores_by_rater={"evaluator1": 5.0},
            official_score=5.0,
        )
        registry = build_detail_rater_registry([known])
        self.assertEqual(registry, (("source-a", "7"), ("source-a", "8")))
        config = legacy_config(
            max_length=64,
            detail_head_mode="categorical",
            detail_expected_loss_weight=0.25,
            detail_rater_loss_weight=0.1,
            detail_rater_registry=registry,
        )

        batch = self._collator(config)([known, unknown])

        self.assertEqual(batch["detail_rater_ids"].shape, (2, 2))
        self.assertEqual(batch["detail_rater_ids"][1].tolist(), [-1, -1])
        self.assertEqual(
            batch["detail_rater_labels"][1, 0].tolist(),
            [DETAIL_MISSING_CLASS] * 9,
        )
        self.assertFalse(batch["detail_rater_mask"][1].any())
        self.assertEqual(batch["detail_rater_ids"][1, 1].item(), DETAIL_PAD_RATER_ID)

    def test_rater_auxiliary_requires_a_train_registry(self) -> None:
        config = legacy_config(
            max_length=64,
            detail_head_mode="categorical",
            detail_expected_loss_weight=0.25,
            detail_rater_loss_weight=0.1,
        )

        with self.assertRaisesRegex(ValueError, "detail_rater_registry"):
            self._collator(config)

    def test_summary_uses_the_same_masks_as_the_collator(self) -> None:
        row = self._row(
            "summary",
            source="nikl_competition",
            official_raters=["evaluator1", "evaluator2"],
            rater_ids={"evaluator1": "100", "evaluator2": "101"},
            scores_by_rater={"evaluator1": 1.0, "evaluator2": 5.0},
            official_score=3.0,
        )
        registry = build_detail_rater_registry([row])

        summary = detail_supervision_summary([row], registry)

        self.assertEqual(summary["criterion_slots"], 9)
        self.assertEqual(summary["criterion_score_valid"], 9)
        self.assertEqual(summary["criterion_distribution_valid"], 9)
        self.assertEqual(summary["rater_registry_count"], 2)
        self.assertEqual(summary["individual_rating_valid"], 18)
        self.assertEqual(summary["max_raters_per_essay"], 2)


if __name__ == "__main__":
    unittest.main()
