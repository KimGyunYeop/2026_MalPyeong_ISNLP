from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from main_code.config import DETAIL_CRITERIA, DETAIL_CRITERIA_BY_TRAIT, TRAITS
from main_code.infer import (
    ScoringOutput,
    align_detail_label_rows,
    criterion_labels,
    evaluate_detail_predictions,
    score_batches,
)


def _labelled_row(essay_id: str, score: float) -> dict:
    traits = {}
    for trait, trait_criteria in zip(TRAITS, DETAIL_CRITERIA_BY_TRAIT, strict=True):
        traits[trait] = {
            "criteria": {
                criterion: {"official_score": score} for criterion in trait_criteria
            }
        }
    return {
        "id": essay_id,
        "prompt_num": "Q1",
        "prompt": "prompt",
        "essay": "essay",
        "score": {trait: score for trait in TRAITS},
        "score_details": {"traits": traits},
    }


class _DetailScorer:
    def eval(self) -> "_DetailScorer":
        return self

    def __call__(self, **kwargs):
        assert kwargs["return_detail_predictions"] is True
        assert kwargs["return_probabilities"] is False
        batch_size = kwargs["input_ids"].shape[0]
        detail_scores = torch.full((batch_size, 9), 3.0)
        detail_probabilities = torch.zeros((batch_size, 9, 5))
        detail_probabilities[..., 2] = 1.0
        return {
            "scores": torch.full((batch_size, 3), 3.0),
            "detail_scores": detail_scores,
            "detail_probabilities": detail_probabilities,
        }


class _RaterSetScorer(_DetailScorer):
    def __call__(self, **kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        probabilities = torch.zeros((batch_size, 2, 9, 5))
        probabilities[:, 0, :, 1] = 1.0
        probabilities[:, 1, :, 3] = 1.0
        return {
            "scores": torch.full((batch_size, 3), 3.0),
            "detail_scores": torch.full((batch_size, 9), 3.0),
            "detail_probabilities": probabilities,
        }


class DetailInferenceTest(unittest.TestCase):
    def test_detail_labels_can_join_flat_input_to_same_canonical_essay(self) -> None:
        raw_row = _labelled_row("e1", 3.0)
        raw_row["essay"] = "첫 문단 둘째 문단"
        del raw_row["score_details"]
        canonical_row = _labelled_row("e1", 3.0)
        canonical_row["essay"] = "첫 문단\n\n둘째 문단"

        aligned = align_detail_label_rows([raw_row], [canonical_row])

        self.assertEqual(aligned[0]["essay"], raw_row["essay"])
        self.assertEqual(set(criterion_labels(aligned[0])), set(DETAIL_CRITERIA))

    def test_detail_label_join_rejects_different_essay(self) -> None:
        raw_row = _labelled_row("e1", 3.0)
        detail_row = _labelled_row("e1", 3.0)
        detail_row["essay"] = "다른 글"

        with self.assertRaisesRegex(ValueError, "본문이 다릅니다"):
            align_detail_label_rows([raw_row], [detail_row])

    def test_score_loop_adds_detail_fields_without_changing_submission_record(
        self,
    ) -> None:
        rows = [_labelled_row("e1", 3.0), _labelled_row("e2", 3.0)]
        batch = {
            "input_ids": torch.ones((2, 3), dtype=torch.long),
            "attention_mask": torch.ones((2, 3), dtype=torch.long),
            "essay_ids": ["e1", "e2"],
            "rows": rows,
        }
        loaded = SimpleNamespace(
            scorer=_DetailScorer(),
            config=SimpleNamespace(
                score_head="regression",
                detail_head_mode="categorical",
            ),
        )

        output = score_batches(
            loaded,
            [batch],
            torch.device("cpu"),
            prompt_lookup={},
            prompt_routing_active=False,
        )

        self.assertEqual(len(output.records), 2)
        self.assertNotIn("criterion_scores", output.records[0])
        self.assertEqual(
            list(output.score_records[0]["criterion_scores"]),
            list(DETAIL_CRITERIA),
        )
        self.assertEqual(
            output.score_records[0]["criterion_distributions"]["content_1"],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        )
        self.assertEqual(output.detail_prediction_arrays[0].shape, (2, 9))
        self.assertEqual(output.detail_probability_arrays[0].shape, (2, 9, 5))

    def test_score_loop_serializes_two_anonymous_rater_distributions(self) -> None:
        rows = [_labelled_row("e1", 3.0)]
        batch = {
            "input_ids": torch.ones((1, 3), dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "essay_ids": ["e1"],
            "rows": rows,
        }
        loaded = SimpleNamespace(
            scorer=_RaterSetScorer(),
            config=SimpleNamespace(
                score_head="regression",
                detail_head_mode="rater_set",
            ),
        )

        output = score_batches(
            loaded,
            [batch],
            torch.device("cpu"),
            prompt_lookup={},
            prompt_routing_active=False,
        )

        record = output.score_records[0]
        self.assertNotIn("criterion_distributions", record)
        self.assertEqual(record["rater_set_class_values"], [1, 2, 3, 4, 5])
        self.assertEqual(
            record["rater_set_distributions"]["predicted_rater_1"]["content_1"],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        )
        self.assertEqual(
            record["rater_set_distributions"]["predicted_rater_2"]["content_1"],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        )
        self.assertEqual(output.detail_probability_arrays[0].shape, (1, 2, 9, 5))

    def test_detail_metrics_include_each_criterion_and_five_two_two_aggregate(
        self,
    ) -> None:
        rows = [
            _labelled_row("e1", 1.0),
            _labelled_row("e2", 3.0),
            _labelled_row("e3", 5.0),
        ]
        predictions = np.asarray(
            [[score] * len(DETAIL_CRITERIA) for score in (1.0, 3.0, 5.0)],
            dtype=np.float32,
        )
        scoring = ScoringOutput(
            records=[],
            score_records=[],
            prediction_arrays=[],
            detail_prediction_arrays=[predictions],
            detail_probability_arrays=[],
            unknown_prompt_count=0,
        )

        metrics = evaluate_detail_predictions(rows, scoring)

        self.assertIsNotNone(metrics)
        assert metrics is not None
        self.assertEqual(set(metrics["criteria"]), set(DETAIL_CRITERIA))
        self.assertEqual(metrics["criteria"]["content_1"]["rmse"], 0.0)
        self.assertEqual(metrics["criteria"]["content_1"]["spearman"], 1.0)
        self.assertEqual(
            metrics["criterion_aggregate_vs_final_labels"]["overall"]["rmse"],
            0.0,
        )

    def test_missing_or_invalid_criterion_labels_are_not_filled(self) -> None:
        row = _labelled_row("missing", 3.0)
        del row["score_details"]["traits"]["content"]["criteria"]["content_1"]
        row["score_details"]["traits"]["content"]["criteria"]["content_2"] = {
            "official_score": float("nan")
        }

        labels = criterion_labels(row)

        self.assertNotIn("content_1", labels)
        self.assertNotIn("content_2", labels)
        self.assertEqual(labels["content_3"], 3.0)


if __name__ == "__main__":
    unittest.main()
