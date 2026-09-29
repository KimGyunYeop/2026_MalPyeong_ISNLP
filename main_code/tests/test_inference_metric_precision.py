from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from main_code.config import TRAITS
from main_code.datasets import RegressionCollator, human_scores
from main_code.infer import evaluate_predictions
from main_code.train import compute_trainer_metrics, overlay_validation_metric_averages
from main_code.utils import regression_metrics
from main_code.tests.config_helpers import legacy_config


class InferenceMetricPrecisionTest(unittest.TestCase):
    class _Tokenizer:
        model_max_length = 64

        def __call__(self, texts, **_kwargs):
            batch_size = len(texts)
            return {
                "input_ids": torch.ones((batch_size, 2), dtype=torch.long),
                "attention_mask": torch.ones((batch_size, 2), dtype=torch.long),
            }

    def test_json_label_precision_is_preserved_for_spearman_ties(self) -> None:
        # The first two content labels are distinct JSON numbers that collapse
        # to the same float32 value.  Inference must rank the original values.
        labels = [
            {"content": 3.00000001, "organization": 1.0, "expression": 1.0},
            {"content": 3.00000002, "organization": 2.0, "expression": 2.0},
            {"content": 4.0, "organization": 3.0, "expression": 3.0},
        ]
        predictions = np.asarray(
            [[3.2, 1.1, 1.1], [3.1, 2.1, 2.1], [4.1, 3.1, 3.1]],
            dtype=np.float32,
        )
        rows = [
            {
                "essay_id": f"e{index}",
                "prompt_num": "Q1",
                "prompt_text": "prompt",
                "essay_text": "essay",
            }
            for index in range(3)
        ]

        metrics, _ = evaluate_predictions(
            rows,
            labels,
            [predictions],
            prompt_lookup={},
            prompt_routing_active=False,
        )
        expected = regression_metrics(
            np.asarray([[row[trait] for trait in TRAITS] for row in labels]),
            predictions.astype(np.float64),
        )
        collapsed = regression_metrics(
            np.asarray(
                [[row[trait] for trait in TRAITS] for row in labels],
                dtype=np.float32,
            ),
            predictions,
        )

        self.assertEqual(metrics["overall"], expected["overall"])
        self.assertNotEqual(
            metrics["traits"]["content"]["spearman"],
            collapsed["traits"]["content"]["spearman"],
        )

    def test_trainer_metric_uses_the_same_float64_tie_contract(self) -> None:
        labels = np.asarray(
            [
                [3.00000001, 1.0, 1.0],
                [3.00000002, 2.0, 2.0],
                [4.0, 3.0, 3.0],
            ],
            dtype=np.float64,
        )
        predictions = np.asarray(
            [[3.2, 1.1, 1.1], [3.1, 2.1, 2.1], [4.1, 3.1, 3.1]],
            dtype=np.float32,
        )

        actual = compute_trainer_metrics(
            SimpleNamespace(label_ids=labels, predictions=predictions)
        )
        expected = regression_metrics(labels, predictions.astype(np.float64))
        collapsed = regression_metrics(labels.astype(np.float32), predictions)

        self.assertEqual(
            actual["content_spearman"],
            expected["traits"]["content"]["spearman"],
        )
        self.assertEqual(actual["overall_spearman"], expected["overall"]["spearman"])
        self.assertNotEqual(
            actual["content_spearman"],
            collapsed["traits"]["content"]["spearman"],
        )

    def test_collator_preserves_json_label_precision_for_trainer(self) -> None:
        row = {
            "id": "e1",
            "prompt_num": "Q1",
            "prompt": "prompt",
            "essay": "essay",
            "score": {
                "content": 3.00000001,
                "organization": 3.25,
                "expression": 4.0,
            },
        }
        batch = RegressionCollator(
            self._Tokenizer(),
            legacy_config(max_length=64),
            include_labels=True,
            include_metadata=False,
        )([row])

        self.assertEqual(batch["labels"].dtype, torch.float64)
        self.assertEqual(batch["labels"][0, 0].item(), 3.00000001)
        self.assertEqual(batch["average_labels"].dtype, torch.float64)
        self.assertEqual(
            batch["average_labels"][0].item(),
            (3.00000001 + 3.25 + 4.0) / 3.0,
        )

    def test_collator_canonicalizes_only_known_competition_mean_grid(self) -> None:
        row = {
            "id": "competition",
            "source_dataset": "nikl_competition",
            "dataset_group": "competition",
            "prompt_num": "Q1",
            "prompt": "prompt",
            "essay": "essay",
            "score": {
                "content": 1.2999999999999998,
                "organization": 3.2499999999999996,
                "expression": 4.0,
            },
            "score_details": {"label_policy": "mean_of_two_raters"},
        }
        batch = RegressionCollator(
            self._Tokenizer(),
            legacy_config(max_length=64),
            include_labels=True,
            include_metadata=False,
        )([row])

        self.assertEqual(batch["labels"].dtype, torch.float64)
        self.assertEqual(batch["labels"][0].tolist(), [1.3, 3.25, 4.0])

    def test_validation_average_overlay_changes_only_metric_average(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        processed = {
            "id": "e1",
            "prompt_num": "Q1",
            "prompt": "prompt",
            "essay": "essay",
            "score": {
                "content": 3.0,
                "organization": 3.25,
                "expression": 4.0,
                "average": (3.0 + 3.25 + 4.0) / 3.0,
            },
        }
        official = {
            **processed,
            "score": {**processed["score"], "average": 3.42},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "validation.jsonl"
            path.write_text(json.dumps(official) + "\n", encoding="utf-8")
            rows = overlay_validation_metric_averages([processed], path)

        self.assertEqual(rows[0]["score"], processed["score"])
        batch = RegressionCollator(
            self._Tokenizer(),
            legacy_config(max_length=64),
            include_labels=True,
            include_metadata=False,
        )(rows)
        self.assertEqual(batch["labels"][0].tolist(), [3.0, 3.25, 4.0])
        self.assertEqual(batch["average_labels"][0].item(), 3.42)

    def test_human_scores_preserves_aihub_two_rater_content_sixths(self) -> None:
        content = 13.0 / 6.0
        row = {
            "id": "aihub25_descriptive:1",
            "source_dataset": "aihub25_descriptive",
            "dataset_group": "external",
            "score": {
                "content": content,
                "organization": 2.25,
                "expression": 3.5,
            },
            "score_details": {"label_policy": "mean_of_two_raters"},
        }

        scores = human_scores(row)

        self.assertIsNotNone(scores)
        assert scores is not None
        self.assertEqual(scores["content"], content)
        self.assertNotEqual(scores["content"], 2.2)


if __name__ == "__main__":
    unittest.main()
