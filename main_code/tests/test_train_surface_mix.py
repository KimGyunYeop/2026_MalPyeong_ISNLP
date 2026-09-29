from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

from main_code.config import load_config
from main_code.datasets import EssayRegressionDataset, RegressionCollator
from main_code.tests.config_helpers import legacy_config


class _PaddingTokenizer:
    model_max_length = 128
    padding_side = "right"

    def pad(self, examples, *, padding, return_tensors):
        self.assert_pad_arguments(padding, return_tensors)
        width = max(len(example["input_ids"]) for example in examples)
        input_ids = []
        attention_mask = []
        for example in examples:
            missing = width - len(example["input_ids"])
            input_ids.append(example["input_ids"] + [0] * missing)
            attention_mask.append(example["attention_mask"] + [0] * missing)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    @staticmethod
    def assert_pad_arguments(padding, return_tensors):
        if padding is not True or return_tensors != "pt":
            raise AssertionError("unexpected tokenizer.pad arguments")


def _row(essay_id: str, essay: str) -> dict:
    return {
        "id": essay_id,
        "prompt_num": "Q1",
        "prompt": "prompt",
        "essay": essay,
        "score": {"content": 3.0, "organization": 3.0, "expression": 3.0},
    }


class TrainSurfaceMixConfigTest(unittest.TestCase):
    def test_default_is_checkpoint_compatible_and_neutral(self) -> None:
        config = legacy_config().validate()
        self.assertEqual(config.train_canonical_surface_probability, 0.0)

        with tempfile.TemporaryDirectory() as directory:
            payload = asdict(config)
            payload.pop("train_canonical_surface_probability")
            path = Path(directory) / "old_config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_config(path)
        self.assertEqual(loaded.train_canonical_surface_probability, 0.0)

    def test_probability_requires_noncanonical_deployment_surface(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a non-canonical"):
            legacy_config(train_canonical_surface_probability=0.5).validate()

        for surface in ("flat", "official_raw"):
            config = legacy_config(
                essay_surface=surface, train_canonical_surface_probability=0.5
            ).validate()
            self.assertEqual(config.train_canonical_surface_probability, 0.5)

    def test_probability_must_be_in_unit_interval(self) -> None:
        for invalid in (-0.01, 1.01, float("nan")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "must be in \\[0, 1\\]"):
                    legacy_config(
                        essay_surface="flat",
                        train_canonical_surface_probability=invalid,
                    ).validate()


class TrainSurfaceMixCollatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = legacy_config(
            essay_surface="flat",
            train_canonical_surface_probability=0.5,
            pooling="essay_mean",
            max_length=128,
        ).validate()
        self.rows = [
            _row("canonical-view", "첫 문단.\n\n둘째  문단."),
            _row("flat-view", "세번째 문단.\n\t네번째 문단."),
        ]

    def _collate_and_capture_essays(self, items, *, include_labels=True):
        calls: list[tuple[str, str]] = []

        def fake_tokenize(_tokenizer, text, essay, *, max_length):
            self.assertEqual(max_length, 128)
            self.assertTrue(text.endswith(essay))
            calls.append((text, essay))
            return [1, 2], [0, 1]

        collator = RegressionCollator(
            _PaddingTokenizer(),
            self.config,
            include_labels=include_labels,
            include_metadata=False,
        )
        with patch(
            "main_code.datasets._tokenize_with_essay_mask",
            side_effect=fake_tokenize,
        ):
            batch = collator(items)
        self.assertEqual(batch["essay_mask"].tolist(), [[0, 1]] * len(items))
        return [essay for _, essay in calls]

    def test_train_rows_sample_views_without_mutating_source_rows(self) -> None:
        originals = [dict(row) for row in self.rows]
        dataset = EssayRegressionDataset(
            self.rows, self.config, split="train", require_labels=True
        )
        with patch("main_code.datasets.random.random", side_effect=[0.25, 0.75]):
            items = [dataset[0], dataset[1]]

        essays = self._collate_and_capture_essays(items)
        self.assertEqual(essays[0], "첫 문단.\n\n둘째  문단.")
        self.assertEqual(essays[1], "세번째 문단. 네번째 문단.")
        self.assertEqual(dataset.surface_view_counts, {"canonical": 1, "flat": 1})
        self.assertEqual(self.rows, originals)
        self.assertIsNot(items[0], self.rows[0])
        self.assertIsNot(items[1], self.rows[1])

    def test_validation_and_inference_always_use_flat_surface(self) -> None:
        expected = ["첫 문단. 둘째 문단.", "세번째 문단. 네번째 문단."]
        for split, require_labels in (("validation", True), ("inference", False)):
            with self.subTest(split=split):
                dataset = EssayRegressionDataset(
                    self.rows,
                    self.config,
                    split=split,
                    require_labels=require_labels,
                )
                with patch(
                    "main_code.datasets.random.random",
                    side_effect=AssertionError("non-train split must not sample"),
                ):
                    items = [dataset[0], dataset[1]]
                essays = self._collate_and_capture_essays(
                    items, include_labels=require_labels
                )
                self.assertEqual(essays, expected)

    def test_official_raw_mix_uses_canonical_only_for_sampled_train_view(self) -> None:
        rows = [
            {
                **_row("raw-mix", "첫 문단.\n\n둘째 문단."),
                "essay_surfaces": {"official_raw": " 첫 문단. 둘째 문단."},
            }
        ]
        config = legacy_config(
            essay_surface="official_raw",
            train_canonical_surface_probability=0.5,
        ).validate()
        dataset = EssayRegressionDataset(
            rows, config, split="train", require_labels=True
        )
        with patch("main_code.datasets.random.random", side_effect=[0.25, 0.75]):
            canonical_item = dataset[0]
            raw_item = dataset[0]

        from main_code.datasets import essay_for_input

        self.assertEqual(
            essay_for_input(canonical_item, config, essay_surface=canonical_item.essay_surface),
            "첫 문단.\n\n둘째 문단.",
        )
        self.assertEqual(
            essay_for_input(raw_item, config, essay_surface=raw_item.essay_surface),
            " 첫 문단. 둘째 문단.",
        )
        self.assertEqual(dataset.surface_view_counts, {"canonical": 1, "official_raw": 1})

    def test_zero_probability_preserves_previous_dataset_items(self) -> None:
        config = legacy_config(essay_surface="flat").validate()
        dataset = EssayRegressionDataset(
            self.rows, config, split="train", require_labels=True
        )
        self.assertIs(dataset[0], self.rows[0])
        self.assertEqual(dataset.surface_view_counts, {"flat": 1})


if __name__ == "__main__":
    unittest.main()
