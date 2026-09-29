from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from main_code.config import DETAIL_CRITERIA, load_config, save_config
from main_code.models import (
    LoadedRegressionModel,
    RegressionScorer,
    _masked_detail_trait_mean,
    save_checkpoint,
)
from main_code.tests.config_helpers import legacy_config


class _TinyBackbone(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, hidden_size)

    def forward(self, input_ids: torch.Tensor, **_kwargs) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _TinyTokenizer:
    def save_pretrained(self, directory: str | Path) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)


def _zero_score_side(scorer: RegressionScorer) -> None:
    with torch.no_grad():
        for name, parameter in scorer.named_parameters():
            if not name.startswith("backbone."):
                parameter.zero_()


class DetailScoreHeadTest(unittest.TestCase):
    def test_default_adds_no_checkpoint_keys(self) -> None:
        config = legacy_config().validate()
        first = RegressionScorer(_TinyBackbone(4), 4, config)
        state = first.scoring_state_dict()

        self.assertFalse(any(name.startswith("detail_") for name in state))
        second = RegressionScorer(_TinyBackbone(4), 4, config)
        incompatible = second.load_state_dict(first.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_scalar_head_outputs_nine_bounded_scores(self) -> None:
        config = legacy_config(
            detail_head_mode="scalar",
            detail_expected_loss_weight=0.25,
        ).validate()
        scorer = RegressionScorer(_TinyBackbone(4), 4, config)
        features = torch.randn(3, 3, 4)

        scores, probabilities, raw = scorer.detail_predictions_from_features(features)

        self.assertEqual(scores.shape, (3, 9))
        self.assertIsNone(probabilities)
        self.assertEqual(raw.shape, (3, 9, 1))
        self.assertTrue(torch.all((scores >= 1) & (scores <= 5)))

    def test_halfstep_head_outputs_nine_by_nine_and_normalized_ce(self) -> None:
        config = legacy_config(
            detail_head_mode="halfstep_categorical",
            detail_final_source="criterion",
            detail_expected_loss_weight=0.25,
            detail_halfstep_loss_weight=0.1,
        ).validate()
        scorer = RegressionScorer(_TinyBackbone(4), 4, config)
        _zero_score_side(scorer)
        features = torch.zeros(1, 3, 4)

        scores, probabilities, raw = scorer.detail_predictions_from_features(features)

        self.assertEqual(len(scorer.detail_heads), 9)
        self.assertTrue(
            all(head.out_features == 9 for head in scorer.detail_heads.values())
        )
        self.assertEqual(scores.shape, (1, 9))
        self.assertEqual(probabilities.shape, (1, 9, 9))
        self.assertEqual(raw.shape, (1, 9, 9))
        torch.testing.assert_close(scores, torch.full((1, 9), 3.0))
        torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones((1, 9)))

        criterion_scores = torch.arange(1.0, 5.01, 0.5).unsqueeze(0)
        criterion_mask = torch.ones((1, 9), dtype=torch.bool)
        loss = scorer._detail_halfstep_cross_entropy(
            raw,
            criterion_scores,
            criterion_mask,
        )
        # Uniform 9-class logits have CE/log(9) == 1.
        self.assertAlmostEqual(loss.item(), 1.0, places=6)
        loss.backward()
        gradients = [head.bias.grad for head in scorer.detail_heads.values()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(
            all(
                torch.isfinite(gradient).all()
                for gradient in gradients
                if gradient is not None
            )
        )

    def test_halfstep_loss_rejects_an_unmasked_off_grid_target(self) -> None:
        logits = torch.zeros((1, 9, 9))
        criterion_scores = torch.full((1, 9), 3.0)
        criterion_scores[0, 0] = 3.25
        criterion_mask = torch.ones((1, 9), dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "1, 1.5, ..., 5"):
            RegressionScorer._detail_halfstep_cross_entropy(
                logits,
                criterion_scores,
                criterion_mask,
            )

    def test_rater_set_has_eighteen_heads_and_two_by_nine_by_five_logits(
        self,
    ) -> None:
        config = legacy_config(
            detail_head_mode="rater_set",
            detail_final_source="criterion",
            detail_expected_loss_weight=0.25,
            detail_rater_set_loss_weight=0.1,
        ).validate()
        scorer = RegressionScorer(_TinyBackbone(4), 4, config)
        features = torch.randn(2, 3, 4)

        scores, probabilities, raw = scorer.detail_predictions_from_features(features)

        expected_names = {
            f"rater{rater_slot}_{criterion}"
            for rater_slot in (1, 2)
            for criterion in DETAIL_CRITERIA
        }
        self.assertIsInstance(scorer.detail_heads, nn.ModuleDict)
        self.assertEqual(set(scorer.detail_heads), expected_names)
        self.assertEqual(len(scorer.detail_heads), 18)
        self.assertTrue(
            all(head.out_features == 5 for head in scorer.detail_heads.values())
        )
        self.assertEqual(scores.shape, (2, 9))
        self.assertEqual(probabilities.shape, (2, 2, 9, 5))
        self.assertEqual(raw.shape, (2, 2, 9, 5))
        torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones((2, 2, 9)))

    def test_rater_set_loss_is_invariant_to_target_slot_swap(self) -> None:
        generator = torch.Generator().manual_seed(17)
        logits = torch.randn((2, 2, 9, 5), generator=generator)
        labels = torch.stack(
            (
                torch.zeros((2, 9), dtype=torch.long),
                torch.full((2, 9), 4, dtype=torch.long),
            ),
            dim=1,
        )
        mask = torch.ones((2, 2, 9), dtype=torch.bool)

        original = RegressionScorer._detail_rater_set_cross_entropy(
            logits,
            labels,
            mask,
        )
        swapped = RegressionScorer._detail_rater_set_cross_entropy(
            logits,
            labels.flip(dims=(1,)),
            mask.flip(dims=(1,)),
        )

        torch.testing.assert_close(original, swapped)

    def test_single_rater_set_target_has_finite_loss_and_gradient(self) -> None:
        generator = torch.Generator().manual_seed(23)
        logits = torch.randn((1, 2, 9, 5), generator=generator, requires_grad=True)
        labels = torch.full((1, 2, 9), -100, dtype=torch.long)
        labels[:, 0] = 3
        mask = torch.zeros((1, 2, 9), dtype=torch.bool)
        mask[:, 0] = True

        loss = RegressionScorer._detail_rater_set_cross_entropy(
            logits,
            labels,
            mask,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        assert logits.grad is not None
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_criterion_final_uses_five_two_two_aggregate_for_primary_loss(self) -> None:
        config = legacy_config(
            detail_head_mode="scalar",
            detail_final_source="criterion",
        ).validate()
        scorer = RegressionScorer(_TinyBackbone(4), 4, config)
        _zero_score_side(scorer)
        with torch.no_grad():
            for head in scorer.heads.values():
                head.bias.fill_(5.0)
        scorer.eval()

        output = scorer(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones((1, 3), dtype=torch.long),
            labels=torch.full((1, 3), 3.0),
        )

        torch.testing.assert_close(output["scores"], torch.full((1, 3), 3.0))
        self.assertAlmostEqual(output["loss"].item(), 0.0, places=7)

    def test_masked_detail_loss_balances_traits_instead_of_nine_heads(self) -> None:
        elementwise = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 5.0, 5.0]])
        mask = torch.ones_like(elementwise, dtype=torch.bool)

        balanced = _masked_detail_trait_mean(elementwise, mask)

        self.assertEqual(balanced.item(), 3.0)
        self.assertNotEqual(balanced.item(), elementwise.mean().item())

    def test_categorical_losses_use_expected_scores_and_centered_rater_severity(
        self,
    ) -> None:
        config = legacy_config(
            detail_head_mode="categorical",
            detail_expected_loss_weight=0.25,
            detail_distribution_loss_weight=0.1,
            detail_hierarchy_loss_weight=0.1,
            detail_rater_loss_weight=0.1,
            detail_rater_registry=(("source", "rater-a"), ("source", "rater-b")),
        ).validate()
        scorer = RegressionScorer(_TinyBackbone(4), 4, config)
        _zero_score_side(scorer)
        scorer.train()
        features = torch.zeros(2, 3, 4)
        scores, trait_probabilities = scorer.predictions_from_features(features)
        detail_scores, detail_probabilities, detail_logits = (
            scorer.detail_predictions_from_features(features)
        )

        criterion_scores = torch.full((2, 9), 3.0)
        criterion_mask = torch.ones((2, 9), dtype=torch.bool)
        criterion_distributions = torch.zeros((2, 9, 5))
        criterion_distributions[0, :, 0] = 1.0
        criterion_distributions[1, :, 4] = 1.0
        distribution_mask = torch.ones((2, 9), dtype=torch.bool)
        rater_ids = torch.tensor([[0], [1]])
        rater_labels = torch.stack(
            (torch.zeros((1, 9)), torch.full((1, 9), 4)), dim=0
        ).long()
        rater_mask = torch.ones((2, 1, 9), dtype=torch.bool)

        loss = scorer._training_loss(
            scores,
            trait_probabilities,
            torch.full((2, 3), 3.0),
            detail_scores=detail_scores,
            detail_probabilities=detail_probabilities,
            detail_logits=detail_logits,
            criterion_scores=criterion_scores,
            criterion_distributions=criterion_distributions,
            criterion_mask=criterion_mask,
            criterion_distribution_mask=distribution_mask,
            detail_rater_ids=rater_ids,
            detail_rater_labels=rater_labels,
            detail_rater_mask=rater_mask,
        )
        loss.backward()

        # Uniform categorical logits have CE/log(5)=1. Expected-score MSE and
        # hierarchy are zero because both direct and criterion scores are 3.
        self.assertAlmostEqual(loss.item(), 0.2, places=6)
        self.assertIsNotNone(scorer.detail_evaluator_severity)
        assert scorer.detail_evaluator_severity is not None
        self.assertGreater(
            scorer.detail_evaluator_severity.grad.abs().sum().item(),
            0.0,
        )

    def test_new_detail_scoring_state_round_trip(self) -> None:
        config = legacy_config(
            detail_head_mode="categorical",
            detail_expected_loss_weight=0.25,
            detail_rater_loss_weight=0.1,
            detail_rater_registry=(("source", "a"), ("source", "b")),
        ).validate()
        first = RegressionScorer(_TinyBackbone(4), 4, config).eval()
        second = RegressionScorer(_TinyBackbone(4), 4, config).eval()
        incompatible = second.load_state_dict(first.scoring_state_dict(), strict=False)

        self.assertTrue(
            all(name.startswith("backbone.") for name in incompatible.missing_keys)
        )
        self.assertEqual(incompatible.unexpected_keys, [])
        features = torch.randn(2, 3, 4)
        first_scores, first_probabilities, _ = first.detail_predictions_from_features(
            features
        )
        second_scores, second_probabilities, _ = (
            second.detail_predictions_from_features(features)
        )
        torch.testing.assert_close(second_scores, first_scores)
        assert first_probabilities is not None and second_probabilities is not None
        torch.testing.assert_close(second_probabilities, first_probabilities)

    def test_inference_uses_base_detail_logits_without_rater_ids(self) -> None:
        config = legacy_config(
            detail_head_mode="categorical",
            detail_rater_loss_weight=0.1,
            detail_rater_registry=(("source", "rater-a"), ("source", "rater-b")),
        ).validate()
        scorer = RegressionScorer(_TinyBackbone(4), 4, config).eval()
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

        ordinary = scorer(**inputs)
        detailed = scorer(**inputs, return_detail_predictions=True)

        self.assertEqual(set(ordinary), {"scores"})
        self.assertEqual(detailed["detail_scores"].shape, (1, 9))
        self.assertEqual(detailed["detail_probabilities"].shape, (1, 9, 5))

    def test_config_round_trip_preserves_source_qualified_raters(self) -> None:
        config = legacy_config(
            detail_head_mode="categorical",
            detail_rater_loss_weight=0.1,
            detail_rater_registry=(
                ("nikl_competition", "7"),
                ("aihub26_essay", "7"),
            ),
        ).validate()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            save_config(config, path)
            restored = load_config(path)

        self.assertEqual(restored.detail_rater_registry, config.detail_rater_registry)
        self.assertEqual(restored.detail_head_mode, "categorical")
        self.assertEqual(restored.detail_rater_loss_weight, 0.1)

    def test_checkpoint_manifest_records_detail_architecture(self) -> None:
        config = legacy_config(
            detail_head_mode="categorical",
            detail_final_source="criterion",
            detail_expected_loss_weight=0.25,
            detail_distribution_loss_weight=0.1,
            detail_rater_loss_weight=0.1,
            detail_rater_registry=(("source", "a"), ("source", "b")),
        ).validate()
        loaded = LoadedRegressionModel(
            scorer=RegressionScorer(_TinyBackbone(4), 4, config),
            tokenizer=_TinyTokenizer(),
            config=config,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_checkpoint(loaded, directory)
            manifest = json.loads(
                (checkpoint / "manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["detail_head_mode"], "categorical")
        self.assertEqual(manifest["detail_final_source"], "criterion")
        self.assertEqual(manifest["detail_rater_registry_count"], 2)
        self.assertEqual(len(manifest["detail_criteria"]), 9)
        self.assertEqual(
            manifest["detail_loss_weights"],
            {
                "expected": 0.25,
                "distribution": 0.1,
                "halfstep": 0.0,
                "rater_set": 0.0,
                "hierarchy": 0.0,
                "rater": 0.1,
            },
        )

    def test_invalid_detail_combinations_fail_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "detail_head_mode"):
            legacy_config(detail_expected_loss_weight=0.1).validate()
        with self.assertRaisesRegex(ValueError, "categorical"):
            legacy_config(
                detail_head_mode="scalar",
                detail_expected_loss_weight=0.1,
                detail_distribution_loss_weight=0.1,
            ).validate()
        with self.assertRaisesRegex(ValueError, "활성 detail_head_mode"):
            legacy_config(detail_final_source="criterion").validate()
        with self.assertRaisesRegex(ValueError, "hierarchy"):
            legacy_config(
                detail_head_mode="scalar",
                detail_final_source="criterion",
                detail_hierarchy_loss_weight=0.1,
            ).validate()
        with self.assertRaisesRegex(ValueError, "halfstep_categorical"):
            legacy_config(
                detail_head_mode="scalar",
                detail_halfstep_loss_weight=0.1,
            ).validate()
        with self.assertRaisesRegex(ValueError, "rater_set"):
            legacy_config(
                detail_head_mode="categorical",
                detail_expected_loss_weight=0.25,
                detail_rater_set_loss_weight=0.1,
            ).validate()
        with self.assertRaisesRegex(ValueError, "detail_expected_loss_weight"):
            legacy_config(
                detail_head_mode="rater_set",
                detail_rater_set_loss_weight=0.1,
            ).validate()
        with self.assertRaisesRegex(ValueError, "categorical"):
            legacy_config(
                detail_head_mode="halfstep_categorical",
                detail_distribution_loss_weight=0.1,
            ).validate()
        with self.assertRaisesRegex(ValueError, "categorical"):
            legacy_config(
                detail_head_mode="rater_set",
                detail_expected_loss_weight=0.25,
                detail_rater_loss_weight=0.1,
            ).validate()
        missing_registry = legacy_config(
            detail_head_mode="categorical",
            detail_rater_loss_weight=0.1,
        ).validate()
        with self.assertRaisesRegex(ValueError, "detail_rater_registry"):
            RegressionScorer(_TinyBackbone(4), 4, missing_registry)


if __name__ == "__main__":
    unittest.main()
