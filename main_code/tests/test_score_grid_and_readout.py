from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from main_code.config import TRAITS, RegressionConfig
from main_code.models import (
    MAX_TRAIT_CLASSES,
    TRAIT_CLASS_COUNTS,
    RegressionScorer,
    trait_native_distribution_targets,
    trait_score_grid,
)
from main_code.tests.config_helpers import legacy_config


class _ConstantBackbone(nn.Module):
    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, input_ids: torch.Tensor, **_):
        hidden = torch.ones((*input_ids.shape, self.hidden_size), dtype=torch.float32)
        return SimpleNamespace(last_hidden_state=hidden + self.anchor * 0)


def _scorer(**updates) -> RegressionScorer:
    return RegressionScorer(
        _ConstantBackbone(), 4, legacy_config(pooling="mean", **updates)
    )


class TraitScoreGridTest(unittest.TestCase):
    """격자는 채점 규칙에서 온다. train 전수 확인값과 코드가 어긋나면 실패한다."""

    def test_grid_matches_the_measured_label_space(self) -> None:
        self.assertEqual(TRAIT_CLASS_COUNTS["content"], 41)
        self.assertEqual(TRAIT_CLASS_COUNTS["organization"], 17)
        self.assertEqual(TRAIT_CLASS_COUNTS["expression"], 17)
        self.assertEqual(MAX_TRAIT_CLASSES, 41)
        for trait in TRAITS:
            grid = trait_score_grid(trait)
            self.assertAlmostEqual(grid[0], 1.0)
            self.assertAlmostEqual(grid[-1], 5.0)

    def test_targets_are_one_hot_on_the_native_grid(self) -> None:
        labels = torch.tensor([[3.4, 3.25, 4.75], [1.0, 5.0, 1.25]])
        targets = trait_native_distribution_targets(labels)
        self.assertEqual(tuple(targets.shape), (2, 3, MAX_TRAIT_CLASSES))
        torch.testing.assert_close(targets.sum(dim=-1), torch.ones(2, 3))
        # content 3.4 -> (3.4-1)/0.1 = 24번 class
        self.assertEqual(int(targets[0, 0].argmax()), 24)
        # organization 3.25 -> (3.25-1)/0.25 = 9번 class
        self.assertEqual(int(targets[0, 1].argmax()), 9)
        # padding class는 절대 켜지지 않는다
        self.assertEqual(float(targets[:, 1, TRAIT_CLASS_COUNTS["organization"] :].sum()), 0.0)


class CategoricalReadoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.input_ids = torch.ones((4, 6), dtype=torch.long)
        self.attention_mask = torch.ones_like(self.input_ids)
        self.labels = torch.tensor(
            [[3.4, 3.25, 4.75], [2.0, 2.5, 3.0], [4.1, 4.0, 2.25], [1.5, 1.0, 5.0]]
        )

    def _scores(self, **updates) -> torch.Tensor:
        scorer = _scorer(distribution_loss_weight=1.0, **updates)
        scorer.eval()
        features = scorer.encode(self.input_ids, self.attention_mask)
        scores, _ = scorer.predictions_from_features(features)
        return scores

    def test_expectation_is_continuous_and_inside_the_range(self) -> None:
        scores = self._scores(score_head="trait_native_distribution")
        self.assertTrue(torch.all(scores >= 1.0) and torch.all(scores <= 5.0))

    def test_argmax_lands_exactly_on_the_native_grid(self) -> None:
        scores = self._scores(
            score_head="trait_native_distribution", categorical_readout="argmax"
        )
        for trait_index, trait in enumerate(TRAITS):
            grid = torch.tensor(trait_score_grid(trait))
            for value in scores[:, trait_index]:
                self.assertTrue(
                    bool(torch.isclose(grid, value, atol=1e-6).any()),
                    f"{trait} 예측 {float(value)}가 격자 위에 없다",
                )

    def test_five_class_argmax_lands_on_integers(self) -> None:
        scores = self._scores(score_head="distribution", categorical_readout="argmax")
        torch.testing.assert_close(scores, scores.round())

    def test_gumbel_is_on_grid_and_still_has_gradient(self) -> None:
        scorer = _scorer(
            score_head="trait_native_distribution",
            categorical_readout="gumbel_straight_through",
            distribution_loss_weight=0.0,
        )
        scorer.train()
        outputs = scorer(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=self.labels,
        )
        self.assertTrue(torch.isfinite(outputs["loss"]))
        outputs["loss"].backward()
        grad = scorer.heads[TRAITS[0]].weight.grad
        self.assertIsNotNone(grad)
        # argmax였다면 MSE만으로는 gradient가 0이다. straight-through라 통과해야 한다.
        self.assertGreater(float(grad.abs().sum()), 0.0)

        scorer.eval()
        features = scorer.encode(self.input_ids, self.attention_mask)
        scores, _ = scorer.predictions_from_features(features)
        grid = torch.tensor(trait_score_grid("organization"))
        for value in scores[:, 1]:
            self.assertTrue(bool(torch.isclose(grid, value, atol=1e-6).any()))

    def test_argmax_without_cross_entropy_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "gumbel_straight_through"):
            legacy_config(
                score_head="distribution",
                categorical_readout="argmax",
                distribution_loss_weight=0.0,
            ).validate()

    def test_readout_needs_a_categorical_final_head(self) -> None:
        with self.assertRaisesRegex(ValueError, "분류일 때만"):
            legacy_config(categorical_readout="argmax").validate()

    def test_score_head_needs_direct_final_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "detail_final_source='direct'"):
            legacy_config(
                score_head="distribution",
                detail_head_mode="scalar",
                detail_final_source="criterion",
                detail_expected_loss_weight=0.25,
            ).validate()


class RaterSetScalarTest(unittest.TestCase):
    def _config(self, **updates) -> RegressionConfig:
        return legacy_config(
            pooling="mean",
            detail_head_mode="rater_set_scalar",
            detail_final_source="criterion",
            detail_expected_loss_weight=0.25,
            detail_rater_set_loss_weight=0.25,
            **updates,
        )

    def test_eighteen_scalar_heads_exist(self) -> None:
        scorer = RegressionScorer(_ConstantBackbone(), 4, self._config())
        self.assertEqual(len(scorer.detail_heads), 18)
        for head in scorer.detail_heads.values():
            self.assertEqual(head.out_features, 1)

    def test_detail_scores_are_bounded_and_averaged_over_raters(self) -> None:
        scorer = RegressionScorer(_ConstantBackbone(), 4, self._config())
        features = torch.randn(3, 4)
        detail_scores, probabilities, raw = scorer.detail_predictions_from_features(
            features
        )
        self.assertIsNone(probabilities)
        self.assertEqual(tuple(raw.shape), (3, 2, 9, 1))
        self.assertEqual(tuple(detail_scores.shape), (3, 9))
        self.assertTrue(torch.all(detail_scores >= 1.0))
        self.assertTrue(torch.all(detail_scores <= 5.0))

    def test_matching_is_permutation_invariant(self) -> None:
        scorer = RegressionScorer(_ConstantBackbone(), 4, self._config())
        predictions = torch.rand(2, 2, 9) * 4 + 1
        labels = torch.randint(0, 5, (2, 2, 9))
        mask = torch.ones_like(labels, dtype=torch.bool)
        straight = scorer._detail_rater_set_squared_error(predictions, labels, mask)
        swapped = scorer._detail_rater_set_squared_error(
            predictions, labels.flip(dims=[1]), mask
        )
        torch.testing.assert_close(straight, swapped)

    def test_config_gate_allows_only_rater_set_modes(self) -> None:
        self._config().validate()
        with self.assertRaisesRegex(ValueError, "detail_rater_set_loss_weight"):
            legacy_config(
                detail_head_mode="scalar",
                detail_final_source="criterion",
                detail_expected_loss_weight=0.25,
                detail_rater_set_loss_weight=0.25,
            ).validate()


if __name__ == "__main__":
    unittest.main()
