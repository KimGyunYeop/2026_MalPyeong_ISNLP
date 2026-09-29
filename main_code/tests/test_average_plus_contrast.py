from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from main_code.config import TRAITS
from main_code.models import (
    AVERAGE_CONTRAST_RANGE,
    RegressionScorer,
    average_plus_contrast_scores,
)
from main_code.tests.config_helpers import legacy_config


class _ConstantBackbone(nn.Module):
    """Return one fixed hidden vector per token so heads decide the output."""

    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, input_ids: torch.Tensor, **_):
        hidden = torch.ones(
            (*input_ids.shape, self.hidden_size), dtype=torch.float32
        )
        return SimpleNamespace(last_hidden_state=hidden + self.anchor * 0)


def _scorer(**updates) -> RegressionScorer:
    config = legacy_config(pooling="mean", **updates).validate()
    return RegressionScorer(_ConstantBackbone(), 4, config)


class AveragePlusContrastFunctionTest(unittest.TestCase):
    def test_trait_mean_equals_the_average_head(self) -> None:
        raw = torch.tensor(
            [[[0.7], [-1.3], [2.0]], [[-2.5], [0.0], [0.4]]], dtype=torch.float32
        )
        scores = average_plus_contrast_scores(raw)
        self.assertEqual(tuple(scores.shape), (2, len(TRAITS)))

        expected_average = 1.0 + 4.0 * torch.sigmoid(raw[:, 0, 0])
        torch.testing.assert_close(scores.mean(dim=-1), expected_average)

    def test_average_is_bounded_and_contrast_is_limited(self) -> None:
        extreme = torch.tensor(
            [[[40.0], [40.0], [-40.0]], [[-40.0], [-40.0], [40.0]]],
            dtype=torch.float32,
        )
        scores = average_plus_contrast_scores(extreme)
        averages = scores.mean(dim=-1)
        self.assertTrue(torch.all(averages >= 1.0))
        self.assertTrue(torch.all(averages <= 5.0))
        deviations = (scores - averages.unsqueeze(-1)).abs()
        # 두 자유 대비는 tanh로 제한되고 세 번째는 그 합의 반대값이다.
        self.assertLessEqual(
            float(deviations.max()), 2.0 * AVERAGE_CONTRAST_RANGE + 1e-5
        )

    def test_rejects_distribution_shaped_raw(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[B,3,1\]"):
            average_plus_contrast_scores(torch.zeros((2, len(TRAITS), 5)))


class AveragePlusContrastModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.input_ids = torch.ones((3, 6), dtype=torch.long)
        self.attention_mask = torch.ones_like(self.input_ids)

    def test_forward_keeps_the_identity_after_head_updates(self) -> None:
        scorer = _scorer(score_parameterization="average_plus_contrast")
        with torch.no_grad():
            for index, trait in enumerate(TRAITS):
                scorer.heads[trait].weight.fill_(0.3 * (index + 1))
                scorer.heads[trait].bias.fill_(-0.5 * index)
        features = scorer.encode(self.input_ids, self.attention_mask)
        scores, probabilities = scorer.predictions_from_features(features)
        self.assertIsNone(probabilities)

        raw = scorer._raw_head_outputs(features)
        expected_average = 1.0 + 4.0 * torch.sigmoid(raw[:, 0, 0])
        torch.testing.assert_close(scores.mean(dim=-1), expected_average)

    def test_default_parameterization_is_unchanged(self) -> None:
        scorer = _scorer()
        self.assertEqual(scorer.regression_config.score_parameterization, "traits")
        features = scorer.encode(self.input_ids, self.attention_mask)
        scores, _ = scorer.predictions_from_features(features)
        raw = scorer._raw_head_outputs(features)
        torch.testing.assert_close(
            scores, 1.0 + 4.0 * torch.sigmoid(raw.squeeze(-1)), rtol=0, atol=0
        )

    def test_no_new_checkpoint_key_is_created(self) -> None:
        traits_keys = set(_scorer().scoring_state_dict())
        reparameterized_keys = set(
            _scorer(score_parameterization="average_plus_contrast").scoring_state_dict()
        )
        self.assertEqual(traits_keys, reparameterized_keys)

    def test_training_loss_reaches_the_average_head(self) -> None:
        scorer = _scorer(score_parameterization="average_plus_contrast")
        labels = torch.tensor(
            [[3.0, 3.5, 4.0], [2.0, 2.5, 2.5], [4.5, 4.0, 3.5]], dtype=torch.float32
        )
        outputs = scorer(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            labels=labels,
        )
        loss = outputs["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        average_head = scorer.heads[TRAITS[0]]
        self.assertIsNotNone(average_head.weight.grad)
        self.assertNotEqual(float(average_head.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
