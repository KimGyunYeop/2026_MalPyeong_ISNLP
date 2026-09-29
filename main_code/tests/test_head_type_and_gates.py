from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from main_code.config import TRAITS, RegressionConfig
from main_code.models import RegressionScorer
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
    config = legacy_config(pooling="mean", **updates)
    return RegressionScorer(_ConstantBackbone(), 4, config)


class IndependentMlpHeadTest(unittest.TestCase):
    """trait를 섞지 않는 2층 head가 실제로 별개 옵션인지 고정한다."""

    def setUp(self) -> None:
        self.input_ids = torch.ones((2, 5), dtype=torch.long)
        self.attention_mask = torch.ones_like(self.input_ids)

    def test_independent_is_one_linear_layer(self) -> None:
        scorer = _scorer(head_type="independent")
        for trait in TRAITS:
            self.assertIsInstance(scorer.heads[trait], nn.Linear)

    def test_independent_mlp_is_two_layers_without_mixing(self) -> None:
        scorer = _scorer(head_type="independent_mlp", head_hidden_size=8)
        for trait in TRAITS:
            head = scorer.heads[trait]
            self.assertIsInstance(head, nn.Sequential)
            self.assertEqual(len(head), 3)
            self.assertIsInstance(head[0], nn.Linear)
            self.assertIsInstance(head[2], nn.Linear)

        # 한 trait head만 바꿔도 다른 trait 점수는 그대로여야 섞이지 않는 것이다.
        features = scorer.encode(self.input_ids, self.attention_mask)
        before, _ = scorer.predictions_from_features(features)
        with torch.no_grad():
            scorer.heads[TRAITS[0]][2].bias.add_(5.0)
        after, _ = scorer.predictions_from_features(features)
        self.assertFalse(torch.equal(after[:, 0], before[:, 0]))
        torch.testing.assert_close(after[:, 1], before[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(after[:, 2], before[:, 2], rtol=0, atol=0)

    def test_mixed_head_v2_does_mix_traits(self) -> None:
        scorer = _scorer(head_type="mixed_head_v2", head_hidden_size=8)
        features = scorer.encode(self.input_ids, self.attention_mask)
        before, _ = scorer.predictions_from_features(features)
        with torch.no_grad():
            scorer.heads[TRAITS[0]][0].bias.add_(5.0)
        after, _ = scorer.predictions_from_features(features)
        self.assertFalse(torch.equal(after[:, 1], before[:, 1]))

    def test_average_plus_contrast_accepts_both_unmixed_heads(self) -> None:
        for head_type in ("independent", "independent_mlp"):
            config = legacy_config(
                score_parameterization="average_plus_contrast", head_type=head_type
            ).validate()
            self.assertEqual(config.head_type, head_type)
        with self.assertRaisesRegex(ValueError, "섞지 않는 head"):
            legacy_config(
                score_parameterization="average_plus_contrast",
                head_type="mixed_head_v2",
            ).validate()


class ParagraphBoundaryGateTest(unittest.TestCase):
    """문단 MTL이 pooling과 불필요하게 묶여 있지 않은지 고정한다."""

    def _config(self, **updates) -> RegressionConfig:
        return legacy_config(
            training_mode="lora_only",
            essay_surface="official_raw",
            paragraph_boundary_loss_weight=0.1,
            **updates,
        )

    def test_token_poolings_are_allowed(self) -> None:
        # boundary head는 token hidden만 쓰므로 mean 말고도 결합할 수 있어야 한다.
        for pooling in ("mean", "first", "last"):
            self._config(pooling=pooling).validate()

    def test_essay_poolings_are_rejected_with_the_real_reason(self) -> None:
        for pooling in ("essay_mean", "essay_attention"):
            with self.assertRaisesRegex(ValueError, "essay_mask"):
                self._config(pooling=pooling).validate()

    def test_non_shared_organization_pooling_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "essay_mask"):
            self._config(organization_pooling="paragraph_mean").validate()


if __name__ == "__main__":
    unittest.main()
