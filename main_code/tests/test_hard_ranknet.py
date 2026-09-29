from __future__ import annotations

import pytest
import torch

from main_code.config import RegressionConfig
from main_code.models import pairwise_ranking_loss


def _hard_rank_config(**updates) -> RegressionConfig:
    values = {
        "batch_size": 2,
        "mse_loss_weight": 0.0,
        "pairwise_loss": "hard_ranknet",
        "pairwise_loss_weight": 1.0,
        "pairwise_temperature": 0.5,
        "ranking_content_weight": 0.0,
        "ranking_organization_weight": 1.0,
        "ranking_expression_weight": 0.0,
    }
    values.update(updates)
    return RegressionConfig(**values).validate()


def test_hard_ranknet_uses_only_order_sign_and_excludes_ties() -> None:
    config = _hard_rank_config()
    scores = torch.tensor(
        [[3.0, 2.0, 3.0], [3.0, 4.0, 3.0]], requires_grad=True
    )
    small_gap = torch.tensor([[3.0, 2.5, 3.0], [3.0, 3.0, 3.0]])
    large_gap = torch.tensor([[1.0, 1.0, 5.0], [5.0, 5.0, 1.0]])

    small_loss = pairwise_ranking_loss(scores, small_gap, config)
    large_loss = pairwise_ranking_loss(scores, large_gap, config)
    assert float(small_loss.detach()) == pytest.approx(float(large_loss.detach()))

    reversed_scores = scores.detach().flip(0)
    assert small_loss < pairwise_ranking_loss(reversed_scores, small_gap, config)

    tied_labels = torch.full((2, 3), 3.0)
    tied_loss = pairwise_ranking_loss(scores, tied_labels, config)
    assert float(tied_loss.detach()) == 0.0
    (small_loss + tied_loss).backward()
    assert torch.isfinite(scores.grad).all()


def test_hard_ranknet_rejects_gap_weighting() -> None:
    with pytest.raises(ValueError, match="점수 간격"):
        _hard_rank_config(pairwise_gap_weighted=True)
