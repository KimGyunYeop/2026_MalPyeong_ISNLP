from __future__ import annotations

import numpy as np
import pytest
import torch

from main_code.config import RegressionConfig
from main_code.models import trait_average_objective
from main_code.utils import regression_metrics
from main_code.tests.config_helpers import legacy_config


def _config(**updates) -> RegressionConfig:
    values = {"batch_size": 4, "pairwise_temperature": 0.5}
    values.update(updates)
    return legacy_config(**values).validate()


def test_default_config_keeps_objective_off() -> None:
    config = _config()
    assert config.trait_average_loss_weight == 0.0
    assert config.trait_average_pairwise_weight == 0.0


def test_negative_weight_is_rejected() -> None:
    with pytest.raises(ValueError):
        _config(trait_average_loss_weight=-0.1)
    with pytest.raises(ValueError):
        _config(trait_average_pairwise_weight=-1.0)


def test_zero_when_trait_average_matches_even_if_traits_are_wrong() -> None:
    """정의 C가 벌하는 것은 평균 오차뿐임을 명시한다.

    trait별 오차가 서로 상쇄되면 per-trait MSE는 크지만 평균 항은 0이어야 한다.
    """

    config = _config(trait_average_loss_weight=1.0)
    labels = torch.tensor([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
    offsetting = torch.tensor([[3.5, 2.5, 3.0], [3.0, 5.0, 4.0]])
    assert torch.square(offsetting - labels).mean() > 0.1
    assert float(trait_average_objective(offsetting, labels, config)) == pytest.approx(
        0.0, abs=1e-6
    )


def test_penalises_shared_bias_across_traits() -> None:
    config = _config(trait_average_loss_weight=1.0)
    labels = torch.tensor([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
    shared_bias = labels + 0.3
    value = float(trait_average_objective(shared_bias, labels, config))
    assert value == pytest.approx(0.09, abs=1e-6)


def test_weight_scales_linearly_and_gradient_flows() -> None:
    labels = torch.tensor([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
    scores = torch.tensor(
        [[3.5, 3.5, 3.5], [4.5, 4.5, 4.5]], requires_grad=True
    )
    half = trait_average_objective(scores, labels, _config(trait_average_loss_weight=0.5))
    full = trait_average_objective(scores, labels, _config(trait_average_loss_weight=1.0))
    assert full.item() == pytest.approx(2 * half.item(), rel=1e-6)
    full.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    # 평균 항이므로 세 trait에 같은 gradient가 흐른다.
    assert scores.grad[0].tolist() == pytest.approx([scores.grad[0][0].item()] * 3)


def test_pairwise_term_prefers_correct_average_order() -> None:
    config = _config(trait_average_loss_weight=0.0, trait_average_pairwise_weight=1.0)
    labels = torch.tensor([[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]])
    correct = torch.tensor([[2.2, 2.1, 2.0], [3.8, 4.1, 4.0]])
    inverted = torch.tensor([[3.8, 4.1, 4.0], [2.2, 2.1, 2.0]])
    assert float(trait_average_objective(correct, labels, config)) < float(
        trait_average_objective(inverted, labels, config)
    )


def test_pairwise_term_ignores_tied_averages() -> None:
    config = _config(trait_average_loss_weight=0.0, trait_average_pairwise_weight=1.0)
    tied = torch.tensor([[3.0, 3.0, 3.0], [2.0, 3.0, 4.0]])
    scores = torch.tensor([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
    assert float(trait_average_objective(scores, tied, config)) == pytest.approx(
        0.0, abs=1e-6
    )


def test_metrics_report_both_definitions_without_changing_existing_keys() -> None:
    labels = np.array([[3.0, 3.5, 4.0], [2.0, 2.5, 3.0], [4.0, 4.5, 5.0]])
    predictions = labels + np.array([[0.5, -0.5, 0.0], [0.0, 0.0, 0.0], [0.2, 0.2, 0.2]])
    metrics = regression_metrics(labels, predictions)

    # 기존 계약: overall은 세 trait RMSE의 산술평균이다.
    per_trait = [metrics["traits"][t]["rmse"] for t in ("content", "organization", "expression")]
    assert metrics["overall"]["rmse"] == pytest.approx(float(np.mean(per_trait)))

    # additive 진단 key: essay별 평균 점수 1개로 계산한다.
    expected = float(
        np.sqrt(np.mean((labels.mean(axis=1) - predictions.mean(axis=1)) ** 2))
    )
    assert metrics["trait_average"]["rmse"] == pytest.approx(expected)
    assert metrics["trait_average"]["spearman"] is not None
    # 첫 essay의 오차가 상쇄되므로 정의 C가 정의 A보다 낮아야 한다.
    assert metrics["trait_average"]["rmse"] < metrics["overall"]["rmse"]
