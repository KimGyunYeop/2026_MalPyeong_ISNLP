"""soft_spearman의 tie 안정성 계약.

batch 안에서 한 trait의 라벨이 전부 동점이면 상관계수 분모 energy가 정확히 0이 된다.
forward는 `valid` 마스크로 그 열을 버리지만, autograd는 모든 원소의 `sqrt` backward를
계산하므로 `0/0`이 그 batch **전체** gradient를 NaN으로 만들었다. batch가 작을수록 동점
확률이 커서 `batch_size=4` run은 첫 eval 전에 전부 죽고 `batch_size=32`는 살아남았다.

이 파일은 (1) tie가 있어도 gradient가 유한하고 (2) tie가 없는 기존 경로의 forward 값이
바뀌지 않았음을 함께 고정한다.
"""

from __future__ import annotations

import torch

from main_code.config import RegressionConfig
from main_code.models import listwise_ranking_loss


def _config(**updates) -> RegressionConfig:
    base = {"listwise_loss": "soft_spearman", "listwise_loss_weight": 0.2}
    base.update(updates)
    return RegressionConfig().with_updates(**base)


def _grad(scores: torch.Tensor, labels: torch.Tensor, config) -> torch.Tensor:
    leaf = scores.clone().requires_grad_(True)
    listwise_ranking_loss(leaf, labels, config).backward()
    assert leaf.grad is not None
    return leaf.grad


def test_tied_trait_column_keeps_gradient_finite() -> None:
    """한 trait 라벨이 전부 동점이어도 gradient가 유한해야 한다."""

    torch.manual_seed(0)
    config = _config()
    for batch_size in (2, 3, 4, 8, 32):
        labels = torch.rand(batch_size, 3) * 4 + 1
        labels[:, 1] = 3.0  # organization 전부 동점 -> energy == 0
        scores = torch.rand(batch_size, 3) * 4 + 1
        gradient = _grad(scores, labels, config)
        assert torch.isfinite(gradient).all(), (
            f"batch_size={batch_size}에서 tie 열이 gradient를 오염시켰습니다"
        )


def test_all_traits_tied_keeps_gradient_finite() -> None:
    """세 trait이 모두 동점이면 loss는 0이고 gradient도 유한해야 한다."""

    config = _config()
    labels = torch.full((4, 3), 3.0)
    scores = torch.rand(4, 3) * 4 + 1
    leaf = scores.clone().requires_grad_(True)
    loss = listwise_ranking_loss(leaf, labels, config)
    loss.backward()
    assert float(loss) == 0.0
    assert leaf.grad is not None and torch.isfinite(leaf.grad).all()


def test_tied_predictions_keep_gradient_finite() -> None:
    """라벨이 아니라 **예측**이 동일해도(학습 초기) gradient가 유한해야 한다."""

    config = _config()
    labels = torch.rand(4, 3) * 4 + 1
    scores = torch.full((4, 3), 3.0)
    gradient = _grad(scores, labels, config)
    assert torch.isfinite(gradient).all()


def test_untied_forward_is_unchanged_by_the_floor() -> None:
    """동점이 없으면 clamp가 항등이라 forward 값이 이전 구현과 같아야 한다.

    하한은 `valid` 문턱과 같은 지점이므로 유효 원소에서 분모가 달라질 수 없다.
    참조값은 clamp 없는 원식으로 이 테스트 안에서 직접 계산한다.
    """

    from main_code.models import _ranking_mean, _ranking_targets, _soft_ranks

    torch.manual_seed(7)
    config = _config()
    scores = torch.rand(16, 3) * 4 + 1
    labels = torch.rand(16, 3) * 4 + 1

    ranked_scores, ranked_labels = _ranking_targets(scores, labels, config)
    predicted = _soft_ranks(ranked_scores.float(), config.listwise_temperature)
    target = _soft_ranks(ranked_labels.float(), config.listwise_temperature)
    predicted_centered = predicted - predicted.mean(dim=0)
    target_centered = target - target.mean(dim=0)
    numerator = (predicted_centered * target_centered).sum(dim=0)
    denominator = torch.sqrt(
        predicted_centered.square().sum(dim=0) * target_centered.square().sum(dim=0)
    )
    valid = denominator > 1e-8
    assert bool(valid.all()), "이 fixture는 동점이 없어야 의미가 있습니다"
    correlation = torch.zeros_like(numerator)
    correlation[valid] = (numerator[valid] / denominator[valid]).clamp(-1.0, 1.0)
    expected = _ranking_mean(1.0 - correlation, config, valid=valid)

    actual = listwise_ranking_loss(scores, labels, config)
    assert torch.equal(actual, expected)


def test_trait_average_target_with_tied_labels_is_finite() -> None:
    """`ranking_target=trait_average` 경로에도 같은 계약이 적용된다."""

    config = _config(ranking_target="trait_average")
    labels = torch.full((4, 3), 3.0)
    scores = torch.rand(4, 3) * 4 + 1
    gradient = _grad(scores, labels, config)
    assert torch.isfinite(gradient).all()
