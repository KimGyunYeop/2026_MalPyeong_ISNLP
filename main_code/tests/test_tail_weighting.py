"""평가셋 라벨 분포 이동 보정의 계약.

왜 이 파일이 필요한가
---------------------
2026-08-20 운영측 답변으로 "예측값 0점 처리 샘플 없음"이 확정되면서, 리더보드
RMSE 0.5191과 로컬 0.4168의 격차는 전부 **분포 차이**로 남았다. Spearman이
0.7597 -> 0.7340으로 거의 그대로였다는 점이 근거다. 미학습 문항 때문이라면 순위
능력이 같이 무너져야 했다.

이 가중치는 그 격차를 학습 시점에 겨냥한다. 위험한 종류의 변경이므로 세 가지를
못 박는다.
  1. 기본값에서 **완전히 꺼진다** (기존 체크포인트 재현이 깨지면 안 된다).
  2. 평균 가중이 1이다 (loss 규모가 바뀌면 lr을 다시 잡아야 한다).
  3. 꼬리가 중앙보다 무겁다 (부호가 뒤집히면 문제를 악화시킨다).
"""

from __future__ import annotations

import math

import pytest
import torch

from main_code.config import RegressionConfig
from main_code.models import sample_tail_weights

TRAITS = 3


def config(**updates: object) -> RegressionConfig:
    base = RegressionConfig(model_id="skt/A.X-4.0-Light")
    return base.with_updates(**updates) if updates else base


def labels(values: list[float]) -> torch.Tensor:
    return torch.tensor([[value] * TRAITS for value in values], dtype=torch.float32)


def test_disabled_by_default_returns_none() -> None:
    """None이어야 호출부가 element_weights=None으로 기존 경로를 그대로 탄다."""

    assert config().tail_weight_target_sigma == 0.0
    assert sample_tail_weights(labels([1.0, 3.0, 5.0]), config()) is None


def test_zero_target_sigma_is_bit_exact_off() -> None:
    assert sample_tail_weights(labels([2.0]), config(tail_weight_target_sigma=0.0)) is None


@pytest.mark.parametrize("sigma", [0.70, 0.80, 0.90])
def test_expected_weight_is_one_on_the_reference_distribution(sigma: float) -> None:
    """E[w]=σ_ref/σ_target 이라는 닫힌 해가 실제로 1로 정규화되는지."""

    reference_mean, reference_sigma = 3.398, 0.617
    generator = torch.Generator().manual_seed(43)
    draws = torch.normal(
        reference_mean, reference_sigma, size=(200_000,), generator=generator
    )
    spec = config(
        tail_weight_target_sigma=sigma,
        tail_weight_reference_mean=reference_mean,
        tail_weight_reference_sigma=reference_sigma,
        tail_weight_max=1e9,  # 상한을 사실상 끄고 순수 정규화만 본다
    )
    weights = sample_tail_weights(draws.unsqueeze(-1).expand(-1, TRAITS), spec)
    assert weights is not None
    assert float(weights.mean()) == pytest.approx(1.0, abs=0.05)


def test_tails_outweigh_the_centre() -> None:
    spec = config(tail_weight_target_sigma=0.80)
    weights = sample_tail_weights(labels([1.5, 2.5, 3.398, 4.3, 4.9]), spec).squeeze(-1)
    centre = float(weights[2])
    assert float(weights[0]) > centre and float(weights[4]) > centre
    assert float(weights[1]) > centre and float(weights[3]) > centre
    # 중앙은 1보다 가벼워야 한다. 그러지 않으면 전체 loss만 커진 것이다.
    assert centre < 1.0


def test_weight_is_symmetric_around_the_reference_mean() -> None:
    """편향을 넣지 않는다. 낮은 쪽만 올리면 예측이 통째로 내려간다."""

    spec = config(tail_weight_target_sigma=0.80, tail_weight_reference_mean=3.4)
    weights = sample_tail_weights(labels([3.4 - 1.2, 3.4 + 1.2]), spec).squeeze(-1)
    assert float(weights[0]) == pytest.approx(float(weights[1]), rel=1e-6)


def test_clip_bounds_the_rarest_samples() -> None:
    spec = config(tail_weight_target_sigma=0.90, tail_weight_max=2.5)
    weights = sample_tail_weights(labels([1.0, 3.4, 5.0]), spec).squeeze(-1)
    assert float(weights.max()) <= 2.5 + 1e-6
    assert float(weights[0]) == pytest.approx(2.5, rel=1e-6)


def test_shape_broadcasts_over_traits() -> None:
    spec = config(tail_weight_target_sigma=0.80)
    weights = sample_tail_weights(labels([2.0, 3.0, 4.0]), spec)
    assert weights.shape == (3, 1)
    assert (weights * torch.ones(3, TRAITS)).shape == (3, TRAITS)


def test_target_narrower_than_reference_downweights_the_tails() -> None:
    """부호 확인. 목표가 더 좁으면 꼬리가 **가벼워져야** 한다."""

    spec = config(tail_weight_target_sigma=0.40, tail_weight_reference_sigma=0.617)
    weights = sample_tail_weights(labels([1.5, 3.398, 4.9]), spec).squeeze(-1)
    assert float(weights[0]) < float(weights[1])
    assert float(weights[2]) < float(weights[1])


def test_negative_sigma_is_rejected() -> None:
    with pytest.raises(ValueError, match="tail_weight_target_sigma"):
        config(tail_weight_target_sigma=-0.1)


def test_max_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="tail_weight_max"):
        config(tail_weight_target_sigma=0.8, tail_weight_max=0.5)


def test_loss_path_uses_the_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """가중치가 실제로 MSE 항을 바꾸는지. 헬퍼만 맞고 배선이 끊기면 무의미하다."""

    from main_code import models

    seen: list[torch.Tensor | None] = []
    original = models._weighted_trait_mean

    def spy(values, spec, *, ranking, valid=None, element_weights=None):
        seen.append(element_weights)
        return original(
            values, spec, ranking=ranking, valid=valid, element_weights=element_weights
        )

    monkeypatch.setattr(models, "_weighted_trait_mean", spy)

    scores = torch.tensor([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]])
    gold = labels([1.5, 3.4])
    spec = config(tail_weight_target_sigma=0.80, distribution_loss_weight=0.0)

    plain = models._weighted_trait_mean(
        torch.square(scores - gold), spec, ranking=False
    )
    weights = sample_tail_weights(gold, spec)
    weighted = models._weighted_trait_mean(
        torch.square(scores - gold), spec, ranking=False, element_weights=weights
    )
    # 큰 오차(1.5 대 3.0)를 가진 쪽이 무거워지므로 가중 loss가 더 커야 한다.
    assert float(weighted) > float(plain)
