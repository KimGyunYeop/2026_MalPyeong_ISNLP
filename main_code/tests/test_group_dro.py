"""문항 GroupDRO의 계약.

왜 이 축인가
    우리 문항별 RMSE는 Q6 0.3236 ~ Q4 0.5020으로 크게 벌어져 있다. 숨은 평가셋의
    문항 구성은 우리 validation과 전혀 다르므로(2025 자료 4,000편의 71%가 우리 학습에
    없는 Q11·Q12) 평균이 아니라 **최악 문항**을 좋게 만드는 것이 구성 변화에 강하다.

    고정 재가중(`tail_weight_*`)은 라벨 값으로 가중치를 미리 정했고 실패했다.
    GroupDRO는 학습 중 관측된 그룹 손실로 가중치를 갱신하므로 어느 문항이 어려운지
    스스로 찾는다. 그리고 새 선택 지표 `worst_prompt_rmse`가 재는 양을 목적함수가
    직접 최적화하게 된다 — 목적과 저울이 같은 것을 본다.
"""

from __future__ import annotations

import pytest
import torch

from main_code.config import RegressionConfig
from main_code.models import (
    GROUP_DRO_SLOTS,
    RegressionScorer,
    group_dro_sample_weights,
)


def config(**updates: object) -> RegressionConfig:
    base = RegressionConfig(model_id="skt/A.X-4.0-Light").with_updates(
        essay_surface="official_raw",
        pooling="mean",
        training_mode="lora_only",
        input_format="baseline_v1",
    )
    return base.with_updates(**updates) if updates else base


def uniform() -> torch.Tensor:
    return torch.full((GROUP_DRO_SLOTS,), 1.0 / GROUP_DRO_SLOTS)


def test_disabled_by_default() -> None:
    assert config().group_dro_step_size == 0.0


def test_returns_none_when_disabled() -> None:
    """None이어야 호출부가 기존 element_weights=None 경로를 그대로 탄다."""

    assert (
        group_dro_sample_weights(
            torch.rand(4, 3), torch.tensor([1, 1, 2, 2]), uniform(), 0.0
        )
        is None
    )


def test_returns_none_without_prompt_ids() -> None:
    assert group_dro_sample_weights(torch.rand(4, 3), None, uniform(), 0.05) is None


def test_returns_none_when_no_group_is_valid() -> None:
    """문항 표기가 없는 batch에서 조용히 이상한 그룹으로 뭉치면 안 된다."""

    assert (
        group_dro_sample_weights(
            torch.rand(3, 3), torch.tensor([-1, 999, 10_000]), uniform(), 0.05
        )
        is None
    )


def test_harder_group_gets_heavier() -> None:
    weights = uniform()
    loss = torch.tensor([[0.1] * 3, [0.1] * 3, [2.0] * 3, [2.0] * 3])
    ids = torch.tensor([1, 1, 2, 2])
    for _ in range(5):
        group_dro_sample_weights(loss, ids, weights, 0.05)
    assert float(weights[2]) > float(weights[1])
    # 등장하지 않은 그룹은 정규화 때문에 줄기만 하고 순서가 뒤집히지 않는다.
    assert float(weights[2]) > float(weights[7])


def test_sample_weights_average_to_one() -> None:
    """loss 규모가 바뀌면 lr을 다시 잡아야 한다. batch 평균 1을 유지해야 한다."""

    weights = uniform()
    loss = torch.tensor([[0.2] * 3, [1.5] * 3, [1.5] * 3, [3.0] * 3])
    ids = torch.tensor([1, 2, 2, 3])
    for _ in range(3):
        sample = group_dro_sample_weights(loss, ids, weights, 0.1)
    assert sample is not None
    assert float(sample.mean()) == pytest.approx(1.0, abs=1e-5)
    assert sample.shape == (4, 1)


def test_weights_stay_a_probability_vector() -> None:
    weights = uniform()
    loss = torch.rand(8, 3) * 5
    ids = torch.tensor([1, 1, 2, 3, 3, 4, 5, 5])
    for _ in range(20):
        group_dro_sample_weights(loss, ids, weights, 0.2)
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-5)
    assert bool((weights > 0).all())


def test_no_gradient_flows_into_the_weights() -> None:
    """가중치는 관측에서 나오는 값이고 최적화 대상이 아니다."""

    weights = uniform()
    loss = torch.rand(4, 3, requires_grad=True)
    sample = group_dro_sample_weights(loss, torch.tensor([1, 1, 2, 2]), weights, 0.05)
    assert sample is not None and not sample.requires_grad
    assert not weights.requires_grad


def test_cannot_combine_with_tail_weighting() -> None:
    """둘이 같은 element_weights 경로를 다투면 무엇을 재는지 흐려진다."""

    with pytest.raises(ValueError, match="group_dro_step_size"):
        config(group_dro_step_size=0.05, tail_weight_target_sigma=0.8)


def test_negative_step_size_rejected() -> None:
    with pytest.raises(ValueError, match="group_dro_step_size"):
        config(group_dro_step_size=-0.1)


def test_weights_are_excluded_from_the_checkpoint() -> None:
    """담기면 step_size=0인 config로 로드할 때 unexpected key로 하드 실패한다."""

    assert "group_dro_weights" in RegressionScorer.TRAINING_ONLY_STATE_PREFIXES

    class Fake:
        TRAINING_ONLY_STATE_PREFIXES = RegressionScorer.TRAINING_ONLY_STATE_PREFIXES
        scoring_state_dict = RegressionScorer.scoring_state_dict

        def state_dict(self):
            return {
                "backbone.w": torch.zeros(1),
                "heads.0.weight": torch.zeros(2),
                "group_dro_weights": torch.zeros(32),
                "prompt_adversary.0.weight": torch.zeros(3),
            }

    assert sorted(Fake().scoring_state_dict()) == ["heads.0.weight"]


def test_training_loss_accepts_prompt_group_ids() -> None:
    """배선이 끊기면 위 계약이 다 무의미하다."""

    import inspect

    assert (
        "prompt_group_ids"
        in inspect.signature(RegressionScorer._training_loss).parameters
    )
