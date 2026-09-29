"""수준 prototype contrastive와 clip 임계 설정화의 계약.

왜 이 두 축인가 (2026-08-21)
    공식 RMSE·Spearman은 세 연속값의 **합** 하나에만 의존한다(400편에서 100% 확인).
    영역 배분을 재조정하는 구조는 지표가 보지 못하므로 외부 논문의 trait dual-view /
    trait expert 권고는 우리 지표에 닿지 않는다. 그 아래에서 살아남는 축이

      1. 수준 prototype: "이 글이 전체적으로 어느 수준인가"의 표현을 다루므로
         합에 직접 작용하고 추론 비용이 0이다 (PLAES / MAPLE 계열).
      2. clip 임계: c02의 grad_norm이 5.841~19.662인데 코드에 1.0이 하드코딩돼
         모든 step이 6~20배로 clip된다. 한 번도 재 본 적 없는 축이다.

    둘 다 학습 전용이라 checkpoint 구조가 바뀌지 않는다.
"""

from __future__ import annotations

import pytest
import torch

from main_code.config import RegressionConfig
from main_code.models import RegressionScorer


def config(**updates: object) -> RegressionConfig:
    base = RegressionConfig(model_id="skt/A.X-4.0-Light").with_updates(
        essay_surface="official_raw",
        pooling="mean",
        training_mode="lora_only",
        input_format="baseline_v1",
    )
    return base.with_updates(**updates) if updates else base


class _Stub:
    level_prototype_loss = RegressionScorer.level_prototype_loss

    def __init__(self, spec: RegressionConfig, hidden: int = 16) -> None:
        self.regression_config = spec
        self.level_prototypes = torch.zeros(spec.level_prototype_count, hidden)


def two_levels() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(43)
    labels = torch.tensor([[1.5] * 3, [1.6] * 3, [1.4] * 3, [4.5] * 3, [4.6] * 3, [4.4] * 3])
    features = torch.cat([torch.randn(3, 16) + 3.0, torch.randn(3, 16) - 3.0])
    return features, labels


# --- clip 임계 --------------------------------------------------------------
def test_max_grad_norm_default_matches_the_previous_hardcoded_value() -> None:
    """기본값이 달라지면 198개 기존 run의 재현이 깨진다."""

    assert config().max_grad_norm == 1.0


def test_max_grad_norm_is_configurable_and_validated() -> None:
    assert config(max_grad_norm=5.0).max_grad_norm == 5.0
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="max_grad_norm"):
            config(max_grad_norm=bad)


def test_trainer_reads_the_config_value() -> None:
    """하드코딩이 남아 있으면 설정을 바꿔도 아무 일이 없다."""

    import inspect

    from main_code import train as train_module

    source = inspect.getsource(train_module)
    assert "max_grad_norm=config.max_grad_norm" in source
    assert "max_grad_norm=1.0," not in source


# --- 수준 prototype ---------------------------------------------------------
def test_disabled_by_default() -> None:
    assert config().level_prototype_weight == 0.0
    features, labels = two_levels()
    assert _Stub(config()).level_prototype_loss(features, labels) is None


def test_loss_is_computed_when_enabled() -> None:
    features, labels = two_levels()
    stub = _Stub(config(level_prototype_weight=0.1))
    for _ in range(3):
        value = stub.level_prototype_loss(features, labels)
    assert value is not None and float(value.detach()) > 0


def test_only_observed_levels_get_prototypes() -> None:
    """관측되지 않은 수준의 prototype이 0으로 남아야 밀어낼 대상에서 빠진다."""

    features, labels = two_levels()
    stub = _Stub(config(level_prototype_weight=0.1, level_prototype_count=5))
    stub.level_prototype_loss(features, labels)
    filled = [i for i in range(5) if float(stub.level_prototypes[i].abs().sum()) > 0]
    assert filled == [0, 4]


def test_single_observed_level_returns_none() -> None:
    """밀어낼 대상이 없으면 loss가 정의되지 않는다. 0을 더하면 안 된다."""

    stub = _Stub(config(level_prototype_weight=0.1))
    labels = torch.full((4, 3), 3.0)
    assert stub.level_prototype_loss(torch.randn(4, 16), labels) is None


def test_gradient_flows_into_features_but_not_prototypes() -> None:
    """prototype은 최적화 대상이 아니라 관측의 요약이다."""

    features, labels = two_levels()
    features = features.clone().requires_grad_(True)
    stub = _Stub(config(level_prototype_weight=0.1))
    for _ in range(2):
        value = stub.level_prototype_loss(features, labels)
    value.backward()
    assert features.grad is not None and float(features.grad.abs().sum()) > 0
    assert not stub.level_prototypes.requires_grad


def test_prototypes_move_toward_the_observed_mean() -> None:
    features, labels = two_levels()
    stub = _Stub(config(level_prototype_weight=0.1, level_prototype_momentum=0.5))
    stub.level_prototype_loss(features, labels)
    first = stub.level_prototypes[0].clone()
    # 첫 관측은 평균으로 초기화된다.
    assert torch.allclose(
        first, torch.nn.functional.normalize(features[:3].float(), p=2, dim=-1).mean(0)
    )


def test_trait_axis_features_are_averaged() -> None:
    _, labels = two_levels()
    stub = _Stub(config(level_prototype_weight=0.1))
    for _ in range(2):
        value = stub.level_prototype_loss(torch.randn(6, 3, 16), labels)
    assert value is not None


def test_weight_scales_the_loss() -> None:
    features, labels = two_levels()
    small = _Stub(config(level_prototype_weight=0.1))
    large = _Stub(config(level_prototype_weight=0.5))
    for _ in range(3):
        a = small.level_prototype_loss(features, labels)
        b = large.level_prototype_loss(features, labels)
    assert float(b.detach()) == pytest.approx(5.0 * float(a.detach()), rel=1e-4)


def test_head_only_training_rejected() -> None:
    with pytest.raises(ValueError, match="prototype"):
        config(
            level_prototype_weight=0.1,
            paragraph_boundary_loss_weight=0.0,
            training_mode="head_only",
        )


def test_prototypes_are_excluded_from_the_checkpoint() -> None:
    assert "level_prototypes" in RegressionScorer.TRAINING_ONLY_STATE_PREFIXES

    class Fake:
        TRAINING_ONLY_STATE_PREFIXES = RegressionScorer.TRAINING_ONLY_STATE_PREFIXES
        scoring_state_dict = RegressionScorer.scoring_state_dict

        def state_dict(self):
            return {
                "backbone.w": torch.zeros(1),
                "heads.0.weight": torch.zeros(2),
                "level_prototypes": torch.zeros(5, 16),
                "group_dro_weights": torch.zeros(32),
            }

    assert sorted(Fake().scoring_state_dict()) == ["heads.0.weight"]
