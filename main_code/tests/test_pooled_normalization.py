"""pooled 표현 정규화 축의 계약.

왜 이 축인가 (2026-08-20 밤)
    c02의 `trainer_state.json`에서 `grad_norm`이 5.841~19.662인데
    `max_grad_norm=1.0`이다. **기록된 모든 step이 6~20배로 clip된다.** 설정한 학습률이
    step마다 다른 배율로 축소되므로 cosine 스케줄이 사실상 작동하지 않는다.

    단서는 경쟁 Docker(Krjin)의 코드 주석에서 나왔다: "풀링된 은닉 상태는 L2 노름이
    300 근처다. 그대로 헤드에 넣으면 헤드의 기울기가 LoRA를 압도해 매 스텝
    max_grad_norm에 걸린다." 그쪽은 parameterless LayerNorm으로 대응했다.

    우리 `normalize_features`는 **L2 정규화**라 크기 정보를 통째로 버리는 다른 연산이고
    (198개 run 중 8개에서 시도), LayerNorm은 **한 번도 쓰이지 않았다**.
"""

from __future__ import annotations

import pytest
import torch

from main_code.config import POOLED_NORMALIZATIONS, RegressionConfig
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
    _normalize_if_requested = RegressionScorer._normalize_if_requested

    def __init__(self, spec: RegressionConfig) -> None:
        self.regression_config = spec


def test_disabled_by_default() -> None:
    assert config().pooled_normalization == "none"
    assert "layernorm" in POOLED_NORMALIZATIONS


def test_default_is_the_identity_in_float() -> None:
    """기본 경로가 바뀌면 198개 기존 run의 재현이 깨진다."""

    pooled = torch.randn(3, 16) * 250.0
    assert torch.allclose(_Stub(config())._normalize_if_requested(pooled), pooled.float())


def test_layernorm_standardizes_each_row() -> None:
    pooled = torch.randn(5, 32) * 250.0 + 40.0
    out = _Stub(config(pooled_normalization="layernorm"))._normalize_if_requested(pooled)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(5), atol=1e-4)
    assert torch.allclose(out.var(dim=-1, unbiased=False), torch.ones(5), atol=1e-3)


def test_layernorm_shrinks_the_norm_by_orders_of_magnitude() -> None:
    """이 축의 존재 이유가 크기다. 줄지 않으면 clip 문제를 못 건드린다."""

    pooled = torch.randn(4, 64) * 300.0
    plain = _Stub(config())._normalize_if_requested(pooled).norm(dim=-1).mean()
    normed = (
        _Stub(config(pooled_normalization="layernorm"))
        ._normalize_if_requested(pooled)
        .norm(dim=-1)
        .mean()
    )
    assert float(normed) < float(plain) / 50


def test_layernorm_differs_from_l2() -> None:
    """둘을 같은 것으로 취급하면 '이미 시도했다'고 잘못 결론 낸다."""

    pooled = torch.randn(4, 32) * 100.0 + 10.0
    layer = _Stub(config(pooled_normalization="layernorm"))._normalize_if_requested(pooled)
    l2 = _Stub(config(normalize_features=True))._normalize_if_requested(pooled)
    assert not torch.allclose(layer, l2, atol=1e-3)
    # L2는 단위 구면으로 보내고 LayerNorm은 sqrt(D) 근처를 유지한다.
    assert float(l2.norm(dim=-1).mean()) == pytest.approx(1.0, abs=1e-4)
    assert float(layer.norm(dim=-1).mean()) > 4.0


def test_layernorm_preserves_relative_order_within_a_row() -> None:
    """LayerNorm은 아핀 변환이라 같은 행 안의 순서를 바꾸지 않는다."""

    pooled = torch.tensor([[1.0, 5.0, 3.0, 9.0, 2.0]]) * 50.0
    out = _Stub(config(pooled_normalization="layernorm"))._normalize_if_requested(pooled)
    assert torch.equal(out.argsort(dim=-1), pooled.argsort(dim=-1))


def test_combining_with_l2_is_rejected() -> None:
    with pytest.raises(ValueError, match="pooled_normalization"):
        config(pooled_normalization="layernorm", normalize_features=True)


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="pooled_normalization"):
        config(pooled_normalization="rmsnorm")


def test_no_parameters_are_added() -> None:
    """parameterless여야 checkpoint 구조가 그대로다. affine이 생기면 배포가 깨진다."""

    import inspect

    source = inspect.getsource(RegressionScorer._normalize_if_requested)
    assert "F.layer_norm" in source
    assert "LayerNorm(" not in source
