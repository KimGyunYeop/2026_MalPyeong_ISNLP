"""근거 접미사를 별도 지분으로 읽는 pooling의 수치 계약.

이 파일이 지키는 것은 하나다. **근거가 없는 view에서 결과가 `pooling="mean"`과
bit-exact하게 같아야 한다.** dual-view 학습은 no-analysis view와 analysis view를
같은 head로 돌리므로, 이 성질이 깨지면 근거를 켜는 순간 c02 baseline 자체가
달라져 비교가 성립하지 않는다.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from main_code.config import RegressionConfig
from main_code.models import RegressionScorer, build_analysis_pool_modules


HIDDEN = 8
TRAITS = 3


class _PoolingStub(nn.Module):
    """`RegressionScorer`의 pooling 부분만 떼어 CPU에서 실행한다.

    method를 복사하지 않고 그대로 빌려온다. 7B backbone 없이 pooling 수치만
    검사하면서도 실제 학습이 쓰는 코드와 같은 코드를 실행하기 위해서다.
    """

    _analysis_pool_share = RegressionScorer._analysis_pool_share
    _analysis_mix_pooled = RegressionScorer._analysis_mix_pooled

    def __init__(self, config: RegressionConfig) -> None:
        super().__init__()
        self.regression_config = config
        attention, logit = build_analysis_pool_modules(config, HIDDEN)
        self.analysis_attention_pool = attention
        if logit is None:
            self.register_parameter("analysis_pool_logit", None)
        else:
            self.analysis_pool_logit = logit

    def pooled(
        self, hidden: torch.Tensor, mask: torch.Tensor, analysis_mask: torch.Tensor
    ) -> torch.Tensor:
        return self._analysis_mix_pooled(hidden, mask, analysis_mask)


def _config(**updates: object) -> RegressionConfig:
    return RegressionConfig().with_updates(**updates)


def _fixture(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[본문+근거, 본문만, 근거 뒤 padding] 3행을 만든다."""

    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(3, 10, HIDDEN, generator=generator)
    mask = torch.zeros(3, 10, dtype=torch.bool)
    mask[0, :10] = True
    mask[1, :6] = True
    mask[2, :8] = True
    analysis = torch.zeros(3, 10, dtype=torch.long)
    analysis[0, 6:10] = 1
    analysis[2, 6:8] = 1
    return hidden, mask, analysis


def _plain_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pooled = (hidden.float() * mask.unsqueeze(-1)).sum(dim=1)
    return pooled / mask.sum(dim=1, keepdim=True).float()


@pytest.mark.parametrize(
    "pooling", ["analysis_mix", "analysis_attention_mix"]
)
def test_no_analysis_row_equals_plain_mean(pooling: str) -> None:
    hidden, mask, analysis = _fixture()
    stub = _PoolingStub(_config(pooling=pooling, analysis_pool_weight=0.9))
    pooled = stub.pooled(hidden, mask, analysis)
    expected = _plain_mean(hidden, mask)
    if pooled.dim() == 3:
        pooled = pooled[:, 0]
    # 두 번째 행에는 근거 token이 없다. 지분 0.9를 줘도 결과가 흔들리면 안 되고,
    # 근사가 아니라 bit-exact해야 한다. c02 baseline과 같은 수를 내는 것이
    # 이 pooling을 실험에 쓸 수 있게 하는 유일한 근거다.
    assert torch.equal(pooled[1], expected[1])


def test_attention_mix_starts_exactly_at_flat_mix() -> None:
    hidden, mask, analysis = _fixture(seed=7)
    flat = _PoolingStub(_config(pooling="analysis_mix", analysis_pool_weight=0.5))
    attention = _PoolingStub(
        _config(pooling="analysis_attention_mix", analysis_pool_weight=0.5)
    )
    flat_pooled = flat.pooled(hidden, mask, analysis)
    attention_pooled = attention.pooled(hidden, mask, analysis)
    # attention 경로는 [B,T,H]로 나오지만 projection이 0이라 균등 가중이므로
    # trait마다 flat 결과와 같아야 한다. 그래야 새 parameter가 학습 시작점에서
    # 아무것도 바꾸지 않는다는 규약이 성립한다.
    assert attention_pooled.shape == (3, TRAITS, HIDDEN)
    for trait in range(TRAITS):
        assert torch.allclose(attention_pooled[:, trait], flat_pooled, atol=1e-6)


def test_fixed_weight_matches_hand_computed_mix() -> None:
    hidden, mask, analysis = _fixture(seed=3)
    weight = 0.25
    stub = _PoolingStub(
        _config(
            pooling="analysis_mix",
            analysis_pool_weight=weight,
            analysis_pool_learnable=False,
        )
    )
    pooled = stub.pooled(hidden, mask, analysis)
    row = 0
    essay = mask[row] & (analysis[row] == 0)
    rationale = mask[row] & (analysis[row] == 1)
    expected = (1 - weight) * hidden[row][essay].mean(dim=0) + weight * hidden[row][
        rationale
    ].mean(dim=0)
    assert torch.allclose(pooled[row], expected, atol=1e-6)


def test_weight_zero_reduces_to_plain_mean_everywhere() -> None:
    hidden, mask, analysis = _fixture(seed=11)
    stub = _PoolingStub(
        _config(
            pooling="analysis_mix",
            analysis_pool_weight=0.0,
            analysis_pool_learnable=False,
        )
    )
    pooled = stub.pooled(hidden, mask, analysis)
    # 지분 0은 본문 구간 평균이지 전체 평균이 아니다. 근거 token을 제외한 평균과
    # 같아야 하며, 이것이 "근거를 안 본다"의 정확한 정의다.
    essay_only = mask & (analysis == 0)
    assert torch.allclose(pooled, _plain_mean(hidden, essay_only), atol=1e-6)


def test_learnable_share_initializes_at_configured_weight() -> None:
    stub = _PoolingStub(_config(pooling="analysis_mix", analysis_pool_weight=0.3))
    assert stub.analysis_pool_logit is not None
    assert stub.analysis_pool_logit.shape == (1,)
    assert math.isclose(
        float(torch.sigmoid(stub.analysis_pool_logit.detach())[0]), 0.3, abs_tol=1e-6
    )
    assert stub.analysis_pool_logit.requires_grad


def test_per_trait_share_is_three_independent_scalars() -> None:
    hidden, mask, analysis = _fixture(seed=5)
    config = _config(
        pooling="analysis_mix", analysis_pool_weight=0.5, analysis_pool_per_trait=True
    )
    stub = _PoolingStub(config)
    assert stub.analysis_pool_logit.shape == (TRAITS,)
    with torch.no_grad():
        stub.analysis_pool_logit.copy_(torch.tensor([-40.0, 0.0, 40.0]))
    pooled = stub.pooled(hidden, mask, analysis)
    assert pooled.shape == (3, TRAITS, HIDDEN)
    essay_only = mask & (analysis == 0)
    rationale_only = mask & (analysis == 1)
    row = 0
    assert torch.allclose(
        pooled[row, 0], hidden[row][essay_only[row]].mean(dim=0), atol=1e-5
    )
    assert torch.allclose(
        pooled[row, 2], hidden[row][rationale_only[row]].mean(dim=0), atol=1e-5
    )


def test_share_receives_gradient() -> None:
    hidden, mask, analysis = _fixture(seed=13)
    stub = _PoolingStub(_config(pooling="analysis_mix", analysis_pool_weight=0.5))
    stub.pooled(hidden, mask, analysis).sum().backward()
    assert stub.analysis_pool_logit.grad is not None
    assert float(stub.analysis_pool_logit.grad.abs().sum()) > 0.0


def test_attention_projection_receives_gradient_from_zero_init() -> None:
    hidden, mask, analysis = _fixture(seed=17)
    stub = _PoolingStub(
        _config(pooling="analysis_attention_mix", analysis_pool_weight=0.5)
    )
    projection = stub.analysis_attention_pool.projection.weight
    assert float(projection.detach().abs().sum()) == 0.0
    stub.pooled(hidden, mask, analysis).sum().backward()
    # 0 초기화가 gradient까지 0으로 만들면 이 경로는 영원히 균등 가중에 갇힌다.
    assert float(projection.grad.abs().sum()) > 0.0


def test_missing_analysis_mask_is_rejected() -> None:
    hidden, mask, _ = _fixture()
    stub = _PoolingStub(_config(pooling="analysis_mix"))
    with pytest.raises(ValueError, match="analysis_mask"):
        stub.pooled(hidden, mask, None)


def test_row_without_essay_tokens_is_rejected() -> None:
    hidden, mask, analysis = _fixture()
    analysis = analysis.clone()
    analysis[1, :6] = 1
    stub = _PoolingStub(_config(pooling="analysis_mix"))
    with pytest.raises(ValueError, match="본문 token"):
        stub.pooled(hidden, mask, analysis)


def test_learnable_weight_rejects_closed_interval_endpoints() -> None:
    with pytest.raises(ValueError, match="analysis_pool_weight"):
        _config(pooling="analysis_mix", analysis_pool_weight=0.0)
    with pytest.raises(ValueError, match="analysis_pool_weight"):
        _config(pooling="analysis_mix", analysis_pool_weight=1.0)


def test_per_trait_flag_requires_a_mix_pooling() -> None:
    with pytest.raises(ValueError, match="analysis_pool_per_trait"):
        _config(pooling="mean", analysis_pool_per_trait=True)
