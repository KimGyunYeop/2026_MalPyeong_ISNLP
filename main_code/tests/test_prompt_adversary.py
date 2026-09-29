"""C7 문항 적대적 학습(gradient reversal)의 계약.

왜 이 축인가
    2025 채점 데이터 4,000편의 71.1%가 우리 학습에 없는 문항(Q11 1,512 / Q12 1,331)이다.
    `prompt_dropout_probability`는 **입력**에서 지문 문자열을 뺀다. 그런데 에세이 본문
    자체가 주제 어휘로 문항을 강하게 드러낸다(문항 9개, 각 1,300여 편). 지문을 빼도
    pooled 표현은 여전히 문항별로 갈린다. 적대적 항은 **표현** 수준에서 그 정보를 지운다.

    위험: content 준거 C1·C2는 논제 대응을 보므로 완전 불변은 해롭다. 그래서
    작은 가중치로 켜고 `unseen_prompt_rmse`로 재야 한다. 아래는 그 실험이 딴 것을
    재지 않도록 계약만 못 박는다.
"""

from __future__ import annotations

import pytest
import torch

from main_code.config import RegressionConfig
from main_code.models import build_prompt_adversary, reverse_gradient


def config(**updates: object) -> RegressionConfig:
    base = RegressionConfig(model_id="skt/A.X-4.0-Light").with_updates(
        essay_surface="official_raw",
        pooling="mean",
        training_mode="lora_only",
        input_format="baseline_v1",
    )
    return base.with_updates(**updates) if updates else base


def test_disabled_by_default() -> None:
    assert config().prompt_adversary_weight == 0.0


def test_no_parameters_are_created_when_disabled() -> None:
    """켜지 않았는데 parameter가 생기면 기존 checkpoint 재현이 깨진다."""

    assert build_prompt_adversary(config(), 64) is None


def test_module_is_created_when_enabled() -> None:
    adversary = build_prompt_adversary(config(prompt_adversary_weight=0.1), 64)
    assert adversary is not None
    assert sum(p.numel() for p in adversary.parameters()) > 0
    assert adversary(torch.zeros(3, 64)).shape == (3, 32)


def test_gradient_is_reversed_and_scaled() -> None:
    tensor = torch.ones(4, 8, requires_grad=True)
    reverse_gradient(tensor, 2.0).sum().backward()
    assert torch.allclose(tensor.grad, torch.full((4, 8), -2.0))


def test_forward_is_the_identity() -> None:
    tensor = torch.randn(5, 7)
    assert torch.allclose(reverse_gradient(tensor, 0.3), tensor)


def test_negative_weight_rejected() -> None:
    with pytest.raises(ValueError, match="prompt_adversary_weight"):
        config(prompt_adversary_weight=-0.1)


def test_too_few_classes_rejected() -> None:
    with pytest.raises(ValueError, match="prompt_adversary_classes"):
        config(prompt_adversary_classes=1)


def test_head_only_training_rejected() -> None:
    """backbone을 학습하지 않으면 표현이 바뀌지 않아 적대적 항이 무의미하다."""

    # 문단경계 게이트가 먼저 같은 이유로 걸리므로 그것을 끄고 본다.
    with pytest.raises(ValueError, match="적대적"):
        config(
            prompt_adversary_weight=0.1,
            paragraph_boundary_loss_weight=0.0,
            training_mode="head_only",
        )


class _Stub:
    """`prompt_adversary_loss`만 떼어내 검사한다. 7B backbone을 띄우지 않는다."""

    from main_code.models import RegressionScorer

    prompt_adversary_loss = RegressionScorer.prompt_adversary_loss

    def __init__(self, spec: RegressionConfig, hidden: int = 16) -> None:
        self.regression_config = spec
        self.prompt_adversary = build_prompt_adversary(spec, hidden)


def test_loss_is_none_when_disabled() -> None:
    stub = _Stub(config())
    assert (
        stub.prompt_adversary_loss(torch.zeros(4, 16), torch.tensor([1, 2, 3, 4]))
        is None
    )


def test_loss_is_none_without_prompt_ids() -> None:
    stub = _Stub(config(prompt_adversary_weight=0.1))
    assert stub.prompt_adversary_loss(torch.zeros(4, 16), None) is None


def test_loss_is_computed_when_enabled() -> None:
    stub = _Stub(config(prompt_adversary_weight=0.1))
    value = stub.prompt_adversary_loss(
        torch.randn(6, 16), torch.tensor([1, 1, 2, 2, 3, 3])
    )
    assert value is not None and float(value.detach()) > 0


def test_rows_without_a_prompt_are_excluded_not_clamped() -> None:
    """조용히 clamp하면 서로 다른 문항이 한 class로 뭉쳐 판별기가 엉뚱한 걸 배운다."""

    stub = _Stub(config(prompt_adversary_weight=0.1, prompt_adversary_classes=4))
    # 전부 범위 밖(-1과 99)이면 계산할 것이 없으므로 None이어야 한다.
    assert (
        stub.prompt_adversary_loss(torch.randn(3, 16), torch.tensor([-1, 99, 100]))
        is None
    )
    # 일부만 유효하면 그 부분만 쓴다.
    assert stub.prompt_adversary_loss(
        torch.randn(3, 16), torch.tensor([-1, 1, 2])
    ) is not None


def test_trait_axis_features_are_averaged() -> None:
    """essay_attention pooling은 [B,T,H]를 준다. 모양이 달라도 죽지 않아야 한다."""

    stub = _Stub(config(prompt_adversary_weight=0.1))
    value = stub.prompt_adversary_loss(
        torch.randn(4, 3, 16), torch.tensor([1, 2, 3, 4])
    )
    assert value is not None


def test_backbone_receives_reversed_gradient() -> None:
    """부호가 뒤집히지 않으면 문항 정보를 **키우는** 방향으로 학습된다."""

    stub = _Stub(config(prompt_adversary_weight=1.0))
    features = torch.randn(8, 16, requires_grad=True)
    plain_logits = stub.prompt_adversary(features.float())
    plain = torch.nn.functional.cross_entropy(
        plain_logits, torch.tensor([1, 1, 2, 2, 3, 3, 4, 4])
    )
    plain.backward()
    direct = features.grad.clone()

    features.grad = None
    reversed_loss = stub.prompt_adversary_loss(
        features, torch.tensor([1, 1, 2, 2, 3, 3, 4, 4])
    )
    reversed_loss.backward()
    assert torch.allclose(features.grad, -direct, atol=1e-5)
