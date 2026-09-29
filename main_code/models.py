from __future__ import annotations

import contextlib
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .config import (
    ANALYSIS_MIX_POOLINGS,
    DETAIL_CRITERIA,
    DETAIL_CRITERIA_BY_TRAIT,
    RATER_SET_HEAD_MODES,
    SCORE_MAX,
    SCORE_MIN,
    TRAIT_SCORE_STEPS,
    TRAITS,
    RegressionConfig,
    load_config,
    lora_target_modules,
    resolve_model,
    save_config,
)
from .quantization_objectives import (
    hard_quantize,
    quantized_mse_risks,
    relaxed_quantize,
)
from .utils import hidden_size_of, resolve_text_model, write_json


TRI21_MODEL_ID = "trillionlabs/Tri-21B"
TRI21_ROPE_COMPAT = {"rope_type": "linear", "factor": 1.0}


# Backbone compatibility -----------------------------------------------------
def load_compatible_backbone_config(
    model_id: str,
    *,
    trust_remote_code: bool,
    revision: str = "main",
) -> Any | None:
    """Load the small model config only when a backbone needs a compatibility fix.

    Tri-21B's remote code expects ``ROPE_INIT_FUNCTIONS['default']``, which was
    removed in Transformers 5.  Linear RoPE with factor 1.0 computes the same
    frequencies as default RoPE and keeps the remote code on a supported entry.
    """

    if model_id != TRI21_MODEL_ID:
        return None
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
        rope_scaling=dict(TRI21_ROPE_COMPAT),
    )


# Representations and training objectives -----------------------------------
class TraitAttentionPool(nn.Module):
    """한 essay token sequence에서 trait별로 다른 weighted mean을 만든다."""

    def __init__(self, hidden_size: int, attention_dim: int):
        super().__init__()
        self.projection = nn.Linear(hidden_size, attention_dim, bias=False)
        self.queries = nn.Parameter(torch.empty(len(TRAITS), attention_dim))
        nn.init.normal_(self.queries, std=attention_dim**-0.5)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not torch.all(mask.bool().any(dim=-1)):
            raise ValueError("essay_attention pooling 대상 token이 비어 있습니다")
        projected = torch.tanh(self.projection(hidden.float()))
        # [B,S,D] x [T,D] -> [B,T,S]. content/organization/expression이
        # 같은 essay에서도 서로 다른 token에 주의를 줄 수 있다.
        logits = torch.einsum("bsd,td->bts", projected, self.queries.float())
        logits = logits.masked_fill(~mask[:, None, :].bool(), -torch.inf)
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("bts,bsh->bth", weights, hidden.float())


class _GradientReversal(torch.autograd.Function):
    """forward는 항등, backward는 gradient에 -coefficient를 곱한다.

    판별기는 문항을 **맞히도록** 학습되고(정상 gradient), backbone은 판별기의 loss를
    **키우도록** 학습된다(뒤집힌 gradient). 그 결과 pooled 표현에서 문항 정보가
    지워진다. 계수는 loss weight와 따로 두지 않는다 — 하나로 충분하고 두 개면
    어느 쪽이 무엇을 하는지 흐려진다.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = float(coefficient)
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.coefficient * grad_output, None


def reverse_gradient(tensor: torch.Tensor, coefficient: float) -> torch.Tensor:
    return _GradientReversal.apply(tensor, coefficient)


def build_prompt_adversary(
    config: RegressionConfig, hidden_size: int
) -> nn.Module | None:
    """문항 판별기. weight가 0이면 만들지 않는다(parameter 수 불변)."""

    if config.prompt_adversary_weight <= 0:
        return None
    return nn.Sequential(
        nn.Linear(hidden_size, config.head_hidden_size),
        nn.Tanh(),
        nn.Linear(config.head_hidden_size, config.prompt_adversary_classes),
    )


# GroupDRO 가중치 벡터의 크기. prompt_group_id(Qn -> n)를 그대로 색인으로 쓰므로
# 등장할 수 있는 문항 번호의 상한이면 된다. 2025 자료의 Q12까지 여유 있게 덮는다.
GROUP_DRO_SLOTS = 32


def group_dro_sample_weights(
    per_sample_loss: torch.Tensor,
    prompt_group_ids: torch.Tensor | None,
    weights: torch.Tensor,
    step_size: float,
) -> torch.Tensor | None:
    """그룹 가중치를 갱신하고 **표본별** 가중치를 만든다.

    Sagawa et al.의 online GroupDRO다. 그룹 g의 가중치를 관측 손실에 비례해
    `w_g <- w_g * exp(step_size * loss_g)`로 올린 뒤 합이 1이 되게 정규화한다.
    반환값은 `[B, 1]`이라 `element_weights`로 그대로 쓸 수 있고, batch 평균이 1이
    되도록 재조정해 loss 규모를 유지한다(lr을 다시 잡지 않아도 되게 한다).

    `weights`는 in-place로 갱신되는 buffer다. gradient는 흐르지 않는다 —
    가중치는 최적화 대상이 아니라 관측에서 나오는 값이다.
    """

    if step_size <= 0 or prompt_group_ids is None:
        return None
    labels = prompt_group_ids.reshape(-1).long()
    valid = (labels >= 0) & (labels < weights.numel())
    if not bool(valid.any()):
        return None
    with torch.no_grad():
        losses = per_sample_loss.detach().reshape(len(labels), -1).mean(dim=-1)
        present = torch.unique(labels[valid])
        for group in present.tolist():
            mask = valid & (labels == group)
            weights[group] = weights[group] * torch.exp(
                step_size * losses[mask].mean().to(weights.dtype)
            )
        weights.clamp_(min=1e-8)
        weights.div_(weights.sum())
        # 존재하는 그룹만 골라 batch 안에서 평균 1이 되도록 재조정한다.
        picked = torch.where(
            valid, weights[labels.clamp(min=0, max=weights.numel() - 1)],
            torch.zeros_like(weights[0]),
        )
        total = picked.sum()
        if float(total) <= 0:
            return None
        scaled = picked * (float(valid.sum()) / total)
    return scaled.unsqueeze(-1).to(per_sample_loss.dtype)


def build_analysis_pool_modules(
    config: RegressionConfig, hidden_size: int
) -> tuple[TraitAttentionPool | None, nn.Parameter | None]:
    """ANALYSIS_MIX_POOLINGS 전용 parameter를 만든다.

    `RegressionScorer.__init__`와 test가 **같은 구성 코드**를 쓰게 하려고 분리했다.
    초기화 규약이 갈라지면 "첫 forward는 baseline과 같다"는 성질이 조용히 깨진다.
    """

    attention: TraitAttentionPool | None = None
    if config.pooling == "analysis_attention_mix":
        # projection을 0으로 두면 logit이 전부 0이라 softmax가 균등해지고 첫 forward는
        # analysis_mix(=접미사 단순 평균)와 bit-exact하게 같다. query는 0이 아니므로
        # projection에 gradient가 흐르고, 필요할 때만 기준별 구획을 고른다.
        attention = TraitAttentionPool(hidden_size, config.attention_pool_dim)
        nn.init.zeros_(attention.projection.weight)

    logit: nn.Parameter | None = None
    if config.pooling in ANALYSIS_MIX_POOLINGS and config.analysis_pool_learnable:
        # sigmoid로 (0,1)에 가두므로 지분이 구간을 벗어날 수 없다. 초기값은
        # analysis_pool_weight를 그대로 재현한다(0.5면 logit 0).
        share = float(config.analysis_pool_weight)
        width = len(TRAITS) if config.analysis_pool_per_trait else 1
        logit = nn.Parameter(
            torch.full((width,), math.log(share / (1.0 - share)), dtype=torch.float32)
        )
    return attention, logit


def _zero_loss(scores: torch.Tensor) -> torch.Tensor:
    """빈 pair처럼 학습할 항이 없을 때 graph를 유지하는 0을 반환한다."""

    return scores.sum() * 0.0


def _trait_weights(
    config: RegressionConfig,
    *,
    device: torch.device,
    ranking: bool,
) -> torch.Tensor:
    """Return content/organization/expression weights in the fixed trait order."""

    if ranking:
        values = (
            config.ranking_content_weight,
            config.ranking_organization_weight,
            config.ranking_expression_weight,
        )
    else:
        values = (
            config.content_loss_weight,
            config.organization_loss_weight,
            config.expression_loss_weight,
        )
    return torch.tensor(values, device=device, dtype=torch.float32)


def _weighted_trait_mean(
    values: torch.Tensor,
    config: RegressionConfig,
    *,
    ranking: bool,
    valid: torch.Tensor | None = None,
    element_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average elementwise losses while keeping the default 1/1/1 scale."""

    if values.shape[-1] != len(TRAITS):
        raise ValueError(
            "trait loss의 마지막 차원은 content/organization/expression이어야 합니다"
        )
    shape = (1,) * (values.ndim - 1) + (len(TRAITS),)
    weights = (
        _trait_weights(config, device=values.device, ranking=ranking)
        .view(shape)
        .expand_as(values)
    )
    if valid is not None:
        weights = weights * valid.to(weights.dtype)
    if element_weights is not None:
        weights = weights * element_weights.to(weights.dtype)
    denominator = weights.sum()
    if float(denominator.detach()) <= 0:
        return _zero_loss(values)
    return (values.float() * weights).sum() / denominator


def sample_tail_weights(
    labels: torch.Tensor, config: RegressionConfig
) -> torch.Tensor | None:
    """라벨 분포를 더 넓은 목표 분포로 옮기는 **표본별** 중요도 가중치.

    왜 필요한가는 `RegressionConfig.tail_weight_target_sigma` 주석에 있다. 요약하면
    리더보드 집합의 정답 SD가 우리 검증 400편(0.653)보다 넓어 보이는데, 우리 학습
    라벨은 SD 0.617에 몰려 있어 꼬리를 배울 기회가 없었다.

    반환값은 ``[B, 1]``이라 ``[B, T]`` 손실에 그대로 broadcast된다. 목표 σ가 0이면
    ``None``을 돌려주고, 호출부는 ``element_weights=None``으로 기존 경로를 탄다 —
    기존 run과 bit-exact 동일해야 하므로 곱셈 1.0조차 끼워 넣지 않는다.

    정규화 상수 σ_ref/σ_target은 y ~ N(μ, σ_ref²)에서 E[w]=σ_target/σ_ref이라는
    닫힌 해에서 나온다. batch 평균으로 나누면 꼬리만 모인 batch에서 가중치가
    통째로 1에 가까워져 효과가 사라지므로 그 방식은 쓰지 않는다.
    """

    sigma_target = float(config.tail_weight_target_sigma)
    if sigma_target <= 0:
        return None
    sigma_reference = float(config.tail_weight_reference_sigma)
    mean = float(config.tail_weight_reference_mean)
    average = labels.float().mean(dim=-1)
    deviation = torch.square(average - mean)
    exponent = deviation * (
        1.0 / (2.0 * sigma_reference**2) - 1.0 / (2.0 * sigma_target**2)
    )
    weights = torch.exp(exponent) * (sigma_reference / sigma_target)
    weights = weights.clamp(max=float(config.tail_weight_max))
    return weights.unsqueeze(-1)


def _ranking_targets(
    scores: torch.Tensor, labels: torch.Tensor, config: RegressionConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """순위 로스가 볼 값을 고른다.

    공식 지표는 essay별 세 trait 평균 점수 **하나**의 순위다. trait_average면 그 양을
    직접 학습하고, per_trait면 trait마다 따로 순위를 맞춘 뒤 평균한다(기존 동작).
    """

    if config.ranking_target == "trait_average":
        return scores.mean(dim=-1, keepdim=True), labels.mean(dim=-1, keepdim=True)
    return scores, labels


def _ranking_mean(
    values: torch.Tensor,
    config: RegressionConfig,
    *,
    valid: torch.Tensor | None = None,
    element_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """순위 로스의 평균. trait_average는 열이 하나뿐이라 trait 가중이 없다."""

    if config.ranking_target == "per_trait":
        return _weighted_trait_mean(
            values, config, ranking=True, valid=valid, element_weights=element_weights
        )
    weights = torch.ones_like(values, dtype=torch.float32)
    if valid is not None:
        weights = weights * valid.to(weights.dtype)
    if element_weights is not None:
        weights = weights * element_weights.to(weights.dtype)
    denominator = weights.sum()
    if float(denominator.detach()) <= 0:
        return _zero_loss(values)
    return (values.float() * weights).sum() / denominator


def _all_pair_differences(
    scores: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """[B,3]에서 중복되지 않는 i<j의 차이 [P,3]만 만든다."""

    batch_size = scores.shape[0]
    indices = torch.triu_indices(batch_size, batch_size, offset=1, device=scores.device)
    left, right = indices[0], indices[1]
    return scores[left] - scores[right], labels[left] - labels[right]


def pairwise_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    config: RegressionConfig,
) -> torch.Tensor:
    """RankNet 또는 score-gap hinge를 batch의 모든 pair에 적용한다."""

    if config.pairwise_loss == "none" or scores.shape[0] < 2:
        return _zero_loss(scores)
    scores, labels = _ranking_targets(scores, labels, config)
    score_diff, label_diff = _all_pair_differences(scores.float(), labels.float())

    if config.pairwise_loss == "soft_ranknet":
        # 예측과 정답에 같은 temperature를 쓰면 scores == labels일 때 두
        # preference probability도 정확히 일치한다. tie target은 0.5다.
        target = torch.sigmoid(label_diff / config.pairwise_temperature)
        elementwise = F.binary_cross_entropy_with_logits(
            score_diff / config.pairwise_temperature,
            target,
            reduction="none",
        )
        valid = torch.ones_like(elementwise, dtype=torch.bool)
    elif config.pairwise_loss == "hard_ranknet":
        # 외부 자료마다 점수 척도와 간격이 달라도 순서 부호만 전달한다.
        # 동점에는 선호 방향이 없으므로 loss와 분모에서 모두 제외한다.
        valid = label_diff != 0
        target = (label_diff > 0).to(score_diff.dtype)
        elementwise = F.binary_cross_entropy_with_logits(
            score_diff / config.pairwise_temperature,
            target,
            reduction="none",
        )
    elif config.pairwise_loss == "hinge":
        # 정답 차이만큼 예측 간격을 확보한다. tie에는 방향이 없으므로 제외한다.
        valid = label_diff != 0
        elementwise = F.relu(label_diff.abs() - label_diff.sign() * score_diff)
    else:  # config.validate()가 막지만 helper를 단독 호출할 때도 명확히 실패한다.
        raise ValueError(f"unsupported pairwise_loss={config.pairwise_loss!r}")

    if not torch.any(valid):
        return _zero_loss(scores)
    gap_weights = None
    if config.pairwise_gap_weighted and config.pairwise_loss != "hard_ranknet":
        # 큰 정답 차이를 조금 더 강조하되 tie도 완전히 버리지 않는 단순 weight다.
        gap_weights = 1.0 + label_diff.abs()
    return _ranking_mean(
        elementwise, config, valid=valid, element_weights=gap_weights
    )


# soft_spearman 분모의 미분 하한. `valid` 문턱 `sqrt(energy) > 1e-8`과 같은 지점이라
# 유효한 원소에서는 clamp가 항등이고 forward는 이전 구현과 bit 단위로 같다.
SOFT_SPEARMAN_ENERGY_FLOOR = 1e-16


def _soft_ranks(values: torch.Tensor, temperature: float) -> torch.Tensor:
    """높은 점수가 rank 1이 되는 differentiable rank를 계산한다."""

    # differences[i,j,t] = value[j,t] - value[i,t]. self 비교의 sigmoid(0)=0.5를
    # 포함하므로 0.5를 더하면 1 + sum_{j!=i} 형태가 된다.
    differences = values.unsqueeze(0) - values.unsqueeze(1)
    return 0.5 + torch.sigmoid(differences / temperature).sum(dim=1)


def trait_average_objective(
    scores: torch.Tensor,
    labels: torch.Tensor,
    config: RegressionConfig,
) -> torch.Tensor:
    """essay별 세 trait 평균 점수 하나를 직접 맞추는 auxiliary objective다.

    per-trait MSE는 각 trait 오차의 제곱만 줄이므로 trait 사이 오차 공분산을 벌하지
    않는다. 리더보드 지표가 trait 평균 점수 1개에서 계산된 것으로 보이는 2026-08-05
    진단에서는 그 공분산이 그대로 점수에 들어가므로 별도 항으로 둔다. 두 weight가
    모두 0이면 호출되지 않는다.
    """

    predicted = scores.mean(dim=-1)
    target = labels.mean(dim=-1)
    total = scores.new_zeros(())
    if config.trait_average_loss_weight > 0:
        total = total + config.trait_average_loss_weight * torch.square(
            predicted - target
        ).mean()
    if config.trait_average_pairwise_weight > 0:
        # gap[i,j] = target[j] - target[i]. 같은 batch 안에서 평균 점수의 대소만 쓴다.
        gap = target.unsqueeze(0) - target.unsqueeze(1)
        margin = predicted.unsqueeze(0) - predicted.unsqueeze(1)
        comparable = gap.abs() > 1e-6
        if bool(comparable.any()):
            temperature = max(float(config.pairwise_temperature), 1e-6)
            logits = margin[comparable] / temperature
            positive = (gap[comparable] > 0).to(logits.dtype)
            pairwise = nn.functional.binary_cross_entropy_with_logits(
                logits, positive
            )
            total = total + config.trait_average_pairwise_weight * pairwise
    return total


def listwise_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    config: RegressionConfig,
) -> torch.Tensor:
    """batch 전체 순서를 맞추는 세 가지 작은 listwise objective다."""

    if config.listwise_loss == "none" or scores.shape[0] < 2:
        return _zero_loss(scores)
    scores, labels = _ranking_targets(scores, labels, config)
    scores = scores.float()
    labels = labels.float()
    temperature = config.listwise_temperature

    if config.listwise_loss == "listnet":
        target = torch.softmax(labels / temperature, dim=0)
        log_prediction = torch.log_softmax(scores / temperature, dim=0)
        per_trait = (target * (target.clamp_min(1e-12).log() - log_prediction)).sum(
            dim=0
        )
        return _ranking_mean(per_trait, config)

    predicted_ranks = _soft_ranks(scores, temperature)
    target_ranks = _soft_ranks(labels, temperature)
    if config.listwise_loss == "soft_rank_mse":
        scale = max(1, scores.shape[0] - 1)
        elementwise = torch.square((predicted_ranks - target_ranks) / scale)
        return _ranking_mean(elementwise, config)
    if config.listwise_loss == "soft_spearman":
        predicted_centered = predicted_ranks - predicted_ranks.mean(dim=0)
        target_centered = target_ranks - target_ranks.mean(dim=0)
        numerator = (predicted_centered * target_centered).sum(dim=0)
        energy = (
            predicted_centered.square().sum(dim=0)
            * target_centered.square().sum(dim=0)
        )
        denominator = torch.sqrt(energy)
        valid = denominator > 1e-8
        if not torch.any(valid):
            return _zero_loss(scores)
        # `sqrt(0)`의 backward는 `grad/(2*sqrt(0))` = 0/0이라 NaN이다. forward에서
        # `valid`로 걸러내도 autograd는 **모든 원소**의 sqrt backward를 계산하므로,
        # batch 안에서 한 trait의 라벨이 전부 동점이면(energy가 정확히 0) 그 한 열이
        # batch 전체 gradient를 NaN으로 만든다. batch가 작을수록 동점 확률이 높아
        # batch_size 4 run은 전부 첫 eval 전에 죽고 32는 살아남았다.
        #
        # `valid` 판정은 원래 값 그대로 두고 미분되는 경로에만 하한을 건다. 하한은
        # `valid` 문턱과 같은 지점(sqrt 1e-8)이므로 valid한 원소에서는 clamp가
        # 항등이고 forward는 bit 단위로 이전과 같다.
        safe_denominator = torch.sqrt(energy.clamp_min(SOFT_SPEARMAN_ENERGY_FLOOR))
        correlation = torch.zeros_like(numerator)
        correlation[valid] = (
            numerator[valid] / safe_denominator[valid]
        ).clamp(-1.0, 1.0)
        return _ranking_mean(1.0 - correlation, config, valid=valid)
    raise ValueError(f"unsupported listwise_loss={config.listwise_loss!r}")


def quantization_objective_active(config: RegressionConfig) -> bool:
    """Return whether the opt-in submitted-surface objective changes training."""

    return config.quantization_rule != "none" and any(
        weight > 0
        for weight in (
            config.quantized_trait_loss_weight,
            config.quantized_mean_loss_weight,
            config.quantized_pooled_loss_weight,
            config.quantized_trait_rank_weight,
            config.quantized_mean_rank_weight,
            config.quantized_pooled_rank_weight,
        )
    )


def _quantization_schedule(
    config: RegressionConfig, step: int
) -> tuple[float, float, float]:
    """Return loss scale, total temperature, and allocation temperature."""

    step = max(0, int(step))
    start = int(config.quantized_loss_start_step)
    if step < start:
        return 0.0, float(config.quantization_temperature), float(
            config.quantization_allocation_temperature
        )
    if config.quantized_loss_ramp_steps > 0:
        scale = min(1.0, max(0.0, (step - start) / config.quantized_loss_ramp_steps))
    else:
        scale = 1.0
    if config.quantization_anneal_steps > 0:
        progress = min(
            1.0, max(0.0, (step - start) / config.quantization_anneal_steps)
        )
    else:
        progress = 0.0
    temperature = (
        config.quantization_temperature
        + (config.quantization_final_temperature - config.quantization_temperature)
        * progress
    )
    allocation_temperature = (
        config.quantization_allocation_temperature
        + (
            config.quantization_final_allocation_temperature
            - config.quantization_allocation_temperature
        )
        * progress
    )
    return scale, float(temperature), float(allocation_temperature)


def _quantized_training_targets(
    labels: torch.Tensor,
    average_labels: torch.Tensor | None,
    config: RegressionConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build raw or explicitly rounded ablation targets without hidden defaults."""

    target_rule = config.quantized_target_rule
    if target_rule == "raw":
        trait_targets = labels.float()
        average_targets = (
            labels.mean(dim=-1).float()
            if average_labels is None
            else average_labels.reshape(-1).to(device=labels.device).float()
        )
        return trait_targets, average_targets
    if target_rule == "same_as_prediction":
        target_rule = config.quantization_rule
    if target_rule not in {"independent_half_up", "average_matched"}:
        raise ValueError(f"unsupported quantized_target_rule={target_rule!r}")
    trait_targets = hard_quantize(labels.float(), target_rule)
    return trait_targets, trait_targets.mean(dim=-1)


def _soft_spearman_columns(
    prediction: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Differentiable 1-rho for each column, excluding constant targets."""

    if prediction.ndim == 1:
        prediction = prediction.unsqueeze(-1)
    if target.ndim == 1:
        target = target.unsqueeze(-1)
    if prediction.shape != target.shape:
        raise ValueError("quantized rank prediction/target shape mismatch")
    if prediction.shape[0] < 2:
        return _zero_loss(prediction)
    prediction = prediction.float()
    target = target.float()
    predicted_ranks = _soft_ranks(prediction, temperature)
    target_ranks = _soft_ranks(target, temperature)
    predicted_centered = predicted_ranks - predicted_ranks.mean(dim=0)
    target_centered = target_ranks - target_ranks.mean(dim=0)
    numerator = (predicted_centered * target_centered).sum(dim=0)
    denominator = torch.sqrt(
        predicted_centered.square().sum(dim=0)
        * target_centered.square().sum(dim=0)
    )
    valid = denominator > 1e-8
    if not torch.any(valid):
        return _zero_loss(prediction)
    correlation = (numerator[valid] / denominator[valid]).clamp(-1.0, 1.0)
    return (1.0 - correlation).mean()


def quantization_aware_objective(
    scores: torch.Tensor,
    labels: torch.Tensor,
    average_labels: torch.Tensor | None,
    config: RegressionConfig,
    *,
    global_step: int = 0,
) -> torch.Tensor:
    """Optimize every rounded-surface metric axis without stochastic sampling.

    The six weights independently select trait-macro, mean-first, and pooled
    error/rank objectives.  Gold remains continuous by default; rounded labels
    are an explicit ablation through ``quantized_target_rule``.
    """

    if not quantization_objective_active(config):
        return _zero_loss(scores)
    scale, temperature, allocation_temperature = _quantization_schedule(
        config, global_step
    )
    if scale <= 0.0:
        return _zero_loss(scores)
    trait_targets, average_targets = _quantized_training_targets(
        labels, average_labels, config
    )
    rule = config.quantization_rule
    straight_through = config.quantization_surrogate == "straight_through"

    if config.quantization_surrogate == "expected_risk":
        risks = quantized_mse_risks(
            scores,
            trait_targets,
            rule,
            average_targets=average_targets,
            temperature=temperature,
            allocation_temperature=allocation_temperature,
            straight_through=False,
        )
        trait_squared = risks.per_trait
        mean_squared = risks.trait_average
    else:
        relaxed = relaxed_quantize(
            scores,
            rule,
            temperature=temperature,
            allocation_temperature=allocation_temperature,
            straight_through=straight_through,
        )
        trait_squared = torch.square(relaxed.scores - trait_targets)
        mean_squared = torch.square(
            relaxed.scores.mean(dim=-1) - average_targets
        )

    def error_form(value: torch.Tensor) -> torch.Tensor:
        if config.quantized_error_form == "mse":
            return value
        return torch.sqrt(value.clamp_min(1e-12))

    total = _zero_loss(scores)
    if config.quantized_trait_loss_weight > 0:
        trait_macro = error_form(trait_squared.mean(dim=0)).mean()
        total = total + config.quantized_trait_loss_weight * trait_macro
    if config.quantized_mean_loss_weight > 0:
        mean_first = error_form(mean_squared.mean())
        total = total + config.quantized_mean_loss_weight * mean_first
    if config.quantized_pooled_loss_weight > 0:
        pooled = error_form(trait_squared.mean())
        total = total + config.quantized_pooled_loss_weight * pooled

    if any(
        weight > 0
        for weight in (
            config.quantized_trait_rank_weight,
            config.quantized_mean_rank_weight,
            config.quantized_pooled_rank_weight,
        )
    ):
        # Exact expected Spearman over 125^B states is intractable.  The
        # deterministic expected score is the low-variance rank surrogate;
        # STE uses the exact hard forward score if explicitly selected.
        rank_scores = relaxed_quantize(
            scores,
            rule,
            temperature=temperature,
            allocation_temperature=allocation_temperature,
            straight_through=straight_through,
        ).scores
        rank_temperature = float(config.quantized_rank_temperature)
        if config.quantized_trait_rank_weight > 0:
            total = total + config.quantized_trait_rank_weight * _soft_spearman_columns(
                rank_scores, trait_targets, rank_temperature
            )
        if config.quantized_mean_rank_weight > 0:
            total = total + config.quantized_mean_rank_weight * _soft_spearman_columns(
                rank_scores.mean(dim=-1), average_targets, rank_temperature
            )
        if config.quantized_pooled_rank_weight > 0:
            total = total + config.quantized_pooled_rank_weight * _soft_spearman_columns(
                rank_scores.reshape(-1), trait_targets.reshape(-1), rank_temperature
            )
    return total * float(scale)


# 공식 train 11,600편에서 실측한 |trait - 세 trait 평균|의 최대는 2.13이다. tanh
# 상한을 그보다 약간 크게 두면 실제 라벨은 모두 표현할 수 있고, 대비가 발산해
# trait 점수가 1~5를 크게 벗어나는 것도 막는다.
AVERAGE_CONTRAST_RANGE = 2.5


def trait_score_grid(trait: str) -> list[float]:
    """trait 라벨이 실제로 놓이는 격자 값을 전부 돌려준다.

    train 11,600편 전수 확인에서 content는 0.1 간격 41개, organization과
    expression은 0.25 간격 17개이며 격자 위탈이 0편이다. 라벨 생성 과정이
    (정수 준거 점수의 평균)이므로 이 격자는 데이터가 아니라 채점 규칙에서 온다.
    """

    step = TRAIT_SCORE_STEPS[trait]
    count = int(round((SCORE_MAX - SCORE_MIN) / step)) + 1
    return [SCORE_MIN + index * step for index in range(count)]


# content 41 / organization 17 / expression 17. 세 head를 하나의 [B,3,K] tensor로
# 다루려고 가장 큰 격자에 맞춰 padding하고, 남는 자리는 mask로 죽인다.
TRAIT_CLASS_COUNTS = {trait: len(trait_score_grid(trait)) for trait in TRAITS}
MAX_TRAIT_CLASSES = max(TRAIT_CLASS_COUNTS.values())


def trait_grid_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    """[3, MAX] 격자 값과 유효 class mask를 만든다."""

    values = torch.zeros(len(TRAITS), MAX_TRAIT_CLASSES, dtype=torch.float32)
    valid = torch.zeros(len(TRAITS), MAX_TRAIT_CLASSES, dtype=torch.bool)
    for trait_index, trait in enumerate(TRAITS):
        grid = trait_score_grid(trait)
        values[trait_index, : len(grid)] = torch.tensor(grid, dtype=torch.float32)
        valid[trait_index, : len(grid)] = True
    return values, valid


def gumbel_straight_through(
    logits: torch.Tensor, temperature: float, *, sample_noise: bool
) -> torch.Tensor:
    """순전파는 one-hot, 역전파는 softmax gradient인 격자 선택을 만든다.

    argmax는 점수 경로의 gradient를 끊어 downstream MSE로 head를 학습할 수 없다.
    straight-through Gumbel은 예측을 격자 위의 한 점으로 유지하면서 gradient는
    통과시킨다. 평가에서는 잡음을 넣지 않아 결정적으로 같은 class를 고른다.
    """

    if sample_noise:
        uniform = torch.rand_like(logits).clamp_min(torch.finfo(logits.dtype).tiny)
        logits = logits - torch.log(-torch.log(uniform))
    soft = torch.softmax(logits / temperature, dim=-1)
    index = logits.argmax(dim=-1, keepdim=True)
    hard = torch.zeros_like(soft).scatter_(-1, index, 1.0)
    return hard + soft - soft.detach()


def average_plus_contrast_scores(raw: torch.Tensor) -> torch.Tensor:
    """세 head 출력을 (평균, 대비1, 대비2)로 읽어 trait 점수를 만든다.

    공식 지표는 essay당 ``(content+organization+expression)/3`` 하나이고, 그 평균
    라벨은 평가자 2인 x 9준거 정수 18개의 평균이라 trait 라벨보다 노이즈가 작다.
    기존 traits 파라미터화에서 평균은 세 head의 파생값이지만, 여기서는 head 하나가
    평균을 직접 낸다. 세 trait 편차의 합이 0이므로 세 trait 평균은 항상 평균 head와
    정확히 같다. 새 parameter를 만들지 않으므로 checkpoint key도 그대로다.
    """

    if raw.ndim != 3 or raw.shape[1] != len(TRAITS) or raw.shape[2] != 1:
        raise ValueError("average_plus_contrast raw는 [B,3,1]이어야 합니다")
    values = raw.squeeze(-1).float()
    average = 1.0 + 4.0 * torch.sigmoid(values[:, 0])
    first = AVERAGE_CONTRAST_RANGE * torch.tanh(values[:, 1])
    second = AVERAGE_CONTRAST_RANGE * torch.tanh(values[:, 2])
    return torch.stack(
        (average + first, average + second, average - first - second), dim=1
    )


def trait_native_distribution_targets(labels: torch.Tensor) -> torch.Tensor:
    """trait 라벨을 자기 네이티브 격자 위의 one-hot으로 바꾼다.

    content는 0.1 간격 41개, organization/expression은 0.25 간격 17개 위에 라벨이
    정확히 놓여 있으므로 5-class처럼 인접 두 bin으로 나눌 이유가 없다. 격자 밖 값은
    가장 가까운 class로 반올림한다(외부 corpus 라벨을 함께 쓸 때만 발생한다).
    """

    if labels.ndim != 2 or labels.shape[1] != len(TRAITS):
        raise ValueError("trait native distribution target label은 [B,3]이어야 합니다")
    targets = torch.zeros(
        labels.shape[0],
        len(TRAITS),
        MAX_TRAIT_CLASSES,
        dtype=torch.float32,
        device=labels.device,
    )
    for trait_index, trait in enumerate(TRAITS):
        step = TRAIT_SCORE_STEPS[trait]
        position = (labels[:, trait_index].float() - SCORE_MIN) / step
        index = position.round().long().clamp(0, TRAIT_CLASS_COUNTS[trait] - 1)
        targets[:, trait_index].scatter_(-1, index.unsqueeze(-1), 1.0)
    return targets


def score_distribution_targets(
    labels: torch.Tensor, *, smoothing: float = 0.0
) -> torch.Tensor:
    """Map float labels to adjacent 1~5 bins, optionally with uniform smoothing.

    Example: 3.7 becomes [0, 0, 0.3, 0.7, 0]. This matches an average human
    score more faithfully than rounding it to a single class.
    """

    values = torch.arange(1, 6, device=labels.device, dtype=torch.float32)
    targets = (1.0 - (labels.float().unsqueeze(-1) - values).abs()).clamp_min(0.0)
    targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    if smoothing:
        targets = (1.0 - smoothing) * targets + smoothing / values.numel()
    return targets


def ordinal_cumulative_targets(
    labels: torch.Tensor, *, steps: int
) -> torch.Tensor:
    """Encode continuous 1~5 labels as soft cumulative ordinal targets.

    With 16 steps the thresholds are 1.00, 1.25, ..., 4.75 and the matching
    expectation is ``1 + 0.25 * sum(probability)``.  A label between grid
    points contributes a fractional target at one boundary instead of being
    rounded before training.  This keeps the historical unrounded gold
    contract while reproducing the competitor's deployed ordinal readout.
    """

    if labels.ndim != 2 or labels.shape[1] != len(TRAITS):
        raise ValueError("ordinal target label은 [B,3]이어야 합니다")
    if steps <= 0:
        raise ValueError("ordinal steps는 양수여야 합니다")
    width = (SCORE_MAX - SCORE_MIN) / steps
    thresholds = SCORE_MIN + width * torch.arange(
        steps, device=labels.device, dtype=torch.float32
    )
    return (
        (labels.float().unsqueeze(-1) - thresholds) / width
    ).clamp(0.0, 1.0)


def _masked_detail_trait_mean(
    elementwise: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Average valid detail entries per essay, then balance the three traits.

    ``elementwise`` may be ``[B,9]`` for criterion losses or ``[B,R,9]``
    for individual-rater losses.  Averaging every essay before the batch mean
    prevents an essay with many raters from receiving a larger weight.
    """

    if elementwise.shape != mask.shape:
        raise ValueError("detail loss 값과 mask shape가 같아야 합니다")
    if elementwise.ndim not in {2, 3} or elementwise.shape[-1] != len(DETAIL_CRITERIA):
        raise ValueError("detail loss는 [B,9] 또는 [B,R,9]여야 합니다")

    mask = mask.bool()
    trait_losses: list[torch.Tensor] = []
    start = 0
    for trait_criteria in DETAIL_CRITERIA_BY_TRAIT:
        stop = start + len(trait_criteria)
        trait_values = elementwise[..., start:stop]
        trait_mask = mask[..., start:stop]
        reduce_dims = tuple(range(1, trait_values.ndim))
        valid_count = trait_mask.sum(dim=reduce_dims)
        per_essay = (trait_values * trait_mask).sum(
            dim=reduce_dims
        ) / valid_count.clamp_min(1)
        valid_essay = valid_count > 0
        if torch.any(valid_essay):
            trait_losses.append(per_essay[valid_essay].mean())
        start = stop

    if not trait_losses:
        return _zero_loss(elementwise)
    return torch.stack(trait_losses).mean()


def _per_essay_detail_trait_mean(
    elementwise: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an equal-trait cost for each essay and whether it has a label.

    This is the assignment cost used by the anonymous two-rater set head.  It
    deliberately keeps the batch dimension so the lower-cost *whole rater
    vector* permutation can be selected independently for every essay.
    """

    if elementwise.shape != mask.shape or elementwise.ndim != 2:
        raise ValueError("rater set cost와 mask는 같은 [B,9] shape여야 합니다")
    if elementwise.shape[1] != len(DETAIL_CRITERIA):
        raise ValueError("rater set cost의 두 번째 차원은 9여야 합니다")

    mask = mask.bool()
    per_trait: list[torch.Tensor] = []
    trait_validity: list[torch.Tensor] = []
    start = 0
    for trait_criteria in DETAIL_CRITERIA_BY_TRAIT:
        stop = start + len(trait_criteria)
        trait_mask = mask[:, start:stop]
        valid_count = trait_mask.sum(dim=-1)
        cost = (elementwise[:, start:stop] * trait_mask).sum(dim=-1)
        per_trait.append(cost / valid_count.clamp_min(1))
        trait_validity.append(valid_count > 0)
        start = stop

    values = torch.stack(per_trait, dim=-1)
    valid = torch.stack(trait_validity, dim=-1)
    valid_trait_count = valid.sum(dim=-1)
    per_essay = (values * valid).sum(dim=-1) / valid_trait_count.clamp_min(1)
    return per_essay, valid_trait_count > 0


def _minimum_rater_assignment_cost(
    assignment_cost: dict[tuple[int, int], torch.Tensor],
    target_valid: dict[int, torch.Tensor],
    *,
    zero_reference: torch.Tensor,
) -> torch.Tensor | None:
    """익명 평가자 2 branch를 순서 없이 매칭해 essay별 최소 비용을 평균한다.

    ``assignment_cost[(예측 slot, 정답 slot)]``는 essay별 [B] 비용이다. 두 정답이
    모두 있으면 직접/교차 배정 중 작은 쪽을, 하나뿐이면 그 정답에 더 잘 맞는
    예측 branch를 고른다. 정답이 하나도 없으면 None을 돌려준다.
    """

    first_valid = target_valid[0]
    second_valid = target_valid[1]
    both = first_valid & second_valid
    first_only = first_valid & ~second_valid
    second_only = second_valid & ~first_valid
    any_target = first_valid | second_valid
    if not torch.any(any_target):
        return None

    selected = zero_reference.reshape(zero_reference.shape[0], -1).sum(dim=-1) * 0.0
    direct = 0.5 * (assignment_cost[(0, 0)] + assignment_cost[(1, 1)])
    swapped = 0.5 * (assignment_cost[(0, 1)] + assignment_cost[(1, 0)])
    selected = torch.where(both, torch.minimum(direct, swapped), selected)
    selected = torch.where(
        first_only,
        torch.minimum(assignment_cost[(0, 0)], assignment_cost[(1, 0)]),
        selected,
    )
    selected = torch.where(
        second_only,
        torch.minimum(assignment_cost[(0, 1)], assignment_cost[(1, 1)]),
        selected,
    )
    return selected[any_target].mean()


def _masked_equal_trait_mean(
    elementwise: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Average ``[B,3]`` values with equal content/org/expression weight."""

    if elementwise.shape != mask.shape or elementwise.ndim != 2:
        raise ValueError("trait loss 값과 mask는 같은 [B,3] shape여야 합니다")
    if elementwise.shape[1] != len(TRAITS):
        raise ValueError("trait loss의 두 번째 차원은 3이어야 합니다")
    mask = mask.bool()
    trait_losses = [
        elementwise[mask[:, index], index].mean()
        for index in range(len(TRAITS))
        if torch.any(mask[:, index])
    ]
    if not trait_losses:
        return _zero_loss(elementwise)
    return torch.stack(trait_losses).mean()


# Unified scorer --------------------------------------------------------------
class RegressionScorer(nn.Module):
    """One shared representation and three trait score heads.

    ``score_head=regression`` keeps the original scalar bounded-regression
    baseline. ``score_head=distribution`` predicts five probabilities per
    trait and returns their expected value, so fractional human means stay
    fractional without rounding. ``score_head=regression_ordinal`` combines
    bounded regression with a cumulative ordinal expected value.
    """

    def __init__(self, backbone: nn.Module, hidden_size: int, config: RegressionConfig):
        super().__init__()
        self.backbone = backbone
        # ``nn.Module.config``는 Hugging Face가 자체 PretrainedConfig 용도로 사용한다.
        # 실험 설정은 별도 이름으로 보관해 Trainer의 config 처리를 방해하지 않는다.
        self.regression_config = config
        # Trainer callback이 optimizer global step을 넣는 train-only 상태다. Tensor나
        # Parameter가 아니므로 checkpoint key와 기존 state fingerprint를 바꾸지 않는다.
        self._quantization_global_step = 0
        # 1~5 grid는 고정 상수다. persistent=False라 기존 scalar checkpoint에
        # 새 missing key가 생기지 않는다.
        self.register_buffer(
            "score_values",
            torch.arange(1, 6, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "detail_halfstep_values",
            torch.arange(1.0, 5.01, 0.5, dtype=torch.float32),
            persistent=False,
        )
        if config.layer_aggregation == "scalar_mix":
            # 0으로 시작하면 처음에는 단순 last-N 평균과 동일하다. 학습이 필요할
            # 때만 softmax weight가 각 layer의 비중을 바꾼다.
            self.layer_mix_logits = nn.Parameter(
                torch.zeros(config.last_n_layers, dtype=torch.float32)
            )
        else:
            self.register_parameter("layer_mix_logits", None)
        self.attention_pool = (
            TraitAttentionPool(hidden_size, config.attention_pool_dim)
            if config.pooling == "essay_attention"
            else None
        )
        attention, logit = build_analysis_pool_modules(config, hidden_size)
        self.analysis_attention_pool = attention
        # C7 문항 판별기. weight=0이면 None이라 parameter 수가 기존과 같다.
        self.prompt_adversary = build_prompt_adversary(config, hidden_size)
        # GroupDRO 가중치. 학습 전용 상태라 checkpoint에서 제외한다
        # (TRAINING_ONLY_STATE_PREFIXES 참조). 균등에서 시작한다.
        self.register_buffer(
            "group_dro_weights",
            torch.full((GROUP_DRO_SLOTS,), 1.0 / GROUP_DRO_SLOTS, dtype=torch.float32),
            persistent=False,
        )
        # 수준 prototype. 0으로 시작하고 첫 관측에서 채워진다(위 helper 참조).
        # 학습 전용 상태라 checkpoint에서 제외한다.
        self.register_buffer(
            "level_prototypes",
            torch.zeros(config.level_prototype_count, hidden_size, dtype=torch.float32),
            persistent=False,
        )
        if logit is None:
            self.register_parameter("analysis_pool_logit", None)
        else:
            self.analysis_pool_logit = logit
        if config.organization_pooling == "first_middle_last":
            # 3H->H projection은 7B backbone에서도 수천만 parameter가 생긴다.
            # 서론/본론/결론의 상대적 중요도만 배우면 되므로 3개 scalar로 줄인다.
            # 0으로 시작해 첫 forward는 기존 shared pooling과 정확히 같다.
            self.organization_segment_weights = nn.Parameter(torch.zeros(3))
        else:
            self.register_parameter("organization_segment_weights", None)
        if config.organization_pooling == "sentence_transition":
            # 0에서 시작하므로 첫 forward는 shared control과 정확히 같다. 큰
            # H->H projection 없이 한 scalar만 인접 문장 변화량의 사용 여부를 배운다.
            self.organization_transition_weight = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("organization_transition_weight", None)
        if config.organization_pooling == "paragraph_mean":
            # 다른 두 organization mode와 같은 관례다. 0에서 시작하므로 첫 forward가
            # shared control과 정확히 같고, scalar 하나만 문단 평균의 사용 여부를 배운다.
            self.organization_paragraph_weight = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("organization_paragraph_weight", None)
        if config.criterion_readout == "textual_anchor_residual":
            # g=0이면 첫 forward에서 각 criterion 표현이 shared control과
            # bit-exact하게 같다. tanh가 residual 크기를 안정적으로 제한한다.
            self.criterion_anchor_gates = nn.Parameter(
                torch.zeros(len(DETAIL_CRITERIA), dtype=torch.float32)
            )
        else:
            self.register_parameter("criterion_anchor_gates", None)
        # trait_native_distribution은 trait마다 class 수가 다르다(41/17/17). 세 head를
        # 한 tensor로 묶으려고 가장 큰 격자에 맞춰 padding하고 mask로 나머지를 죽인다.
        if config.score_head == "distribution":
            output_size = 5
        elif config.score_head == "trait_native_distribution":
            output_size = MAX_TRAIT_CLASSES
        elif config.score_head == "regression_ordinal":
            output_size = 1 + config.ordinal_steps
        else:
            output_size = 1
        # Pardro's deployed composite head normalizes the pooled last-token
        # representation immediately before its regression/ordinal branches.
        # Keep this module exclusive to the new head so every historical
        # score head retains its exact parameter set and forward graph.
        self.direct_head_norm = (
            nn.LayerNorm(hidden_size)
            if config.score_head == "regression_ordinal"
            else None
        )
        if config.score_head == "trait_native_distribution":
            grid_values, grid_valid = trait_grid_tensors()
            self.register_buffer("trait_grid_values", grid_values, persistent=False)
            self.register_buffer("trait_grid_valid", grid_valid, persistent=False)
        if config.head_type in {"independent_mlp", "mixed_head_v2"}:
            self.heads = nn.ModuleDict(
                {
                    trait: nn.Sequential(
                        nn.Linear(hidden_size, config.head_hidden_size),
                        nn.GELU(),
                        nn.Linear(config.head_hidden_size, output_size),
                    )
                    for trait in TRAITS
                }
            )
        else:
            self.heads = nn.ModuleDict(
                {trait: nn.Linear(hidden_size, output_size) for trait in TRAITS}
            )

        # 세부 채점 기준 head는 기존 direct 3-trait head를 대체하지 않는다.
        # 기본 none에서는 빈 ModuleDict라 과거 checkpoint에 새 state key가 생기지 않는다.
        self.detail_heads = nn.ModuleDict()
        if config.detail_head_mode in RATER_SET_HEAD_MODES:
            # 평가자 원점수는 100% 정수 1~5다. rater_set은 그 5-class를 그대로
            # 모형화하고, rater_set_scalar는 같은 18개 단위를 bounded scalar로 낸다.
            rater_output_size = 1 if config.detail_head_mode == "rater_set_scalar" else 5
            self.detail_heads.update(
                {
                    f"rater{rater_slot + 1}_{criterion}": nn.Linear(
                        hidden_size, rater_output_size
                    )
                    for rater_slot in range(2)
                    for criterion in DETAIL_CRITERIA
                }
            )
        elif config.detail_head_mode != "none":
            detail_output_size = {
                "scalar": 1,
                "categorical": 5,
                "halfstep_categorical": 9,
            }[config.detail_head_mode]
            self.detail_heads.update(
                {
                    criterion: nn.Linear(hidden_size, detail_output_size)
                    for criterion in DETAIL_CRITERIA
                }
            )

        # 평가자 ID는 individual-rating CE를 위한 train-only severity다. 동일한
        # base criterion logits가 evaluator ID 없는 validation/test prediction이다.
        self.register_parameter("detail_evaluator_severity", None)
        if config.detail_rater_loss_weight > 0:
            if not config.detail_rater_registry:
                raise ValueError(
                    "평가자 auxiliary에는 train data에서 만든 "
                    "detail_rater_registry가 필요합니다"
                )
            self.detail_evaluator_severity = nn.Parameter(
                torch.zeros(
                    len(config.detail_rater_registry),
                    len(DETAIL_CRITERIA),
                    dtype=torch.float32,
                )
            )

        self.prompt_traits = (
            tuple(TRAITS) if config.prompt_head_traits == "all" else ("organization",)
        )
        self.prompt_heads = nn.ModuleList()
        self.register_parameter("prompt_bias", None)
        if config.prompt_head_mode != "none" and not config.prompt_registry:
            raise ValueError(
                "문제별 head를 사용할 때 train prompt_registry가 필요합니다"
            )
        if config.prompt_head_mode == "average":
            # Copy init이면 학습 시작점에서 공용/문제별 평균이 기존 head와 같다.
            self.prompt_heads.extend(
                nn.ModuleDict(
                    {
                        trait: copy.deepcopy(self.heads[trait])
                        for trait in self.prompt_traits
                    }
                )
                for _ in config.prompt_registry
            )
        elif config.prompt_head_mode == "bias":
            self.prompt_bias = nn.Parameter(
                torch.zeros(
                    len(config.prompt_registry),
                    len(TRAITS),
                    output_size,
                    dtype=torch.float32,
                )
            )

        # 점수와 같은 backbone token hidden을 쓰는 train-only auxiliary다. 모든
        # score module을 먼저 만든 뒤 독립 RNG 구간에서 초기화한다. 따라서 이
        # head의 on/off가 direct/detail head 초기값이나 학습 시작 RNG를 바꾸지
        # 않아 fresh control과 단일 seed로도 정확히 paired 비교할 수 있다.
        self.paragraph_boundary_head = None
        if config.paragraph_boundary_loss_weight > 0:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(config.seed) + 126_127)
                self.paragraph_boundary_head = nn.Linear(hidden_size, 1)
        self.current_stage = "head"

    def scoring_parameters(self) -> tuple[nn.Parameter, ...]:
        """backbone 밖의 head/pooling/layer-mix parameter를 한곳에서 반환한다."""

        backbone_ids = {id(parameter) for parameter in self.backbone.parameters()}
        return tuple(
            parameter
            for parameter in self.parameters()
            if id(parameter) not in backbone_ids
        )

    # 학습 전용 보조 module. checkpoint에 담지 않는다 — 추론에서 호출되지 않고,
    # 담으면 `prompt_adversary_weight=0`인 config로 로드할 때 unexpected key로
    # 하드 실패한다. 즉 적대적 학습으로 이긴 모델을 표준 배포 manifest로 못 쓰게 된다.
    # 판별기는 loss 항과 같은 성질이라 가중치를 보존할 이유가 없다.
    TRAINING_ONLY_STATE_PREFIXES = (
        "prompt_adversary.",
        "group_dro_weights",
        "level_prototypes",
    )

    def scoring_state_dict(self) -> dict[str, torch.Tensor]:
        """동결 backbone과 학습 전용 보조 module을 제외한 checkpoint state다."""

        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if not name.startswith("backbone.")
            and not name.startswith(self.TRAINING_ONLY_STATE_PREFIXES)
        }

    def train(self, mode: bool = True) -> "RegressionScorer":
        super().train(mode)
        if mode and self.current_stage == "head":
            self.backbone.eval()
        return self

    def set_training_stage(self, stage: str) -> None:
        if stage not in {"head", "joint"}:
            raise ValueError("stage must be head or joint")
        self.current_stage = stage
        self.backbone.requires_grad_(False)
        if stage == "joint":
            adapters = 0
            for name, parameter in self.backbone.named_parameters():
                if "lora_" in name:
                    parameter.requires_grad_(True)
                    adapters += parameter.numel()
            if adapters == 0:
                raise ValueError("joint stage에는 LoRA adapter가 필요합니다")
            self.backbone.train()
        else:
            self.backbone.eval()
        for parameter in self.scoring_parameters():
            parameter.requires_grad_(True)

    def set_quantization_global_step(self, step: int) -> None:
        """Set the optimizer step used by the optional quantization schedule."""

        if int(step) < 0:
            raise ValueError("quantization global step must be non-negative")
        self._quantization_global_step = int(step)

    def _final_norm(self, hidden: torch.Tensor) -> torch.Tensor:
        norm = getattr(resolve_text_model(self.backbone), "norm", None)
        return norm(hidden) if isinstance(norm, nn.Module) else hidden

    def _aggregate_layers(self, outputs: Any) -> torch.Tensor:
        config = self.regression_config
        if config.layer_aggregation == "last":
            return outputs.last_hidden_state
        states = getattr(outputs, "hidden_states", None)
        if not states or len(states) < config.last_n_layers:
            raise ValueError(
                f"{config.layer_aggregation}에는 마지막 {config.last_n_layers}개 "
                "hidden state가 필요합니다"
            )
        selected = list(states[-config.last_n_layers :])
        # Hugging Face decoder의 최종 state는 보통 이미 final norm을 통과한다.
        # 중간 layer만 같은 final norm으로 scale을 맞춘 뒤 평균/혼합한다.
        normalized = [
            state if index == len(selected) - 1 else self._final_norm(state)
            for index, state in enumerate(selected)
        ]
        if config.layer_aggregation == "last_n_mean":
            return torch.stack([state.float() for state in normalized], dim=0).mean(0)
        weights = torch.softmax(self.layer_mix_logits.float(), dim=0)
        return sum(
            weight * state.float()
            for weight, state in zip(weights, normalized, strict=True)
        )

    def _analysis_pool_share(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """접미사 지분을 [K] (K=1 또는 len(TRAITS))로 반환한다."""

        if self.analysis_pool_logit is not None:
            return torch.sigmoid(self.analysis_pool_logit.to(device=device, dtype=dtype))
        width = len(TRAITS) if self.regression_config.analysis_pool_per_trait else 1
        return torch.full(
            (width,),
            float(self.regression_config.analysis_pool_weight),
            device=device,
            dtype=dtype,
        )

    def _analysis_mix_pooled(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
        analysis_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """채점 구간을 본문/접미사로 나눠 각각 pooling한 뒤 지분으로 섞는다.

        접미사가 없는 row는 지분이 정확히 0이 되어 결과가 `pooling="mean"`과
        bit-exact하게 같다. dual-view 학습에서 no-analysis view가 baseline과
        어긋나지 않아야 두 view를 한 head로 학습할 수 있기 때문이다.
        """

        if analysis_mask is None or analysis_mask.shape != mask.shape:
            raise ValueError(
                f"pooling={self.regression_config.pooling!r}에는 attention_mask와 "
                "같은 shape의 analysis_mask가 필요합니다"
            )
        analysis_select = mask & analysis_mask.bool()
        essay_select = mask & ~analysis_mask.bool()
        if not torch.all(essay_select.any(dim=-1)):
            raise ValueError("분석 pooling에 본문 token이 하나도 없는 row가 있습니다")

        # 축약 방식을 "mean" 분기와 **글자 그대로 같게** 유지한다. einsum으로 바꾸면
        # 합산 순서가 달라져 근거 없는 row의 결과가 baseline과 1e-8 어긋나고, 그러면
        # 이 pooling의 유일한 안전장치인 "no-analysis view는 c02와 동일" 계약이 깨진다.
        # 임시 tensor는 순차 생성이라 peak memory가 "mean" 분기와 같다.
        hidden_float = hidden.float()

        def _masked_mean(selection: torch.Tensor) -> torch.Tensor:
            summed = (hidden_float * selection.unsqueeze(-1)).sum(dim=1)
            counts = selection.sum(dim=1, keepdim=True).float()
            return summed / counts.clamp(min=1.0)

        essay_pooled = _masked_mean(essay_select)
        analysis_pooled = _masked_mean(analysis_select)
        if self.analysis_attention_pool is not None:
            # 접미사가 없는 row는 softmax가 정의되지 않으므로 본문 구간을 먹여
            # 유한한 값을 만들고, 지분 0으로 결과에서 완전히 제거한다.
            has_row = analysis_select.any(dim=-1, keepdim=True)
            attention_select = torch.where(has_row, analysis_select, essay_select)
            analysis_pooled = self.analysis_attention_pool(
                hidden_float, attention_select
            )

        share = self._analysis_pool_share(
            device=hidden_float.device, dtype=hidden_float.dtype
        )
        has_analysis = analysis_select.any(dim=-1).to(hidden_float.dtype)
        per_trait = analysis_pooled.dim() == 3 or share.numel() > 1
        if not per_trait:
            weight = (share[0] * has_analysis).unsqueeze(-1)
            return (1.0 - weight) * essay_pooled + weight * analysis_pooled

        traits = len(TRAITS)
        if share.numel() == 1:
            share = share.expand(traits)
        if essay_pooled.dim() == 2:
            essay_pooled = essay_pooled.unsqueeze(1).expand(-1, traits, -1)
        if analysis_pooled.dim() == 2:
            analysis_pooled = analysis_pooled.unsqueeze(1).expand(-1, traits, -1)
        weight = share.view(1, traits, 1) * has_analysis.view(-1, 1, 1)
        return (1.0 - weight) * essay_pooled + weight * analysis_pooled

    def _normalize_if_requested(self, pooled: torch.Tensor) -> torch.Tensor:
        config = self.regression_config
        if config.normalize_features:
            return F.normalize(pooled.float(), p=2, dim=-1)
        if getattr(config, "pooled_normalization", "none") == "layernorm":
            # parameterless. affine parameter가 없으므로 state_dict가 그대로다.
            value = pooled.float()
            return F.layer_norm(value, value.shape[-1:])
        return pooled.float()

    def _add_organization_features(
        self,
        hidden: torch.Tensor,
        pooled: torch.Tensor,
        attention_mask: torch.Tensor,
        essay_mask: torch.Tensor | None,
        sentence_ids: torch.Tensor | None,
        paragraph_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the optional organization-only representation branch."""

        mode = self.regression_config.organization_pooling
        if mode == "shared":
            return pooled
        if essay_mask is None or essay_mask.shape != attention_mask.shape:
            raise ValueError("organization 전용 pooling에는 essay_mask가 필요합니다")

        essay_tokens = attention_mask.bool() & essay_mask.bool()
        if not torch.all(essay_tokens.any(dim=-1)):
            raise ValueError("organization pooling 대상 essay token이 비어 있습니다")
        hidden_float = hidden.float()
        essay_mean = (hidden_float * essay_tokens.unsqueeze(-1)).sum(
            dim=1
        ) / essay_tokens.sum(dim=1, keepdim=True).float()
        organization_features = essay_mean

        if mode == "first_middle_last":
            lengths = essay_tokens.sum(dim=1, keepdim=True)
            ranks = essay_tokens.long().cumsum(dim=1) - 1
            segment_ids = (ranks * 3 // lengths).clamp(min=0, max=2)
            segment_means = []
            for segment_index in range(3):
                segment_mask = essay_tokens & (segment_ids == segment_index)
                denominator = segment_mask.sum(dim=1, keepdim=True)
                segment_mean = (hidden_float * segment_mask.unsqueeze(-1)).sum(
                    dim=1
                ) / denominator.clamp_min(1).float()
                segment_means.append(
                    torch.where(denominator > 0, segment_mean, essay_mean)
                )
            assert self.organization_segment_weights is not None
            segments = torch.stack(segment_means, dim=1)
            residual = torch.einsum(
                "k,bkh->bh",
                self.organization_segment_weights.float(),
                segments - essay_mean[:, None, :],
            )
            base = pooled[:, 1] if pooled.ndim == 3 else pooled
            organization_features = base.float() + residual

        if mode == "paragraph_mean":
            if paragraph_ids is None or paragraph_ids.shape != attention_mask.shape:
                raise ValueError(
                    "paragraph_mean에는 attention_mask와 같은 shape의 paragraph_ids가 "
                    "필요합니다"
                )
            if torch.any(essay_tokens & (paragraph_ids < 0)):
                raise ValueError("essay token의 paragraph_ids가 비어 있습니다")
            # 문단 수는 실측 최대 13개라 명시적인 loop가 index_add보다 읽기 쉽다.
            paragraph_count = int(paragraph_ids.max().detach().item()) + 1
            paragraph_means = []
            paragraph_present = []
            for paragraph_index in range(paragraph_count):
                paragraph_mask = essay_tokens & (paragraph_ids == paragraph_index)
                token_count = paragraph_mask.sum(dim=1, keepdim=True)
                paragraph_sum = (hidden_float * paragraph_mask.unsqueeze(-1)).sum(dim=1)
                paragraph_means.append(
                    paragraph_sum / token_count.clamp_min(1).float()
                )
                paragraph_present.append(token_count > 0)
            means = torch.stack(paragraph_means, dim=1)
            present = torch.stack(paragraph_present, dim=1).float()
            # 문단마다 같은 가중치를 준다. token 수로 평균하는 essay_mean과 달리
            # 긴 문단이 표현을 지배하지 않는다.
            paragraph_mean = (means * present).sum(dim=1) / present.sum(
                dim=1
            ).clamp_min(1.0)
            assert self.organization_paragraph_weight is not None
            # 문단 cue가 없어 한 문단인 글은 paragraph_mean == essay_mean이므로
            # residual이 0이고 shared control과 정확히 같아진다.
            base = pooled[:, 1] if pooled.ndim == 3 else pooled
            organization_features = base.float() + (
                self.organization_paragraph_weight.float()
                * (paragraph_mean - essay_mean)
            )

        if mode == "sentence_transition":
            if sentence_ids is None or sentence_ids.shape != attention_mask.shape:
                raise ValueError(
                    "sentence_transition에는 attention_mask와 같은 sentence_ids가 "
                    "필요합니다"
                )
            if torch.any(essay_tokens & (sentence_ids < 0)):
                raise ValueError("essay token의 sentence_ids가 비어 있습니다")
            valid_sentences = essay_tokens & (sentence_ids >= 0)
            safe_sentence_ids = sentence_ids.masked_fill(~valid_sentences, 0)
            sentence_count = int(sentence_ids.max().detach().item()) + 1
            batch_size, _, hidden_size = hidden_float.shape
            row_offsets = (
                torch.arange(batch_size, device=hidden.device) * sentence_count
            ).unsqueeze(1)
            flat_sentence_ids = (safe_sentence_ids + row_offsets).reshape(-1)
            flat_sentence_sums = hidden_float.new_zeros(
                (batch_size * sentence_count, hidden_size)
            )
            flat_sentence_sums.index_add_(
                0,
                flat_sentence_ids,
                (
                    hidden_float * valid_sentences.unsqueeze(-1)
                ).reshape(-1, hidden_size),
            )
            sentence_sums = flat_sentence_sums.reshape(
                batch_size, sentence_count, hidden_size
            )
            flat_sentence_token_counts = hidden_float.new_zeros(
                (batch_size * sentence_count, 1)
            )
            flat_sentence_token_counts.index_add_(
                0,
                flat_sentence_ids,
                valid_sentences.reshape(-1, 1).float(),
            )
            sentence_token_counts = flat_sentence_token_counts.reshape(
                batch_size, sentence_count, 1
            )
            sentence_means = sentence_sums / sentence_token_counts.clamp_min(1.0)
            sentence_means = F.layer_norm(sentence_means, (hidden_size,))
            adjacent_valid = (
                (sentence_token_counts[:, :-1, 0] > 0)
                & (sentence_token_counts[:, 1:, 0] > 0)
            )
            adjacent_differences = torch.abs(
                sentence_means[:, 1:] - sentence_means[:, :-1]
            ) * adjacent_valid.unsqueeze(-1)
            adjacent_counts = adjacent_valid.sum(dim=1, keepdim=True)
            transitions = adjacent_differences.sum(dim=1) / adjacent_counts.clamp_min(
                1
            )
            transitions = torch.where(
                adjacent_counts > 0,
                transitions,
                torch.zeros_like(transitions),
            )
            assert self.organization_transition_weight is not None
            base = pooled[:, 1] if pooled.ndim == 3 else pooled
            organization_features = base.float() + (
                self.organization_transition_weight.float() * transitions
            )

        if pooled.ndim == 3:
            return torch.stack(
                (pooled[:, 0], organization_features, pooled[:, 2]), dim=1
            )
        return torch.stack((pooled, organization_features, pooled), dim=1)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        essay_mask: torch.Tensor | None = None,
        analysis_mask: torch.Tensor | None = None,
        sentence_ids: torch.Tensor | None = None,
        paragraph_ids: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        shared_pooling_mask: torch.Tensor | None = None,
        criterion_anchor_positions: torch.Tensor | None = None,
        return_token_hidden: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]
    ):
        grad_enabled = self.current_stage == "joint" and torch.is_grad_enabled()
        model_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
            "output_hidden_states": (
                self.regression_config.layer_aggregation != "last"
            ),
        }
        # BERT류 encoder는 use_cache를 받지 않는다. decoder는 cache를 끄면
        # 학습/hidden-state 추출 메모리를 아낄 수 있다.
        if self.regression_config.backbone_type == "decoder":
            model_inputs["use_cache"] = False
        if (
            token_type_ids is not None
            and self.regression_config.backbone_type != "decoder"
        ):
            model_inputs["token_type_ids"] = token_type_ids
        # head-only에서는 큰 backbone forward만 no-grad로 막는다. 이 블록 밖의
        # scalar mix와 attention pooling은 score-side 학습 대상이라 gradient를 둔다.
        with torch.set_grad_enabled(grad_enabled):
            outputs = self.backbone(**model_inputs)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden_states = getattr(outputs, "hidden_states", None)
            if not hidden_states:
                raise ValueError("backbone output에 hidden state가 없습니다")
        hidden = self._aggregate_layers(outputs)
        mask = attention_mask.bool()
        if self.regression_config.input_format == "rubric_conditioned_v1":
            if shared_pooling_mask is None:
                raise ValueError(
                    "rubric_conditioned_v1에는 suffix를 제외하는 shared_pooling_mask가 "
                    "필요합니다"
                )
        if shared_pooling_mask is not None:
            if shared_pooling_mask.shape != attention_mask.shape:
                raise ValueError(
                    "shared_pooling_mask는 attention_mask와 같은 shape여야 합니다"
                )
            if not torch.all(
                (shared_pooling_mask == 0) | (shared_pooling_mask == 1)
            ):
                raise ValueError("shared_pooling_mask 값은 0 또는 1이어야 합니다")
            scoring_tokens = shared_pooling_mask.bool()
            if torch.any(scoring_tokens & ~mask):
                raise ValueError("shared_pooling_mask가 padding token을 포함합니다")
            mask = mask & scoring_tokens
        if self.regression_config.pooling.startswith("essay_"):
            if essay_mask is None or essay_mask.shape != attention_mask.shape:
                raise ValueError(
                    "essay pooling에는 attention_mask와 같은 shape의 essay_mask가 필요합니다"
                )
            mask = mask & essay_mask.bool()
        if not torch.all(mask.any(dim=-1)):
            raise ValueError("pooling 대상 token이 비어 있습니다")

        pooling = self.regression_config.pooling
        if pooling in {"mean", "essay_mean"}:
            pooled = (hidden.float() * mask.unsqueeze(-1)).sum(dim=1)
            pooled = pooled / mask.sum(dim=1, keepdim=True).float()
        elif pooling in ANALYSIS_MIX_POOLINGS:
            pooled = self._analysis_mix_pooled(hidden, mask, analysis_mask)
        elif pooling == "essay_attention":
            assert self.attention_pool is not None
            pooled = self.attention_pool(hidden, mask)
        else:
            token_positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(
                0
            )
            if pooling == "first":
                positions = (
                    torch.where(mask, token_positions, mask.shape[1]).min(dim=1).values
                )
            else:  # last
                positions = torch.where(mask, token_positions, -1).max(dim=1).values
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            pooled = hidden[rows, positions]

        pooled = self._add_organization_features(
            hidden, pooled, attention_mask, essay_mask, sentence_ids, paragraph_ids
        )
        pooled = self._normalize_if_requested(pooled)
        if self.regression_config.criterion_readout != "textual_anchor_residual":
            return (pooled, None, hidden) if return_token_hidden else pooled
        if criterion_anchor_positions is None:
            raise ValueError(
                "textual_anchor_residual에는 criterion_anchor_positions가 필요합니다"
            )
        expected_shape = (input_ids.shape[0], len(DETAIL_CRITERIA))
        if tuple(criterion_anchor_positions.shape) != expected_shape:
            raise ValueError(
                "criterion_anchor_positions는 [B,9]여야 합니다: "
                f"expected={expected_shape}, actual={tuple(criterion_anchor_positions.shape)}"
            )
        if criterion_anchor_positions.dtype != torch.long:
            raise ValueError("criterion_anchor_positions dtype은 torch.long이어야 합니다")
        if torch.any(criterion_anchor_positions < 0) or torch.any(
            criterion_anchor_positions >= input_ids.shape[1]
        ):
            raise ValueError("criterion anchor token 위치가 sequence 범위를 벗어났습니다")
        if torch.any(
            criterion_anchor_positions[:, 1:]
            <= criterion_anchor_positions[:, :-1]
        ):
            raise ValueError("criterion anchor token 위치는 행마다 고유한 증가 순서여야 합니다")
        rows = torch.arange(input_ids.shape[0], device=input_ids.device).unsqueeze(1)
        anchor_attention = attention_mask.bool()[rows, criterion_anchor_positions]
        if not torch.all(anchor_attention):
            raise ValueError("criterion anchor가 padding token을 가리킵니다")
        criterion_anchor_hidden = hidden[rows, criterion_anchor_positions]
        normalized_anchors = self._normalize_if_requested(criterion_anchor_hidden)
        if return_token_hidden:
            return pooled, normalized_anchors, hidden
        return pooled, normalized_anchors

    def _criterion_readout_features(
        self,
        shared_features: torch.Tensor,
        criterion_anchor_hidden: torch.Tensor | None,
    ) -> torch.Tensor:
        """Blend shared and criterion-specific text anchors for detail heads."""

        if self.regression_config.criterion_readout == "shared":
            return shared_features
        if criterion_anchor_hidden is None:
            raise ValueError("textual anchor hidden state가 없습니다")
        if criterion_anchor_hidden.ndim != 3 or criterion_anchor_hidden.shape[1] != len(
            DETAIL_CRITERIA
        ):
            raise ValueError("textual anchor hidden state는 [B,9,H]여야 합니다")
        if shared_features.ndim == 2:
            criterion_shared = shared_features.unsqueeze(1).expand(
                -1, len(DETAIL_CRITERIA), -1
            )
        elif shared_features.ndim == 3 and shared_features.shape[1] == len(TRAITS):
            trait_indices = torch.tensor(
                tuple(
                    trait_index
                    for trait_index, trait_criteria in enumerate(
                        DETAIL_CRITERIA_BY_TRAIT
                    )
                    for _ in trait_criteria
                ),
                device=shared_features.device,
            )
            criterion_shared = shared_features[:, trait_indices]
        else:
            raise ValueError("shared feature는 [B,H] 또는 [B,3,H]여야 합니다")
        assert self.criterion_anchor_gates is not None
        gates = torch.tanh(self.criterion_anchor_gates.float()).view(1, -1, 1)
        return criterion_shared.float() + gates * (
            criterion_anchor_hidden.float() - criterion_shared.float()
        )

    def _raw_head_outputs(
        self,
        features: torch.Tensor,
        heads: nn.ModuleDict | None = None,
    ) -> torch.Tensor:
        """Return [batch, trait, 1 or 5] before score conversion."""

        features = features.float()
        if self.direct_head_norm is not None:
            features = self.direct_head_norm(features)
        if features.ndim not in {2, 3}:
            raise ValueError("pooled feature는 [B,H] 또는 [B,3,H]여야 합니다")
        if features.ndim == 3 and features.shape[1] != len(TRAITS):
            raise ValueError("trait별 pooled feature의 두 번째 차원은 3이어야 합니다")

        def trait_features(index: int) -> torch.Tensor:
            return features if features.ndim == 2 else features[:, index]

        selected_heads = self.heads if heads is None else heads
        if self.regression_config.head_type == "mixed_head_v2":
            middle = torch.stack(
                [
                    selected_heads[trait][1](
                        selected_heads[trait][0](trait_features(index))
                    )
                    for index, trait in enumerate(TRAITS)
                ],
                dim=1,
            )
            other_mean = (middle.sum(dim=1, keepdim=True) - middle) / (len(TRAITS) - 1)
            weight = self.regression_config.mixed_head_weight
            mixed = weight * middle + (1.0 - weight) * other_mean
            return torch.stack(
                [
                    selected_heads[trait][2](mixed[:, index])
                    for index, trait in enumerate(TRAITS)
                ],
                dim=1,
            )

        return torch.stack(
            [
                selected_heads[trait](trait_features(index))
                for index, trait in enumerate(TRAITS)
            ],
            dim=1,
        )

    def _predictions_from_raw(
        self, raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Convert raw outputs to scores and a loss-side auxiliary tensor.

        Distribution heads return normalized probabilities.  The composite
        head returns *raw ordinal logits* so training can use numerically
        stable BCE-with-logits even when sigmoid would saturate at 0 or 1.
        """

        config = self.regression_config
        if config.score_head == "regression_ordinal":
            regression = 1.0 + 4.0 * torch.sigmoid(raw[..., 0])
            ordinal_probabilities = torch.sigmoid(raw[..., 1:])
            ordinal = 1.0 + (
                4.0 / config.ordinal_steps
            ) * ordinal_probabilities.sum(dim=-1)
            weight = config.ordinal_blend_weight
            scores = weight * ordinal + (1.0 - weight) * regression
            return scores, raw[..., 1:]
        if config.score_head in {"distribution", "trait_native_distribution"}:
            if config.score_head == "distribution":
                values, valid = self.score_values, None
            else:
                values, valid = self.trait_grid_values, self.trait_grid_valid
            if config.head_type == "weighted_mixed":
                # 확률을 먼저 섞으면 각 trait의 결과도 여전히 합 1인 분포다.
                # validate()가 이 조합에서 readout을 expectation으로 고정한다.
                masked = raw if valid is None else raw.masked_fill(~valid, float("-inf"))
                probabilities = torch.softmax(masked, dim=-1)
                other_mean = (
                    probabilities.sum(dim=1, keepdim=True) - probabilities
                ) / (len(TRAITS) - 1)
                weight = config.mixed_head_weight
                probabilities = weight * probabilities + (1.0 - weight) * other_mean
                scores = (probabilities * values.to(probabilities.device)).sum(dim=-1)
                return scores, probabilities
            return self._categorical_score(raw, values, valid)

        if config.score_parameterization == "average_plus_contrast":
            # validate()가 distribution head와 trait 사이를 섞는 head를 이미
            # 막았으므로 아래 weighted_mixed 혼합을 다시 적용하지 않는다.
            return average_plus_contrast_scores(raw), None

        scores = 1.0 + 4.0 * torch.sigmoid(raw.squeeze(-1))
        if config.head_type == "weighted_mixed":
            other_mean = (scores.sum(dim=1, keepdim=True) - scores) / (len(TRAITS) - 1)
            weight = config.mixed_head_weight
            scores = weight * scores + (1.0 - weight) * other_mean
        return scores, None

    def _prompt_raw_outputs(
        self,
        features: torch.Tensor,
        global_raw: torch.Tensor,
        prompt_ids: torch.Tensor,
    ) -> torch.Tensor:
        """문제별 branch raw output을 만들고 unknown은 공용 raw로 되돌린다."""

        if prompt_ids.ndim != 1 or prompt_ids.shape[0] != features.shape[0]:
            raise ValueError("prompt_ids는 batch와 같은 길이의 [B] tensor여야 합니다")
        prompt_count = len(self.regression_config.prompt_registry)
        known = (prompt_ids >= 0) & (prompt_ids < prompt_count)
        safe_ids = prompt_ids.clamp(min=0, max=max(0, prompt_count - 1))

        if self.regression_config.prompt_head_mode == "bias":
            assert self.prompt_bias is not None
            selected_bias = self.prompt_bias[safe_ids]
            selected_bias = torch.where(
                known[:, None, None],
                selected_bias,
                torch.zeros_like(selected_bias),
            )
            active_traits = torch.tensor(
                [trait in self.prompt_traits for trait in TRAITS],
                device=global_raw.device,
                dtype=global_raw.dtype,
            ).view(1, len(TRAITS), 1)
            return global_raw + selected_bias * active_traits

        # 모든 P개 linear head를 작은 score-side에서 계산한 뒤 ID로 gather한다.
        # 같은-question batch에서도 각 head가 graph에 남아 DDP unused parameter를
        # 만들지 않으며, P=9라 backbone에 비해 추가 연산은 매우 작다.
        rows = torch.arange(features.shape[0], device=features.device)
        prompt_raw_traits = []
        for trait_index, trait in enumerate(TRAITS):
            if trait not in self.prompt_traits:
                prompt_raw_traits.append(global_raw[:, trait_index])
                continue
            trait_features = (
                features if features.ndim == 2 else features[:, trait_index]
            )
            all_prompt_outputs = torch.stack(
                [
                    prompt_heads[trait](trait_features.float())
                    for prompt_heads in self.prompt_heads
                ],
                dim=1,
            )
            selected = all_prompt_outputs[rows, safe_ids]
            prompt_raw_traits.append(
                torch.where(
                    known[:, None],
                    selected,
                    global_raw[:, trait_index],
                )
            )
        return torch.stack(prompt_raw_traits, dim=1)

    def predictions_from_features(
        self,
        features: torch.Tensor,
        prompt_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """공용/문제별 branch를 score 또는 probability 단계에서 평균한다."""

        global_raw = self._raw_head_outputs(features)
        global_scores, global_probabilities = self._predictions_from_raw(global_raw)
        config = self.regression_config
        if config.prompt_head_mode == "none":
            return global_scores, global_probabilities
        if prompt_ids is None:
            raise ValueError("문제별 head가 활성화되면 prompt_ids가 필요합니다")

        prompt_raw = self._prompt_raw_outputs(features, global_raw, prompt_ids)
        prompt_scores, prompt_probabilities = self._predictions_from_raw(prompt_raw)
        weight = config.prompt_head_weight
        if global_probabilities is not None:
            assert prompt_probabilities is not None
            probabilities = (
                1.0 - weight
            ) * global_probabilities + weight * prompt_probabilities
            scores = (probabilities * self.score_values.to(probabilities.device)).sum(
                dim=-1
            )
            return scores, probabilities
        scores = (1.0 - weight) * global_scores + weight * prompt_scores
        return scores, None

    def score_from_features(
        self,
        features: torch.Tensor,
        prompt_ids: torch.Tensor | None = None,
        criterion_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        direct_scores, _ = self.predictions_from_features(
            features, prompt_ids=prompt_ids
        )
        if self.regression_config.detail_final_source == "direct":
            return direct_scores
        if (
            self.regression_config.criterion_readout == "textual_anchor_residual"
            and criterion_features is None
        ):
            raise ValueError(
                "textual_anchor_residual score에는 criterion_features가 필요합니다"
            )
        detail_scores, _, _ = self.detail_predictions_from_features(
            features if criterion_features is None else criterion_features
        )
        return self.aggregate_detail_scores(detail_scores)

    def _detail_raw_outputs(self, features: torch.Tensor) -> torch.Tensor:
        """Return criterion logits, or anonymous-rater logits for ``rater_set``.

        Scalar/categorical modes return ``[B,9,C]``.  ``rater_set`` returns
        ``[B,2,9,5]`` so the two complete nine-score vectors can be matched as
        a set rather than assigning meaning to evaluator storage order.
        """

        if self.regression_config.detail_head_mode == "none":
            raise ValueError("detail_head_mode='none'에는 detail output이 없습니다")
        features = features.float()
        if features.ndim not in {2, 3}:
            raise ValueError("detail feature는 [B,H], [B,3,H], [B,9,H]여야 합니다")
        if features.ndim == 3 and features.shape[1] not in {
            len(TRAITS),
            len(DETAIL_CRITERIA),
        }:
            raise ValueError("detail feature의 두 번째 차원은 3 또는 9여야 합니다")

        def outputs_for_rater(rater_slot: int | None) -> torch.Tensor:
            outputs = []
            criterion_index = 0
            for trait_index, trait_criteria in enumerate(DETAIL_CRITERIA_BY_TRAIT):
                for criterion in trait_criteria:
                    if features.ndim == 2:
                        criterion_features = features
                    elif features.shape[1] == len(DETAIL_CRITERIA):
                        criterion_features = features[:, criterion_index]
                    else:
                        criterion_features = features[:, trait_index]
                    key = (
                        criterion
                        if rater_slot is None
                        else f"rater{rater_slot + 1}_{criterion}"
                    )
                    outputs.append(self.detail_heads[key](criterion_features))
                    criterion_index += 1
            return torch.stack(outputs, dim=1)

        if self.regression_config.detail_head_mode in RATER_SET_HEAD_MODES:
            return torch.stack([outputs_for_rater(0), outputs_for_rater(1)], dim=1)
        return outputs_for_rater(None)

    def _categorical_score(
        self, logits: torch.Tensor, values: torch.Tensor, valid: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """분류 logits을 격자 점수와 확률로 바꾼다. 세 readout이 여기 한 곳에 있다.

        ``values``는 마지막 축과 broadcast되는 격자 값이고, ``valid``가 주어지면
        padding class를 softmax 전에 죽인다. 반환되는 확률은 readout과 무관하게
        항상 softmax 확률이라 cross entropy 경로가 영향을 받지 않는다.
        """

        readout = self.regression_config.categorical_readout
        # persistent=False buffer는 checkpoint에 저장되지 않아 추론에서 CPU에 남을 수
        # 있다. 기존 score_values도 같은 이유로 매번 device를 맞춘다.
        if valid is not None:
            logits = logits.masked_fill(~valid.to(logits.device), float("-inf"))
        probabilities = torch.softmax(logits, dim=-1)
        values = values.to(probabilities.device)

        if readout == "expectation":
            return (probabilities * values).sum(dim=-1), probabilities
        if readout == "argmax":
            index = probabilities.argmax(dim=-1, keepdim=True)
            selected = values.expand_as(probabilities).gather(-1, index)
            return selected.squeeze(-1), probabilities
        if readout == "gumbel_straight_through":
            one_hot = gumbel_straight_through(
                logits,
                self.regression_config.gumbel_temperature,
                sample_noise=self.training,
            )
            return (one_hot * values).sum(dim=-1), probabilities
        raise ValueError(f"unsupported categorical_readout={readout!r}")

    def detail_predictions_from_features(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Return criterion expected scores, optional probabilities, and raw logits."""

        raw = self._detail_raw_outputs(features)
        mode = self.regression_config.detail_head_mode
        if mode == "categorical":
            detail_scores, probabilities = self._categorical_score(
                raw, self.score_values, None
            )
            return detail_scores, probabilities, raw
        if mode == "halfstep_categorical":
            detail_scores, probabilities = self._categorical_score(
                raw, self.detail_halfstep_values, None
            )
            return detail_scores, probabilities, raw
        if mode == "rater_set":
            per_rater_scores, probabilities = self._categorical_score(
                raw, self.score_values, None
            )
            # 두 익명 평가자 예측의 평균이 criterion 점수다. 라벨도 같은 평균이다.
            return per_rater_scores.mean(dim=1), probabilities, raw
        if mode == "rater_set_scalar":
            per_rater_scores = 1.0 + 4.0 * torch.sigmoid(raw.squeeze(-1))
            return per_rater_scores.mean(dim=1), None, raw
        detail_scores = 1.0 + 4.0 * torch.sigmoid(raw.squeeze(-1))
        return detail_scores, None, raw

    @staticmethod
    def aggregate_detail_scores(detail_scores: torch.Tensor) -> torch.Tensor:
        """Aggregate the canonical 5/2/2 criterion order into ``[B,3]``."""

        if detail_scores.ndim != 2 or detail_scores.shape[1] != len(DETAIL_CRITERIA):
            raise ValueError("detail_scores는 [B,9]여야 합니다")
        trait_scores = []
        start = 0
        for trait_criteria in DETAIL_CRITERIA_BY_TRAIT:
            stop = start + len(trait_criteria)
            trait_scores.append(detail_scores[:, start:stop].mean(dim=-1))
            start = stop
        return torch.stack(trait_scores, dim=-1)

    @staticmethod
    def _validate_criterion_targets(
        criterion_scores: torch.Tensor,
        criterion_mask: torch.Tensor,
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if criterion_scores.shape != (batch_size, len(DETAIL_CRITERIA)):
            raise ValueError("criterion_scores는 [B,9]여야 합니다")
        if criterion_mask.shape != criterion_scores.shape:
            raise ValueError(
                "criterion_mask는 criterion_scores와 같은 [B,9]여야 합니다"
            )
        targets = criterion_scores.float()
        mask = criterion_mask.bool()
        valid_targets = targets[mask]
        if valid_targets.numel() and (
            not torch.all(torch.isfinite(valid_targets))
            or torch.any((valid_targets < 1) | (valid_targets > 5))
        ):
            raise ValueError("유효 criterion score는 유한한 1~5 값이어야 합니다")
        return targets, mask

    @staticmethod
    def _validate_distribution_targets(
        criterion_distributions: torch.Tensor,
        criterion_distribution_mask: torch.Tensor,
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected_shape = (batch_size, len(DETAIL_CRITERIA), 5)
        if criterion_distributions.shape != expected_shape:
            raise ValueError("criterion_distributions는 [B,9,5]여야 합니다")
        if criterion_distribution_mask.shape != expected_shape[:2]:
            raise ValueError("criterion_distribution_mask는 [B,9]여야 합니다")
        targets = criterion_distributions.float()
        mask = criterion_distribution_mask.bool()
        valid_targets = targets[mask]
        if valid_targets.numel():
            if not torch.all(torch.isfinite(valid_targets)) or torch.any(
                valid_targets < 0
            ):
                raise ValueError(
                    "유효 criterion distribution은 유한한 음이 아닌 값이어야 합니다"
                )
            totals = valid_targets.sum(dim=-1)
            if not torch.allclose(
                totals, torch.ones_like(totals), atol=1e-5, rtol=1e-5
            ):
                raise ValueError(
                    "유효 criterion distribution의 class 합은 1이어야 합니다"
                )
        return targets, mask

    @staticmethod
    def _detail_halfstep_cross_entropy(
        detail_logits: torch.Tensor,
        criterion_scores: torch.Tensor,
        criterion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Classify official criterion means on the exact 0.5-point grid."""

        batch_size = detail_logits.shape[0]
        expected_shape = (batch_size, len(DETAIL_CRITERIA), 9)
        if detail_logits.shape != expected_shape:
            raise ValueError("halfstep detail logits는 [B,9,9]여야 합니다")
        targets, mask = RegressionScorer._validate_criterion_targets(
            criterion_scores,
            criterion_mask,
            batch_size=batch_size,
        )
        class_positions = 2.0 * (targets - 1.0)
        rounded_positions = class_positions.round()
        if torch.any(mask & ~torch.isclose(class_positions, rounded_positions)):
            raise ValueError(
                "halfstep categorical target은 1, 1.5, ..., 5 중 하나여야 합니다"
            )
        labels = rounded_positions.long().clamp(min=0, max=8)
        cross_entropy = F.cross_entropy(
            detail_logits.reshape(-1, 9),
            labels.reshape(-1),
            reduction="none",
        ).reshape(batch_size, len(DETAIL_CRITERIA))
        return _masked_detail_trait_mean(cross_entropy, mask) / math.log(9.0)

    @staticmethod
    def _detail_rater_set_cross_entropy(
        detail_logits: torch.Tensor,
        official_rater_labels: torch.Tensor,
        official_rater_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Match two anonymous predicted rater vectors to one or two officials.

        Matching is performed once for the complete nine-criterion vector,
        never criterion by criterion.  This preserves within-rater scoring
        patterns while removing the arbitrary evaluator1/evaluator2 order.
        """

        batch_size = detail_logits.shape[0]
        logit_shape = (batch_size, 2, len(DETAIL_CRITERIA), 5)
        target_shape = logit_shape[:-1]
        if detail_logits.shape != logit_shape:
            raise ValueError("rater_set detail logits는 [B,2,9,5]여야 합니다")
        if official_rater_labels.shape != target_shape:
            raise ValueError("official_rater_labels는 [B,2,9]여야 합니다")
        if official_rater_mask.shape != target_shape:
            raise ValueError("official_rater_mask는 [B,2,9]여야 합니다")

        mask = official_rater_mask.bool()
        raw_labels = official_rater_labels[mask].float()
        if raw_labels.numel() and (
            not torch.all(torch.isfinite(raw_labels))
            or not torch.all(raw_labels == raw_labels.round())
        ):
            raise ValueError("유효 official rater label은 정수 class index여야 합니다")
        labels = official_rater_labels.long()
        valid_labels = labels[mask]
        if valid_labels.numel() and torch.any((valid_labels < 0) | (valid_labels > 4)):
            raise ValueError("유효 official rater label class index는 0..4여야 합니다")
        safe_labels = labels.clamp(min=0, max=4)

        assignment_cost: dict[tuple[int, int], torch.Tensor] = {}
        target_valid: dict[int, torch.Tensor] = {}
        for target_slot in range(2):
            target_valid[target_slot] = mask[:, target_slot].any(dim=-1)
            for prediction_slot in range(2):
                cross_entropy = F.cross_entropy(
                    detail_logits[:, prediction_slot].reshape(-1, 5),
                    safe_labels[:, target_slot].reshape(-1),
                    reduction="none",
                ).reshape(batch_size, len(DETAIL_CRITERIA))
                assignment_cost[(prediction_slot, target_slot)], _ = (
                    _per_essay_detail_trait_mean(
                        cross_entropy,
                        mask[:, target_slot],
                    )
                )

        matched = _minimum_rater_assignment_cost(
            assignment_cost, target_valid, zero_reference=detail_logits
        )
        if matched is None:
            return _zero_loss(detail_logits)
        return matched / math.log(5.0)

    def _detail_rater_set_squared_error(
        self,
        per_rater_scores: torch.Tensor,
        official_rater_labels: torch.Tensor,
        official_rater_mask: torch.Tensor,
    ) -> torch.Tensor:
        """rater_set_scalar의 순서 없는 MSE다. 매칭 규칙은 CE 판과 같다.

        label은 class index(0..4)로 저장되어 있으므로 1~5 점수로 되돌려 비교한다.
        """

        batch_size = per_rater_scores.shape[0]
        expected_shape = (batch_size, 2, len(DETAIL_CRITERIA))
        if per_rater_scores.shape != expected_shape:
            raise ValueError("rater_set_scalar 예측은 [B,2,9]여야 합니다")
        if official_rater_labels.shape != expected_shape:
            raise ValueError("official_rater_labels는 [B,2,9]여야 합니다")
        if official_rater_mask.shape != expected_shape:
            raise ValueError("official_rater_mask는 [B,2,9]여야 합니다")

        mask = official_rater_mask.bool()
        targets = official_rater_labels.float() + 1.0

        assignment_cost: dict[tuple[int, int], torch.Tensor] = {}
        target_valid: dict[int, torch.Tensor] = {}
        for target_slot in range(2):
            target_valid[target_slot] = mask[:, target_slot].any(dim=-1)
            for prediction_slot in range(2):
                squared = torch.square(
                    per_rater_scores[:, prediction_slot] - targets[:, target_slot]
                )
                assignment_cost[(prediction_slot, target_slot)], _ = (
                    _per_essay_detail_trait_mean(squared, mask[:, target_slot])
                )

        matched = _minimum_rater_assignment_cost(
            assignment_cost, target_valid, zero_reference=per_rater_scores
        )
        if matched is None:
            return _zero_loss(per_rater_scores)
        return matched

    def _detail_rater_cross_entropy(
        self,
        detail_logits: torch.Tensor,
        detail_rater_ids: torch.Tensor,
        detail_rater_labels: torch.Tensor,
        detail_rater_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute train-only evaluator-severity CE without changing base logits."""

        assert self.detail_evaluator_severity is not None
        batch_size = detail_logits.shape[0]
        if detail_rater_ids.ndim != 2 or detail_rater_ids.shape[0] != batch_size:
            raise ValueError("detail_rater_ids는 [B,R]여야 합니다")
        expected_shape = (
            batch_size,
            detail_rater_ids.shape[1],
            len(DETAIL_CRITERIA),
        )
        if detail_rater_labels.shape != expected_shape:
            raise ValueError("detail_rater_labels는 [B,R,9]여야 합니다")
        if detail_rater_mask.shape != expected_shape:
            raise ValueError(
                "detail_rater_mask는 detail_rater_labels와 같은 [B,R,9]여야 합니다"
            )

        mask = detail_rater_mask.bool()
        used_raters = mask.any(dim=-1)
        rater_count = self.detail_evaluator_severity.shape[0]
        valid_raw_ids = detail_rater_ids[used_raters].float()
        if valid_raw_ids.numel() and (
            not torch.all(torch.isfinite(valid_raw_ids))
            or not torch.all(valid_raw_ids == valid_raw_ids.round())
        ):
            raise ValueError("유효 detail rater ID index는 유한한 정수여야 합니다")
        ids = detail_rater_ids.long()
        if torch.any(used_raters & ((ids < 0) | (ids >= rater_count))):
            raise ValueError("유효 detail rater ID가 registry 범위를 벗어났습니다")

        valid_raw_labels = detail_rater_labels[mask].float()
        if valid_raw_labels.numel() and (
            not torch.all(torch.isfinite(valid_raw_labels))
            or not torch.all(valid_raw_labels == valid_raw_labels.round())
        ):
            raise ValueError("유효 detail_rater_labels는 정수 class index여야 합니다")
        labels = detail_rater_labels.long()
        valid_labels = labels[mask]
        if valid_labels.numel() and torch.any((valid_labels < 0) | (valid_labels > 4)):
            raise ValueError("유효 detail_rater_labels class index는 0..4여야 합니다")

        # evaluator 축 평균을 빼 base logits와 severity의 식별 가능성을 유지한다.
        severity = self.detail_evaluator_severity.float()
        centered_severity = severity - severity.mean(dim=0, keepdim=True)
        safe_ids = ids.clamp(min=0, max=rater_count - 1)
        selected_severity = centered_severity[safe_ids]
        class_direction = self.score_values.to(detail_logits.device) - 3.0
        rater_logits = detail_logits[:, None, :, :] + (
            selected_severity[..., None] * class_direction
        )
        safe_labels = labels.clamp(min=0, max=4)
        cross_entropy = F.cross_entropy(
            rater_logits.reshape(-1, 5),
            safe_labels.reshape(-1),
            reduction="none",
        ).reshape(expected_shape)
        return _masked_detail_trait_mean(cross_entropy, mask) / math.log(5.0)

    def _detail_training_loss(
        self,
        scores: torch.Tensor,
        detail_scores: torch.Tensor,
        detail_probabilities: torch.Tensor | None,
        detail_logits: torch.Tensor,
        *,
        criterion_scores: torch.Tensor | None,
        criterion_distributions: torch.Tensor | None,
        criterion_mask: torch.Tensor | None,
        criterion_distribution_mask: torch.Tensor | None,
        official_rater_labels: torch.Tensor | None,
        official_rater_mask: torch.Tensor | None,
        detail_rater_ids: torch.Tensor | None,
        detail_rater_labels: torch.Tensor | None,
        detail_rater_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Combine the explicitly enabled 9-criterion auxiliary objectives."""

        config = self.regression_config
        loss = _zero_loss(detail_scores)
        score_targets: torch.Tensor | None = None
        score_mask: torch.Tensor | None = None
        needs_score_mask = (
            config.detail_expected_loss_weight > 0
            or config.detail_halfstep_loss_weight > 0
            or config.detail_hierarchy_loss_weight > 0
        )
        if needs_score_mask:
            if criterion_scores is None or criterion_mask is None:
                raise ValueError(
                    "detail expected/hierarchy loss에는 criterion_scores와 "
                    "criterion_mask가 필요합니다"
                )
            score_targets, score_mask = self._validate_criterion_targets(
                criterion_scores,
                criterion_mask,
                batch_size=detail_scores.shape[0],
            )

        if config.detail_expected_loss_weight > 0:
            assert score_targets is not None and score_mask is not None
            expected_mse = _masked_detail_trait_mean(
                torch.square(detail_scores - score_targets),
                score_mask,
            )
            loss = loss + config.detail_expected_loss_weight * expected_mse

        if config.detail_distribution_loss_weight > 0:
            if detail_probabilities is None:
                raise ValueError(
                    "detail distribution loss에는 categorical detail head가 필요합니다"
                )
            if criterion_distributions is None or criterion_distribution_mask is None:
                raise ValueError(
                    "detail distribution loss에는 criterion_distributions와 "
                    "criterion_distribution_mask가 필요합니다"
                )
            distribution_targets, distribution_mask = (
                self._validate_distribution_targets(
                    criterion_distributions,
                    criterion_distribution_mask,
                    batch_size=detail_scores.shape[0],
                )
            )
            cross_entropy = -(
                distribution_targets * detail_probabilities.clamp_min(1e-8).log()
            ).sum(dim=-1)
            distribution_loss = _masked_detail_trait_mean(
                cross_entropy,
                distribution_mask,
            ) / math.log(5.0)
            loss = loss + config.detail_distribution_loss_weight * distribution_loss

        if config.detail_halfstep_loss_weight > 0:
            if criterion_scores is None or criterion_mask is None:
                raise ValueError(
                    "detail halfstep loss에는 criterion_scores와 criterion_mask가 "
                    "필요합니다"
                )
            halfstep_loss = self._detail_halfstep_cross_entropy(
                detail_logits,
                criterion_scores,
                criterion_mask,
            )
            loss = loss + config.detail_halfstep_loss_weight * halfstep_loss

        if config.detail_rater_set_loss_weight > 0:
            if official_rater_labels is None or official_rater_mask is None:
                raise ValueError(
                    "detail rater set loss에는 official_rater_labels/mask가 "
                    "필요합니다"
                )
            if config.detail_head_mode == "rater_set_scalar":
                per_rater_scores = 1.0 + 4.0 * torch.sigmoid(detail_logits.squeeze(-1))
                rater_set_loss = self._detail_rater_set_squared_error(
                    per_rater_scores,
                    official_rater_labels,
                    official_rater_mask,
                )
            else:
                rater_set_loss = self._detail_rater_set_cross_entropy(
                    detail_logits,
                    official_rater_labels,
                    official_rater_mask,
                )
            loss = loss + config.detail_rater_set_loss_weight * rater_set_loss

        if config.detail_hierarchy_loss_weight > 0:
            assert score_mask is not None
            aggregated_scores = []
            complete_trait_mask = []
            start = 0
            for trait_criteria in DETAIL_CRITERIA_BY_TRAIT:
                stop = start + len(trait_criteria)
                aggregated_scores.append(detail_scores[:, start:stop].mean(dim=-1))
                complete_trait_mask.append(score_mask[:, start:stop].all(dim=-1))
                start = stop
            hierarchy_scores = torch.stack(aggregated_scores, dim=-1)
            hierarchy_mask = torch.stack(complete_trait_mask, dim=-1)
            hierarchy_loss = _masked_equal_trait_mean(
                torch.square(hierarchy_scores - scores),
                hierarchy_mask,
            )
            loss = loss + config.detail_hierarchy_loss_weight * hierarchy_loss

        if config.detail_rater_loss_weight > 0:
            if any(
                tensor is None
                for tensor in (
                    detail_rater_ids,
                    detail_rater_labels,
                    detail_rater_mask,
                )
            ):
                raise ValueError(
                    "detail rater loss에는 detail_rater_ids/labels/mask가 필요합니다"
                )
            assert detail_rater_ids is not None
            assert detail_rater_labels is not None
            assert detail_rater_mask is not None
            rater_loss = self._detail_rater_cross_entropy(
                detail_logits,
                detail_rater_ids,
                detail_rater_labels,
                detail_rater_mask,
            )
            loss = loss + config.detail_rater_loss_weight * rater_loss
        return loss

    def _paragraph_boundary_training_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Balanced BCE on deployable sentence/gap candidate token positions."""

        if logits.ndim != 2:
            raise ValueError("paragraph boundary logits는 [B,S]여야 합니다")
        if labels.shape != logits.shape or mask.shape != logits.shape:
            raise ValueError(
                "paragraph boundary labels/mask는 logits와 같은 [B,S]여야 합니다"
            )
        valid = mask.bool()
        if not torch.any(valid):
            return _zero_loss(logits)
        targets = labels.float()[valid]
        if torch.any((targets < 0) | (targets > 1)):
            raise ValueError("paragraph boundary label은 0 또는 1이어야 합니다")
        selected_logits = logits.float()[valid]
        positive = targets == 1
        negative = targets == 0
        class_losses = []
        if torch.any(positive):
            class_losses.append(F.softplus(-selected_logits[positive]).mean())
        if torch.any(negative):
            class_losses.append(F.softplus(selected_logits[negative]).mean())
        return torch.stack(class_losses).mean()

    def level_prototype_loss(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor | None:
        """수준 prototype contrastive. 자기 수준에 가깝고 다른 수준에서 멀어지게 한다.

        prototype은 EMA buffer다. gradient는 **표현으로만** 흐르고 prototype으로는
        흐르지 않는다 — prototype은 최적화 대상이 아니라 관측의 요약이다. 그래서
        표현이 prototype을 따라가고, prototype이 표현을 따라간다(교대 갱신).

        `features`가 trait 축을 가지면 평균한다. 이 loss는 "글 전체 수준"을 다루고
        영역별 배분은 공식 지표가 보지 못하므로(2026-08-21 부록) trait을 나누지 않는다.
        """

        config = self.regression_config
        weight = float(config.level_prototype_weight)
        if weight <= 0:
            return None
        pooled = features if features.ndim == 2 else features.mean(dim=1)
        pooled = F.normalize(pooled.float(), p=2, dim=-1)
        count = int(config.level_prototype_count)
        # 라벨 평균을 [1,5]에서 count개 등폭 bin으로 나눈다.
        average = labels.float().mean(dim=-1).clamp(1.0, 5.0)
        level = ((average - 1.0) / 4.0 * count).floor().clamp(0, count - 1).long()

        prototypes = self.level_prototypes
        if prototypes.shape[-1] != pooled.shape[-1]:
            raise ValueError("level_prototypes 차원이 pooled 표현과 다릅니다")
        with torch.no_grad():
            momentum = float(config.level_prototype_momentum)
            for index in torch.unique(level).tolist():
                mean = pooled[level == index].mean(dim=0)
                if prototypes[index].abs().sum() == 0:
                    prototypes[index] = mean
                else:
                    prototypes[index] = (
                        momentum * prototypes[index] + (1.0 - momentum) * mean
                    )
            ready = prototypes.abs().sum(dim=-1) > 0
        if int(ready.sum()) < 2:
            # prototype이 하나뿐이면 밀어낼 대상이 없어 loss가 정의되지 않는다.
            return None
        anchors = F.normalize(prototypes[ready].float(), p=2, dim=-1)
        # 음의 제곱거리를 logit으로 쓰는 prototypical classification.
        distance = torch.cdist(pooled, anchors).square()
        logits = -distance / float(config.level_prototype_temperature)
        # ready로 걸러 낸 뒤의 색인으로 target을 옮긴다.
        mapping = torch.full(
            (count,), -1, dtype=torch.long, device=pooled.device
        )
        mapping[ready] = torch.arange(int(ready.sum()), device=pooled.device)
        target = mapping[level]
        valid = target >= 0
        if not bool(valid.any()):
            return None
        return weight * F.cross_entropy(logits[valid], target[valid])

    def prompt_adversary_loss(
        self, features: torch.Tensor, prompt_group_ids: torch.Tensor | None
    ) -> torch.Tensor | None:
        """문항 판별 cross-entropy. backbone에는 뒤집힌 gradient가 흐른다.

        `features`가 trait 축을 가진 경우(예: essay_attention pooling) trait 평균을
        쓴다. 어느 trait의 표현에서 문항을 지울지 고르는 것은 별개의 실험이고,
        기본 pooling=mean에서는 세 trait이 같은 벡터라 차이가 없다.
        """

        adversary = getattr(self, "prompt_adversary", None)
        if adversary is None or prompt_group_ids is None:
            return None
        config = self.regression_config
        pooled = features if features.ndim == 2 else features.mean(dim=1)
        labels = prompt_group_ids.reshape(-1).long()
        # 문항 표기가 없는 행(-1)과 class 범위를 넘는 행은 제외한다. 조용히 clamp하면
        # 서로 다른 문항이 같은 class로 뭉쳐 판별기가 엉뚱한 것을 배운다.
        valid = (labels >= 0) & (labels < config.prompt_adversary_classes)
        if not bool(valid.any()):
            return None
        reversed_features = reverse_gradient(
            pooled[valid].float(), config.prompt_adversary_weight
        )
        logits = adversary(reversed_features)
        return F.cross_entropy(logits, labels[valid])

    def _training_loss(
        self,
        scores: torch.Tensor,
        probabilities: torch.Tensor | None,
        labels: torch.Tensor,
        *,
        prompt_group_ids: torch.Tensor | None = None,
        average_labels: torch.Tensor | None = None,
        detail_scores: torch.Tensor | None = None,
        detail_probabilities: torch.Tensor | None = None,
        detail_logits: torch.Tensor | None = None,
        criterion_scores: torch.Tensor | None = None,
        criterion_distributions: torch.Tensor | None = None,
        criterion_mask: torch.Tensor | None = None,
        criterion_distribution_mask: torch.Tensor | None = None,
        official_rater_labels: torch.Tensor | None = None,
        official_rater_mask: torch.Tensor | None = None,
        detail_rater_ids: torch.Tensor | None = None,
        detail_rater_labels: torch.Tensor | None = None,
        detail_rater_mask: torch.Tensor | None = None,
        direct_scores: torch.Tensor | None = None,
        paragraph_boundary_logits: torch.Tensor | None = None,
        paragraph_boundary_labels: torch.Tensor | None = None,
        paragraph_boundary_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Combine enabled objectives; every optional method is visible here."""

        config = self.regression_config
        # 꼬리 가중은 **점별(pointwise) 목표**에만 건다. 순위 로스는 batch 안의
        # 상대 비교라 표본 가중을 곱하면 무엇을 최적화하는지가 흐려진다.
        tail_weights = sample_tail_weights(labels, config)
        if tail_weights is None and config.group_dro_step_size > 0:
            # GroupDRO도 같은 element_weights 경로를 쓴다. config 검증이 둘의 동시
            # 사용을 막으므로 여기서 경쟁하지 않는다.
            tail_weights = group_dro_sample_weights(
                torch.square(scores - labels),
                prompt_group_ids,
                self.group_dro_weights,
                config.group_dro_step_size,
            )
        weighted_mse = _weighted_trait_mean(
            torch.square(scores - labels),
            config,
            ranking=False,
            element_weights=tail_weights,
        )
        loss = config.mse_loss_weight * weighted_mse

        if probabilities is not None:
            if config.score_head == "regression_ordinal":
                if config.ordinal_loss_weight > 0:
                    targets = ordinal_cumulative_targets(
                        labels, steps=config.ordinal_steps
                    )
                    ordinal_bce = F.binary_cross_entropy_with_logits(
                        probabilities.float(), targets, reduction="none"
                    ).mean(dim=-1)
                    loss = loss + config.ordinal_loss_weight * _weighted_trait_mean(
                        ordinal_bce, config, ranking=False
                    )
            elif config.distribution_loss_weight > 0:
                targets = (
                    trait_native_distribution_targets(labels)
                    if config.score_head == "trait_native_distribution"
                    else score_distribution_targets(
                        labels, smoothing=config.distribution_label_smoothing
                    )
                )
                cross_entropy = -(
                    targets * probabilities.clamp_min(1e-8).log()
                ).sum(dim=-1)
                loss = loss + config.distribution_loss_weight * _weighted_trait_mean(
                    cross_entropy, config, ranking=False, element_weights=tail_weights
                )

        if (
            config.trait_average_loss_weight > 0
            or config.trait_average_pairwise_weight > 0
        ):
            loss = loss + trait_average_objective(scores, labels, config)
        if config.pairwise_loss_weight > 0:
            loss = loss + config.pairwise_loss_weight * pairwise_ranking_loss(
                scores, labels, config
            )
        if config.listwise_loss_weight > 0:
            loss = loss + config.listwise_loss_weight * listwise_ranking_loss(
                scores, labels, config
            )
        if quantization_objective_active(config):
            loss = loss + quantization_aware_objective(
                scores,
                labels,
                average_labels,
                config,
                global_step=self._quantization_global_step,
            )
        if config.detail_head_mode != "none":
            if detail_scores is None or detail_logits is None:
                raise ValueError("활성 detail head의 학습 prediction이 없습니다")
            loss = loss + self._detail_training_loss(
                scores if direct_scores is None else direct_scores,
                detail_scores,
                detail_probabilities,
                detail_logits,
                criterion_scores=criterion_scores,
                criterion_distributions=criterion_distributions,
                criterion_mask=criterion_mask,
                criterion_distribution_mask=criterion_distribution_mask,
                official_rater_labels=official_rater_labels,
                official_rater_mask=official_rater_mask,
                detail_rater_ids=detail_rater_ids,
                detail_rater_labels=detail_rater_labels,
                detail_rater_mask=detail_rater_mask,
            )
        if config.paragraph_boundary_loss_weight > 0:
            if any(
                value is None
                for value in (
                    paragraph_boundary_logits,
                    paragraph_boundary_labels,
                    paragraph_boundary_mask,
                )
            ):
                raise ValueError(
                    "paragraph boundary multi-task에는 logits/labels/mask가 필요합니다"
                )
            assert paragraph_boundary_logits is not None
            assert paragraph_boundary_labels is not None
            assert paragraph_boundary_mask is not None
            boundary_loss = self._paragraph_boundary_training_loss(
                paragraph_boundary_logits,
                paragraph_boundary_labels,
                paragraph_boundary_mask,
            )
            loss = loss + config.paragraph_boundary_loss_weight * boundary_loss
        return loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        average_labels: torch.Tensor | None = None,
        # 문항별 지표 집계에만 쓰이는 입력. Trainer가 batch의 모든 key를 forward로
        # 넘기므로 여기서 받아 두되 loss에는 쓰지 않는다. average_labels와 같은 규약이다.
        prompt_group_ids: torch.Tensor | None = None,
        essay_mask: torch.Tensor | None = None,
        analysis_mask: torch.Tensor | None = None,
        sentence_ids: torch.Tensor | None = None,
        paragraph_ids: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        shared_pooling_mask: torch.Tensor | None = None,
        criterion_anchor_positions: torch.Tensor | None = None,
        prompt_ids: torch.Tensor | None = None,
        criterion_scores: torch.Tensor | None = None,
        criterion_distributions: torch.Tensor | None = None,
        criterion_mask: torch.Tensor | None = None,
        criterion_distribution_mask: torch.Tensor | None = None,
        official_rater_labels: torch.Tensor | None = None,
        official_rater_mask: torch.Tensor | None = None,
        detail_rater_ids: torch.Tensor | None = None,
        detail_rater_labels: torch.Tensor | None = None,
        detail_rater_mask: torch.Tensor | None = None,
        paragraph_boundary_labels: torch.Tensor | None = None,
        paragraph_boundary_mask: torch.Tensor | None = None,
        return_probabilities: bool = False,
        return_detail_predictions: bool = False,
    ) -> dict[str, torch.Tensor]:
        # ``average_labels``는 기본/역사 경로에서는 metric-only다. 명시적인
        # quantized mean-first loss를 켠 경우에만 아래 training loss로 전달된다.
        use_paragraph_boundary_auxiliary = (
            self.training
            and self.regression_config.paragraph_boundary_loss_weight > 0
        )
        encoded_features = self.encode(
            input_ids,
            attention_mask,
            essay_mask=essay_mask,
            analysis_mask=analysis_mask,
            sentence_ids=sentence_ids,
            paragraph_ids=paragraph_ids,
            token_type_ids=token_type_ids,
            shared_pooling_mask=shared_pooling_mask,
            criterion_anchor_positions=criterion_anchor_positions,
            return_token_hidden=use_paragraph_boundary_auxiliary,
        )
        token_hidden: torch.Tensor | None = None
        if isinstance(encoded_features, tuple) and len(encoded_features) == 3:
            features, criterion_anchor_hidden, token_hidden = encoded_features
        elif isinstance(encoded_features, tuple):
            features, criterion_anchor_hidden = encoded_features
        else:
            features = encoded_features
            criterion_anchor_hidden = None
        direct_scores, direct_probabilities = self.predictions_from_features(
            features, prompt_ids=prompt_ids
        )

        detail_scores: torch.Tensor | None = None
        detail_probabilities: torch.Tensor | None = None
        detail_logits: torch.Tensor | None = None
        needs_detail = self.regression_config.detail_head_mode != "none" and (
            self.training
            or return_detail_predictions
            or self.regression_config.detail_final_source == "criterion"
        )
        if needs_detail:
            detail_features = self._criterion_readout_features(
                features, criterion_anchor_hidden
            )
            (
                detail_scores,
                detail_probabilities,
                detail_logits,
            ) = self.detail_predictions_from_features(detail_features)

        if self.regression_config.detail_final_source == "criterion":
            if detail_scores is None:
                raise ValueError(
                    "criterion final score에 필요한 detail prediction이 없습니다"
                )
            scores = self.aggregate_detail_scores(detail_scores)
        else:
            scores = direct_scores

        paragraph_boundary_logits: torch.Tensor | None = None
        if use_paragraph_boundary_auxiliary:
            if self.paragraph_boundary_head is None or token_hidden is None:
                raise ValueError("활성 paragraph boundary head의 token hidden이 없습니다")
            paragraph_boundary_logits = self.paragraph_boundary_head(
                token_hidden.float()
            ).squeeze(-1)

        output = {"scores": scores}
        # probabilities는 direct 3-trait distribution의 부가 출력이다. criterion
        # aggregate final에는 하나의 고유한 3-trait class 분포가 없으므로 노출하지 않는다.
        if (
            return_probabilities
            and self.regression_config.detail_final_source == "direct"
            and direct_probabilities is not None
        ):
            output["probabilities"] = direct_probabilities
        if return_detail_predictions:
            if detail_scores is None:
                raise ValueError(
                    "detail prediction을 요청하려면 detail_head_mode를 켜야 합니다"
                )
            output["detail_scores"] = detail_scores
            if detail_probabilities is not None:
                output["detail_probabilities"] = detail_probabilities
        if labels is not None:
            labels = labels.float()
            if self.training:
                loss = self._training_loss(
                    scores,
                    direct_probabilities,
                    labels,
                    prompt_group_ids=prompt_group_ids,
                    average_labels=average_labels,
                    detail_scores=detail_scores,
                    detail_probabilities=detail_probabilities,
                    detail_logits=detail_logits,
                    criterion_scores=criterion_scores,
                    criterion_distributions=criterion_distributions,
                    criterion_mask=criterion_mask,
                    criterion_distribution_mask=criterion_distribution_mask,
                    official_rater_labels=official_rater_labels,
                    official_rater_mask=official_rater_mask,
                    detail_rater_ids=detail_rater_ids,
                    detail_rater_labels=detail_rater_labels,
                    detail_rater_mask=detail_rater_mask,
                    direct_scores=direct_scores,
                    paragraph_boundary_logits=paragraph_boundary_logits,
                    paragraph_boundary_labels=paragraph_boundary_labels,
                    paragraph_boundary_mask=paragraph_boundary_mask,
                )
                # C7: 문항 적대적 항. weight=0이면 helper가 None을 돌려주고 loss가
                # 문자 그대로 바뀌지 않는다. training일 때만, 즉 validation 지표에는
                # 절대 섞이지 않는다.
                adversary = self.prompt_adversary_loss(features, prompt_group_ids)
                if adversary is not None:
                    loss = loss + adversary
                # 수준 prototype contrastive. training일 때만 걸리므로 validation
                # 지표에는 섞이지 않는다.
                prototype = self.level_prototype_loss(features, labels)
                if prototype is not None:
                    loss = loss + prototype
            else:
                # validation 지표/베스트 checkpoint 기준은 기존과 동일한 MSE다.
                loss = F.mse_loss(scores, labels)
            output["loss"] = loss
        return output


@dataclass
class LoadedRegressionModel:
    scorer: RegressionScorer
    tokenizer: Any
    config: RegressionConfig
    # Submission-only path. Training and historical checkpoint loading keep this
    # ``None`` and therefore retain their original ``AutoModel`` behavior.
    causal_lm: nn.Module | None = None


# Model construction and checkpoints ----------------------------------------
def _adapter_targets(model: nn.Module, config: RegressionConfig) -> list[str]:
    """Choose familiar decoder or encoder attention projection suffixes."""

    if config.lora_targets == "all_linear":
        # build_model()은 multimodal container에서 text model을 먼저 분리한 뒤 이
        # helper를 호출한다. 따라서 여기서 찾는 Linear는 vision tower나 score
        # head가 아니라 language backbone의 attention/MLP projection뿐이다.
        targets = [
            name
            for name, module in model.named_modules()
            if name and isinstance(module, nn.Linear)
        ]
        if not targets:
            raise ValueError("language backbone에서 LoRA용 Linear module을 찾지 못했습니다")
        return targets

    if config.lora_targets != "auto":
        targets = [
            item.strip() for item in config.lora_targets.split(",") if item.strip()
        ]
        if not targets:
            raise ValueError("lora_targets에 module 이름이 없습니다")
        return targets

    decoder_targets = lora_target_modules(config.lora_include_mlp)
    encoder_target_groups = [
        ["query", "key", "value"],  # BERT/DeBERTa/RoBERTa 계열
        ["Wqkv", "Wo"],  # ModernBERT 계열
        ["q", "k", "v", "o"],  # T5/ByT5 encoder 계열
    ]
    candidates = (
        [*encoder_target_groups, decoder_targets]
        if config.backbone_type == "encoder"
        else [decoder_targets, *encoder_target_groups]
    )
    module_names = [name for name, _ in model.named_modules()]
    for targets in candidates:
        if any(
            name == target or name.endswith(f".{target}")
            for name in module_names
            for target in targets
        ):
            return targets
    raise ValueError(
        "LoRA target module을 자동으로 찾지 못했습니다. "
        "--lora-targets에 comma-separated suffix를 지정하세요"
    )


def _prepare_for_adapter_training(
    model: nn.Module, config: RegressionConfig
) -> nn.Module:
    try:
        from peft import prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError("LoRA/QLoRA에는 peft가 필요합니다") from exc
    if config.use_qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
    if config.gradient_checkpointing:
        # checkpointing은 activation memory를 줄이는 대신 forward를 다시 계산한다.
        # VRAM이 남는 실험은 --no-gradient-checkpointing으로 이 branch만 끌 수 있다.
        # PEFT는 ``is_loaded_in_4bit``가 있는 model에만 이 둘을 켠다. Qwen/Gemma
        # multimodal container에서 분리한 실제 text child에는 marker가 없으므로,
        # config 계약을 직접 적용해야 activation이 95GiB까지 커지는 silent fallback을
        # 막을 수 있다. 이미 PEFT가 켠 일반 model에도 아래 호출은 idempotent다.
        enable_checkpointing = getattr(model, "gradient_checkpointing_enable", None)
        enable_input_grads = getattr(model, "enable_input_require_grads", None)
        if not callable(enable_checkpointing) or not callable(enable_input_grads):
            raise RuntimeError(
                "gradient_checkpointing=True인데 text backbone이 checkpointing/input "
                "gradient API를 제공하지 않습니다"
            )
        checkpointing_indicator = getattr(model, "is_gradient_checkpointing", False)
        if callable(checkpointing_indicator):
            checkpointing_indicator = checkpointing_indicator()
        checkpointing_active = bool(
            checkpointing_indicator
            or getattr(model, "gradient_checkpointing_enabled", False)
            or any(
                bool(getattr(module, "gradient_checkpointing", False))
                for module in model.modules()
            )
        )
        if not checkpointing_active:
            enable_checkpointing()

        input_grads_active = bool(
            getattr(model, "input_grads_enabled", False)
            or getattr(model, "_require_grads_hook", None) is not None
            or bool(getattr(model, "_require_grads_hooks", ()))
        )
        if not input_grads_active:
            enable_input_grads()

        checkpointing_indicator = getattr(model, "is_gradient_checkpointing", False)
        if callable(checkpointing_indicator):
            checkpointing_indicator = checkpointing_indicator()
        checkpointing_active = bool(
            checkpointing_indicator
            or getattr(model, "gradient_checkpointing_enabled", False)
            or any(
                bool(getattr(module, "gradient_checkpointing", False))
                for module in model.modules()
            )
        )
        input_grads_active = bool(
            getattr(model, "input_grads_enabled", False)
            or getattr(model, "_require_grads_hook", None) is not None
            or bool(getattr(model, "_require_grads_hooks", ()))
        )
        if not checkpointing_active or not input_grads_active:
            raise RuntimeError(
                "gradient checkpointing 활성화 검증 실패: "
                f"checkpointing={checkpointing_active}, "
                f"input_grads={input_grads_active}"
            )
    model_config = getattr(model, "config", None)
    if model_config is not None and hasattr(model_config, "use_cache"):
        model_config.use_cache = False
    return model


def _attach_new_adapter(model: nn.Module, config: RegressionConfig) -> nn.Module:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("LoRA/QLoRA에는 peft가 필요합니다") from exc
    model = _prepare_for_adapter_training(model, config)
    return get_peft_model(
        model,
        LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=_adapter_targets(model, config),
            bias="none",
            task_type="FEATURE_EXTRACTION",
        ),
    )


def _load_trainable_adapter(
    model: nn.Module, config: RegressionConfig
) -> nn.Module:
    """Load preadapted LoRA weights while leaving the current score head fresh."""

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("LoRA adapter warm start에는 peft가 필요합니다") from exc
    model = _prepare_for_adapter_training(model, config)
    return PeftModel.from_pretrained(
        model,
        config.initial_lora_adapter,
        is_trainable=True,
    )


def build_model(
    config: RegressionConfig,
    *,
    device: torch.device,
    adapter_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    load_in_4bit: bool | None = None,
    retain_causal_lm: bool = False,
) -> LoadedRegressionModel:
    config.validate()
    spec = resolve_model(config.model_id)
    model_id = spec.model_id
    # train.resolved_config (and legacy config migration) already records the
    # catalog default. Runtime must honor an explicit --no-trust-remote-code.
    trust_remote_code = config.trust_remote_code
    quantized = config.use_qlora if load_in_4bit is None else load_in_4bit
    if quantized and device.type != "cuda":
        raise RuntimeError("4-bit model loading에는 CUDA GPU가 필요합니다")
    try:
        from transformers import (
            AutoModel,
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        raise RuntimeError("transformers가 필요합니다") from exc

    tokenizer_source = str(tokenizer_path) if tokenizer_path is not None else model_id
    tokenizer_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
    }
    if tokenizer_path is None:
        tokenizer_kwargs["revision"] = config.model_revision
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer에 pad 또는 eos token이 필요합니다")
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "revision": config.model_revision,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    compatible_config = load_compatible_backbone_config(
        model_id,
        trust_remote_code=trust_remote_code,
        revision=config.model_revision,
    )
    if compatible_config is not None:
        model_kwargs["config"] = compatible_config
    if quantized:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": device.index or 0}

    if retain_causal_lm and config.backbone_type != "decoder":
        raise ValueError(
            "제출 CausalLM 공유 경로는 decoder checkpoint만 지원합니다: "
            f"backbone_type={config.backbone_type!r}"
        )
    model_factory = AutoModelForCausalLM if retain_causal_lm else AutoModel
    container = model_factory.from_pretrained(model_id, **model_kwargs)
    causal_lm: nn.Module | None = container if retain_causal_lm else None
    if retain_causal_lm and bool(getattr(container.config, "is_encoder_decoder", False)):
        raise ValueError("제출 CausalLM 공유 경로는 encoder-decoder 모델을 지원하지 않습니다")
    # T5처럼 encoder-decoder인 AutoModel은 decoder_input_ids 없이 전체 forward를
    # 호출할 수 없다. encoder baseline에서는 encoder 부분만 뽑아 기존 scorer의
    # input_ids/attention_mask 인터페이스를 그대로 사용한다.
    is_encoder_decoder = bool(
        getattr(getattr(container, "config", None), "is_encoder_decoder", False)
    )
    if (
        config.backbone_type == "encoder"
        and is_encoder_decoder
        and hasattr(container, "get_encoder")
    ):
        backbone = container.get_encoder()
    else:
        backbone = resolve_text_model(container)
    if retain_causal_lm and backbone is container:
        raise RuntimeError(
            f"{model_id}: CausalLM wrapper에서 독립 text backbone을 찾지 못했습니다"
        )
    hidden_size = hidden_size_of(backbone)
    # encoder tokenizer가 알리는 512/8192 등의 실제 한계를 보존한다. tokenizer가
    # 무제한 sentinel을 쓸 때는 model config의 position limit로 보완한다.
    position_limit = getattr(backbone.config, "max_position_embeddings", None)
    if isinstance(position_limit, int) and 0 < position_limit < 1_000_000:
        tokenizer_limit = getattr(tokenizer, "model_max_length", None)
        if not isinstance(tokenizer_limit, int) or tokenizer_limit >= 1_000_000:
            tokenizer.model_max_length = position_limit
        else:
            tokenizer.model_max_length = min(tokenizer_limit, position_limit)

    if adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("adapter reload에는 peft가 필요합니다") from exc
        backbone = PeftModel.from_pretrained(backbone, adapter_path, is_trainable=False)
        verify_loaded_peft_adapter_exact(backbone, adapter_path)
    elif config.initial_lora_adapter:
        backbone = _load_trainable_adapter(backbone, config)
    elif config.training_mode != "head_only":
        backbone = _attach_new_adapter(backbone, config)
    else:
        backbone.requires_grad_(False)

    if not quantized:
        # CausalLM 컨테이너를 보존하는 제출 경로에서는 text backbone과 untied LM
        # head를 함께 옮긴다. backbone은 같은 module 객체를 참조하므로 두 벌이 아니다.
        (causal_lm if causal_lm is not None else backbone).to(device)
    scorer = RegressionScorer(backbone, hidden_size, config)
    # Quantized backbone 전체에 `.to()`를 호출할 수 없으므로 새 score-side
    # module과 scalar parameter만 명시적으로 backbone 장치로 옮긴다.
    for name, module in scorer.named_children():
        if name != "backbone":
            module.to(device)
    # layer_mix/prompt_bias/organization 위치·transition weight는 child module 안이 아닌
    # scorer 최상위 Parameter다. QLoRA backbone에 `.to()`하지 않으면서 이 작은
    # score-side parameter만 GPU로 보내야 bias/FML mode도 device가 일치한다.
    for _, parameter in scorer.named_parameters(recurse=False):
        parameter.data = parameter.data.to(device)
    if quantized:
        # Trainer/Accelerate must not call .to() on an already-dispatched 4-bit model.
        for name in ("hf_device_map", "quantization_method", "is_loaded_in_4bit"):
            value = getattr(backbone, name, None)
            if value is None:
                value = getattr(container, name, None)
            if value is not None:
                setattr(scorer, name, value)
        if getattr(scorer, "hf_device_map", None) is None:
            scorer.hf_device_map = {"": device.index or 0}
    initial_stage = "joint" if config.training_mode == "lora_only" else "head"
    scorer.set_training_stage(initial_stage)
    if causal_lm is not None:
        causal_lm.requires_grad_(False)
        causal_lm.eval()
    return LoadedRegressionModel(
        scorer=scorer,
        tokenizer=tokenizer,
        config=config,
        causal_lm=causal_lm,
    )


def save_checkpoint(loaded: LoadedRegressionModel, directory: str | Path) -> Path:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    save_config(loaded.config, output / "config.json")
    torch.save(
        {
            "schema_version": 2,
            "scoring_state": loaded.scorer.scoring_state_dict(),
        },
        output / "heads.pt",
    )
    loaded.tokenizer.save_pretrained(output / "tokenizer")
    has_adapter = loaded.config.training_mode != "head_only"
    if has_adapter:
        loaded.scorer.backbone.save_pretrained(
            output / "adapter", save_embedding_layers=False
        )
    write_json(
        output / "manifest.json",
        {
            "model_id": loaded.config.model_id,
            "model_slug": loaded.config.model_slug,
            "model_revision": loaded.config.model_revision,
            "model_source_run": loaded.config.model_source_run,
            "training_mode": loaded.config.training_mode,
            "use_qlora": loaded.config.use_qlora,
            "backbone_type": loaded.config.backbone_type,
            "input_format": loaded.config.input_format,
            "rubric_profile": loaded.config.rubric_profile,
            "criterion_readout": loaded.config.criterion_readout,
            "criterion_anchor_count": (
                len(DETAIL_CRITERIA)
                if loaded.config.criterion_readout == "textual_anchor_residual"
                else 0
            ),
            "essay_surface": loaded.config.essay_surface,
            "head_type": loaded.config.head_type,
            "prompt_head_mode": loaded.config.prompt_head_mode,
            "prompt_head_traits": loaded.config.prompt_head_traits,
            "prompt_head_weight": loaded.config.prompt_head_weight,
            "prompt_count": len(loaded.config.prompt_registry),
            "organization_pooling": loaded.config.organization_pooling,
            "score_head": loaded.config.score_head,
            "score_values": [1, 2, 3, 4, 5],
            "distribution_loss_weight": (loaded.config.distribution_loss_weight),
            "distribution_label_smoothing": (
                loaded.config.distribution_label_smoothing
            ),
            "ordinal_head": {
                "steps": loaded.config.ordinal_steps,
                "blend_weight": loaded.config.ordinal_blend_weight,
                "loss_weight": loaded.config.ordinal_loss_weight,
                "active": loaded.config.score_head == "regression_ordinal",
            },
            "detail_head_mode": loaded.config.detail_head_mode,
            "detail_final_source": loaded.config.detail_final_source,
            "detail_criteria": list(DETAIL_CRITERIA),
            "detail_loss_weights": {
                "expected": loaded.config.detail_expected_loss_weight,
                "distribution": loaded.config.detail_distribution_loss_weight,
                "halfstep": loaded.config.detail_halfstep_loss_weight,
                "rater_set": loaded.config.detail_rater_set_loss_weight,
                "hierarchy": loaded.config.detail_hierarchy_loss_weight,
                "rater": loaded.config.detail_rater_loss_weight,
            },
            "detail_rater_registry_count": len(loaded.config.detail_rater_registry),
            "mixed_head_weight": loaded.config.mixed_head_weight,
            "head_hidden_size": loaded.config.head_hidden_size,
            "pooling": loaded.config.pooling,
            "normalize_features": loaded.config.normalize_features,
            "layer_aggregation": loaded.config.layer_aggregation,
            "last_n_layers": loaded.config.last_n_layers,
            "trait_loss_weights": {
                "content": loaded.config.content_loss_weight,
                "organization": loaded.config.organization_loss_weight,
                "expression": loaded.config.expression_loss_weight,
            },
            "ranking_trait_weights": {
                "content": loaded.config.ranking_content_weight,
                "organization": loaded.config.ranking_organization_weight,
                "expression": loaded.config.ranking_expression_weight,
            },
            "quantization_objective": {
                "rule": loaded.config.quantization_rule,
                "surrogate": loaded.config.quantization_surrogate,
                "target_rule": loaded.config.quantized_target_rule,
                "error_form": loaded.config.quantized_error_form,
                "trait_loss_weight": loaded.config.quantized_trait_loss_weight,
                "mean_loss_weight": loaded.config.quantized_mean_loss_weight,
                "pooled_loss_weight": loaded.config.quantized_pooled_loss_weight,
                "trait_rank_weight": loaded.config.quantized_trait_rank_weight,
                "mean_rank_weight": loaded.config.quantized_mean_rank_weight,
                "pooled_rank_weight": loaded.config.quantized_pooled_rank_weight,
                "temperature": loaded.config.quantization_temperature,
                "final_temperature": loaded.config.quantization_final_temperature,
                "allocation_temperature": (
                    loaded.config.quantization_allocation_temperature
                ),
                "final_allocation_temperature": (
                    loaded.config.quantization_final_allocation_temperature
                ),
                "rank_temperature": loaded.config.quantized_rank_temperature,
                "start_step": loaded.config.quantized_loss_start_step,
                "ramp_steps": loaded.config.quantized_loss_ramp_steps,
                "anneal_steps": loaded.config.quantization_anneal_steps,
            },
            "lora_targets": loaded.config.lora_targets,
            "initial_lora_adapter": loaded.config.initial_lora_adapter,
            "has_adapter": has_adapter,
            "traits": list(TRAITS),
        },
    )
    return output


def load_checkpoint(
    directory: str | Path,
    *,
    device: torch.device,
    load_in_4bit: bool | None = None,
    retain_causal_lm: bool = False,
) -> LoadedRegressionModel:
    checkpoint = Path(directory)
    config = load_config(checkpoint / "config.json")
    adapter = checkpoint / "adapter"
    expects_adapter = config.training_mode != "head_only"
    if expects_adapter and not adapter.is_dir():
        raise FileNotFoundError(f"LoRA checkpoint에 adapter 폴더가 없습니다: {adapter}")
    tokenizer = checkpoint / "tokenizer"
    if not tokenizer.is_dir():
        raise FileNotFoundError(f"checkpoint에 tokenizer 폴더가 없습니다: {tokenizer}")
    loaded = build_model(
        config,
        device=device,
        adapter_path=adapter if expects_adapter else None,
        tokenizer_path=tokenizer,
        load_in_4bit=load_in_4bit,
        retain_causal_lm=retain_causal_lm,
    )
    load_score_state(loaded.scorer, checkpoint / "heads.pt")
    loaded.scorer.requires_grad_(False)
    loaded.scorer.eval()
    return loaded


def verify_loaded_peft_adapter_exact(
    backbone: nn.Module,
    adapter_path: str | Path,
    *,
    adapter_name: str = "default",
) -> None:
    """첫 LoRA도 저장 tensor와 실제 메모리 tensor가 완전히 같은지 검증한다.

    ``PeftModel.from_pretrained``는 missing adapter key를 경고만 하고 계속 실행할 수 있다.
    제출에서는 일부가 무작위 초기값인 adapter를 정상 모델처럼 서빙하면 안 되므로, 저장된
    state와 PEFT가 다시 내보내는 활성 adapter state의 key/shape/value를 모두 대조한다.
    """

    try:
        from peft.utils.save_and_load import (
            get_peft_model_state_dict,
            load_peft_weights,
        )
    except ImportError as exc:
        raise RuntimeError("adapter strict 검증에는 peft가 필요합니다") from exc

    expected = load_peft_weights(str(adapter_path), device="cpu")
    actual = get_peft_model_state_dict(backbone, adapter_name=adapter_name)
    expected_keys = set(expected)
    actual_keys = set(actual)
    missing = sorted(actual_keys - expected_keys)
    unexpected = sorted(expected_keys - actual_keys)
    if missing or unexpected:
        raise ValueError(
            "LoRA adapter tensor key가 불완전합니다: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )

    for key in sorted(expected_keys):
        saved = expected[key].detach().cpu()
        loaded = actual[key].detach().cpu()
        if saved.shape != loaded.shape:
            raise ValueError(
                f"LoRA adapter tensor shape가 다릅니다: {key}: "
                f"saved={tuple(saved.shape)}, loaded={tuple(loaded.shape)}"
            )
        if not torch.equal(saved.to(dtype=loaded.dtype), loaded):
            raise ValueError(f"LoRA adapter tensor 값이 다릅니다: {key}")


def load_score_state(
    scorer: "RegressionScorer",
    heads_path: str | Path,
    *,
    allow_fresh: Sequence[str] = (),
) -> None:
    """`heads.pt`의 score-side weight를 scorer에 싣는다.

    `load_checkpoint`와 공유 backbone loader가 **같은 함수**를 쓰게 하려고 분리했다. 두 경로가
    head를 다르게 읽으면 같은 checkpoint에서 다른 점수가 나온다.

    `allow_fresh`는 checkpoint에 없어도 되는 score-side parameter 이름을 **명시적으로**
    나열한다. pooling을 바꿔 warm start할 때(예: mean으로 학습된 c02에서
    `analysis_attention_mix`로 이어 학습) 새 pooling parameter는 원래 존재할 수 없다.
    기본값이 비어 있으므로 기존 경로의 "하나라도 빠지면 실패" 계약은 그대로다.
    """

    allowed_fresh = set(allow_fresh)

    state = torch.load(heads_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and state.get("schema_version") == 2:
        scoring_state = state.get("scoring_state")
        if not isinstance(scoring_state, dict):
            raise ValueError("heads.pt에 scoring_state가 없습니다")
        incompatible = scorer.load_state_dict(scoring_state, strict=False)
        missing_score_state = [
            name
            for name in incompatible.missing_keys
            if not name.startswith("backbone.")
            and name not in allowed_fresh
            # 학습 전용 보조 module은 저장하지 않으므로 로드 때 항상 missing이다.
            # 새로 초기화되는 것이 정상이고, 추론에서는 호출되지 않는다.
            and not name.startswith(RegressionScorer.TRAINING_ONLY_STATE_PREFIXES)
        ]
        if missing_score_state or incompatible.unexpected_keys:
            raise ValueError(
                "score-side checkpoint 구조가 config와 다릅니다: "
                f"missing={missing_score_state}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
    else:
        # 기존 baseline checkpoint는 heads.state_dict()만 저장했다. 기본
        # last-layer/mean 구조에서는 그대로 읽어 이전 실험을 깨지 않는다.
        scorer.heads.load_state_dict(state)


# 공유 backbone 다중 어댑터 -----------------------------------------------------
def _load_peft_adapter_strict(
    backbone: nn.Module,
    adapter_path: str | Path,
    *,
    adapter_name: str,
    context: str,
    key_mapping: dict[str, str] | None = None,
) -> Any:
    """Load one inference adapter and reject every missing/unexpected tensor.

    PEFT supports adapter-specific rank, alpha, and target-module sets.  Those
    differences are therefore not load errors.  State-dict incompatibility is:
    accepting even one missing LoRA tensor would silently run a partly random
    adapter.  A failed load may already have injected the named adapter, so it
    is deleted before propagating the error.
    """

    configs = getattr(backbone, "peft_config", {})
    if adapter_name in configs:
        raise ValueError(f"adapter 이름이 이미 존재합니다: {adapter_name}")

    def rollback() -> None:
        if adapter_name not in getattr(backbone, "peft_config", {}):
            return
        if not hasattr(backbone, "delete_adapter"):
            raise RuntimeError(
                f"{context}: 실패한 adapter를 제거할 delete_adapter가 없습니다"
            )
        backbone.delete_adapter(adapter_name)

    try:
        result = backbone.load_adapter(
            str(adapter_path),
            adapter_name=adapter_name,
            is_trainable=False,
            key_mapping=key_mapping,
        )
        missing = list(getattr(result, "missing_keys", ()) or ())
        unexpected = list(getattr(result, "unexpected_keys", ()) or ())
        if missing or unexpected:
            raise ValueError(
                f"{context} tensor key가 backbone과 호환되지 않습니다: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return result
    except Exception:
        try:
            rollback()
        except Exception as rollback_exc:
            raise RuntimeError(
                f"{context}: adapter load 실패 뒤 rollback도 실패했습니다"
            ) from rollback_exc
        raise


@dataclass
class SharedMember:
    """공유 backbone 위의 한 앙상블 구성원."""

    name: str
    adapter_name: str
    scorer: "RegressionScorer"
    tokenizer: Any
    config: RegressionConfig
    checkpoint: Path


class SharedBackboneEnsemble:
    """backbone을 한 번만 올리고 LoRA 어댑터만 갈아 끼우는 앙상블.

    핵심 사실 두 개에 의존한다.

    1. LoRA 주입은 backbone 모듈 트리의 `nn.Linear`를 제자리에서 교체한다. 따라서 어댑터 여러
       개를 올려 두고 `set_adapter`로 활성 어댑터만 바꾸면, 같은 가중치 한 벌로 서로 다른 모델
       N개를 순차 실행할 수 있다.
    2. score head(`heads.pt`)는 backbone과 별개인 작은 모듈이다. 구성원마다 자기 head를 갖고
       공유 backbone을 참조하면 된다.

    그래서 VRAM은 `backbone 1개 + head N개`이고, head는 보통 수 MB다. 멤버마다 backbone을
    새로 로드하는 것과 비교해 7B 기준 약 14.5GiB를 멤버당 절약한다.

    `with_lm_head=True`면 `AutoModelForCausalLM`으로 올려 같은 가중치에서 생성도 한다. 근거
    생성 어댑터를 같은 backbone에 얹을 수 있다는 뜻이다. decoder 모델의 `AutoModel`과
    `AutoModelForCausalLM.model`은 같은 transformer이므로 점수는 달라지지 않는다.
    """

    def __init__(
        self,
        backbone: nn.Module,
        members: list[SharedMember],
        *,
        causal_lm: nn.Module | None = None,
    ) -> None:
        self.backbone = backbone
        self.members = members
        self.causal_lm = causal_lm
        self._active: str | None = None

    # --- 활성 어댑터 -----------------------------------------------------
    def activate(self, adapter_name: str) -> None:
        """활성 어댑터를 바꾼다. 이미 활성이면 아무 일도 하지 않는다."""

        if self._active == adapter_name:
            return
        if not hasattr(self.backbone, "set_adapter"):
            raise RuntimeError("공유 backbone이 PeftModel이 아니어서 어댑터를 바꿀 수 없습니다")
        self.backbone.set_adapter(adapter_name)
        self._active = adapter_name

    @contextlib.contextmanager
    def using(self, adapter_name: str) -> Iterator[None]:
        """어댑터를 켜고 끝나면 이전 어댑터로 되돌린다."""

        previous = self._active
        self.activate(adapter_name)
        try:
            yield
        finally:
            if previous is not None and previous != adapter_name:
                self.activate(previous)

    @property
    def adapter_names(self) -> list[str]:
        return [member.adapter_name for member in self.members]

    @property
    def active_adapter(self) -> str | None:
        """Runner-visible adapter state used by submission restore guards."""

        return self._active

    def member(self, name: str) -> SharedMember:
        for candidate in self.members:
            if candidate.name == name:
                return candidate
        raise KeyError(f"구성원을 찾을 수 없습니다: {name}")

    def load_causal_adapter(
        self,
        adapter_path: str | Path,
        *,
        adapter_name: str = "rationale",
    ) -> str:
        """Load a CausalLM LoRA on the exact score text backbone, fail-closed.

        Score adapters were trained on ``AutoModel`` and are saved below
        ``base_model.model.layers``. Rationale adapters were trained on
        ``AutoModelForCausalLM`` and therefore contain one additional ``model``
        namespace. PEFT's explicit ``key_mapping`` removes exactly that level;
        missing or unexpected tensors are never accepted.

        PEFT stores rank/alpha/targets per adapter.  The submitted expanded-Y1
        score adapter is r32/a64 attention+MLP while the rationale adapter is
        r16/a32 attention-only.  Different rank/alpha are valid, and the
        rationale targets are allowed when they are a subset of the score
        targets.  The unverified reverse direction (new rationale-only target
        injection) remains fail-closed.
        """

        if self.causal_lm is None:
            raise RuntimeError("CausalLM 컨테이너 없이 생성 어댑터를 공유할 수 없습니다")
        if not hasattr(self.backbone, "load_adapter"):
            raise RuntimeError("공유 score backbone이 PeftModel이 아닙니다")
        if adapter_name in getattr(self.backbone, "peft_config", {}):
            raise ValueError(f"adapter 이름이 이미 존재합니다: {adapter_name}")

        path = Path(adapter_path)
        config_path = path / "adapter_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"근거 adapter_config.json이 없습니다: {config_path}")
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        expected_model = resolve_model(self.members[0].config.model_id).model_id
        if raw.get("base_model_name_or_path") != expected_model:
            raise ValueError(
                "근거 LoRA base model이 score backbone과 다릅니다: "
                f"{raw.get('base_model_name_or_path')!r} != {expected_model!r}"
            )
        if raw.get("peft_type") != "LORA" or raw.get("task_type") != "CAUSAL_LM":
            raise ValueError(
                "공유 근거 adapter는 CAUSAL_LM LoRA여야 합니다: "
                f"peft_type={raw.get('peft_type')!r}, task_type={raw.get('task_type')!r}"
            )
        if raw.get("modules_to_save") not in (None, []):
            raise ValueError("modules_to_save가 있는 근거 LoRA는 backbone 공유 대상이 아닙니다")

        rationale_targets_raw = raw.get("target_modules")
        if not isinstance(rationale_targets_raw, list) or not rationale_targets_raw:
            raise ValueError("공유 근거 adapter target_modules는 비어 있지 않은 list여야 합니다")
        if not all(isinstance(item, str) and item for item in rationale_targets_raw):
            raise ValueError("공유 근거 adapter target_modules에 빈/비문자 값이 있습니다")
        for field in ("r", "lora_alpha"):
            value = raw.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"공유 근거 adapter {field}는 양수여야 합니다: {value!r}")
        if raw.get("bias") != "none" or raw.get("lora_bias") not in (None, False):
            raise ValueError("bias를 바꾸는 근거 LoRA는 안전하게 adapter switch할 수 없습니다")
        if raw.get("rank_pattern") not in (None, {}):
            raise ValueError("rank_pattern이 있는 근거 LoRA는 제출 공유 경로에서 지원하지 않습니다")
        if raw.get("alpha_pattern") not in (None, {}):
            raise ValueError("alpha_pattern이 있는 근거 LoRA는 제출 공유 경로에서 지원하지 않습니다")

        score_config = self.backbone.peft_config[self.members[0].adapter_name]
        score_targets = set(getattr(score_config, "target_modules", ()) or ())
        rationale_targets = set(rationale_targets_raw)
        if not rationale_targets.issubset(score_targets):
            raise ValueError(
                "근거 LoRA target은 score LoRA target의 subset이어야 합니다: "
                f"score={sorted(score_targets)}, rationale={sorted(rationale_targets)}"
            )

        _load_peft_adapter_strict(
            self.backbone,
            path,
            adapter_name=adapter_name,
            context="근거 LoRA",
            # A CausalLM adapter has keys such as
            # base_model.model.model.layers.*; the score PeftModel wraps the
            # text model and expects base_model.model.layers.*.
            key_mapping={r"^model\.": ""},
        )

        loaded_config = self.backbone.peft_config.get(adapter_name)
        loaded_targets = set(getattr(loaded_config, "target_modules", ()) or ())
        if (
            loaded_config is None
            or getattr(loaded_config, "r", None) != raw.get("r")
            or getattr(loaded_config, "lora_alpha", None) != raw.get("lora_alpha")
            or loaded_targets != rationale_targets
        ):
            self.backbone.delete_adapter(adapter_name)
            raise ValueError(
                "PEFT가 읽은 근거 adapter config가 artifact와 다릅니다: "
                f"artifact=(r={raw.get('r')}, alpha={raw.get('lora_alpha')}, "
                f"targets={sorted(rationale_targets)}), "
                f"loaded=(r={getattr(loaded_config, 'r', None)}, "
                f"alpha={getattr(loaded_config, 'lora_alpha', None)}, "
                f"targets={sorted(loaded_targets)})"
            )
        self.backbone.eval()
        self.backbone.requires_grad_(False)
        return adapter_name


def _shared_backbone_identity(config: RegressionConfig) -> tuple[Any, ...]:
    """backbone 가중치를 공유할 수 있는지 판정하는 key."""

    return (
        resolve_model(config.model_id).model_id,
        config.model_revision,
        config.trust_remote_code,
        config.backbone_type,
        config.use_qlora,
    )


def load_shared_backbone_checkpoints(
    directories: Sequence[str | Path],
    *,
    device: torch.device,
    load_in_4bit: bool | None = None,
    with_lm_head: bool = False,
    names: Sequence[str] | None = None,
) -> SharedBackboneEnsemble:
    """같은 backbone을 쓰는 checkpoint 여러 개를 backbone 한 벌로 올린다.

    모든 checkpoint의 backbone identity가 같아야 한다. 다르면 즉시 중단한다. 조용히 첫 번째
    backbone으로 다른 checkpoint를 돌리면 그 구성원의 점수가 학습 때와 달라진다.
    """

    paths = [Path(directory) for directory in directories]
    if not paths:
        raise ValueError("checkpoint가 하나도 없습니다")
    configs = [load_config(path / "config.json") for path in paths]
    identities = {_shared_backbone_identity(config) for config in configs}
    if len(identities) != 1:
        raise ValueError(
            "backbone identity가 다른 checkpoint는 공유할 수 없습니다: "
            f"{sorted(str(item) for item in identities)}"
        )
    for path, config in zip(paths, configs, strict=True):
        if config.training_mode == "head_only":
            raise ValueError(
                f"{path}: head_only checkpoint에는 adapter가 없어 공유 대상이 아닙니다"
            )
        if not (path / "adapter").is_dir():
            raise FileNotFoundError(f"{path}: adapter 폴더가 없습니다")
    labels = list(names) if names is not None else [path.name for path in paths]
    if len(labels) != len(paths):
        raise ValueError("names 길이가 checkpoint 수와 다릅니다")
    if len(set(labels)) != len(labels):
        raise ValueError(f"구성원 이름이 중복됩니다: {labels}")

    # 첫 구성원은 기존 검증된 경로를 그대로 쓴다. 여기서 backbone과 PeftModel이 만들어진다.
    first = load_checkpoint(
        paths[0],
        device=device,
        load_in_4bit=load_in_4bit,
        retain_causal_lm=with_lm_head,
    )
    backbone = first.scorer.backbone
    if not hasattr(backbone, "load_adapter"):
        raise RuntimeError(
            "첫 checkpoint의 backbone이 PeftModel이 아닙니다. 어댑터 공유가 불가능합니다."
        )
    hidden_size = hidden_size_of(resolve_text_model(backbone))

    causal_lm = first.causal_lm
    if with_lm_head and causal_lm is None:
        raise RuntimeError("CausalLM 보존 로더가 생성 모델을 반환하지 않았습니다")

    default_name = "default"
    members = [
        SharedMember(
            name=labels[0],
            adapter_name=default_name,
            scorer=first.scorer,
            tokenizer=first.tokenizer,
            config=first.config,
            checkpoint=paths[0],
        )
    ]
    for index in range(1, len(paths)):
        path, config, label = paths[index], configs[index], labels[index]
        adapter_name = f"member_{index}"
        _load_peft_adapter_strict(
            backbone,
            path / "adapter",
            adapter_name=adapter_name,
            context=f"{path}: score LoRA",
        )
        tokenizer_dir = path / "tokenizer"
        if not tokenizer_dir.is_dir():
            raise FileNotFoundError(f"{path}: tokenizer 폴더가 없습니다")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir), trust_remote_code=config.trust_remote_code
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        scorer = RegressionScorer(backbone, hidden_size, config)
        for child_name, module in scorer.named_children():
            if child_name != "backbone":
                module.to(device)
        for _, parameter in scorer.named_parameters(recurse=False):
            parameter.data = parameter.data.to(device)
        load_score_state(scorer, path / "heads.pt")
        # 순서가 중요하다. `set_training_stage`는 공유 backbone의 requires_grad와 train 모드를
        # 건드리므로 반드시 `eval()` **앞에** 온다. 뒤에 오면 backbone이 train 모드로 남아 LoRA
        # dropout이 켜지고, 공유 backbone이므로 **다른 구성원 점수까지** 달라진다.
        scorer.set_training_stage(
            "joint" if config.training_mode == "lora_only" else "head"
        )
        scorer.requires_grad_(False)
        scorer.eval()
        members.append(
            SharedMember(
                name=label,
                adapter_name=adapter_name,
                scorer=scorer,
                tokenizer=tokenizer,
                config=config,
                checkpoint=path,
            )
        )
    # 공유 backbone은 한 번 더 명시적으로 추론 모드에 고정한다. 구성원 하나라도 train 모드를
    # 남기면 LoRA dropout이 켜져 모든 구성원의 점수가 흔들린다.
    backbone.eval()
    backbone.requires_grad_(False)
    for member in members:
        member.scorer.eval()
    ensemble = SharedBackboneEnsemble(backbone, members, causal_lm=causal_lm)
    ensemble.activate(default_name)
    return ensemble
