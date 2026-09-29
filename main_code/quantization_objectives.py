"""Deterministic, differentiable objectives for the submitted score surface.

The scoring model predicts three continuous values, while the submission emits
three integers in ``[1, 5]``.  This module keeps those two surfaces explicit:

* ``independent_half_up`` independently rounds every trait;
* ``average_matched`` first rounds the three-score total and then chooses the
  closest integer triple having that total.

All soft objectives enumerate the complete ``5 ** 3 == 125`` state space.  No
sampling, Gumbel noise, or global random state is used.  The returned state
probabilities therefore support both exact expected squared risk and a
hard-forward/soft-backward straight-through estimator (STE).

This file deliberately has no dependency on the trainer or model classes.  It
can be integrated into the current score head without changing inference, and
it can also be imported by later multi-stage training code.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Literal

import torch
from torch import Tensor


QuantizationRule = Literal["independent_half_up", "average_matched"]
ObjectiveScope = Literal["per_trait", "trait_average", "both"]
Reduction = Literal["none", "mean", "sum"]

QUANTIZATION_RULES: tuple[QuantizationRule, ...] = (
    "independent_half_up",
    "average_matched",
)
OBJECTIVE_SCOPES: tuple[ObjectiveScope, ...] = (
    "per_trait",
    "trait_average",
    "both",
)

# Canonical order: content changes slowest and expression changes fastest.  It
# agrees with itertools.product(range(1, 6), repeat=3) used in the existing
# postprocess oracle tests.
_CPU_INTEGER_STATES = torch.tensor(
    tuple(product(range(1, 6), repeat=3)), dtype=torch.int64
)


@dataclass(frozen=True)
class QuantizationRelaxation:
    """The complete deterministic relaxation of one prediction tensor.

    ``state_probabilities`` has shape ``[..., 125]``.  ``soft_scores`` is its
    expectation and ``hard_scores`` is the exact inference result.  ``scores``
    equals ``soft_scores`` normally; with STE it has hard values in the forward
    pass and the gradient of ``soft_scores`` in the backward pass.
    """

    state_probabilities: Tensor
    soft_scores: Tensor
    hard_scores: Tensor
    scores: Tensor


@dataclass(frozen=True)
class QuantizedMseRisks:
    """Unreduced squared risks on both plausible competition metric axes.

    ``per_trait`` has shape ``[..., 3]`` and ``trait_average`` has shape
    ``[...]``.  When STE is enabled their forward values are the risks of the
    hard inference scores, while their gradients come from exact expected risk
    under all 125 states.
    """

    per_trait: Tensor
    trait_average: Tensor


@dataclass(frozen=True)
class QuantizedRankInputs:
    """Quantized prediction tensors ready for a rank surrogate.

    A field excluded by ``scope`` is ``None``.  ``per_trait`` is ``[..., 3]``;
    ``trait_average`` is ``[...]``.  The caller can feed these values to its
    existing RankNet/SoftSpearman/listwise loss without duplicating the
    quantization rule.
    """

    per_trait: Tensor | None
    trait_average: Tensor | None


def integer_score_states(*, device: torch.device | str, dtype: torch.dtype) -> Tensor:
    """Return the canonical ``[125, 3]`` integer-score state tensor."""

    if not dtype.is_floating_point:
        raise TypeError("integer_score_states dtype must be floating point")
    return _CPU_INTEGER_STATES.to(device=device, dtype=dtype)


def _validate_predictions(predictions: Tensor) -> None:
    if not isinstance(predictions, Tensor):
        raise TypeError("predictions must be a torch.Tensor")
    if not predictions.dtype.is_floating_point:
        raise TypeError("predictions must have a floating-point dtype")
    if predictions.ndim < 1 or predictions.shape[-1] != 3:
        raise ValueError("predictions must have shape [..., 3]")


def _validate_rule(rule: str) -> QuantizationRule:
    if rule not in QUANTIZATION_RULES:
        raise ValueError(f"rule choices={QUANTIZATION_RULES}, got {rule!r}")
    return rule  # type: ignore[return-value]


def _validate_scope(scope: str) -> ObjectiveScope:
    if scope not in OBJECTIVE_SCOPES:
        raise ValueError(f"scope choices={OBJECTIVE_SCOPES}, got {scope!r}")
    return scope  # type: ignore[return-value]


def _work_dtype(dtype: torch.dtype) -> torch.dtype:
    # fp16/bfloat16 softmaxes become brittle at the small temperatures useful
    # for quantization.  Promotion is differentiable and the result remains on
    # the input device.
    return torch.float64 if dtype == torch.float64 else torch.float32


def _stable_temperature(value: float, dtype: torch.dtype, name: str) -> float:
    try:
        temperature = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a positive finite scalar") from exc
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    # Below this scale a float32 derivative can overflow even though the
    # stabilized softmax value is well defined.  Saturating the effective
    # temperature preserves the intended zero-temperature limit and guarantees
    # finite gradients.
    return max(temperature, 16.0 * torch.finfo(dtype).eps)


def _working_predictions(predictions: Tensor, *, clip_input: bool) -> Tensor:
    _validate_predictions(predictions)
    values = predictions.to(dtype=_work_dtype(predictions.dtype))
    return values.clamp(1.0, 5.0) if clip_input else values


@torch.no_grad()
def hard_quantize(
    predictions: Tensor,
    rule: QuantizationRule,
    *,
    clip_input: bool = True,
) -> Tensor:
    """Apply the exact inference quantizer with deterministic tie handling.

    The default ``clip_input=True`` matches :class:`ScorePostprocessor`.  For
    direct parity with ``utils.average_matched_integer_scores`` on deliberately
    out-of-range values, pass ``clip_input=False``; in-range model predictions
    are identical either way.

    Half ties are rounded upward.  Average-matched allocation ties are resolved
    content, then organization, then expression, exactly like NumPy's first
    ``argmax``/``argmin`` in the existing oracle.
    """

    _validate_predictions(predictions)
    rule = _validate_rule(rule)
    # NumPy's production oracle promotes to float64 before summing/rounding.
    # Doing the same also prevents autocast from changing boundary decisions.
    values = predictions.detach().to(dtype=torch.float64)
    if clip_input:
        values = values.clamp(1.0, 5.0)

    scores = torch.floor(values + 0.5).clamp(1.0, 5.0)
    if rule == "independent_half_up":
        return scores.to(dtype=predictions.dtype)

    target_total = torch.floor(values.sum(dim=-1) + 0.5).clamp(3.0, 15.0)
    flat_values = values.reshape(-1, 3)
    flat_scores = scores.reshape(-1, 3).clone()
    flat_target = target_total.reshape(-1)
    rows = torch.arange(flat_scores.shape[0], device=flat_scores.device)

    # Both the starting and target sums are in [3, 15], so twelve adjustments
    # are a strict upper bound even for adversarial out-of-range inputs.
    for _ in range(12):
        need_up = flat_scores.sum(dim=-1) < flat_target
        if bool(need_up.any()):
            residual = flat_values - flat_scores
            residual = residual.masked_fill(flat_scores >= 5.0, -torch.inf)
            chosen = residual.argmax(dim=-1)
            active_rows = rows[need_up]
            flat_scores[active_rows, chosen[need_up]] += 1.0

    for _ in range(12):
        need_down = flat_scores.sum(dim=-1) > flat_target
        if bool(need_down.any()):
            residual = flat_values - flat_scores
            residual = residual.masked_fill(flat_scores <= 1.0, torch.inf)
            chosen = residual.argmin(dim=-1)
            active_rows = rows[need_down]
            flat_scores[active_rows, chosen[need_down]] -= 1.0

    return flat_scores.reshape_as(values).to(dtype=predictions.dtype)


def _independent_state_probabilities(values: Tensor, temperature: float) -> Tensor:
    states_1d = torch.arange(1, 6, device=values.device, dtype=values.dtype)
    squared_distance = torch.square(values.unsqueeze(-1) - states_1d)
    # Centering guarantees at least one zero logit even at a vanishingly small
    # temperature; the others may safely saturate to -inf.
    squared_distance = squared_distance - squared_distance.amin(
        dim=-1, keepdim=True
    )
    marginal = torch.softmax(-squared_distance / temperature, dim=-1)
    joint = (
        marginal[..., 0, :, None, None]
        * marginal[..., 1, None, :, None]
        * marginal[..., 2, None, None, :]
    )
    return joint.reshape(*values.shape[:-1], 125)


def _average_matched_state_probabilities(
    values: Tensor,
    total_temperature: float,
    allocation_temperature: float,
) -> Tensor:
    states = integer_score_states(device=values.device, dtype=values.dtype)
    state_totals = states.sum(dim=-1).to(dtype=torch.int64)
    totals = torch.arange(3, 16, device=values.device, dtype=values.dtype)

    # Soft relaxation of T = half_up(sum(x)).
    total_distance = torch.square(values.sum(dim=-1, keepdim=True) - totals)
    total_distance = total_distance - total_distance.amin(dim=-1, keepdim=True)
    total_probability = torch.softmax(
        -total_distance / total_temperature, dim=-1
    )

    # For every possible T, normalize only over integer triples whose sum is T.
    # This is exact 125-state enumeration, not independent marginals and not a
    # stochastic sample.
    allocation_distance = torch.square(
        values.unsqueeze(-2) - states
    ).sum(dim=-1)
    valid = state_totals.unsqueeze(0) == torch.arange(
        3, 16, device=values.device, dtype=torch.int64
    ).unsqueeze(1)
    expanded_distance = allocation_distance.unsqueeze(-2).expand(
        *allocation_distance.shape[:-1], 13, 125
    )
    view_shape = (1,) * (expanded_distance.ndim - 2) + valid.shape
    expanded_valid = valid.view(view_shape)
    masked_distance = expanded_distance.masked_fill(~expanded_valid, torch.inf)
    minimum = masked_distance.amin(dim=-1, keepdim=True)
    conditional_logits = -(masked_distance - minimum) / allocation_temperature
    conditional_probability = torch.softmax(conditional_logits, dim=-1)

    joint = (total_probability.unsqueeze(-1) * conditional_probability).sum(dim=-2)
    # The expression is normalized analytically.  A final division removes only
    # floating accumulation drift (important to strict tests in float32).
    return joint / joint.sum(dim=-1, keepdim=True)


def soft_state_probabilities(
    predictions: Tensor,
    rule: QuantizationRule,
    *,
    temperature: float = 0.1,
    allocation_temperature: float | None = None,
    clip_input: bool = True,
) -> Tensor:
    """Return a deterministic probability over all 125 integer triples.

    For ``independent_half_up``, the joint distribution is the product of three
    soft nearest-integer categoricals.  For ``average_matched``, it is

    ``p(total | sum(x)) * p(integer triple | total, x)``.

    The latter has separate total and allocation temperatures; omitting
    ``allocation_temperature`` shares ``temperature``.  Squared-distance
    logits make the zero-temperature mode equal to the hard rule away from its
    measure-zero decision boundaries.
    """

    rule = _validate_rule(rule)
    values = _working_predictions(predictions, clip_input=clip_input)
    total_temperature = _stable_temperature(
        temperature, values.dtype, "temperature"
    )
    allocation_temperature = _stable_temperature(
        temperature if allocation_temperature is None else allocation_temperature,
        values.dtype,
        "allocation_temperature",
    )

    if rule == "independent_half_up":
        return _independent_state_probabilities(values, total_temperature)
    return _average_matched_state_probabilities(
        values, total_temperature, allocation_temperature
    )


def relaxed_quantize(
    predictions: Tensor,
    rule: QuantizationRule,
    *,
    temperature: float = 0.1,
    allocation_temperature: float | None = None,
    straight_through: bool = False,
    clip_input: bool = True,
) -> QuantizationRelaxation:
    """Build soft scores, hard scores, and an optional quantization STE."""

    probabilities = soft_state_probabilities(
        predictions,
        rule,
        temperature=temperature,
        allocation_temperature=allocation_temperature,
        clip_input=clip_input,
    )
    states = integer_score_states(
        device=probabilities.device, dtype=probabilities.dtype
    )
    soft_scores = probabilities @ states
    hard_scores = hard_quantize(
        predictions, rule, clip_input=clip_input
    ).to(dtype=soft_scores.dtype)
    scores = soft_scores
    if straight_through:
        scores = soft_scores + (hard_scores - soft_scores).detach()
    return QuantizationRelaxation(
        state_probabilities=probabilities,
        soft_scores=soft_scores,
        hard_scores=hard_scores,
        scores=scores,
    )


def _validate_targets(predictions: Tensor, targets: Tensor) -> None:
    if not isinstance(targets, Tensor):
        raise TypeError("targets must be a torch.Tensor")
    if targets.shape != predictions.shape:
        raise ValueError("targets must have the same [..., 3] shape as predictions")
    if not targets.dtype.is_floating_point:
        raise TypeError("targets must have a floating-point dtype")


def quantized_mse_risks(
    predictions: Tensor,
    targets: Tensor,
    rule: QuantizationRule,
    *,
    average_targets: Tensor | None = None,
    temperature: float = 0.1,
    allocation_temperature: float | None = None,
    straight_through: bool = False,
    clip_input: bool = True,
) -> QuantizedMseRisks:
    """Compute exact expected squared risk over the 125 integer triples.

    ``targets`` are never rounded.  If the dataset's stored ``score.average``
    is available, pass it as ``average_targets``; otherwise the arithmetic mean
    of the three trait targets is used.  With ``straight_through=True``, forward
    values are hard inference risks while backward gradients are those of the
    exact expected risks.
    """

    _validate_predictions(predictions)
    _validate_targets(predictions, targets)
    probabilities = soft_state_probabilities(
        predictions,
        rule,
        temperature=temperature,
        allocation_temperature=allocation_temperature,
        clip_input=clip_input,
    )
    work_targets = targets.to(device=predictions.device, dtype=probabilities.dtype)
    states = integer_score_states(
        device=probabilities.device, dtype=probabilities.dtype
    )

    state_trait_error = torch.square(
        states.view((1,) * (work_targets.ndim - 1) + states.shape)
        - work_targets.unsqueeze(-2)
    )
    soft_per_trait = torch.sum(
        probabilities.unsqueeze(-1) * state_trait_error, dim=-2
    )

    if average_targets is None:
        work_average_targets = work_targets.mean(dim=-1)
    else:
        if not isinstance(average_targets, Tensor):
            raise TypeError("average_targets must be a torch.Tensor")
        if average_targets.shape != predictions.shape[:-1]:
            raise ValueError(
                "average_targets must have shape predictions.shape[:-1]"
            )
        if not average_targets.dtype.is_floating_point:
            raise TypeError("average_targets must have a floating-point dtype")
        work_average_targets = average_targets.to(
            device=predictions.device, dtype=probabilities.dtype
        )
    state_average = states.mean(dim=-1)
    state_average_error = torch.square(
        state_average.view((1,) * work_average_targets.ndim + state_average.shape)
        - work_average_targets.unsqueeze(-1)
    )
    soft_trait_average = torch.sum(
        probabilities * state_average_error, dim=-1
    )

    if not straight_through:
        return QuantizedMseRisks(
            per_trait=soft_per_trait, trait_average=soft_trait_average
        )

    hard_scores = hard_quantize(
        predictions, rule, clip_input=clip_input
    ).to(dtype=probabilities.dtype)
    hard_per_trait = torch.square(hard_scores - work_targets)
    hard_trait_average = torch.square(
        hard_scores.mean(dim=-1) - work_average_targets
    )
    return QuantizedMseRisks(
        per_trait=soft_per_trait + (hard_per_trait - soft_per_trait).detach(),
        trait_average=soft_trait_average
        + (hard_trait_average - soft_trait_average).detach(),
    )


def _reduce(values: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "none":
        return values
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    raise ValueError("reduction choices=('none', 'mean', 'sum')")


def quantized_mse_loss(
    predictions: Tensor,
    targets: Tensor,
    rule: QuantizationRule,
    *,
    scope: ObjectiveScope = "both",
    average_targets: Tensor | None = None,
    temperature: float = 0.1,
    allocation_temperature: float | None = None,
    straight_through: bool = False,
    per_trait_weight: float = 1.0,
    trait_average_weight: float = 1.0,
    reduction: Reduction = "mean",
    clip_input: bool = True,
) -> Tensor:
    """Reduce quantized MSE risk on ``per_trait``, ``trait_average``, or both.

    The per-trait branch first averages C/O/E, so with ``reduction='none'`` all
    scopes return one value per leading example.  ``both`` is the explicit
    weighted sum of the two branches; it is not silently re-normalized.
    """

    scope = _validate_scope(scope)
    for name, weight in (
        ("per_trait_weight", per_trait_weight),
        ("trait_average_weight", trait_average_weight),
    ):
        if not math.isfinite(float(weight)) or float(weight) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    risks = quantized_mse_risks(
        predictions,
        targets,
        rule,
        average_targets=average_targets,
        temperature=temperature,
        allocation_temperature=allocation_temperature,
        straight_through=straight_through,
        clip_input=clip_input,
    )
    per_example_trait = risks.per_trait.mean(dim=-1)
    if scope == "per_trait":
        selected = float(per_trait_weight) * per_example_trait
    elif scope == "trait_average":
        selected = float(trait_average_weight) * risks.trait_average
    else:
        if per_trait_weight == 0.0 and trait_average_weight == 0.0:
            raise ValueError("both scope requires at least one positive weight")
        selected = (
            float(per_trait_weight) * per_example_trait
            + float(trait_average_weight) * risks.trait_average
        )
    return _reduce(selected, reduction)


def quantized_rank_inputs(
    predictions: Tensor,
    rule: QuantizationRule,
    *,
    scope: ObjectiveScope = "both",
    temperature: float = 0.1,
    allocation_temperature: float | None = None,
    straight_through: bool = False,
    clip_input: bool = True,
) -> QuantizedRankInputs:
    """Return deterministic soft/STE predictions for an external rank loss."""

    scope = _validate_scope(scope)
    scores = relaxed_quantize(
        predictions,
        rule,
        temperature=temperature,
        allocation_temperature=allocation_temperature,
        straight_through=straight_through,
        clip_input=clip_input,
    ).scores
    return QuantizedRankInputs(
        per_trait=scores if scope in ("per_trait", "both") else None,
        trait_average=(
            scores.mean(dim=-1)
            if scope in ("trait_average", "both")
            else None
        ),
    )


__all__ = [
    "OBJECTIVE_SCOPES",
    "QUANTIZATION_RULES",
    "ObjectiveScope",
    "QuantizationRelaxation",
    "QuantizationRule",
    "QuantizedMseRisks",
    "QuantizedRankInputs",
    "hard_quantize",
    "integer_score_states",
    "quantized_mse_loss",
    "quantized_mse_risks",
    "quantized_rank_inputs",
    "relaxed_quantize",
    "soft_state_probabilities",
]
