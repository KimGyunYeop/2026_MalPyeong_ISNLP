from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from main_code.config import RegressionConfig, load_config
from main_code.datasets import RegressionCollator
from main_code.models import (
    RegressionScorer,
    _quantization_schedule,
    _quantized_training_targets,
    quantization_aware_objective,
    quantization_objective_active,
)
from main_code.quantization_objectives import hard_quantize, quantized_mse_risks
from main_code.tests.config_helpers import legacy_config
from main_code.train import (
    QuantizationScheduleCallback,
    overlay_training_metric_averages,
)


QUANTIZED_WEIGHT_FIELDS = (
    "quantized_trait_loss_weight",
    "quantized_mean_loss_weight",
    "quantized_pooled_loss_weight",
    "quantized_trait_rank_weight",
    "quantized_mean_rank_weight",
    "quantized_pooled_rank_weight",
)

# RMSE and MSE differ only for the three error scopes.  Rank scopes do not read
# quantized_error_form, so testing them once avoids redundant parameter cases.
OBJECTIVE_CASES = tuple(
    (field, error_form)
    for field in QUANTIZED_WEIGHT_FIELDS
    for error_form in (
        ("mse", "rmse") if field.endswith("loss_weight") else ("mse",)
    )
)


def _scores(*, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(
        [
            [1.82, 2.13, 2.44],
            [2.21, 2.76, 3.18],
            [2.73, 3.11, 3.69],
            [3.16, 3.74, 4.21],
            [3.68, 4.09, 4.63],
            [4.17, 4.52, 4.88],
        ],
        dtype=torch.float32,
        requires_grad=requires_grad,
    )


def _labels() -> torch.Tensor:
    return torch.tensor(
        [
            [1.50, 2.00, 2.50],
            [2.00, 2.50, 3.00],
            [2.50, 3.00, 3.50],
            [3.00, 3.50, 4.00],
            [3.50, 4.00, 4.50],
            [4.00, 4.50, 5.00],
        ],
        dtype=torch.float32,
    )


def _stored_averages() -> torch.Tensor:
    # Deliberately not the arithmetic trait means.  This catches accidental
    # fallback to labels.mean(-1) in the official mean-first branch.
    return torch.tensor([2.03, 2.47, 3.08, 3.61, 4.04, 4.53])


def _quantized_config(**updates: object) -> RegressionConfig:
    values: dict[str, object] = {
        "mse_loss_weight": 0.0,
        "quantization_rule": "average_matched",
        "quantized_error_form": "mse",
    }
    values.update(updates)
    return legacy_config(**values).validate()


def test_default_and_zero_weight_rule_are_complete_training_noops() -> None:
    defaults = RegressionConfig().validate()
    historical = legacy_config().validate()
    dormant_rule = legacy_config(quantization_rule="average_matched").validate()
    assert not quantization_objective_active(defaults)
    assert not quantization_objective_active(historical)
    assert not quantization_objective_active(dormant_rule)
    assert all(getattr(defaults, field) == 0.0 for field in QUANTIZED_WEIGHT_FIELDS)

    scores = _scores(requires_grad=True)
    reference_scores = _scores(requires_grad=True)
    labels = _labels()
    scorer = RegressionScorer(nn.Identity(), hidden_size=4, config=dormant_rule)
    reference_scorer = RegressionScorer(
        nn.Identity(), hidden_size=4, config=historical
    )
    with patch(
        "main_code.models.quantization_aware_objective",
        side_effect=AssertionError("zero-weight path called quantization"),
    ):
        without_average = scorer._training_loss(scores, None, labels)
        with_average = scorer._training_loss(
            scores, None, labels, average_labels=_stored_averages()
        )
    legacy_loss = torch.square(scores - labels).mean()
    assert torch.equal(without_average, legacy_loss)
    assert torch.equal(with_average, legacy_loss)

    gradient = torch.autograd.grad(with_average, scores)[0]
    reference_loss = reference_scorer._training_loss(
        reference_scores, None, labels
    )
    reference_gradient = torch.autograd.grad(reference_loss, reference_scores)[0]
    assert torch.equal(with_average, reference_loss)
    assert torch.equal(gradient, reference_gradient)


def test_old_mixed_arm_config_missing_new_fields_loads_as_noop(tmp_path: Path) -> None:
    """Queued pre-integration configs inherit only additive no-op defaults."""

    payload = asdict(RegressionConfig())
    for key in tuple(payload):
        if key.startswith("quantization_") or key.startswith("quantized_"):
            payload.pop(key)
    config_path = tmp_path / "pre_quantization_resolved_config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_config(config_path).validate()
    assert loaded.quantization_rule == "none"
    assert all(getattr(loaded, field) == 0.0 for field in QUANTIZED_WEIGHT_FIELDS)
    assert not quantization_objective_active(loaded)


@pytest.mark.parametrize("rule", ("independent_half_up", "average_matched"))
@pytest.mark.parametrize(
    "surrogate", ("soft", "straight_through", "expected_risk")
)
@pytest.mark.parametrize(
    ("weight_field", "error_form"),
    OBJECTIVE_CASES,
    ids=[f"{field}-{form}" for field, form in OBJECTIVE_CASES],
)
def test_every_rule_surrogate_and_metric_scope_has_finite_gradient(
    rule: str,
    surrogate: str,
    weight_field: str,
    error_form: str,
) -> None:
    weights = {field: 0.0 for field in QUANTIZED_WEIGHT_FIELDS}
    weights[weight_field] = 1.0
    config = _quantized_config(
        quantization_rule=rule,
        quantization_surrogate=surrogate,
        quantized_error_form=error_form,
        quantization_temperature=0.35,
        quantization_allocation_temperature=0.27,
        **weights,
    )
    scores = _scores(requires_grad=True)
    loss = quantization_aware_objective(
        scores,
        _labels(),
        _stored_averages(),
        config,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    # The fixture avoids decision boundaries, constant rank columns, and exact
    # target matches, so every supported objective must provide a useful signal.
    assert scores.grad.abs().sum().item() > 0.0


@pytest.mark.parametrize("prediction_rule", ("independent_half_up", "average_matched"))
@pytest.mark.parametrize(
    "target_rule",
    ("raw", "same_as_prediction", "independent_half_up", "average_matched"),
)
def test_every_target_rule_is_explicit_and_finite(
    prediction_rule: str, target_rule: str
) -> None:
    config = _quantized_config(
        quantization_rule=prediction_rule,
        quantization_surrogate="soft",
        quantized_target_rule=target_rule,
        quantized_trait_loss_weight=0.5,
        quantized_mean_loss_weight=0.5,
    )
    labels = _labels()
    stored = _stored_averages()
    trait_targets, average_targets = _quantized_training_targets(
        labels, stored, config
    )
    if target_rule == "raw":
        assert torch.equal(trait_targets, labels)
        assert torch.equal(average_targets, stored)
    else:
        effective_rule = (
            prediction_rule if target_rule == "same_as_prediction" else target_rule
        )
        expected = hard_quantize(labels, effective_rule)
        assert torch.equal(trait_targets, expected)
        assert torch.equal(average_targets, expected.mean(dim=-1))

    scores = _scores(requires_grad=True)
    loss = quantization_aware_objective(scores, labels, stored, config)
    loss.backward()
    assert torch.isfinite(loss)
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


@pytest.mark.parametrize("rule", ("independent_half_up", "average_matched"))
def test_expected_risk_mean_scope_uses_stored_average_labels_exactly(
    rule: str,
) -> None:
    config = _quantized_config(
        quantization_rule=rule,
        quantization_surrogate="expected_risk",
        quantized_mean_loss_weight=1.0,
        quantization_temperature=0.31,
        quantization_allocation_temperature=0.23,
    )
    scores = _scores()
    labels = _labels()
    stored = _stored_averages()
    actual = quantization_aware_objective(scores, labels, stored, config)
    risks = quantized_mse_risks(
        scores,
        labels,
        rule,
        average_targets=stored,
        temperature=0.31,
        allocation_temperature=0.23,
    )
    fallback_risks = quantized_mse_risks(
        scores,
        labels,
        rule,
        average_targets=labels.mean(dim=-1),
        temperature=0.31,
        allocation_temperature=0.23,
    )
    assert torch.allclose(actual, risks.trait_average.mean())
    assert not torch.allclose(actual, fallback_risks.trait_average.mean())


def test_training_average_overlay_reaches_collator_without_changing_traits(
    tmp_path: Path,
) -> None:
    processed = {
        "id": "official-1",
        "prompt_num": "Q1",
        "prompt": "prompt",
        "essay": "essay",
        "score": {
            "content": 3.0,
            "organization": 3.25,
            "expression": 4.0,
            "average": (3.0 + 3.25 + 4.0) / 3.0,
        },
    }
    extra = {
        "id": "extra-1",
        "prompt_num": "Q1",
        "prompt": "prompt",
        "essay": "another essay",
        "score": {"content": 2.0, "organization": 2.5, "expression": 3.0},
    }
    official = {
        **processed,
        "score": {**processed["score"], "average": 3.42},
    }
    overlay_file = tmp_path / "official_train.jsonl"
    overlay_file.write_text(json.dumps(official) + "\n", encoding="utf-8")

    rows = overlay_training_metric_averages([processed, extra], overlay_file)
    assert "_official_metric_average" not in processed
    assert rows[0]["score"] == processed["score"]
    assert rows[0]["_official_metric_average"] == 3.42
    assert rows[1] is extra

    class Tokenizer:
        model_max_length = 64
        padding_side = "right"

        def __call__(self, texts, **kwargs):
            assert isinstance(texts, list)
            return {
                "input_ids": torch.tensor([[1, 2]] * len(texts)),
                "attention_mask": torch.tensor([[1, 1]] * len(texts)),
            }

    batch = RegressionCollator(
        Tokenizer(),
        legacy_config(max_length=64),
        include_labels=True,
        include_metadata=False,
    )(rows)
    assert batch["labels"].dtype == torch.float64
    assert batch["labels"][0].tolist() == [3.0, 3.25, 4.0]
    assert batch["average_labels"].tolist() == pytest.approx(
        [3.42, (2.0 + 2.5 + 3.0) / 3.0]
    )


def test_schedule_has_exact_start_ramp_and_independent_temperature_anneal() -> None:
    config = _quantized_config(
        quantized_mean_loss_weight=1.0,
        quantized_loss_start_step=10,
        quantized_loss_ramp_steps=20,
        quantization_anneal_steps=40,
        quantization_temperature=0.8,
        quantization_final_temperature=0.2,
        quantization_allocation_temperature=0.6,
        quantization_final_allocation_temperature=0.1,
    )
    expected = {
        0: (0.0, 0.8, 0.6),
        9: (0.0, 0.8, 0.6),
        10: (0.0, 0.8, 0.6),
        20: (0.5, 0.65, 0.475),
        30: (1.0, 0.5, 0.35),
        50: (1.0, 0.2, 0.1),
        100: (1.0, 0.2, 0.1),
    }
    for step, values in expected.items():
        assert _quantization_schedule(config, step) == pytest.approx(values)


def test_schedule_scales_objective_and_trainer_callback_updates_model_step() -> None:
    config = _quantized_config(
        quantized_mean_loss_weight=1.0,
        quantized_loss_start_step=5,
        quantized_loss_ramp_steps=10,
        quantization_anneal_steps=0,
    )
    labels = _labels()
    stored = _stored_averages()
    before_scores = _scores(requires_grad=True)
    before = quantization_aware_objective(
        before_scores, labels, stored, config, global_step=4
    )
    before.backward()
    assert before.item() == 0.0
    assert before_scores.grad is not None
    assert torch.equal(before_scores.grad, torch.zeros_like(before_scores.grad))

    half = quantization_aware_objective(
        _scores(), labels, stored, config, global_step=10
    )
    full = quantization_aware_objective(
        _scores(), labels, stored, config, global_step=15
    )
    assert half.item() == pytest.approx(0.5 * full.item())

    scorer = RegressionScorer(nn.Identity(), hidden_size=4, config=config)
    callback = QuantizationScheduleCallback(scorer)
    callback.on_train_begin(None, SimpleNamespace(global_step=4), None)
    assert scorer._quantization_global_step == 4
    inactive = scorer._training_loss(
        _scores(), None, labels, average_labels=stored
    )
    assert inactive.item() == 0.0

    callback.on_step_begin(None, SimpleNamespace(global_step=15), None)
    assert scorer._quantization_global_step == 15
    active = scorer._training_loss(_scores(), None, labels, average_labels=stored)
    assert active.item() == pytest.approx(full.item())


def test_quantization_options_do_not_change_checkpoint_parameter_schema() -> None:
    dormant = RegressionScorer(
        nn.Identity(),
        hidden_size=4,
        config=legacy_config(quantization_rule="average_matched").validate(),
    )
    active = RegressionScorer(
        nn.Identity(),
        hidden_size=4,
        config=_quantized_config(quantized_mean_loss_weight=1.0),
    )
    assert list(dormant.state_dict()) == list(active.state_dict())
    assert all(
        dormant.state_dict()[key].shape == active.state_dict()[key].shape
        for key in dormant.state_dict()
    )
    assert "_quantization_global_step" not in dormant.state_dict()
