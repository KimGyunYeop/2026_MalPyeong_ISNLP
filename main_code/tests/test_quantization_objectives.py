from __future__ import annotations

import numpy as np
import pytest
import torch

from main_code.postprocess import ScorePostprocessor
from main_code.quantization_objectives import (
    QUANTIZATION_RULES,
    hard_quantize,
    integer_score_states,
    quantized_mse_loss,
    quantized_mse_risks,
    quantized_rank_inputs,
    relaxed_quantize,
    soft_state_probabilities,
)
from main_code.utils import (
    average_matched_integer_scores,
    per_trait_integer_scores,
)


def test_integer_state_space_is_complete_and_canonical() -> None:
    states = integer_score_states(device="cpu", dtype=torch.float64)
    assert states.shape == (125, 3)
    assert torch.unique(states, dim=0).shape == (125, 3)
    assert states.min().item() == 1.0
    assert states.max().item() == 5.0
    assert states[0].tolist() == [1.0, 1.0, 1.0]
    assert states[-1].tolist() == [5.0, 5.0, 5.0]


def test_independent_hard_quantizer_exactly_matches_existing_oracle() -> None:
    values = np.asarray(
        [
            [-10.0, 0.49, 1.0],
            [1.49, 1.50, 2.50],
            [3.49, 3.50, 4.50],
            [4.51, 5.00, 100.0],
        ],
        dtype=np.float64,
    )
    actual = hard_quantize(
        torch.from_numpy(values), "independent_half_up", clip_input=False
    )
    assert np.array_equal(actual.numpy(), per_trait_integer_scores(values))


def test_average_matched_hard_quantizer_matches_utils_and_postprocess() -> None:
    rng = np.random.default_rng(943)
    values = np.concatenate(
        [
            rng.uniform(-2.0, 8.0, size=(500, 3)),
            np.asarray(
                [
                    [3.49, 3.49, 3.49],
                    [4.51, 4.51, 5.0],
                    [1.5, 2.5, 3.5],
                    [0.0, 7.0, 3.5],
                ]
            ),
        ]
    ).astype(np.float64)
    tensor = torch.from_numpy(values)

    direct = hard_quantize(tensor, "average_matched", clip_input=False)
    assert np.array_equal(direct.numpy(), average_matched_integer_scores(values))

    deployed = hard_quantize(tensor, "average_matched")
    expected_deployed = ScorePostprocessor("average_matched").apply(values)
    assert np.array_equal(deployed.numpy(), expected_deployed)
    assert deployed[-4].tolist() == [4.0, 3.0, 3.0]
    assert deployed[-3].tolist() == [4.0, 5.0, 5.0]


@pytest.mark.parametrize("rule", QUANTIZATION_RULES)
def test_soft_relaxation_is_a_normalized_125_state_distribution(rule: str) -> None:
    values = torch.tensor(
        [[1.20, 2.80, 4.15], [3.31, 3.78, 2.44]], dtype=torch.float64
    )
    probabilities = soft_state_probabilities(values, rule, temperature=0.2)
    assert probabilities.shape == (2, 125)
    assert torch.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    assert probabilities.sum(dim=-1).tolist() == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize("rule", QUANTIZATION_RULES)
def test_zero_temperature_mode_converges_to_hard_rule_away_from_ties(
    rule: str,
) -> None:
    values = torch.tensor(
        [
            [1.13, 2.27, 4.09],
            [2.82, 3.14, 4.71],
            [4.22, 3.63, 1.91],
        ],
        dtype=torch.float64,
    )
    probabilities = soft_state_probabilities(values, rule, temperature=1e-4)
    states = integer_score_states(device="cpu", dtype=torch.float64)
    modal_states = states[probabilities.argmax(dim=-1)]
    assert torch.equal(modal_states, hard_quantize(values, rule))


@pytest.mark.parametrize("rule", QUANTIZATION_RULES)
@pytest.mark.parametrize("temperature", [1e-30, 1e-4, 0.1, 100.0])
def test_temperatures_have_finite_values_and_gradients(
    rule: str, temperature: float
) -> None:
    values = torch.tensor(
        [[1.23, 2.71, 4.16], [3.18, 3.83, 2.27]],
        dtype=torch.float32,
        requires_grad=True,
    )
    probabilities = soft_state_probabilities(
        values, rule, temperature=temperature
    )
    coefficients = torch.linspace(
        -1.0, 1.0, 125, device=values.device, dtype=probabilities.dtype
    )
    (probabilities * coefficients).sum().backward()
    assert torch.isfinite(probabilities).all()
    assert probabilities.sum(dim=-1).detach().tolist() == pytest.approx([1.0, 1.0])
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


@pytest.mark.parametrize("rule", QUANTIZATION_RULES)
def test_relaxation_is_deterministic_and_does_not_consume_rng(rule: str) -> None:
    values = torch.tensor([[2.37, 3.61, 4.12]], dtype=torch.float32)
    torch.manual_seed(123)
    state_before = torch.random.get_rng_state().clone()
    first = soft_state_probabilities(values, rule, temperature=0.13)
    state_after = torch.random.get_rng_state().clone()
    torch.manual_seed(999)
    second = soft_state_probabilities(values, rule, temperature=0.13)
    assert torch.equal(state_before, state_after)
    assert torch.equal(first, second)


@pytest.mark.parametrize("rule", QUANTIZATION_RULES)
def test_score_ste_is_hard_forward_and_soft_backward(rule: str) -> None:
    values = torch.tensor(
        [[2.42, 3.61, 4.18], [3.36, 2.73, 1.89]], requires_grad=True
    )
    result = relaxed_quantize(
        values, rule, temperature=0.2, straight_through=True
    )
    assert torch.equal(result.scores, result.hard_scores)
    assert not torch.equal(result.soft_scores, result.hard_scores)
    result.scores.square().mean().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    assert values.grad.abs().sum().item() > 0.0


@pytest.mark.parametrize("rule", QUANTIZATION_RULES)
def test_exact_expected_mse_risks_match_manual_state_sum(rule: str) -> None:
    predictions = torch.tensor(
        [[2.31, 3.72, 4.11], [3.66, 2.29, 1.82]], dtype=torch.float64
    )
    targets = torch.tensor(
        [[2.50, 3.25, 4.00], [3.50, 2.50, 2.00]], dtype=torch.float64
    )
    average_targets = torch.tensor([3.17, 2.71], dtype=torch.float64)
    probabilities = soft_state_probabilities(
        predictions, rule, temperature=0.3
    )
    states = integer_score_states(device="cpu", dtype=torch.float64)
    expected_traits = (
        probabilities.unsqueeze(-1)
        * torch.square(states.unsqueeze(0) - targets.unsqueeze(1))
    ).sum(dim=1)
    expected_average = (
        probabilities
        * torch.square(
            states.mean(dim=-1).unsqueeze(0) - average_targets.unsqueeze(1)
        )
    ).sum(dim=1)

    risks = quantized_mse_risks(
        predictions,
        targets,
        rule,
        average_targets=average_targets,
        temperature=0.3,
    )
    assert torch.allclose(risks.per_trait, expected_traits)
    assert torch.allclose(risks.trait_average, expected_average)


def test_mse_scopes_and_stored_average_target_are_kept_separate() -> None:
    predictions = torch.tensor([[2.3, 3.6, 4.1], [3.7, 2.4, 1.9]])
    targets = torch.tensor([[2.5, 3.5, 4.0], [3.5, 2.5, 2.0]])
    stored_average = torch.tensor([3.1, 2.8])
    risks = quantized_mse_risks(
        predictions,
        targets,
        "average_matched",
        average_targets=stored_average,
        temperature=0.25,
    )
    per_trait = quantized_mse_loss(
        predictions,
        targets,
        "average_matched",
        scope="per_trait",
        average_targets=stored_average,
        temperature=0.25,
        reduction="none",
    )
    average = quantized_mse_loss(
        predictions,
        targets,
        "average_matched",
        scope="trait_average",
        average_targets=stored_average,
        temperature=0.25,
        reduction="none",
    )
    both = quantized_mse_loss(
        predictions,
        targets,
        "average_matched",
        scope="both",
        average_targets=stored_average,
        temperature=0.25,
        per_trait_weight=0.25,
        trait_average_weight=0.75,
        reduction="none",
    )
    assert torch.allclose(per_trait, risks.per_trait.mean(dim=-1))
    assert torch.allclose(average, risks.trait_average)
    assert torch.allclose(both, 0.25 * per_trait + 0.75 * average)


@pytest.mark.parametrize("rule", QUANTIZATION_RULES)
def test_mse_risk_ste_has_hard_forward_value_and_finite_soft_gradient(
    rule: str,
) -> None:
    predictions = torch.tensor(
        [[2.42, 3.61, 4.18], [3.36, 2.73, 1.89]], requires_grad=True
    )
    targets = torch.tensor([[2.2, 3.4, 4.0], [3.5, 2.5, 2.1]])
    actual = quantized_mse_loss(
        predictions,
        targets,
        rule,
        scope="both",
        temperature=0.2,
        straight_through=True,
    )
    hard = hard_quantize(predictions, rule)
    expected = (
        torch.square(hard - targets).mean()
        + torch.square(hard.mean(dim=-1) - targets.mean(dim=-1)).mean()
    )
    assert actual.detach().item() == pytest.approx(expected.item())
    actual.backward()
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()
    assert predictions.grad.abs().sum().item() > 0.0


@pytest.mark.parametrize("rule", QUANTIZATION_RULES)
def test_rank_inputs_support_every_scope_and_ste(rule: str) -> None:
    predictions = torch.tensor(
        [[2.42, 3.61, 4.18], [3.36, 2.73, 1.89]], requires_grad=True
    )
    per_trait = quantized_rank_inputs(
        predictions, rule, scope="per_trait", temperature=0.2
    )
    assert per_trait.per_trait is not None
    assert per_trait.per_trait.shape == (2, 3)
    assert per_trait.trait_average is None

    average = quantized_rank_inputs(
        predictions,
        rule,
        scope="trait_average",
        temperature=0.2,
        straight_through=True,
    )
    assert average.per_trait is None
    assert average.trait_average is not None
    assert average.trait_average.tolist() == pytest.approx(
        hard_quantize(predictions, rule).mean(dim=-1).tolist()
    )

    both = quantized_rank_inputs(predictions, rule, scope="both", temperature=0.2)
    assert both.per_trait is not None
    assert both.trait_average is not None
    assert torch.allclose(both.trait_average, both.per_trait.mean(dim=-1))


@pytest.mark.parametrize("rule", QUANTIZATION_RULES)
def test_half_precision_inputs_are_promoted_for_stable_soft_objectives(rule: str) -> None:
    predictions = torch.tensor(
        [[2.49, 3.51, 4.02]], dtype=torch.float16, requires_grad=True
    )
    probabilities = soft_state_probabilities(
        predictions, rule, temperature=1e-6
    )
    assert probabilities.dtype == torch.float32
    probabilities.square().sum().backward()
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()


def test_invalid_quantization_arguments_fail_closed() -> None:
    predictions = torch.ones(2, 3)
    targets = torch.ones(2, 3)
    with pytest.raises(ValueError, match="rule choices"):
        soft_state_probabilities(predictions, "unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive and finite"):
        soft_state_probabilities(predictions, "average_matched", temperature=0.0)
    with pytest.raises(ValueError, match="scope choices"):
        quantized_mse_loss(
            predictions, targets, "average_matched", scope="unknown"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="average_targets"):
        quantized_mse_risks(
            predictions,
            targets,
            "average_matched",
            average_targets=torch.ones(2, 1),
        )
