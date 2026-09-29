from __future__ import annotations

import itertools

import numpy as np
import pytest

from main_code.config import RegressionConfig
from main_code.postprocess import (
    DEFAULT_SCORE_POSTPROCESS,
    SCORE_POSTPROCESS_RULES,
    ScorePostprocessor,
)


def config(**overrides) -> RegressionConfig:
    return RegressionConfig(
        model_id="skt/A.X-4.0-Light", training_mode="lora_only", **overrides
    ).validate()


def test_default_rule_is_the_metric_optimal_one() -> None:
    assert config().score_postprocess == DEFAULT_SCORE_POSTPROCESS == "average_matched"


def test_unknown_rule_is_rejected() -> None:
    with pytest.raises(ValueError, match="score_postprocess"):
        config(score_postprocess="bogus")


def test_every_rule_stays_inside_the_official_range() -> None:
    scores = np.array([[0.4, 6.2, 3.0], [1.0, 5.0, 2.5]])
    for rule in SCORE_POSTPROCESS_RULES:
        out = ScorePostprocessor(rule).apply(scores)
        assert out.min() >= 1.0 and out.max() <= 5.0


def test_integer_rules_emit_integers() -> None:
    scores = np.array([[3.257, 3.584, 3.805]])
    for rule in ("per_trait_round", "average_matched"):
        out = ScorePostprocessor(rule).apply(scores)
        assert np.array_equal(out, np.round(out))


def test_average_matched_tracks_the_continuous_sum_more_closely() -> None:
    """평균 정합의 존재 이유가 그대로 성립하는지 본다."""

    rng = np.random.default_rng(0)
    scores = np.clip(rng.normal(3.4, 0.6, size=(2000, 3)), 1, 5)
    target = scores.mean(axis=1)
    per_trait = ScorePostprocessor("per_trait_round").apply(scores).mean(axis=1)
    matched = ScorePostprocessor("average_matched").apply(scores).mean(axis=1)
    assert np.abs(matched - target).std() < np.abs(per_trait - target).std()


def test_average_matched_differs_at_the_half_up_total() -> None:
    """합 10.47은 사사오입 합 10으로 보존하며 동률은 C/O/E 순으로 푼다."""

    scores = np.asarray([[3.49, 3.49, 3.49]], dtype=np.float64)
    independent = ScorePostprocessor("per_trait_round").apply(scores)
    matched = ScorePostprocessor("average_matched").apply(scores)

    assert independent.tolist() == [[3.0, 3.0, 3.0]]
    assert matched.tolist() == [[4.0, 3.0, 3.0]]
    assert matched.sum(axis=1).tolist() == [10.0]


def test_average_matched_respects_upper_bound_while_matching_total() -> None:
    """5점 경계에서는 움직일 수 있는 trait만 조정하고 합 14를 유지한다."""

    scores = np.asarray([[4.51, 4.51, 5.0]], dtype=np.float64)
    independent = ScorePostprocessor("per_trait_round").apply(scores)
    matched = ScorePostprocessor("average_matched").apply(scores)

    assert independent.tolist() == [[5.0, 5.0, 5.0]]
    assert matched.tolist() == [[4.0, 5.0, 5.0]]
    assert matched.sum(axis=1).tolist() == [14.0]


def test_average_matched_is_minimum_squared_residual_at_the_target_total() -> None:
    """Greedy residual update가 제약된 1..5 정수 전수조사의 최솟값과 같다."""

    rows = np.asarray(
        [
            [1.0, 1.49, 1.49],
            [2.21, 3.87, 4.44],
            [3.49, 3.49, 3.49],
            [4.51, 4.51, 5.0],
        ],
        dtype=np.float64,
    )
    converted = ScorePostprocessor("average_matched").apply(rows)

    for raw, integer_scores in zip(rows, converted, strict=True):
        target_total = int(np.clip(np.floor(raw.sum() + 0.5), 3, 15))
        candidates = [
            np.asarray(candidate, dtype=np.float64)
            for candidate in itertools.product(range(1, 6), repeat=3)
            if sum(candidate) == target_total
        ]
        best_cost = min(
            float(np.square(candidate - raw).sum()) for candidate in candidates
        )
        actual_cost = float(np.square(integer_scores - raw).sum())

        assert integer_scores.sum() == target_total
        assert actual_cost == pytest.approx(best_cost)


def test_apply_rows_keeps_the_raw_scores() -> None:
    rows = [
        {
            "essay_id": "a",
            "scores": {"content": 3.3, "organization": 3.6, "expression": 3.8},
        }
    ]
    ScorePostprocessor("average_matched").apply_rows(rows)
    assert rows[0]["scores"] == {"content": 3.3, "organization": 3.6, "expression": 3.8}
    assert rows[0]["submitted_scores"] == {
        "content": 3.0,
        "organization": 4.0,
        "expression": 4.0,
    }
    assert rows[0]["score_postprocess"] == "average_matched"


def test_apply_row_matches_apply() -> None:
    row = {"content": 2.2, "organization": 4.7, "expression": 3.1}
    single = ScorePostprocessor().apply_row(row)
    batch = ScorePostprocessor().apply(np.array([[2.2, 4.7, 3.1]]))[0]
    assert [single["content"], single["organization"], single["expression"]] == list(
        batch
    )


def test_serving_engine_uses_the_same_rule() -> None:
    """서빙과 연구 경로가 갈라지지 않는지 확인한다. 예전 결함의 회귀 방지다."""

    from main_code_submission.engine import _submitted_integer_scores

    row = {"content": 3.257, "organization": 3.584, "expression": 3.805}
    assert _submitted_integer_scores(row) == ScorePostprocessor().apply_row(row)
