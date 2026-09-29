"""정수 총점 offset (2026-08-21).

이 축이 존재하는 이유와 왜 **정수**인지는
`main_code/utils.py`의 `average_matched_integer_scores` docstring에 있다.
여기서는 그 문서가 약속한 성질을 코드가 실제로 지키는지 못 박는다.

가장 중요한 두 가지
  1. offset=0이면 기존 배포와 **bit-exact** 동일하다 (c02 재현이 깨지면 안 된다).
  2. offset=+1이면 저장된 실제 400편에서 Spearman이 **비트 단위로** 같다.
     이것이 이 설계 전체의 근거다 — 연속 offset은 이 성질이 없다.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from main_code.postprocess import ScorePostprocessor
from main_code.utils import TRAITS, average_matched_integer_scores, round_half_up

ROOT = Path(__file__).resolve().parents[2]
C02_PREDICTIONS = (
    ROOT
    / "main_code/results/new_proposed/c02_soup_v1/base_c02/score_predictions.jsonl"
)

RNG = np.random.default_rng(20260821)


def _random_scores(n: int = 512) -> np.ndarray:
    return RNG.uniform(1.0, 5.0, size=(n, len(TRAITS)))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or sorted_values[index] != sorted_values[start]:
            ranks[order[start:index]] = (start + index - 1) / 2.0
            start = index
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(_average_ranks(a), _average_ranks(b))[0, 1])


# --------------------------------------------------------------- 기본 계약


def test_zero_offset_is_bit_exact_with_no_offset():
    """0이면 아무 일도 하지 않는다. c02 재현이 이 성질에 걸려 있다."""

    scores = _random_scores()
    baseline = average_matched_integer_scores(scores)
    assert np.array_equal(baseline, average_matched_integer_scores(scores, 0))
    assert np.array_equal(
        baseline, ScorePostprocessor("average_matched", 0).apply(scores)
    )
    assert np.array_equal(baseline, ScorePostprocessor("average_matched").apply(scores))


def test_offset_moves_target_total_by_exactly_the_offset_when_unclipped():
    scores = RNG.uniform(1.5, 4.0, size=(256, len(TRAITS)))  # clip에서 멀리
    base_total = average_matched_integer_scores(scores).sum(axis=1)
    for offset in (-2, -1, 1, 2):
        moved = average_matched_integer_scores(scores, offset).sum(axis=1)
        assert np.array_equal(moved, base_total + offset), offset


def test_target_total_matches_round_half_up_plus_offset():
    scores = _random_scores()
    for offset in (-1, 0, 1, 2):
        result = average_matched_integer_scores(scores, offset)
        expected = np.clip(round_half_up(scores.sum(axis=1)) + offset, 3.0, 15.0)
        assert np.array_equal(result.sum(axis=1), expected), offset


def test_traits_stay_in_range_and_integral_under_offset():
    scores = _random_scores(1024)
    for offset in (-4, -2, -1, 0, 1, 2, 4):
        result = average_matched_integer_scores(scores, offset)
        assert result.min() >= 1.0, offset
        assert result.max() <= 5.0, offset
        assert np.array_equal(result, np.round(result)), offset


def test_extreme_offset_saturates_without_leaving_the_grid():
    """clip 경계에서 배분 루프가 5를 넘거나 1 아래로 내려가지 않는다."""

    high = np.full((8, len(TRAITS)), 4.9)
    low = np.full((8, len(TRAITS)), 1.1)
    for offset in (5, 12, -5, -12):
        for scores in (high, low):
            result = average_matched_integer_scores(scores, offset)
            assert result.min() >= 1.0
            assert result.max() <= 5.0
            assert result.sum(axis=1).min() >= 3.0
            assert result.sum(axis=1).max() <= 15.0


# ------------------------------------------------------------- 잘못된 입력


@pytest.mark.parametrize("bad", [0.5, 1 / 3, -0.5, 1.5])
def test_non_integer_offset_is_rejected(bad):
    """1/3의 배수가 아닌 이동은 동점 구조를 바꾼다. 조용히 잘라내지 않는다."""

    with pytest.raises(ValueError, match="integer"):
        average_matched_integer_scores(_random_scores(4), bad)
    with pytest.raises(ValueError, match="integer"):
        ScorePostprocessor("average_matched", bad)


@pytest.mark.parametrize("rule", ["none", "per_trait_round"])
def test_offset_rejected_for_rules_without_a_target_total(rule):
    """조용히 무시하면 배포에서 offset이 사라진 것을 알 수 없다."""

    with pytest.raises(ValueError, match="average_matched"):
        ScorePostprocessor(rule, 1)
    ScorePostprocessor(rule, 0)  # 0은 어느 규칙에서도 무해하다


# ------------------------------------------------------- 산출물 스키마 보존


def test_apply_rows_records_offset_only_when_nonzero():
    rows = [{"scores": {t: 3.2 for t in TRAITS}} for _ in range(3)]
    ScorePostprocessor("average_matched", 0).apply_rows(rows)
    assert all("integer_total_offset" not in row for row in rows)

    rows = [{"scores": {t: 3.2 for t in TRAITS}} for _ in range(3)]
    ScorePostprocessor("average_matched", 1).apply_rows(rows)
    assert all(row["integer_total_offset"] == 1 for row in rows)


# ------------------------------------- 설계 전체의 근거: 실제 400편에서의 성질


@pytest.mark.skipif(
    not C02_PREDICTIONS.is_file(), reason="배포 c02 예측 파일이 없는 환경"
)
def test_offset_one_preserves_spearman_bitwise_on_deployed_predictions():
    """이 테스트가 이 축의 존재 이유다.

    저장된 배포 c02 예측 400편에서 정수 총점을 1 올리면 순위 벡터가 **비트 단위로**
    같다. 연속 offset(1/3의 배수가 아닌 이동)은 이 성질이 없다 — 같은 파일에서
    δ=0.15는 Spearman을 0.0447 떨어뜨린다.
    """

    rows = [
        json.loads(line)
        for line in C02_PREDICTIONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    continuous = np.array(
        [[row["scores"][trait] for trait in TRAITS] for row in rows], dtype=np.float64
    )
    gold = np.array(
        [
            sum(row["labels"][trait] for trait in TRAITS) / 3.0
            for row in rows
        ],
        dtype=np.float64,
    )

    base = average_matched_integer_scores(continuous).mean(axis=1)
    moved = average_matched_integer_scores(continuous, 1).mean(axis=1)

    # 모든 편이 정확히 1/3 올라간다 (clip에 걸리는 편이 없다)
    assert np.allclose(moved - base, 1.0 / 3.0), "clip에 걸린 편이 생겼다"

    rho_base = _spearman(base, gold)
    rho_moved = _spearman(moved, gold)
    assert rho_moved == rho_base, (rho_base, rho_moved)
    assert np.array_equal(
        np.argsort(np.argsort(base)), np.argsort(np.argsort(moved))
    )

    # RMSE는 실제로 움직인다 (편향이 양수인 방향으로)
    rmse_base = math.sqrt(float(((base - gold) ** 2).mean()))
    rmse_moved = math.sqrt(float(((moved - gold) ** 2).mean()))
    assert rmse_moved > rmse_base, "로컬 편향은 0에 가까우므로 로컬에서는 나빠져야 한다"


@pytest.mark.skipif(
    not C02_PREDICTIONS.is_file(), reason="배포 c02 예측 파일이 없는 환경"
)
def test_deployed_predictions_have_clip_headroom_for_plus_one():
    """+1이 clip에 걸리지 않는다는 배포 전제를 못 박는다."""

    rows = [
        json.loads(line)
        for line in C02_PREDICTIONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    continuous = np.array(
        [[row["scores"][trait] for trait in TRAITS] for row in rows], dtype=np.float64
    )
    totals = average_matched_integer_scores(continuous).sum(axis=1)
    assert totals.max() <= 14.0, f"T 최대 {totals.max()} — +1이 clip된다"
