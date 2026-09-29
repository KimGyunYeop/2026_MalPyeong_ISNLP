"""운영측이 공지로 제공한 평가 지표 산출 코드. **수정 금지.**

출처와 확정 사항
----------------
- 2026-08-19 답변(집계 순서·필드·반올림):
    1) content/organization/expression 세 영역 점수를 먼저 평균하여 에세이 1건당
       하나의 예측 점수(pred_avg)를 만들고, 전체 샘플에 대해 RMSE와 Spearman을
       각각 한 번씩 계산한다.
    2) 예측값은 **반올림된** C/O/E의 평균이고, 정답값은 C/O/E를 다시 평균한 값이
       아니라 원본 데이터의 ``score.average`` 필드를 그대로 쓴다.
    3) 반올림은 영역별로 먼저 하고 그 뒤에 평균한다. 함수는 ``math.floor(x + 0.5)``다.
    5) 파싱 불가한 출력은 2회 재호출 후에도 실패하면 그 샘플을 제외하지 않고
       **예측값 0점**으로 집계에 포함한다.
- 2026-08-19 답변(라이브러리·핵심 코드): 아래 세 함수 원문.
- 2026-07-20 답변: Spearman 동점은 ``scipy.stats.spearmanr`` 기본(average rank)을 따른다.

아래 ``OFFICIAL`` 구역의 함수는 공지 원문을 그대로 옮긴 것이다. 서식(들여쓰기)만
파이썬 문법에 맞게 복원했고 이름·연산·반환 규약은 손대지 않는다. 이 파일을 고쳐
우리 숫자를 좋게 만들면 우리만 다른 척도를 보게 된다.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


TRAITS = ("content", "organization", "expression")


# --- OFFICIAL: 공지 원문 (수정 금지) ------------------------------------------
def round_half_up(x: float) -> int:
    # Python 기본 round()는 짝수로 반올림(banker's rounding)하므로 별도 구현
    return math.floor(x + 0.5)


def _compute_rmse(items):
    if not items:
        return 0.0
    return float(np.sqrt(np.mean([(pred - human) ** 2 for pred, human in items])))


def agg_rmse_average(items):
    return _compute_rmse(items)


def agg_spearman_average(items):
    if len(items) < 2:
        return 0.0
    preds = [p for p, _ in items]
    humans = [h for _, h in items]
    corr, _ = spearmanr(preds, humans)
    return float(0.0 if np.isnan(corr) else corr)
# --- OFFICIAL 끝 ---------------------------------------------------------------


# 파싱 실패 시 집계에 들어가는 예측값. 2026-08-19 답변 5번.
PARSE_FAILURE_PREDICTION = 0.0


def official_pred_avg(trait_scores: Sequence[float] | Mapping[str, float]) -> float:
    """영역별로 먼저 반올림한 뒤 평균한다(2026-08-19 답변 3번)."""

    if isinstance(trait_scores, Mapping):
        values = [float(trait_scores[trait]) for trait in TRAITS]
    else:
        values = [float(value) for value in trait_scores]
    if len(values) != len(TRAITS):
        raise ValueError(f"trait 점수는 {len(TRAITS)}개여야 합니다")
    return float(np.mean([round_half_up(value) for value in values]))


def official_items(
    predictions: Iterable[Sequence[float] | Mapping[str, float] | None],
    gold_averages: Iterable[float],
) -> list[tuple[float, float]]:
    """``agg_*`` 함수가 받는 (예측값, 정답값) 쌍 목록을 만든다.

    예측이 ``None``이면 파싱 실패로 보고 예측값 0을 넣는다. 샘플을 **빼지 않는다**.
    운영측이 그렇게 집계한다고 답변했고, 한 행을 0으로 잃는 비용이 상수로 찍는
    비용의 30배라는 사실이 이 계약에서 나온다.
    """

    items: list[tuple[float, float]] = []
    for prediction, gold in zip(predictions, gold_averages, strict=True):
        pred = (
            PARSE_FAILURE_PREDICTION
            if prediction is None
            else official_pred_avg(prediction)
        )
        items.append((pred, float(gold)))
    return items


def official_metrics(
    predictions: Iterable[Sequence[float] | Mapping[str, float] | None],
    gold_averages: Iterable[float],
) -> dict[str, float]:
    """공식 정의 그대로의 RMSE와 Spearman."""

    items = official_items(predictions, gold_averages)
    return {
        "count": len(items),
        "parse_failures": sum(1 for pred, _ in items if pred == PARSE_FAILURE_PREDICTION),
        "rmse": agg_rmse_average(items),
        "spearman": agg_spearman_average(items),
    }
