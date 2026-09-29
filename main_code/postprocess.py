"""모델 원점수를 제출 점수로 바꾸는 **단 하나의** 규칙 모음.

지금까지 이 변환은 세 곳에 흩어져 있었다. `utils._rounding_variant_metrics`(지표 계산),
`main_code_submission.engine._submitted_integer_scores`(서빙), 그리고 `infer`의
`submission.json`(**변환을 아예 안 했다**). 세 번째가 실수 원점수를 그대로 써서,
연구 산출물과 실제 서빙 출력이 `.4249` vs `.4477`로 갈렸다.

여기 한 곳으로 모아 세 경로가 같은 함수를 부르게 한다. 규칙을 옵션으로 둔 이유는
두 가지다. 대회 채점 방식이 또 바뀌면 코드가 아니라 인자만 바꾸면 되고, Docker 배포에서
서빙 설정이 연구 설정과 같은 이름을 쓰게 된다.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .config import TRAITS
from .utils import average_matched_integer_scores, per_trait_integer_scores

#: 각 규칙이 무엇을 하는지. `--score-postprocess` 선택지와 같은 순서다.
SCORE_POSTPROCESS_RULES: dict[str, str] = {
    # 모델 원점수를 그대로 낸다. 2026-08-06 공지 이전 동작이고, 지금 제출하면 평가
    # 서버가 영역별로 사사오입하므로 실질적으로 per_trait_round와 같아진다.
    "none": "실수 원점수 그대로",
    # 영역별 독립 사사오입. 평가 서버가 실수 제출에 적용하는 바로 그 변환이다.
    "per_trait_round": "영역별 사사오입 (평가 서버가 실수 제출에 하는 것)",
    # 세 정수의 합을 연속 예측 합에 맞춘다. 단계 수는 13으로 같지만 평균의 배치 오차가
    # sd .167 -> .096으로 준다. 공식 지표가 세 영역 평균 하나만 보므로 이것이 최적이다.
    "average_matched": "평균 정합 정수 삼중 (기본, 공식 지표 최적)",
}

DEFAULT_SCORE_POSTPROCESS = "average_matched"


class ScorePostprocessor:
    """원점수 배열/딕셔너리를 제출 점수로 바꾼다.

    상태가 없으므로 서빙에서 한 번 만들어 재사용해도 되고 매번 새로 만들어도 된다.
    """

    def __init__(
        self,
        rule: str = DEFAULT_SCORE_POSTPROCESS,
        integer_total_offset: int = 0,
    ):
        if rule not in SCORE_POSTPROCESS_RULES:
            raise ValueError(
                f"score_postprocess choices={tuple(SCORE_POSTPROCESS_RULES)}"
            )
        offset = int(integer_total_offset)
        if offset != integer_total_offset:
            raise ValueError("integer_total_offset must be an integer")
        if offset and rule != "average_matched":
            # 다른 규칙에는 "목표 합"이 없다. 조용히 무시하면 배포에서 offset이 사라진다.
            raise ValueError(
                "integer_total_offset은 average_matched에서만 쓸 수 있습니다: "
                f"rule={rule!r}"
            )
        self.rule = rule
        # 목표 정수 합에 더하는 값. 근거는 average_matched_integer_scores docstring.
        self.integer_total_offset = offset

    def apply(self, scores: np.ndarray) -> np.ndarray:
        """``[N,3]`` 원점수를 같은 shape의 제출 점수로 바꾼다."""

        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim != 2 or scores.shape[1] != len(TRAITS):
            raise ValueError("score_postprocess 입력은 [N,3]이어야 합니다")
        if self.rule == "none":
            # 규칙이 없어도 범위는 지킨다. 1~5 밖의 값은 어떤 채점 방식에서도 무효다.
            return np.clip(scores, 1.0, 5.0)
        if self.rule == "per_trait_round":
            return per_trait_integer_scores(scores)
        return average_matched_integer_scores(
            np.clip(scores, 1.0, 5.0), self.integer_total_offset
        )

    def apply_row(self, scores: dict[str, float]) -> dict[str, float]:
        """trait 이름 딕셔너리 하나를 변환한다. 서빙이 쓰는 형태다."""

        matrix = np.asarray([[float(scores[trait]) for trait in TRAITS]], dtype=np.float64)
        converted = self.apply(matrix)[0]
        return {trait: float(converted[index]) for index, trait in enumerate(TRAITS)}

    def apply_rows(self, rows: Iterable[dict[str, Any]], key: str = "scores") -> None:
        """예측 record 목록에 ``submitted_scores``를 **덧붙인다**(원본은 보존).

        원점수를 지우지 않는 것이 중요하다. 축 판정은 연속값으로 하고 보고는 정수로
        하므로 둘 다 남아 있어야 나중에 재측정할 수 있다.
        """

        rows = list(rows)
        if not rows:
            return
        matrix = np.asarray(
            [[float(row[key][trait]) for trait in TRAITS] for row in rows],
            dtype=np.float64,
        )
        converted = self.apply(matrix)
        for row, values in zip(rows, converted, strict=True):
            row["submitted_scores"] = {
                trait: float(values[index]) for index, trait in enumerate(TRAITS)
            }
            row["score_postprocess"] = self.rule
            if self.integer_total_offset:
                # 0이 아닐 때만 기록해 기존 산출물의 스키마를 건드리지 않는다.
                row["integer_total_offset"] = self.integer_total_offset
