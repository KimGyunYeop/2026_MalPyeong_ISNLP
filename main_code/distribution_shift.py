"""리더보드 집합의 라벨 분포로 옮겨 재는 지표. **상수 수정 금지.**

왜 이 파일이 있는가
-------------------
2026-08-20 운영측 답변: "확인 결과, 예측값이 0점으로 처리된 샘플은 없습니다."
이걸로 유실 가설이 죽었고, 남은 격차는 전부 분포 차이다. 실측 두 쌍을 나란히 놓으면
설명이 하나로 좁혀진다.

    로컬 공식 400편   RMSE 0.41682   Spearman 0.75974   정답 SD 0.6533
    리더보드          RMSE 0.5191    Spearman 0.7340    정답 SD 미공개

RMSE^2 = σ_p^2 + σ_g^2 − 2·r·σ_p·σ_g 에 로컬 상수(σ_p=0.4936, r=0.7703, 편향 −0.0095)를
그대로 두고 σ_g만 0.79로 바꾸면 **0.5168**이 나온다(실측 0.5191, 오차 0.002).
Spearman이 0.7597 → 0.7340으로 거의 그대로였다는 점이 이 해석의 핵심 근거다. 미학습
문항 탓이라면 순위 능력이 함께 무너져야 하는데 그러지 않았다. Pearson≈Spearman으로
두고 두 식을 같이 풀면 (σ_g, r) = (0.768, 0.744)로 떨어진다.

그래서 우리는 지금까지 **틀린 저울로 체크포인트를 골라 왔다.** 185개 run을 전부
`official_matched_rmse`(σ_g=0.653 집합)로 선택했는데, 채점은 σ_g≈0.77 집합에서 된다.
이 파일은 그 저울을 하나 더 놓는다. 기존 지표는 손대지 않는다 — 과거 표와 비교
가능성이 사라지면 안 되기 때문이다.

가중치
------
    w(y) = exp( (y−μ)²/(2σ_ref²) − (y−μ)²/(2σ_target²) ) · (σ_ref/σ_target)

정답 분포 N(μ, σ_ref²)를 N(μ, σ_target²)로 옮기는 중요도 가중치다. (σ_ref/σ_target)는
y ~ N(μ, σ_ref²)에서 E[w] = σ_target/σ_ref 라는 닫힌 해에서 나온 정규화 상수라
표본 구성에 흔들리지 않는다.

**아래 상수는 실험마다 바꾸면 안 된다.** 저울이 run마다 달라지면 run 사이 비교가
불가능해진다. 학습 쪽 가중치(`RegressionConfig.tail_weight_*`)는 이것과 별개의 축이고
그쪽은 조정 대상이다 — 학습 라벨 분포(11,600편, SD 0.617)와 검증 라벨 분포(400편,
SD 0.6533)가 다르기 때문에 참조값도 다르다.
"""

from __future__ import annotations

import math

import numpy as np

# 공식 validation 400편의 score.average 평균과 표준편차. 측정값이다.
SHIFT_REFERENCE_MEAN = 3.3978
SHIFT_REFERENCE_SIGMA = 0.6533
# 위 문서화한 두 식을 함께 풀어 얻은 리더보드 집합의 추정 정답 SD.
SHIFT_TARGET_SIGMA = 0.77
# 상한이 없으면 최저점 소수 표본이 지표를 독점해 잡음이 커진다. 400편에서 이 상한에
# 걸리는 것은 정답 1.5 미만 소수다.
SHIFT_WEIGHT_MAX = 4.0


def shift_weights(
    gold_averages: np.ndarray,
    *,
    reference_mean: float = SHIFT_REFERENCE_MEAN,
    reference_sigma: float = SHIFT_REFERENCE_SIGMA,
    target_sigma: float = SHIFT_TARGET_SIGMA,
    weight_max: float = SHIFT_WEIGHT_MAX,
) -> np.ndarray:
    """정답값마다의 중요도 가중치. 평균이 1이 되도록 정규화되어 있다."""

    if reference_sigma <= 0 or target_sigma <= 0:
        raise ValueError("reference_sigma와 target_sigma는 양수여야 합니다")
    gold = np.asarray(gold_averages, dtype=np.float64)
    deviation = np.square(gold - reference_mean)
    exponent = deviation * (
        1.0 / (2.0 * reference_sigma**2) - 1.0 / (2.0 * target_sigma**2)
    )
    weights = np.exp(exponent) * (reference_sigma / target_sigma)
    return np.minimum(weights, weight_max)


def shifted_rmse(predictions: np.ndarray, gold_averages: np.ndarray) -> float:
    """리더보드 분포로 옮겨 잰 RMSE.

    ``predictions``는 공식 정의대로 이미 영역별 반올림 후 평균낸 essay당 한 값이다.
    """

    prediction = np.asarray(predictions, dtype=np.float64)
    gold = np.asarray(gold_averages, dtype=np.float64)
    if prediction.shape != gold.shape:
        raise ValueError("predictions와 gold_averages의 모양이 같아야 합니다")
    if prediction.size == 0:
        return 0.0
    weights = shift_weights(gold)
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    return math.sqrt(float((weights * np.square(prediction - gold)).sum()) / total)
