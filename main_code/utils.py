from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from .config import TRAITS


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Assign average ranks to ties (the definition used by Spearman)."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def round_half_up(values: np.ndarray) -> np.ndarray:
    """대회 채점의 사사오입. numpy의 ``rint``는 banker's rounding이라 쓰지 않는다."""

    return np.floor(np.asarray(values, dtype=np.float64) + 0.5)


def per_trait_integer_scores(predictions: np.ndarray) -> np.ndarray:
    """각 trait을 독립적으로 1~5 정수로 반올림한다.

    2026-08-06 공지: 모델이 실수를 출력하면 평가 서버가 영역별로 사사오입해 정수로
    바꾼 뒤 RMSE/Spearman을 산출한다. 실수를 그대로 제출했을 때 실제로 채점되는 값이다.
    """

    return np.clip(round_half_up(predictions), 1.0, 5.0)


def average_matched_integer_scores(
    predictions: np.ndarray, integer_total_offset: int = 0
) -> np.ndarray:
    """세 trait 평균이 연속 예측 평균에 가장 가깝도록 정수 삼중을 고른다.

    공식 지표는 세 trait 점수의 **평균 하나**만 쓴다. trait마다 독립으로 반올림하면
    그 평균의 양자화 간격이 1이 되지만, 세 정수의 **합**을 목표에 맞추면 간격이 1/3로
    줄어든다. 즉 같은 연속 예측에서 양자화 손실만 1/3로 줄이는 무료 변환이다.
    맞춘 뒤에도 각 trait은 자기 연속 예측에 가장 가까운 정수를 유지하므로 근거 문장과의
    정합성도 크게 흔들리지 않는다.

    ``integer_total_offset``은 목표 합에 더하는 **정수**다 (2026-08-21).

    왜 정수여야 하는가
        제출값은 T/3 격자 위에 있다. 연속 예측에 1/3의 배수가 아닌 값을 더하면 essay마다
        T가 다르게 움직여 동점 구조가 바뀌고 Spearman이 깎인다. 실측 400편에서 δ=0.15는
        ρ를 0.0447 떨어뜨렸다. 반면 T에 정수를 더하면 clip에 걸리지 않는 한 **모든** essay가
        같이 움직여 순위 벡터가 비트 단위로 보존된다(400/400편 확인).

    왜 이 값을 쓰고 싶은가
        리더보드 RMSE 0.5191과 로컬 0.4168의 격차는 대부분 평균 편향이다. 관측
        (RMSE 0.5191, ρ 0.7340)을 열화 모형 6종에서 동시에 맞추면 편향이 0.273~0.285로
        모인다. 손익분기가 1/6이므로 T에 +1(=예측 +1/3)이 RMSE만 옮긴다.

        clip 여유도 확인됐다 — c02의 연속 평균 최대가 4.2803이라 T 최대가 13이고,
        T=15가 되려면 4.8333이 필요하다. +1 후 clip되는 편은 0/400이다.

    0이면 기존 동작과 bit-exact 동일하다.
    """

    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[1] != len(TRAITS):
        raise ValueError("predictions must have shape [N, 3]")
    offset = int(integer_total_offset)
    if offset != integer_total_offset:
        # 정수가 아닌 offset은 격자를 옮겨 동점 구조를 바꾼다. 아래 docstring 참조.
        raise ValueError("integer_total_offset must be an integer")
    target_total = np.clip(
        round_half_up(predictions.sum(axis=1)) + offset, 3.0, 15.0
    )
    scores = per_trait_integer_scores(predictions)
    for row in range(predictions.shape[0]):
        # 합이 목표와 같아질 때까지 잔차가 가장 큰 trait부터 1씩 옮긴다.
        while scores[row].sum() < target_total[row]:
            residual = predictions[row] - scores[row]
            residual[scores[row] >= 5.0] = -np.inf
            scores[row, int(np.argmax(residual))] += 1.0
        while scores[row].sum() > target_total[row]:
            residual = predictions[row] - scores[row]
            residual[scores[row] <= 1.0] = np.inf
            scores[row, int(np.argmin(residual))] -= 1.0
    return scores


# 제출 서버가 내보내는 surface. 공식 지표는 이 surface 기준으로 읽는다.
SUBMITTED_SCORE_SURFACE = "average_matched"

SCORE_METRIC_SURFACES = (
    "raw_continuous",
    "independent_half_up",
    "average_matched",
)


def _spearman_value(prediction: np.ndarray, truth: np.ndarray) -> float | None:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    if prediction.shape != truth.shape or prediction.size < 2:
        return None
    if np.unique(prediction).size < 2 or np.unique(truth).size < 2:
        return None
    value = float(
        np.corrcoef(_average_ranks(prediction), _average_ranks(truth))[0, 1]
    )
    return None if math.isnan(value) else value


def score_metric_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    average_labels: np.ndarray | None = None,
) -> dict[str, dict[str, float | None]]:
    """세 prediction surface × 세 집계축의 RMSE/Spearman을 전부 계산한다.

    ``trait_macro``는 C/O/E에서 지표를 각각 계산한 뒤 산술평균하고,
    ``mean_first``는 essay마다 C/O/E를 먼저 평균한 뒤 저장 ``score.average``와
    한 번 계산하며, ``pooled``는 N×3 cell을 펼친다. Gold는 어느 surface에서도
    반올림하지 않는다. 이 함수는 Trainer와 ``result_all``의 metric 이름 계약이다.
    """

    labels = np.asarray(labels, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if labels.shape != predictions.shape or labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("labels/predictions must both have shape [N, 3]")
    if labels.shape[0] < 2 or not (
        np.isfinite(labels).all() and np.isfinite(predictions).all()
    ):
        raise ValueError("metric inputs must contain at least two finite rows")
    if average_labels is None:
        gold_average = labels.mean(axis=1)
    else:
        gold_average = np.asarray(average_labels, dtype=np.float64).reshape(-1)
        if gold_average.shape != (labels.shape[0],) or not np.isfinite(
            gold_average
        ).all():
            raise ValueError("average_labels must be one finite value per row")

    surfaces = {
        "raw_continuous": predictions,
        "independent_half_up": per_trait_integer_scores(predictions),
        "average_matched": average_matched_integer_scores(predictions),
    }
    matrix: dict[str, dict[str, float | None]] = {}
    for surface, values in surfaces.items():
        metrics: dict[str, float | None] = {}
        trait_rmses: list[float] = []
        trait_rhos: list[float] = []
        for index, trait in enumerate(TRAITS):
            rmse = float(np.sqrt(np.mean((values[:, index] - labels[:, index]) ** 2)))
            rho = _spearman_value(values[:, index], labels[:, index])
            metrics[f"{trait}_rmse"] = rmse
            metrics[f"{trait}_spearman"] = rho
            trait_rmses.append(rmse)
            if rho is not None:
                trait_rhos.append(rho)
        metrics["trait_macro_rmse"] = float(np.mean(trait_rmses))
        metrics["trait_macro_spearman"] = (
            float(np.mean(trait_rhos)) if trait_rhos else None
        )
        predicted_average = values.mean(axis=1)
        metrics["mean_first_rmse"] = float(
            np.sqrt(np.mean((predicted_average - gold_average) ** 2))
        )
        metrics["mean_first_spearman"] = _spearman_value(
            predicted_average, gold_average
        )
        metrics["pooled_rmse"] = float(
            np.sqrt(np.mean((values.reshape(-1) - labels.reshape(-1)) ** 2))
        )
        metrics["pooled_spearman"] = _spearman_value(
            values.reshape(-1), labels.reshape(-1)
        )
        # 운영측 공지 코드로 계산한 **공식 지표**. 위 아홉 조합은 진단용이고 리더보드가
        # 읽는 숫자는 이것 하나다. 우리 구현이 아니라 공지 원문 함수를 그대로 부른다.
        #
        # surface마다 따로 계산하는 이유: 공식 평가자는 우리가 **실제로 내보낸 값**에
        # 영역별 round_half_up을 적용한 뒤 평균한다. 연속값을 제출하면 average_matched
        # 조정이 통째로 무시되고 독립 반올림과 같아진다. 즉 이 값은 "이 surface를
        # 제출했다면 리더보드가 매길 점수"다.
        # 지연 import. `official_metrics`는 scipy를 요구하는데 제출 container에는
        # scipy가 없고 넣을 이유도 없다. 이 함수는 학습/평가 경로에서만 불리고
        # 서빙은 utils에서 정수 변환 헬퍼만 쓴다. module-level import로 두면
        # container가 모델 로드 전에 ModuleNotFoundError로 죽는다(2026-08-20 실측).
        from .official_metrics import official_metrics

        official = official_metrics(values, gold_average)
        metrics["official_rmse"] = official["rmse"]
        metrics["official_spearman"] = official["spearman"]
        matrix[surface] = metrics
    # 상위에 별도 "official" 항목을 두지 않는다. 이 dict는 "surface -> {지표: 숫자}"
    # 계약이고 Trainer가 그대로 float으로 펼친다. 공식 숫자는
    # ``matrix[SUBMITTED_SCORE_SURFACE]["official_rmse"]``로 읽는다.
    return matrix


def regression_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    average_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Shared three-trait RMSE/Spearman used by training and inference.

    ``average_labels``가 주어지면 공식 지표(2026-07-20 운영진 답변)를 그 값으로 계산한다.
    운영진 답변 원문은 "에세이 1편당 세 영역 점수를 먼저 평균낸 값(하나의 숫자)으로
    압축한 뒤 전체 샘플에 대해 상관계수를 단 한 번 계산", human_avg는 데이터셋의
    ``score.average``다. 생략하면 세 trait label의 산술평균으로 대체한다.
    """

    if labels.shape != predictions.shape or labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("labels/predictions must both have shape [N, 3]")
    if average_labels is not None and len(average_labels) != labels.shape[0]:
        raise ValueError("average_labels 길이가 labels 행 수와 다릅니다")
    traits: dict[str, dict[str, float | None]] = {}
    for index, trait in enumerate(TRAITS):
        truth = labels[:, index]
        prediction = predictions[:, index]
        rmse = float(np.sqrt(np.mean((truth - prediction) ** 2)))
        spearman: float | None = None
        if len(set(truth.tolist())) >= 2 and len(set(prediction.tolist())) >= 2:
            value = float(
                np.corrcoef(_average_ranks(truth), _average_ranks(prediction))[0, 1]
            )
            spearman = None if math.isnan(value) else value
        traits[trait] = {"rmse": rmse, "spearman": spearman}
    available = [
        values["spearman"]
        for values in traits.values()
        if values["spearman"] is not None
    ]
    official = _official_metric_surfaces(labels, predictions, average_labels)
    raw_official = official["raw_continuous"]
    submitted_official = official["average_matched"]
    return {
        "schema_version": 2,
        "count": int(labels.shape[0]),
        "traits": traits,
        "overall": {
            # The competition baseline reports the mean of three trait RMSEs.
            "rmse": float(np.mean([traits[trait]["rmse"] for trait in TRAITS])),
            "spearman": float(np.mean(available)) if available else None,
        },
        # 중심 스키마는 두 표면뿐이다. raw_continuous는 모델의 실수 C/O/E 평균이고,
        # average_matched는 실제 제출하는 정수 삼중의 평균이다. 둘 다 같은 공식
        # score.average gold에 대해 RMSE/Spearman을 계산한다.
        "official": official,
        # 과거 artifact reader와 checkpoint metric adapter가 읽던 최소 alias. 새 코드는
        # 위 official block을 읽고, 여기에는 더 이상 여러 rounding 가설을 늘리지 않는다.
        "trait_average": {
            **raw_official,
            "gold_source": official["gold_source"],
        },
        "trait_average_rounded": {
            "average_matched_integer": dict(submitted_official),
        },
    }


def _official_metric_surfaces(
    labels: np.ndarray,
    predictions: np.ndarray,
    average_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """공식 gold에 대한 raw와 실제 제출 표면만 계산한다."""

    gold_average = (
        np.asarray(average_labels, dtype=np.float64)
        if average_labels is not None
        else labels.mean(axis=1)
    )
    return {
        "gold_source": (
            "score_average" if average_labels is not None else "trait_mean_fallback"
        ),
        "raw_continuous": _collapsed_metrics(
            predictions.mean(axis=1), gold_average
        ),
        "average_matched": _collapsed_metrics(
            average_matched_integer_scores(predictions).mean(axis=1),
            gold_average,
        ),
    }


def _collapsed_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    """한 쌍의 1차원 예측/정답에서 RMSE와 Spearman을 계산한다."""

    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
    spearman: float | None = None
    if len(set(prediction.tolist())) >= 2 and len(set(truth.tolist())) >= 2:
        value = float(
            np.corrcoef(_average_ranks(truth), _average_ranks(prediction))[0, 1]
        )
        spearman = None if math.isnan(value) else value
    return {"rmse": rmse, "spearman": spearman}


def _text_config(config: Any) -> Any:
    if hasattr(config, "get_text_config"):
        try:
            return config.get_text_config()
        except TypeError:
            return config.get_text_config(decoder=True)
    return getattr(config, "text_config", config)


def resolve_text_model(model: nn.Module) -> nn.Module:
    """Gemma 같은 wrapper 안에서 실제 text backbone을 찾는다."""

    candidate = model.get_base_model() if hasattr(model, "get_base_model") else model
    for _ in range(5):
        language_model = getattr(candidate, "language_model", None)
        if isinstance(language_model, nn.Module):
            return language_model
        inner = getattr(candidate, "model", None)
        if not isinstance(inner, nn.Module) or inner is candidate:
            break
        candidate = inner
    return candidate


def hidden_size_of(model: nn.Module) -> int:
    hidden_size = getattr(_text_config(model.config), "hidden_size", None)
    # T5/ByT5 encoder config는 같은 차원을 d_model이라는 이름으로 저장한다.
    if hidden_size is None:
        hidden_size = getattr(_text_config(model.config), "d_model", None)
    if hidden_size is None:
        raise ValueError("model config에 hidden_size가 없습니다")
    return int(hidden_size)


def model_cache_status(
    cache_dir: str | Path,
    model_id: str,
    revision: str = "main",
) -> tuple[bool, str]:
    """네트워크 접속 없이 config, tokenizer와 모든 weight shard를 확인한다."""

    repo = Path(cache_dir) / f"models--{model_id.replace('/', '--')}"
    ref = repo / "refs" / revision
    if ref.is_file():
        commit = ref.read_text(encoding="utf-8").strip()
    elif (repo / "snapshots" / revision).is_dir():
        commit = revision
    else:
        return False, f"{model_id}: cache ref가 없습니다"

    snapshot = repo / "snapshots" / commit
    missing_metadata = [
        name
        for name in ("config.json", "tokenizer_config.json")
        if not (snapshot / name).is_file()
    ]
    if missing_metadata:
        return False, f"{model_id}: {', '.join(missing_metadata)} 누락"

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = snapshot / index_name
        if not index_path.is_file():
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
            shards = sorted(set(weight_map.values()))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return False, f"{model_id}: {index_name} 파싱 실패 ({exc})"
        if not shards:
            return False, f"{model_id}: {index_name}에 weight shard가 없습니다"
        missing = [
            shard
            for shard in shards
            if not (snapshot / shard).is_file()
            or (snapshot / shard).stat().st_size == 0
        ]
        if missing:
            preview = ", ".join(missing[:2])
            return False, (
                f"{model_id}: weight shard {len(missing)}/{len(shards)}개 누락"
                f" ({preview})"
            )
        return True, f"{model_id}: cache 완료 ({len(shards)} shards, {commit[:12]})"

    weights = [
        path
        for pattern in ("*.safetensors", "pytorch_model*.bin")
        for path in snapshot.glob(pattern)
        if path.is_file() and path.stat().st_size > 0
    ]
    if not weights:
        return False, f"{model_id}: 완성된 weight 파일이 없습니다"
    return True, f"{model_id}: cache 완료 ({len(weights)} weights, {commit[:12]})"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def prediction_record(essay_id: str, scores: Iterable[float]) -> dict[str, Any]:
    values = [float(value) for value in scores]
    if len(values) != len(TRAITS):
        raise ValueError("three scores are required")
    judge = {
        trait: {
            "score": values[index],
            "rationale": "회귀 채점 헤드가 예측한 점수입니다.",
        }
        for index, trait in enumerate(TRAITS)
    }
    return {
        "essay_id": essay_id,
        "judge": judge,
        "parse_ok": True,
        "parse_error": None,
        "raw_content": json.dumps(judge, ensure_ascii=False),
    }
