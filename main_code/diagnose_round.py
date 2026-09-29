"""라운드 결과를 "왜 안 움직였는가"까지 분해하는 읽기 전용 진단기.

`aggregate_results.py`가 arm별 지표를 표로 만드는 반면 이 스크립트는 **오차 예산**을
답한다. 어떤 arm이 이겼는지가 아니라 남은 오차가 어디에 있고 어떤 축이 그것을 건드릴 수
있는지를 본다. 다음 다섯 가지를 계산한다.

1. 오차 예산   전체 RMSE를 라벨 노이즈 / 모든 arm이 공유하는 오차 / arm 고유 오차로 나눈다.
               공유 오차가 지배적이면 head·목적함수 sweep은 원리적으로 이길 수 없다.
2. 앙상블 천장 arm 간 잔차 상관에서 예측 평균으로 얻을 수 있는 최대 이득을 미리 계산한다.
3. 제출 규칙   정수 삼중 선택 규칙이 RMSE와 Spearman을 어떻게 맞바꾸는지 직접 잰다.
4. 표면 신호   잔차가 길이·문단수 같은 표면 특징으로 설명되면 그 정보는 입력에 없는 것이다.
5. 선택 편향   400편 하나로 checkpoint 선택과 arm 순위를 동시에 정하면 얼마나 낙관적인지를
               절반 분할로 잰다.

사용:
    PYTHONPATH=$PWD python3 main_code/diagnose_round.py \
        --round results/new_proposed/ax4light_r4_trait3_int5_r32mlp_s1104_official_raw_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import PACKAGE_ROOT, TRAITS
from .utils import average_matched_integer_scores, per_trait_integer_scores

VALIDATION_PATH = PACKAGE_ROOT / "datasets" / "processed_dataset" / "validation.jsonl"


# --- 공통 계산 --------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    # str.splitlines()는  / 같은 문자에서도 쪼개 official_raw essay를 깨뜨린다.
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    """utils의 tie-aware 평균 rank를 그대로 쓴다."""

    from .utils import _average_ranks

    if len(set(left.tolist())) < 2 or len(set(right.tolist())) < 2:
        return float("nan")
    return float(np.corrcoef(_average_ranks(left), _average_ranks(right))[0, 1])


def _rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(((prediction - truth) ** 2).mean()))


def load_arms(round_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """arm 이름 -> [N,3] 예측, 그리고 공통 gold를 돌려준다.

    essay_id로 정렬해 arm 사이 행 순서가 어긋나지 않게 한다.
    """

    # 정답은 예측 파일 안의 labels를 쓴다. holdout을 붙인 라운드는 평가 집합이 공식
    # 400편이 아니므로 외부 파일에서 찾으면 KeyError가 난다. labels가 없는 옛 artifact만
    # 공식 validation으로 되돌린다.
    fallback = {row["id"]: row for row in _read_jsonl(VALIDATION_PATH)}
    arms: dict[str, np.ndarray] = {}
    gold: np.ndarray | None = None
    order: list[str] | None = None
    for arm_dir in sorted(p for p in round_dir.iterdir() if p.is_dir()):
        hits = sorted(arm_dir.glob("*/*/eval/score_predictions.jsonl"))
        if not hits:
            continue
        rows = sorted(_read_jsonl(hits[0]), key=lambda row: row["essay_id"])
        ids = [row["essay_id"] for row in rows]
        if order is None:
            order = ids
            gold = np.array(
                [
                    [(row.get("labels") or fallback[row["essay_id"]]["score"])[t] for t in TRAITS]
                    for row in rows
                ],
                dtype=np.float64,
            )
        elif ids != order:
            raise ValueError(f"{arm_dir.name}의 essay 집합이 다른 arm과 다릅니다")
        arms[arm_dir.name] = np.array(
            [[row["scores"][t] for t in TRAITS] for row in rows], dtype=np.float64
        )
    if gold is None or order is None:
        raise SystemExit(f"{round_dir}에 예측이 없습니다")
    return arms, gold, order


def label_noise_floor(essay_ids: list[str]) -> float | None:
    """2인 평가자 평균 라벨의 노이즈 sd = 어떤 모델도 내려갈 수 없는 RMSE 하한.

    두 평가자의 세 trait 평균 차이 d에 대해 var(d)=2*sigma_r^2 이고 2인 평균의 노이즈는
    sigma_r/sqrt(2) = sd(d)/2 다.
    """

    gold_rows = {row["id"]: row for row in _read_jsonl(VALIDATION_PATH)}
    differences = []
    for essay_id in essay_ids:
        if essay_id not in gold_rows:  # holdout essay는 공식 validation에 없다
            continue
        details = gold_rows[essay_id]["score_details"]
        table = details.get("rater_trait_scores") or {}
        raters = details.get("primary_raters") or []
        if len(raters) != 2 or any(name not in table for name in raters):
            continue
        values = [
            [table[name][trait] for trait in TRAITS]
            for name in raters
        ]
        if any(value is None for row in values for value in row):
            continue
        differences.append(np.mean(values[0]) - np.mean(values[1]))
    if len(differences) < 30:
        return None
    return float(np.std(differences, ddof=1) / 2.0)


# --- 1. 오차 예산 -----------------------------------------------------------
def error_budget(arms: dict[str, np.ndarray], gold: np.ndarray, floor: float | None) -> None:
    truth = gold.mean(axis=1)
    residuals = {name: pred.mean(axis=1) - truth for name, pred in arms.items()}
    names = list(residuals)
    total = float(np.mean([np.sqrt((residuals[n] ** 2).mean()) for n in names]))

    pairs = [
        float(np.corrcoef(residuals[a], residuals[b])[0, 1])
        for index, a in enumerate(names)
        for b in names[index + 1 :]
    ]
    shared_fraction = float(np.mean(pairs)) if pairs else 1.0
    shared_variance = shared_fraction * total ** 2
    unique_sd = float(np.sqrt(max(total ** 2 - shared_variance, 0.0)))

    print("\n[1] 오차 예산")
    print(f"  arm {len(names)}개 평균 RMSE            {total:.4f}")
    print(f"  arm 간 잔차 상관 평균                {shared_fraction:.4f}")
    if floor is not None:
        model_shared = float(np.sqrt(max(shared_variance - floor ** 2, 0.0)))
        print(f"  = 라벨 노이즈 (내릴 수 없음)        {floor:.4f}")
        print(f"  + 모든 arm이 공유하는 모델 오차     {model_shared:.4f}   <- 여기가 진짜 목표")
    print(f"  + arm 고유 오차                     {unique_sd:.4f}")
    if unique_sd > 0:
        print(
            f"  -> arm 하나를 바꿔서 움직일 수 있는 최대치는 {unique_sd:.4f} 규모다. "
            f"공유 오차를 건드리는 축(backbone/데이터/입력/예산)이 아니면 이 위로 못 간다."
        )


# --- 2. 앙상블 천장 ---------------------------------------------------------
def ensemble_ceiling(arms: dict[str, np.ndarray], gold: np.ndarray) -> None:
    truth = gold.mean(axis=1)
    names = list(arms)
    print("\n[2] 앙상블 천장 (예측 평균)")
    best_single = min(_rmse(arms[n].mean(axis=1), truth) for n in names)
    for count in (2, 3, len(names)):
        if count > len(names) or count < 2:
            continue
        chosen = names[:count] if count == len(names) else None
        if chosen is None:
            # 탐색 없이 상한만 본다: 가장 상관이 낮은 조합을 고른다.
            residuals = {n: arms[n].mean(axis=1) - truth for n in names}
            picked = [min(names, key=lambda n: _rmse(arms[n].mean(axis=1), truth))]
            while len(picked) < count:
                picked.append(
                    min(
                        (n for n in names if n not in picked),
                        key=lambda n: float(
                            np.mean([np.corrcoef(residuals[n], residuals[p])[0, 1] for p in picked])
                        ),
                    )
                )
            chosen = picked
        averaged = np.mean([arms[n] for n in chosen], axis=0)
        matched = average_matched_integer_scores(np.clip(averaged, 1, 5)).mean(axis=1)
        print(
            f"  {count:2d}개 평균  연속 {_rmse(averaged.mean(axis=1), truth):.4f}"
            f"  제출 {_rmse(matched, truth):.4f}/{_spearman(matched, truth):.4f}"
            f"   ({', '.join(chosen[:3])}{'...' if len(chosen) > 3 else ''})"
        )
    print(f"  단일 최고 연속 RMSE {best_single:.4f} 대비 개선폭이 seed 노이즈보다 작으면 "
          f"앙상블은 비용만 늘린다.")


# --- 3. 제출 규칙 -----------------------------------------------------------
def submission_rule(arms: dict[str, np.ndarray], gold: np.ndarray, arm: str) -> None:
    truth = gold.mean(axis=1)
    pred = arms[arm]
    print(f"\n[3] 제출 규칙 ({arm})")
    print(f"  {'규칙':30} {'RMSE':>7} {'rho':>7} {'단계':>5} {'동점쌍':>7}")
    for name, out in (
        ("연속(제출 불가, 참고)", pred),
        ("영역별 사사오입", per_trait_integer_scores(pred)),
        ("평균 정합 정수 삼중(현행)", average_matched_integer_scores(pred)),
    ):
        mean = out.mean(axis=1)
        _, counts = np.unique(np.round(mean, 6), return_counts=True)
        ties = float((counts * (counts - 1)).sum() / (len(mean) * (len(mean) - 1)))
        print(
            f"  {name:30} {_rmse(mean, truth):7.4f} {_spearman(mean, truth):7.4f}"
            f" {len(counts):5d} {ties:7.1%}"
        )
    ceiling = average_matched_integer_scores(gold).mean(axis=1)
    print(f"  gold를 같은 격자에 올렸을 때의 rho 천장 {_spearman(ceiling, truth):.4f}")

    # 단조 확대는 순위를 보존하므로 Spearman만 바꾼다. RMSE와의 교환비를 직접 본다.
    center = float(pred.mean())
    print(f"  {'단조 확대 k':30} {'RMSE':>7} {'rho':>7}")
    for k in (0.9, 1.0, 1.1, 1.25):
        out = average_matched_integer_scores(np.clip(center + k * (pred - center), 1, 5))
        mean = out.mean(axis=1)
        print(f"  {'  k=' + format(k, '.2f'):30} {_rmse(mean, truth):7.4f} {_spearman(mean, truth):7.4f}")


# --- 4. 표면 신호 -----------------------------------------------------------
def surface_signal(arms: dict[str, np.ndarray], gold: np.ndarray, essay_ids: list[str], arm: str) -> None:
    gold_rows = {row["id"]: row for row in _read_jsonl(VALIDATION_PATH)}
    residual = arms[arm].mean(axis=1) - gold.mean(axis=1)
    metadata = [gold_rows[i]["metadata"] for i in essay_ids]
    features = {
        key: np.array([float(m.get(key) or 0) for m in metadata])
        for key in ("written_length", "paragraph_count", "sentence_count", "grade")
    }
    columns = [np.ones(len(residual))] + [
        (v - v.mean()) / (v.std() + 1e-9) for v in features.values() if v.std() > 0
    ]
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, residual, rcond=None)
    explained = 1.0 - ((residual - design @ coefficients) ** 2).mean() / (residual ** 2).mean()
    print(f"\n[4] 표면 특징이 잔차를 설명하는 정도 ({arm})")
    for key, values in features.items():
        if values.std() == 0:
            continue
        print(f"  corr(잔차, {key:16}) = {_spearman(values, residual):+.3f}")
    print(f"  선형결합 R2 = {explained:.2%}"
          f"   (특징 {len(columns)-1}개/표본 {len(residual)}개면 우연만으로도 "
          f"{(len(columns)-1)/len(residual):.2%})")


# --- 5. 선택 편향 -----------------------------------------------------------
def selection_bias(arms: dict[str, np.ndarray], gold: np.ndarray, seed: int = 0) -> None:
    """절반에서 고른 승자를 나머지 절반에서 재평가한다.

    같은 400편으로 arm을 고르고 그 값을 보고하면 최고값은 낙관적으로 치우친다.
    분할을 여러 번 반복해 그 치우침의 크기를 잰다.
    """

    truth = gold.mean(axis=1)
    names = list(arms)
    if len(names) < 2:
        return
    rng = np.random.default_rng(seed)
    optimism = []
    for _ in range(200):
        order = rng.permutation(len(truth))
        left, right = order[: len(order) // 2], order[len(order) // 2 :]
        scores = {n: _rmse(arms[n].mean(axis=1)[left], truth[left]) for n in names}
        winner = min(scores, key=scores.get)
        held_out = _rmse(arms[winner].mean(axis=1)[right], truth[right])
        optimism.append(held_out - scores[winner])
    print("\n[5] 승자 선택의 낙관 편향 (절반 선택 -> 나머지 절반 재평가, 200회)")
    print(f"  중앙값 {np.median(optimism):+.4f}   평균 {np.mean(optimism):+.4f}"
          f"   (arm {len(names)}개)")
    print("  같은 표본으로 고르고 보고한 값은 이만큼 실제보다 좋아 보인다.")


# --- 6. 승격 판정 -----------------------------------------------------------
def _paired_se(arm: np.ndarray, control: np.ndarray, truth: np.ndarray, draws: int) -> float:
    """ΔRMSE의 짝지은 bootstrap 표준오차.

    seed 노이즈는 "같은 검증 집합에서 seed를 바꾸면"만 잰다. 목표는 히든 test이고
    그쪽도 400편 표본이므로 **검증 표본 오차**가 반드시 들어가야 한다.
    """

    rng = np.random.default_rng(0)
    boots = []
    for _ in range(draws):
        index = rng.integers(0, len(truth), len(truth))
        boots.append(_rmse(arm[index], truth[index]) - _rmse(control[index], truth[index]))
    return float(np.std(boots, ddof=1))


def promotion(
    arms: dict[str, np.ndarray],
    gold: np.ndarray,
    control: str,
    *,
    exclude: tuple[str, ...] = (),
    draws: int = 2000,
) -> None:
    """유의성 문턱 대신 경험적 베이즈 사후평균으로 승격을 판정한다.

    최종 점수는 지표별 **순위**이고 리더보드 인접 팀 간격이 RMSE 약 `.006`이다.
    무효 옵션을 채택하는 비용은 0이고 진짜 개선을 버리는 비용은 순위 하나이므로
    "유의한가"가 아니라 "사후평균이 유리한가"를 물어야 한다.

    팔 사이 진짜 효과 분산 `tau^2 = max(0, var(delta) - mean(SE^2))`를 추정하고
    `delta * tau^2/(tau^2+SE^2)`로 수축한다. 모든 팔이 사실 무효면 `tau^2`가 0이 되어
    자동으로 아무것도 승격되지 않는다. 반대로 축이 살아 있으면 1σ 미만도 승격된다.

    ``exclude``에는 **기전이 다른** 팔을 넣는다(예: argmax는 readout 자체를 깬다).
    그런 팔은 후보가 아니면서 `tau^2`만 부풀려 나머지 팔에 없는 신뢰를 준다.
    """

    truth = gold.mean(axis=1)
    reference = arms[control].mean(axis=1)
    names = [name for name in arms if name != control and name not in exclude]
    if not names:
        return
    deltas = np.array([_rmse(arms[n].mean(axis=1), truth) - _rmse(reference, truth) for n in names])
    errors = np.array([_paired_se(arms[n].mean(axis=1), reference, truth, draws) for n in names])
    noise = float((errors ** 2).mean())
    tau2 = max(float(deltas.var(ddof=1)) - noise, 0.0)
    factor = tau2 / (tau2 + noise) if tau2 + noise else 0.0

    print(f"\n[6] 승격 판정 (control={control}, 경험적 베이즈)")
    if exclude:
        print(f"  tau^2 추정에서 제외한 팔: {', '.join(exclude)}")
    print(f"  관측 delta 분산 {deltas.var(ddof=1):.3e}   노이즈 분산 {noise:.3e}")
    print(f"  tau^2 {tau2:.3e}   수축계수 {factor:.3f}")
    if factor == 0:
        print("  -> 팔 사이 흩어짐이 노이즈보다 작다. 이 축에는 승격할 것이 없다.")
        print("     자를 바꾸지 않는 한(검증 확대) 몇 번을 돌려도 같은 결과가 나온다.")
    print(f"  {'arm':30} {'ΔRMSE':>9} {'SE':>8} {'사후평균':>10} {'판정':>6}")
    for index in np.argsort(deltas):
        posterior = factor * deltas[index]
        verdict = "승격" if posterior < 0 else ("기각" if posterior > 0 else "동률")
        print(f"  {names[index]:30} {deltas[index]:+9.4f} {errors[index]:8.4f}"
              f" {posterior:+10.4f} {verdict:>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", required=True, help="arm 폴더들을 담은 라운드 디렉터리")
    parser.add_argument("--arm", default="", help="규칙/잔차 분석에 쓸 arm (기본: 연속 RMSE 최소)")
    parser.add_argument("--skip", default="", help="제외할 arm 이름 (쉼표 구분)")
    parser.add_argument("--control", default="", help="승격 판정의 대조 arm (기본: 이름에 baseline/control)")
    parser.add_argument(
        "--exclude-from-tau",
        default="",
        help="tau^2 추정에서 뺄 arm (쉼표 구분). 기전이 다른 팔만 넣는다",
    )
    args = parser.parse_args()

    round_dir = Path(args.round)
    if not round_dir.is_absolute():
        for candidate in (Path.cwd() / round_dir, PACKAGE_ROOT / round_dir):
            if candidate.is_dir():
                round_dir = candidate
                break
    arms, gold, essay_ids = load_arms(round_dir)
    for name in filter(None, (s.strip() for s in args.skip.split(","))):
        arms.pop(name, None)
    truth = gold.mean(axis=1)
    arm = args.arm or min(arms, key=lambda n: _rmse(arms[n].mean(axis=1), truth))

    print(f"# {round_dir.name}   arm {len(arms)}개, essay {len(truth)}편")
    floor = label_noise_floor(essay_ids)
    error_budget(arms, gold, floor)
    ensemble_ceiling(arms, gold)
    submission_rule(arms, gold, arm)
    surface_signal(arms, gold, essay_ids, arm)
    selection_bias(arms, gold)

    control = args.control or next(
        (name for name in arms if "baseline" in name or "control" in name), ""
    )
    if control:
        promotion(
            arms,
            gold,
            control,
            exclude=tuple(filter(None, (s.strip() for s in args.exclude_from_tau.split(",")))),
        )
    else:
        print("\n[6] 승격 판정: control arm을 찾지 못했습니다 (--control로 지정하세요)")


if __name__ == "__main__":
    main()
