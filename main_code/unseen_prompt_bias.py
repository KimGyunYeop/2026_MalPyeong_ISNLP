"""문항을 보류한 fold에서 **미학습 문항의 편향 δ**를 추정한다.

왜 이게 값어치가 있는가
    리더보드 격차 MSE 0.0957의 분해에서 평균 이동이 28.7%였다. 그런데 예측에
    상수를 더하는 것은 **Spearman을 전혀 바꾸지 않는다**(단조 변환). 공식 종합은
    RMSE 45% + Spearman 45%이므로, δ를 알면 45%를 개선하면서 45%를 하나도 잃지
    않는다. 이만큼 비대칭적으로 유리한 수는 다른 데 없다.

    문제는 δ를 모른다는 것이었다. 2025 보고서의 평균(3.5547)에서 역산하는 것은
    도박이다 — 그 차이의 92%가 신규 문항 Q11·Q12가 우리 전체보다 +0.20 높은 데서
    오고, 숨은 test의 문항 구성을 알 수 없다.

    이 파일은 다른 경로를 쓴다. 문항을 학습에서 **보류한** 모델이 그 문항에서
    보이는 편향을 직접 재면, "문항을 못 본 모델이 그 문항에서 갖는 편향"의 표본이
    된다. 보고서 숫자에 기대지 않고 우리 데이터만으로 추정한다.

    같은 fold에서 **본 문항**의 편향도 함께 재는 것이 핵심이다. 두 값의 차이가
    "문항을 못 봤다는 사실만의 효과"이고, 모델·seed·step에 공통인 편향은 상쇄된다.

이것이 내 사후 보정 부정 결과와 모순되지 않는 이유
    앞서 공식 400편을 5겹으로 나눠 isotonic/아핀을 교차적합했더니 전부 악화됐다.
    그건 **본 문항 안에서의** 보정이었고 거기에는 고칠 오차가 없었다
    (E[정답|예측]이 예측값과 ±0.08 이내로 일치). 지금 재는 것은 **못 본 문항의**
    편향이다. 서로 다른 양이고, 앞의 결과가 이것을 반증하지 않는다.

usage: python -m main_code.unseen_prompt_bias
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .official_metrics import official_pred_avg

ROOT = Path(__file__).resolve().parent.parent
GROUP = ROOT / "main_code/results/new_proposed/c02_lopo_v1"
BASELINE = (
    ROOT
    / "main_code/results/new_proposed/y6_final_combo_noext_s43_v1/c02_avg_mse025"
    / "ax4_light/a228dbe60ed8/eval_best_official_rmse/score_predictions.jsonl"
)
# arm 이름 -> (보류한 문항, held 평가 디렉터리, seen 평가 디렉터리)
ARMS = {
    "x01_lopo_q89": (("Q8", "Q9"), "eval_Q89", "eval_Q1toQ7"),
    "x02_lopo_q89_essayonly": (("Q8", "Q9"), "eval_Q89", "eval_Q1toQ7"),
    "f123_plain": (("Q1", "Q2", "Q3"), "eval_held", "eval_seen"),
    "f123_essayonly": (("Q1", "Q2", "Q3"), "eval_held", "eval_seen"),
    "f45_plain": (("Q4", "Q5"), "eval_held", "eval_seen"),
    "f45_essayonly": (("Q4", "Q5"), "eval_held", "eval_seen"),
}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def official_rows() -> dict[str, dict]:
    source = glob.glob(str(ROOT / "datasets/*2026_validation.jsonl"))[0]
    return {row["id"]: row for row in _rows(Path(source))}


def load(path: Path) -> dict[str, float]:
    return {
        row["essay_id"]: float(official_pred_avg(row["submitted_scores"]))
        for row in _rows(path)
    }


def summarize(prediction: dict[str, float], official: dict[str, dict]) -> dict:
    ids = sorted(prediction)
    p = np.array([prediction[i] for i in ids])
    g = np.array([float(official[i]["score"]["average"]) for i in ids])
    return {
        "n": len(ids),
        "bias": float(np.mean(p - g)),
        "rmse": math.sqrt(float(np.mean((p - g) ** 2))),
        "spearman": float(spearmanr(p, g)[0]) if len(set(p.tolist())) > 1 else float("nan"),
        "pred_sd": float(p.std()),
        "gold_sd": float(g.std()),
    }


def main() -> int:
    official = official_rows()
    base = load(BASELINE)

    print("=== 배포 c02 (모든 문항을 학습에서 본 모델)의 문항군별 편향 ===")
    for arm, (held, _, _) in ARMS.items():
        if not arm.endswith("_plain") and arm != "x01_lopo_q89":
            continue
        ids = [i for i in base if official[i]["prompt_num"] in held]
        p = np.array([base[i] for i in ids])
        g = np.array([float(official[i]["score"]["average"]) for i in ids])
        print(
            f"  {'+'.join(held):12s} n={len(ids):3d}  편향 {float(np.mean(p-g)):+.4f}  "
            f"RMSE {math.sqrt(float(np.mean((p-g)**2))):.5f}"
        )
    print()

    print("=== 문항 보류 fold: 보류 문항(held) vs 본 문항(seen) ===")
    header = (
        f"{'arm':24s} {'보류':12s} | {'held n':>6s} {'held 편향':>9s} {'held RMSE':>9s} "
        f"{'held ρ':>7s} | {'seen 편향':>9s} | {'δ = held-seen':>13s}"
    )
    print(header)
    print("-" * len(header))
    deltas: list[tuple[str, float]] = []
    for arm, (held, held_dir, seen_dir) in ARMS.items():
        held_path = GROUP / arm / held_dir / "score_predictions.jsonl"
        seen_path = GROUP / arm / seen_dir / "score_predictions.jsonl"
        if not held_path.is_file() or not seen_path.is_file():
            print(f"{arm:24s} {'+'.join(held):12s} | (아직 없음)")
            continue
        h = summarize(load(held_path), official)
        s = summarize(load(seen_path), official)
        delta = h["bias"] - s["bias"]
        deltas.append((arm, delta))
        print(
            f"{arm:24s} {'+'.join(held):12s} | {h['n']:6d} {h['bias']:+9.4f} "
            f"{h['rmse']:9.5f} {h['spearman']:7.4f} | {s['bias']:+9.4f} | {delta:+13.4f}"
        )
    print()

    if not deltas:
        print("fold 결과가 아직 없습니다.")
        return 1

    plain = [d for a, d in deltas if not a.endswith("essayonly")]
    essay = [d for a, d in deltas if a.endswith("essayonly")]
    print("=== δ 추정 (미학습 문항이라는 사실만의 편향) ===")
    for name, values in (("지문 사용 (기본)", plain), ("지문 제거 (essay_only)", essay)):
        if not values:
            continue
        array = np.array(values)
        print(
            f"  {name:24s} fold {len(array)}개  평균 {array.mean():+.4f}  "
            f"폭 {array.min():+.4f}~{array.max():+.4f}  표준편차 {array.std(ddof=1) if len(array) > 1 else float('nan'):.4f}"
        )
    print()
    print("해석 지침")
    print("  · |평균 δ|가 fold 사이 폭보다 크면 방향이 일관된 것이고, 그만큼을 배포")
    print("    모델 예측에 더하는 것이 Spearman을 잃지 않고 RMSE를 줄이는 수가 된다.")
    print("  · 폭이 평균보다 크면 δ의 방향이 문항마다 다르다는 뜻이므로 상수 보정을")
    print("    하면 안 된다. 그 경우 이 경로는 닫힌다.")
    print("  · 지문 제거 쪽의 |δ|가 더 작으면, 미학습 문항 편향의 상당 부분이 지문을")
    print("    읽는 데서 온다는 증거다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
