"""clip 임계 arm이 전체 데이터 재학습을 받을 자격이 있는지 사전 등록 규칙으로 판정한다.

왜 이 판정이 필요한가
--------------------
c02의 `trainer_state.json`에서 grad_norm이 5.841~19.662인데 `max_grad_norm`이 1.0으로
하드코딩돼 있었다. **기록된 모든 step이 6~20배로 clip된다.** 설정한 학습률이 step마다
다른 배율로 축소되므로 cosine 스케줄이 사실상 작동하지 않는다. 한 번도 재 본 적 없는 축이다.

대조군이 정확히 존재한다
    `g1_h89_base`는 clip=1.0(당시 하드코딩 값), `g2b_clip5`/`g2b_clip10`은 5.0/10.0이고
    그 밖의 모든 설정이 같다(Q8·Q9 보류, 990 step, seed 43). resolved_config 차이는
    `max_grad_norm`과 그때 없던 새 키의 no-op 기본값뿐이다.

왜 짝지은 부트스트랩인가
    같은 400편에 대한 두 예측은 오차가 강하게 상관된다(형제 checkpoint에서 0.998).
    절대 RMSE의 SE는 0.0175지만 **차이**의 SE는 0.0032다. 절대 SE로 판정하면 실재하는
    차이를 전부 놓친다.

판정은 **연속** official RMSE로 한다. `average_matched`의 1/3 격자 스냅이 선택 잡음을
키우기 때문이다(같은 궤적에서 dip 초과분 중앙값 0.01383 -> 0.00978).

`--emit-clip`을 주면 승자의 clip 값만 stdout에 낸다(없으면 아무것도 내지 않는다).
큐 스크립트가 그 값으로 재학습을 걸 수 있다.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GROUP = ROOT / "main_code/results/final_proposed_datasplited"
TRAITS = ("content", "organization", "expression")

# 사전 등록 임계. 차이의 SE는 짝지은 부트스트랩으로 매번 다시 잰다.
RMSE_SE_MULTIPLE = 2.0
SPEARMAN_SE_MULTIPLE = 1.0
BOOTSTRAP_DRAWS = 4000
HELD = ("Q8", "Q9")


def official_gold() -> dict[str, float]:
    matches = sorted((ROOT / "datasets").glob("*2026_validation.jsonl"))
    if not matches:
        raise SystemExit("공식 validation 파일을 찾지 못했습니다")
    gold: dict[str, float] = {}
    for line in matches[0].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        gold[str(row["id"])] = float(row["score"]["average"])
    return gold


def load_arm(name: str, gold: dict[str, float], evaluation: str = "eval_best"):
    path = GROUP / name / evaluation / "score_predictions.jsonl"
    if not path.is_file():
        return None
    ids, prediction, truth, prompts = [], [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row["essay_id"])
        if key not in gold:
            continue
        ids.append(key)
        prediction.append(sum(row["scores"][t] for t in TRAITS) / 3.0)
        truth.append(gold[key])
        prompts.append(row["prompt_num"])
    order = np.argsort(ids)
    return {
        "name": name,
        "ids": np.asarray(ids)[order],
        "prediction": np.asarray(prediction)[order],
        "truth": np.asarray(truth)[order],
        "prompts": np.asarray(prompts)[order],
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or sorted_values[index] != sorted_values[start]:
            ranks[order[start:index]] = (start + index - 1) / 2.0
            start = index
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    return float(np.corrcoef(average_ranks(a), average_ranks(b))[0, 1])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return math.sqrt(float(((a - b) ** 2).mean()))


def paired_bootstrap(control, arm, draws: int = BOOTSTRAP_DRAWS):
    """같은 essay를 함께 리샘플링해 **차이**의 SE를 잰다."""

    if not np.array_equal(control["ids"], arm["ids"]):
        raise SystemExit(f"{arm['name']}: essay 집합이 대조군과 다릅니다")
    truth = control["truth"]
    rng = np.random.default_rng(20260821)
    n = len(truth)
    d_rmse, d_rho = [], []
    for _ in range(draws):
        pick = rng.integers(0, n, n)
        t = truth[pick]
        c, a = control["prediction"][pick], arm["prediction"][pick]
        d_rmse.append(rmse(a, t) - rmse(c, t))
        d_rho.append(spearman(a, t) - spearman(c, t))
    return {
        "rmse_delta": rmse(arm["prediction"], truth) - rmse(control["prediction"], truth),
        "rmse_se": float(np.std(d_rmse, ddof=1)),
        "spearman_delta": spearman(arm["prediction"], truth)
        - spearman(control["prediction"], truth),
        "spearman_se": float(np.nanstd(d_rho, ddof=1)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", default="g1_h89_base")
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["g2b_clip5", "g2b_clip10"],
        help="arm 이름 (clip 값은 이름 끝의 숫자로 읽는다)",
    )
    parser.add_argument("--emit-clip", action="store_true")
    args = parser.parse_args(argv)

    gold = official_gold()
    control = load_arm(args.control, gold)
    if control is None:
        if not args.emit_clip:
            print(f"대조군 {args.control}의 평가 산출물이 없습니다")
        return 2

    def report(*parts):
        if not args.emit_clip:
            print(*parts)

    held_mask = np.isin(control["prompts"], HELD)
    report(f"대조군 {args.control} (clip 1.0)")
    report(f"  전체 400 연속 RMSE {rmse(control['prediction'], control['truth']):.5f}"
           f"  ρ {spearman(control['prediction'], control['truth']):.5f}")
    report(f"  외부 Q8+Q9        RMSE "
           f"{rmse(control['prediction'][held_mask], control['truth'][held_mask]):.5f}")
    report("")

    winners = []
    for name in args.arms:
        arm = load_arm(name, gold)
        if arm is None:
            report(f"{name}: 아직 평가 산출물 없음 — 건너뜀")
            continue
        stats = paired_bootstrap(control, arm)
        held = np.isin(arm["prompts"], HELD)
        rmse_gate = stats["rmse_delta"] <= -RMSE_SE_MULTIPLE * stats["rmse_se"]
        rho_gate = stats["spearman_delta"] >= -SPEARMAN_SE_MULTIPLE * stats["spearman_se"]
        report(f"{name}")
        report(f"  전체 400 연속 RMSE {rmse(arm['prediction'], arm['truth']):.5f}"
               f"  ΔRMSE {stats['rmse_delta']:+.5f}  (2SE {2*stats['rmse_se']:.5f})"
               f"  {'통과' if rmse_gate else '실패'}")
        report(f"  전체 400 ρ         {spearman(arm['prediction'], arm['truth']):.5f}"
               f"  Δρ    {stats['spearman_delta']:+.5f}  (1SE {stats['spearman_se']:.5f})"
               f"  {'통과' if rho_gate else '실패'}")
        report(f"  외부 Q8+Q9        RMSE "
               f"{rmse(arm['prediction'][held], arm['truth'][held]):.5f}")
        if rmse_gate and rho_gate:
            digits = "".join(ch for ch in name if ch.isdigit())
            winners.append((stats["rmse_delta"], float(digits) if digits else 1.0, name))
        report("")

    if not winners:
        report("사전 등록 규칙을 통과한 arm이 없습니다 → 전체 데이터 재학습을 하지 않습니다.")
        return 1
    winners.sort()
    delta, clip, name = winners[0]
    if args.emit_clip:
        print(f"{clip:g}")
    else:
        print(f"승자: {name}  clip={clip:g}  ΔRMSE {delta:+.5f}")
        print("→ 전체 9문항 데이터로 재학습할 자격이 있습니다 (제출 2 후보).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
