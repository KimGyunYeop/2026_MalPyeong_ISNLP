"""제출 후보를 **곡선 통계 + 실제 평가**를 합쳐 줄 세운다.

왜 last3와 최저를 **둘 다** 보나 (2026-08-21 사용자 지정)
    최저값만 보면 c02처럼 한 점 튄 것을 고른다(최저 0.41652, last3 0.43675, 튐 +0.0202).
    그런데 last3만 보면 반대 실수를 한다 — **아직 하강 중인 곡선의 끝**에서 최저가 나온
    경우, 그 최저는 튐이 아니고 "학습이 덜 끝났다"는 뜻이다. 실측 예:
      plat_e2_s43: 최저 0.42729 @ step 1024/1104(93%), last3 0.42794 -> 차이 0.0007.
      **최저가 곧 평탄이다.** 이때는 최저 checkpoint를 실어도 된다.
    그래서 판정은 (last3, 최저, 튐폭, 최저 위치) 네 개를 같이 읽는다.

판정 규칙
    1. 게이트: `튐폭 > 0.008` **이고** `최저 위치 < 0.75`면 탈락(앞쪽 고립 딥).
    2. 1차 정렬은 `last3`. `최저`는 확인·동점 처리에 쓴다.
    3. last3와 최저가 가까운 후보(`튐폭` 작음)를 선호한다 — 그 숫자가 믿을 만하다.
    4. **연속 RMSE를 최종 확인에 쓴다.** 제출 표면은 1/3 격자 때문에 ±0.004의 양자화
       잡음이 얹힌다(실측: 같은 e2에서 평탄 0.42932 / 최저 0.42509인데 연속은
       0.42675 / 0.42679로 사실상 동일). 재실행 SD도 연속 0.0055 대 제출 0.0138이다.

usage:
    python -m main_code.rank_candidates                       # results/final_proposed
    python -m main_code.rank_candidates <group_dir> ...
"""

from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .analyze_eval_plateau import curve

ROOT = Path(__file__).resolve().parent.parent
TRAITS = ("content", "organization", "expression")
SPIKE_MAX = 0.008
ARGMIN_MIN = 0.75
# 배포 c02. R15의 반례로 표 아래에 항상 같이 찍는다.
C02 = {"last3": 0.43675, "best": 0.41652, "spike": 0.0202, "argmin": 0.57,
       "sub": 0.416824, "cont": 0.425007}


def gold_map() -> dict[str, float]:
    source = glob.glob(str(ROOT / "datasets/*2026_validation.jsonl"))[0]
    out = {}
    for line in open(source, encoding="utf-8"):
        row = json.loads(line)
        out[str(row["id"])] = float(row["score"]["average"])
    return out


GOLD = gold_map()


def measure(pred_path: Path) -> dict | None:
    rows = []
    for line in open(pred_path, encoding="utf-8"):
        row = json.loads(line)
        rid = str(row.get("essay_id") or row.get("id") or "")
        if rid in GOLD and "scores" in row:
            rows.append((rid, [float(row["scores"][t]) for t in TRAITS]))
    if len(rows) != len(GOLD):
        return None
    rows.sort()
    cont = np.array([v for _, v in rows])
    gold = np.array([GOLD[i] for i, _ in rows])
    matched = np.clip(np.floor(cont.sum(axis=1) + 0.5), 3.0, 15.0) / 3.0
    mean_cont = cont.mean(axis=1)
    return {
        "sub_rmse": math.sqrt(float(((matched - gold) ** 2).mean())),
        "sub_rho": float(spearmanr(matched, gold).statistic),
        "cont_rmse": math.sqrt(float(((mean_cont - gold) ** 2).mean())),
        "cont_rho": float(spearmanr(mean_cont, gold).statistic),
        "pred_mean": float(matched.mean()),
        "pred_sd": float(matched.std(ddof=1)),
    }


def main() -> int:
    groups = [Path(a) for a in sys.argv[1:]] or [ROOT / "main_code/results/final_proposed"]
    rows = []
    for group in groups:
        for run_dir in sorted(p for p in group.glob("*") if p.is_dir() and not p.name.startswith("_")):
            state = run_dir / "trainer/trainer_state.json"
            c = curve(state) if state.is_file() else None
            evals = {}
            for tag in ("plateau", "spike", "micro", "last"):
                pred = run_dir / f"eval_{tag}/score_predictions.jsonl"
                if pred.is_file():
                    got = measure(pred)
                    if got:
                        evals[tag] = got
            if not c and not evals:
                continue
            rows.append({"arm": run_dir.name, "curve": c, "evals": evals})

    if not rows:
        print("후보 없음 (아직 학습 중이거나 평가 전)")
        return 0

    done = [r for r in rows if r["curve"]]
    done.sort(key=lambda r: r["curve"]["last3"])
    print(f"후보 {len(rows)}개 (곡선 있음 {len(done)}개)\n")
    print("=== 곡선 통계 — last3와 최저를 같이 본다 ===")
    print(f"{'arm':16s} {'last3':>8} {'최저':>8} {'튐폭':>8} {'최저위치':>8} {'eval':>4} {'게이트':>6}")
    print("-" * 70)
    for r in done:
        c = r["curve"]
        gate = "탈락" if (c["spike"] > SPIKE_MAX and c["argmin_frac"] < ARGMIN_MIN) else "통과"
        print(f"{r['arm'][:16]:16s} {c['last3']:8.5f} {c['best']:8.5f} {c['spike']:+8.5f} "
              f"{c['argmin_frac']:8.2f} {c['n_eval']:4d} {gate:>6}")
    print(f"{'배포 c02 (참조)':16s} {C02['last3']:8.5f} {C02['best']:8.5f} {C02['spike']:+8.5f} "
          f"{C02['argmin']:8.2f} {7:4d} {'탈락':>6}")

    print(f"\n=== 실제 평가 — 연속 RMSE가 1차 확인 지표 (제출 표면은 ±0.004 격자 잡음) ===")
    print(f"{'arm':16s} {'checkpoint':10s} {'연속RMSE':>9} {'연속rho':>8} {'제출RMSE':>9} {'제출rho':>8} {'예측평균':>8}")
    print("-" * 82)
    for r in done + [x for x in rows if not x["curve"]]:
        for tag, v in sorted(r["evals"].items()):
            print(f"{r['arm'][:16]:16s} {tag:10s} {v['cont_rmse']:9.5f} {v['cont_rho']:8.5f} "
                  f"{v['sub_rmse']:9.5f} {v['sub_rho']:8.5f} {v['pred_mean']:8.4f}")
    print(f"{'배포 c02':16s} {'step736':10s} {C02['cont']:9.5f} {0.750039:8.5f} "
          f"{C02['sub']:9.5f} {0.759737:8.5f} {3.3883:8.4f}")

    passing = [r for r in done
               if not (r["curve"]["spike"] > SPIKE_MAX and r["curve"]["argmin_frac"] < ARGMIN_MIN)]
    if passing:
        print(f"\n=== 게이트 통과 후보 {len(passing)}개, last3 순 ===")
        for i, r in enumerate(passing, 1):
            c = r["curve"]
            cont = min((v["cont_rmse"] for v in r["evals"].values()), default=float("nan"))
            print(f"  {i}. {r['arm']:18s} last3 {c['last3']:.5f}  최저 {c['best']:.5f} "
                  f"(튐 {c['spike']:+.5f}, 위치 {c['argmin_frac']:.2f})  연속 {cont:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
