"""제출 후보 집계 — 곡선 통계(last3·last5·최저)와 실제 평가를 한 표로 낸다.

왜 별도 집계인가
    기존 `aggregate_results*.py`는 `run.json`만 읽어서 metric별 **best 하나**만 안다.
    곡선은 `trainer/trainer_state.json`의 `log_history`에만 있고(294개 생존) 집계기가
    그것을 안 읽는다. 그래서 "앞에서 한 번 튄 것"과 "평탄하게 좋은 것"을 구분할 수
    없었고, 그 결과 배포 c02(57% 지점 튐)를 골랐다.

무엇을 합치나
    1. `results/final_proposed/` 아래 새로 학습한 후보
    2. `--reference` 로 지정한 **기존 실험 기록**(재학습 없이 복사). 데이터 구성과
       학습량이 같아야 같은 표에 세울 수 있으므로 profile·holdout·surface·steps를
       함께 적고, 대상 학습량과 다르면 `steps_diff` 열에 표시한다.

판정 규약 (R15 / R15-a)
    - 1차: **연속** 표면. 제출 표면은 1/3 격자에서 ±0.004 양자화 잡음이 얹힌다.
    - `last3`의 창 폭은 eval 횟수에 의존하므로 `final`도 같이 본다.
    - 게이트: `튐폭 > 0.008` **이고** `최저 위치 < 0.75`면 탈락(앞쪽 고립 딥).
    - 게이트를 통과하면 최저 checkpoint == 사실상 마지막이므로 그것을 쓴다.

usage:
    python -m main_code.aggregate_candidates             # 표 출력 + md/csv 저장
    python -m main_code.aggregate_candidates --quiet      # 저장만 (watcher용)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .analyze_eval_plateau import curve

ROOT = Path(__file__).resolve().parent.parent
TRAITS = ("content", "organization", "expression")
GROUP = ROOT / "main_code/results/final_proposed"
SPIKE_MAX, ARGMIN_MIN = 0.008, 0.75
TARGET_STEPS_NOTE = "학습량이 다르면 같은 줄에서 성능만 비교하지 않는다"

# 재학습 없이 기록만 가져올 기존 실험. 전부 primary_data_profile=full / official_raw /
# holdout 없음 — 즉 **데이터 구성이 오늘 후보와 동일**하다. 학습량(step)만 다르다.
REFERENCES = {
    "ref_y1f4_s42": "new_proposed/ax4light_final_combo_s1288_official_raw_v1/y1_f4_rank_metric/ax4_light/5f475fbd3458",
    "ref_e5_s42": "new_proposed/ax4light_r4_trait3_int5_r32mlp_s1104_official_raw_v1/e5_listwise_spearman020/ax4_light/a2f177a5bea4",
    "ref_attn_s42": "new_proposed/y1_postsubmit_final10_noext_s42_v1/a1_y1_r32_attention/ax4_light/941c52b06aef",
    "ref_e11_s42": "new_proposed/ax4light_r4_trait3_int5_r32mlp_s1104_official_raw_v1/e11_pairwise_avg_target/ax4_light/ea573c750533",
    "ref_e3_s42": "new_proposed/ax4light_r4_trait3_int5_r32mlp_s1104_official_raw_v1/e3_label_smoothing_010/ax4_light/10339354644a",
    "ref_e4_s42": "new_proposed/ax4light_r4_trait3_int5_r32mlp_s1104_official_raw_v1/e4_pairwise_ranknet020/ax4_light/f1bf81204c28",
    "ref_e1_s42": "new_proposed/ax4light_r4_trait3_int5_r32mlp_s1104_official_raw_v1/e1_trait_average_050/ax4_light/612d99e1107a",
    "ref_d6_s42": "new_proposed/ax4light_r3_headunit_r32mlp_s1104_official_raw_v1/d6_rater18_int5/ax4_light/fbc8b9145c11",
    "ref_d9_s42": "new_proposed/ax4light_r3_headunit_r32mlp_s1104_official_raw_v1/d9_crit9_auxiliary/ax4_light/8be98f11eb62",
    "ref_e2_s42": "new_proposed/ax4light_r4_trait3_int5_r32mlp_s1104_official_raw_v1/e2_distribution_weight_050/ax4_light/047cd6b4f6b2",
    "ref_c02_s43": "new_proposed/y6_final_combo_noext_s43_v1/c02_avg_mse025/ax4_light/a228dbe60ed8",
    "ref_y6_s43": "new_proposed/ax4light_final_combo_s1288_official_raw_v1/y6_int5_rater18aux_seed43/ax4_light/87078367e276",
}


def gold_map() -> dict[str, float]:
    source = glob.glob(str(ROOT / "datasets/*2026_validation.jsonl"))[0]
    return {str(json.loads(l)["id"]): float(json.loads(l)["score"]["average"])
            for l in open(source, encoding="utf-8")}


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
    }


def collect_one(name: str, run_dir: Path, kind: str) -> dict | None:
    cfg_path = run_dir / "resolved_config.json"
    if not cfg_path.is_file():
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    state = run_dir / "trainer/trainer_state.json"
    c = curve(state) if state.is_file() else None
    evals = {}
    for pred in sorted(run_dir.glob("eval*/score_predictions.jsonl")):
        got = measure(pred)
        if got:
            evals[pred.parent.name.replace("eval_", "").replace("eval", "best")] = got
    if not c and not evals:
        return None
    has_weights = bool(list(run_dir.glob("**/adapter_model*.safetensors")))
    return {
        "name": name, "kind": kind, "path": str(run_dir.relative_to(ROOT)),
        "seed": cfg.get("seed"), "steps": cfg.get("max_train_steps"),
        "profile": cfg.get("primary_data_profile"),
        "surface": cfg.get("essay_surface"),
        "holdout": cfg.get("unseen_prompt_holdout") or "",
        "weights": has_weights, "curve": c, "evals": evals,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    rows = []
    for run_dir in sorted(p for p in GROUP.glob("*") if p.is_dir() and not p.name.startswith("_")):
        got = collect_one(run_dir.name, run_dir, "학습")
        if got:
            rows.append(got)
    for name, rel in REFERENCES.items():
        got = collect_one(name, ROOT / "main_code/results" / rel, "기록복사")
        if got:
            rows.append(got)

    # 데이터 구성이 다른 행은 성능 비교에서 빼고 경고만 남긴다
    bad = [r for r in rows if r["profile"] != "full" or r["holdout"]
           or r["surface"] not in (None, "official_raw")]
    ok = [r for r in rows if r not in bad]
    ranked = sorted((r for r in ok if r["curve"] and r["curve"].get("cont_final") is not None),
                    key=lambda r: r["curve"]["cont_final"])

    lines = [
        "# 제출 후보 집계 — 평탄 수준과 최저를 같이 본다",
        "",
        f"- 생성 {datetime.now().isoformat(timespec='seconds')}",
        f"- 후보 {len(rows)}개 (새 학습 {sum(1 for r in rows if r['kind']=='학습')}, "
        f"기록복사 {sum(1 for r in rows if r['kind']=='기록복사')})",
        "- **판정은 연속 표면**이다. 제출 표면은 1/3 격자에서 ±0.004 양자화 잡음이 얹히고",
        "  재실행 SD도 0.0138 대 0.0055로 2.5배 흐리다.",
        f"- `last3`의 창 폭은 eval 횟수에 의존한다 — `final`을 같이 본다 (R15-a).",
        f"- 게이트: 튐폭 > {SPIKE_MAX} **이고** 최저 위치 < {ARGMIN_MIN}면 탈락(앞쪽 고립 딥).",
        f"- 기록복사 행은 재학습하지 않았다. 데이터 구성(profile=full / official_raw / holdout 없음)은",
        f"  같고 학습량(step)만 다르다 — {TARGET_STEPS_NOTE}.",
        "",
        "## 연속 표면 순위",
        "",
        "| # | 후보 | 종류 | step | seed | eval | final | last3 | last5 | 최저 | 튐폭 | 위치 | 게이트 | 가중치 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for i, r in enumerate(ranked, 1):
        c = r["curve"]
        gate = "탈락" if (c["cont_spike"] > SPIKE_MAX and c["cont_argmin_frac"] < ARGMIN_MIN) else "통과"
        lines.append(
            f"| {i} | `{r['name']}` | {r['kind']} | {r['steps']} | {r['seed']} | {c['n_eval']} | "
            f"**{c['cont_final']:.5f}** | {c['cont_last3']:.5f} | {c['cont_last5']:.5f} | "
            f"{c['cont_min']:.5f} | {c['cont_spike']:+.5f} | {c['cont_argmin_frac']:.2f} | "
            f"{gate} | {'있음' if r['weights'] else '**없음**'} |")

    lines += ["", "## 실제 평가 (checkpoint별)", "",
              "| 후보 | checkpoint | 연속 RMSE | 연속 ρ | 제출 RMSE | 제출 ρ | 예측 평균 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in ranked + [x for x in ok if x not in ranked]:
        for tag, v in sorted(r["evals"].items()):
            lines.append(f"| `{r['name']}` | {tag} | {v['cont_rmse']:.5f} | {v['cont_rho']:.5f} | "
                         f"{v['sub_rmse']:.5f} | {v['sub_rho']:.5f} | {v['pred_mean']:.4f} |")

    if bad:
        lines += ["", "## 데이터 구성이 달라 같은 표에 세우지 않은 행", ""]
        for r in bad:
            lines.append(f"- `{r['name']}` profile={r['profile']} surface={r['surface']} holdout={r['holdout']}")

    (GROUP / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (GROUP / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "kind", "steps", "seed", "n_eval", "cont_final", "cont_last3",
                    "cont_last5", "cont_min", "cont_spike", "cont_argmin_frac", "sub_final",
                    "sub_last3", "sub_last5", "sub_min", "gate", "weights", "path"])
        for r in ranked:
            c = r["curve"]
            gate = "fail" if (c["cont_spike"] > SPIKE_MAX and c["cont_argmin_frac"] < ARGMIN_MIN) else "pass"
            w.writerow([r["name"], r["kind"], r["steps"], r["seed"], c["n_eval"],
                        f"{c['cont_final']:.6f}", f"{c['cont_last3']:.6f}", f"{c['cont_last5']:.6f}",
                        f"{c['cont_min']:.6f}", f"{c['cont_spike']:.6f}", f"{c['cont_argmin_frac']:.4f}",
                        f"{c['sub_final']:.6f}", f"{c['sub_last3']:.6f}", f"{c['sub_last5']:.6f}",
                        f"{c['sub_min']:.6f}", gate, int(r["weights"]), r["path"]])

    if not args.quiet:
        print("\n".join(lines[:14 + len(ranked)]))
        print(f"\n-> {GROUP/'results.md'}  /  {GROUP/'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
