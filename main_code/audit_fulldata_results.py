"""전체 데이터(비분할) 실험 전부를 배포 c02와 같은 척도로 감사하고 요약을 만든다.

왜 코드로 만드는가
    손으로 적은 결과 목록은 곧 틀린다(R9와 같은 이유). 이 스크립트는 결과 트리의
    `resolved_config.json`과 `score_predictions.jsonl`만 읽어 표를 다시 만든다.

두 가지를 반드시 지킨다 (2026-08-21에 둘 다 틀렸다가 고쳤다)

1. **지표 표면**은 제출 표면 `average_matched`다. 연속 세 값의 **합**을 사사오입해
   3~15로 clip하고 3으로 나눈다. 영역별로 먼저 반올림한 뒤 평균하면 같은 c02가
   0.41682 -> 0.45902로 **0.042 나빠진다**(400편 중 111편이 다르다). 우리가 내는
   것은 이미 정수 삼중이므로 운영측 사사오입은 무연산이고, 따라서 official 지표는
   average_matched 평균과 정확히 같다. 배포 c02의 docker HTTP 실측
   0.416823637099 / 0.759736661325와 12자리 일치를 매 실행 검증한다.

2. **배포 가능성 등급**을 나눈다. `NOT_DEPLOYABLE_canonical__` 아래 run은 원천
   문단 경계를 입력에 넣은 것이라 제출에서 재현할 수 없는데, 연속 RMSE 상위를
   독점한다. 등급을 안 나누면 "0.4177이 최고"라는 잘못된 결론이 나온다.

판정 척도
    연속 official RMSE(재실행 SD 0.0055)를 1차로 본다. 제출 표면은 1/3 격자 양자화
    때문에 SD가 0.0138이라 축 비교에는 너무 흐리다(R1, R3).

usage:
    python -m main_code.audit_fulldata_results            # 표만 출력
    python -m main_code.audit_fulldata_results --write    # md/json도 쓴다
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
TRAITS = ("content", "organization", "expression")

# 배포 c02. `main_code_submission/CURRENT_RELEASE_STATUS.md`가 권위 기준이다.
CTRL_MARK = "y6_final_combo_noext_s43_v1/c02_avg_mse025"
CTRL_EXPECT = (0.416823637099, 0.759736661325)

# 재실행 SD (R1). 연속 표면 기준으로 판정한다.
SD_CONTINUOUS = 0.0055

DERIVED_SURFACES = ("flat", "gap_newline", "kiwi")
CHECKPOINT_PRIORITY = ("best_checkpoint_official_matched_rmse", "best_checkpoint")

# 성능과 무관해서 축 이름에서 빼는 키
IGNORE_KEYS = {
    "seed", "output_dir", "experiment_case", "experiment_config_id",
    "experiment_description", "experiment_base", "experiment_suite",
    "dataset_root", "extended_data_dir", "checkpoint_retention",
    "schema_version", "model_cache_dir", "train_file", "device",
    "smoke_rows_per_split", "detail_rater_registry", "prompt_registry",
    "model_source_run", "dataset_sha256", "score_postprocess", "model_revision",
}


def gold_map() -> dict[str, float]:
    # 디스크 파일명이 NFD라 소스의 NFC 리터럴과 안 맞는다. glob으로 찾는다.
    source = glob.glob(str(ROOT / "datasets/*2026_validation.jsonl"))[0]
    out = {}
    for line in open(source, encoding="utf-8"):
        row = json.loads(line)
        out[str(row["id"])] = float(row["score"]["average"])
    return out


GOLD = gold_map()


def grade(path: str, cfg: dict) -> str:
    """제출 계약과 같은 입력 표면을 썼는가."""
    if cfg.get("allow_validation_leaked") or cfg.get("primary_data_profile") == "validation_leaked":
        return "label_leak"
    if "NOT_DEPLOYABLE_canonical__" in path or cfg.get("essay_surface") == "canonical":
        return "leaked"
    if cfg.get("essay_surface") == "official_raw":
        return "contract"
    if cfg.get("essay_surface") in DERIVED_SURFACES:
        return "derived"
    return "unknown"          # essay_surface 필드가 없던 과거 run


def surfaces(pred_path: str):
    """(제출 average_matched, 연속 평균, gold) — essay_id 정렬 순."""
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
    return matched, cont.mean(axis=1), gold


def metrics(values: np.ndarray, gold: np.ndarray) -> tuple[float, float]:
    rmse = math.sqrt(float(((values - gold) ** 2).mean()))
    return rmse, float(spearmanr(values, gold).statistic)


def checkpoint_of(eval_dir: str) -> str:
    try:
        manifest = json.load(open(os.path.join(eval_dir, "inference_manifest.json")))
    except Exception:
        return ""
    return str(manifest.get("checkpoint", "")).rstrip("/").split("/")[-1]


def normalize(value):
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def collect() -> list[dict]:
    runs = []
    pattern = str(ROOT / "main_code/results/**/resolved_config.json")
    for cfg_path in glob.glob(pattern, recursive=True):
        rel = os.path.relpath(cfg_path, ROOT)
        if any(mark in rel for mark in ("_ABANDONED", "_SUPERSEDED", "_abandoned")):
            continue
        try:
            cfg = json.load(open(cfg_path))
        except Exception:
            continue
        if cfg.get("unseen_prompt_holdout"):
            continue                      # 분할 실험은 별도 트리에서 집계한다
        base = os.path.dirname(cfg_path)
        evals = []
        for pred in sorted(glob.glob(os.path.join(base, "**", "score_predictions.jsonl"), recursive=True)):
            got = surfaces(pred)
            if got:
                evals.append((os.path.dirname(pred), checkpoint_of(os.path.dirname(pred)), got))
        if not evals:
            continue
        pick = None
        for want in CHECKPOINT_PRIORITY:
            pick = next((e for e in evals if e[1] == want), None)
            if pick:
                break
        runs.append({"path": rel, "cfg": cfg, "grade": grade(rel, cfg),
                     "pick": pick or evals[0], "n_evals": len(evals)})
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="md/json을 결과 트리에 쓴다")
    args = ap.parse_args()

    runs = collect()
    by_grade = collections.Counter(r["grade"] for r in runs)

    control = next(r for r in runs if CTRL_MARK in r["path"])
    c_match, c_cont, c_gold = control["pick"][2]
    ctrl_sub = metrics(c_match, c_gold)
    ctrl_cont = metrics(c_cont, c_gold)
    if abs(ctrl_sub[0] - CTRL_EXPECT[0]) > 1e-9 or abs(ctrl_sub[1] - CTRL_EXPECT[1]) > 1e-9:
        raise SystemExit(
            f"대조군 지표가 배포 실측과 다르다: {ctrl_sub} != {CTRL_EXPECT}. "
            "표면 정의가 바뀌었는지 확인하라."
        )

    usable = [r for r in runs if r["grade"] in ("contract", "unknown")]
    ctrl_cfg = control["cfg"]

    axes = collections.defaultdict(list)
    for run in usable:
        cfg = run["cfg"]
        diff = [
            f"{k}={normalize(cfg.get(k))}"
            for k in sorted(set(cfg) | set(ctrl_cfg))
            if k not in IGNORE_KEYS and normalize(cfg.get(k)) != normalize(ctrl_cfg.get(k))
        ]
        axes[" ; ".join(diff) if diff else "(c02 대조군과 동일)"].append(run)

    rows = []
    for key, items in axes.items():
        vals = []
        for run in items:
            matched, cont, gold = run["pick"][2]
            sub_r, sub_rho = metrics(matched, gold)
            con_r, con_rho = metrics(cont, gold)
            vals.append({"sub_rmse": sub_r, "sub_rho": sub_rho,
                         "cont_rmse": con_r, "cont_rho": con_rho,
                         "seed": run["cfg"].get("seed"), "path": run["path"],
                         "grade": run["grade"], "backbone": run["cfg"].get("model_slug"),
                         "checkpoint": run["pick"][1]})
        vals.sort(key=lambda v: v["cont_rmse"])
        seeds = sorted({v["seed"] for v in vals})
        cont_list = [v["cont_rmse"] for v in vals]
        sub_list = [v["sub_rmse"] for v in vals]
        rows.append({
            "key": key, "n_runs": len(vals), "n_seeds": len(seeds), "seeds": seeds,
            "cont_mean": float(np.mean(cont_list)), "cont_best": min(cont_list),
            "cont_sd": float(np.std(cont_list, ddof=1)) if len(cont_list) > 1 else None,
            "sub_mean": float(np.mean(sub_list)), "sub_best": min(sub_list),
            "sub_sd": float(np.std(sub_list, ddof=1)) if len(sub_list) > 1 else None,
            "backbones": sorted({str(v["backbone"]) for v in vals}),
            "grades": sorted({v["grade"] for v in vals}),
            "runs": vals,
        })
    rows.sort(key=lambda r: r["cont_mean"])

    print(f"등급별 실행 수: {dict(by_grade)}")
    print(f"비교 대상(contract + unknown) {len(usable)}개, 서로 다른 설정 {len(rows)}개")
    print(f"배포 c02  제출 {ctrl_sub[0]:.6f}/{ctrl_sub[1]:.6f}  연속 {ctrl_cont[0]:.6f}/{ctrl_cont[1]:.6f}")
    print()

    replicated = [r for r in rows if r["n_seeds"] > 1]
    replicated.sort(key=lambda r: r["cont_mean"])
    print(f"=== seed 반복이 있는 설정 {len(replicated)}개 — **판정 가능한 유일한 표** ===")
    print(f"{'연속평균':>8} {'연속SD':>7} {'제출평균':>8} {'제출SD':>7} {'seed':>10}  축")
    for r in replicated:
        csd = f"{r['cont_sd']:.4f}" if r["cont_sd"] is not None else "   -  "
        ssd = f"{r['sub_sd']:.4f}" if r["sub_sd"] is not None else "   -  "
        key = r["key"] if len(r["key"]) <= 78 else r["key"][:75] + "..."
        print(f"{r['cont_mean']:8.5f} {csd:>7} {r['sub_mean']:8.5f} {ssd:>7} {str(r['seeds']):>10}  {key}")

    print()
    print("=== backbone (contract 등급만, 연속 RMSE) ===")
    bb = collections.defaultdict(list)
    for run in usable:
        if run["grade"] != "contract":
            continue
        matched, cont, gold = run["pick"][2]
        bb[run["cfg"].get("model_slug")].append((metrics(cont, gold)[0], metrics(matched, gold)[0]))
    print(f"{'backbone':20s} {'n':>4} {'최저':>9} {'중앙':>9} {'제출최저':>9}")
    for name, vals in sorted(bb.items(), key=lambda kv: min(v[0] for v in kv[1])):
        c = sorted(v[0] for v in vals)
        print(f"{str(name):20s} {len(vals):4d} {c[0]:9.5f} {c[len(c)//2]:9.5f} {min(v[1] for v in vals):9.5f}")

    if args.write:
        out_dir = ROOT / "main_code/results"
        (out_dir / "fulldata_audit.json").write_text(
            json.dumps({"control": {"submitted": ctrl_sub, "continuous": ctrl_cont},
                        "grades": dict(by_grade), "axes": rows},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n-> {out_dir / 'fulldata_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
