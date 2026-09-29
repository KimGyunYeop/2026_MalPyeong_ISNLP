"""학습 중 eval 곡선에서 **평탄 구간(plateau) 성능**을 뽑아 arm을 다시 줄 세운다.

왜 필요한가 (2026-08-21 사용자 지정 원칙)
    배포 c02는 1288 step 중 **736 step**(epoch 0.571)에서 최저를 찍었다. 앞쪽
    체크포인트에서 한 번 튄 값이다. 400편으로 여러 step을 재고 그중 최소를 고르면
    **평가셋에 대한 선택 과적합**이고, 그 최소값은 숨은 test로 옮겨 가지 않는다.
    실제로 c02 레시피 3 seed의 로컬 ρ는 0.7597(s43) / 0.7346(s42) / 0.7279(s44)인데
    리더보드 실측 ρ는 **0.7340** — 레시피 평균 0.7407에 가깝고 배포 draw의 0.7597과는
    0.026 떨어져 있다. **튄 값은 전이되지 않았다.**

    그래서 선택 기준을 바꾼다: 학습 후반까지 **평탄하게 유지되면서** 좋은 arm을 고른다.

무엇을 계산하나
    - `last3`   : 마지막 3개 eval의 평균. 사용자가 지정한 1차 기준.
    - `plateau` : 후반 절반 eval의 평균과 SD. 평탄한지 본다.
    - `spike`   : (후반 절반 평균) − (전체 최소). 클수록 "앞에서 한 번 튄" 모양이다.
    - `argmin_frac` : 최소값이 나온 step / 총 step. 작으면 앞쪽에서 튄 것이다.

usage:
    python -m main_code.analyze_eval_plateau                 # 전체 데이터 run
    python -m main_code.analyze_eval_plateau --split         # 분할 run
    python -m main_code.analyze_eval_plateau --write         # json 저장
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# 제출 표면(average_matched) 그대로. **키 이름이 시기별로 다르다** — 08-11 이전 run은
# `eval_official_matched_*`, 이후는 `eval_average_matched_official_*`다. 둘 다 같은
# 정의(세 정수 합 사사오입 / 3)이고, 하나만 찾으면 배포 c02와 y6가 표에서 빠진다.
KEY_RMSE_CANDIDATES = (
    "eval_average_matched_official_rmse",
    "eval_official_matched_rmse",
)
KEY_RHO_CANDIDATES = (
    "eval_average_matched_official_spearman",
    "eval_official_matched_spearman",
)
# 연속(raw) 표면. **판정은 이쪽으로 한다** — 제출 표면은 1/3 격자에서 ±0.004 양자화
# 잡음이 얹히고 재실행 SD도 0.0138 대 0.0055로 2.5배 흐리다.
KEY_CONT_CANDIDATES = (
    "eval_official_rmse",
    "eval_raw_continuous_official_rmse",
)


def curve(state_path: Path) -> dict | None:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    history = state.get("log_history", [])
    key_r = next((k for k in KEY_RMSE_CANDIDATES if any(k in e for e in history)), None)
    if key_r is None:
        return None
    key_h = next((k for k in KEY_RHO_CANDIDATES if any(k in e for e in history)), None)
    key_c = next((k for k in KEY_CONT_CANDIDATES if any(k in e for e in history)), None)
    rows = []
    for entry in history:
        if key_r in entry:
            rows.append((int(entry.get("step", 0)), float(entry.get("epoch", 0.0)),
                         float(entry[key_r]),
                         float(entry.get(key_h, float("nan"))) if key_h else float("nan"),
                         float(entry.get(key_c, float("nan"))) if key_c else float("nan")))
    if len(rows) < 3:
        return None
    rows.sort()
    steps = np.array([r[0] for r in rows])
    epochs = np.array([r[1] for r in rows])
    rmse = np.array([r[2] for r in rows])
    rho = np.array([r[3] for r in rows])
    cont = np.array([r[4] for r in rows])
    half = max(2, len(rmse) // 2)
    late_r, late_h = rmse[-half:], rho[-half:]
    last3_r, last3_h = rmse[-3:], rho[-3:]
    argmin = int(np.argmin(rmse))
    def stats(v: np.ndarray, prefix: str) -> dict:
        """창 폭이 eval 횟수에 의존하므로 final·last3·last5를 **모두** 낸다 (R15-a)."""
        if not np.isfinite(v).all():
            return {}
        j = int(np.argmin(v))
        hi = max(2, len(v) // 2)
        return {
            f"{prefix}final": float(v[-1]),
            f"{prefix}last3": float(v[-3:].mean()),
            f"{prefix}last5": float(v[-5:].mean()) if len(v) >= 5 else float(v.mean()),
            f"{prefix}min": float(v.min()),
            f"{prefix}min_step": int(steps[j]),
            f"{prefix}argmin_frac": float(steps[j] / max(steps[-1], 1)),
            f"{prefix}late_mean": float(v[hi:].mean()),
            f"{prefix}late_sd": float(v[hi:].std(ddof=1)) if len(v[hi:]) > 1 else None,
            f"{prefix}spike": float(v[hi:].mean() - v.min()),
        }

    out = {
        "metric_key": key_r,
        "cont_key": key_c,
        "n_eval": len(rmse),
        "max_step": int(state.get("max_steps") or steps[-1]),
        "best": float(rmse.min()),
        "best_step": int(steps[argmin]),
        "best_epoch": float(epochs[argmin]),
        "argmin_frac": float(steps[argmin] / max(steps[-1], 1)),
        "last3": float(last3_r.mean()),
        "last3_rho": float(last3_h.mean()),
        "last3_sd": float(last3_r.std(ddof=1)) if len(last3_r) > 1 else None,
        "late_mean": float(late_r.mean()),
        "late_sd": float(late_r.std(ddof=1)) if len(late_r) > 1 else None,
        "late_rho": float(late_h.mean()),
        "spike": float(late_r.mean() - rmse.min()),
        "final": float(rmse[-1]),
        "curve": [[int(s), round(float(e), 3), round(float(v), 5)] for s, e, v in zip(steps, epochs, rmse)],
        "cont_curve": ([round(float(v), 5) for v in cont] if key_c else []),
    }
    out.update(stats(rmse, "sub_"))
    out.update(stats(cont, "cont_") if key_c else {})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="store_true", help="분할(holdout) run만")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    out = []
    for state_path in glob.glob(str(ROOT / "main_code/results/**/trainer/trainer_state.json"), recursive=True):
        rel = os.path.relpath(state_path, ROOT)
        if any(m in rel for m in ("_ABANDONED", "_SUPERSEDED", "_abandoned", "NOT_DEPLOYABLE")):
            continue
        run_dir = Path(state_path).parent.parent
        cfg_path = run_dir / "resolved_config.json"
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        is_split = bool(cfg.get("unseen_prompt_holdout"))
        if is_split != args.split:
            continue
        if cfg.get("allow_validation_leaked") or cfg.get("primary_data_profile") == "validation_leaked":
            continue
        if cfg.get("essay_surface") not in (None, "official_raw"):
            continue
        c = curve(Path(state_path))
        if c is None:
            continue
        parts = rel.split("/")
        c["run"] = "/".join(parts[3:5]) if len(parts) > 5 else rel
        c["seed"] = cfg.get("seed")
        c["backbone"] = cfg.get("model_slug")
        c["holdout"] = cfg.get("unseen_prompt_holdout") or ""
        c["path"] = str(run_dir.relative_to(ROOT))
        out.append(c)

    label = "분할" if args.split else "전체 데이터"
    print(f"{label} run {len(out)}개에서 eval 곡선을 읽었다 (제출 표면 average_matched)\n")
    if not out:
        return 0

    # --- 연속 표면 재순위. 판정은 이쪽이다 (R15-a) ---
    withc = [r for r in out if r.get("cont_final") is not None]
    withc.sort(key=lambda r: r["cont_last3"])
    print(f"=== 연속 표면 재순위 ({len(withc)} run) — 창 폭이 eval 횟수에 의존하므로 final/last3/last5를 같이 본다 ===")
    print(f"{'last3':>8} {'last5':>8} {'final':>8} {'최저':>8} {'튐폭':>8} {'위치':>5} {'eval':>4} {'seed':>4}  run")
    print("-" * 124)
    for r in withc[: args.top]:
        print(f"{r['cont_last3']:8.5f} {r['cont_last5']:8.5f} {r['cont_final']:8.5f} {r['cont_min']:8.5f} "
              f"{r['cont_spike']:+8.5f} {r['cont_argmin_frac']:5.2f} {r['n_eval']:4d} {str(r['seed']):>4}  {r['run'][:52]}")
    print()

    out.sort(key=lambda r: r["last3"])
    print(f"=== 제출 표면 (참고 — 1/3 격자 잡음 ±0.004) ===")
    print(f"{'last3':>8} {'후반평균':>8} {'후반SD':>7} {'최저':>8} {'튐폭':>7} {'최저위치':>8} {'eval':>4} {'seed':>4}  run")
    print("-" * 132)
    for r in out[: args.top]:
        sd = f"{r['late_sd']:.4f}" if r["late_sd"] is not None else "  -   "
        print(f"{r['last3']:8.5f} {r['late_mean']:8.5f} {sd:>7} {r['best']:8.5f} "
              f"{r['spike']:+7.4f} {r['argmin_frac']:8.2f} {r['n_eval']:4d} {str(r['seed']):>4}  {r['run'][:56]}")

    print(f"\n=== 배포 c02의 모양 ===")
    for r in out:
        if "c02_avg_mse025" in r["path"] and "y6_final_combo" in r["path"]:
            print(f"  최저 {r['best']:.5f} @ step {r['best_step']}/{r['max_step']} (epoch {r['best_epoch']:.2f}, {r['argmin_frac']:.0%} 지점)")
            print(f"  후반 절반 평균 {r['late_mean']:.5f} (SD {r['late_sd']:.5f})  ->  **튐폭 {r['spike']:+.4f}**")
            print(f"  마지막 3 eval 평균 {r['last3']:.5f}")
            print(f"  곡선: {r['curve']}")

    if args.write:
        name = "eval_plateau_split.json" if args.split else "eval_plateau_full.json"
        (ROOT / "main_code/results" / name).write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n-> main_code/results/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
