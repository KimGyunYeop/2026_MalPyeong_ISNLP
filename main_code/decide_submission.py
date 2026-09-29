"""제출 후보를 배포 c02와 **짝지어** 비교하고 사전 등록한 규칙으로 판정한다.

왜 짝짓기인가
    같은 에세이를 두 모델이 함께 본다. 형제 checkpoint끼리 오차 상관이 0.998이라
    절대 RMSE의 표준오차보다 **차이의 표준오차가 훨씬 작다**(400편에서 0.0175 대
    0.0032). 절대값만 보면 0.005 차이는 잡음에 묻히지만 짝지어 보면 신호다.

무엇을 1차 기준으로 쓰는가 (2026-08-21 재정정)
    2025 최종보고서 표 V-6에서 채점 데이터 4,000편의 71.1%가 우리 학습에 없는
    문항(Q11 1,512편, Q12 1,331편)임을 확인했다.

    ※ 08-20에 여기 적었던 "순위 손실 53.9% / 평균 이동 28.7%" 분해는 **틀렸다.**
      귀속 순서에 의존하는 계산이었을 뿐 측정이 아니다. 확정된 것은 다음이다.
      리더보드 ρ 0.7340이 **미선택 seed 42의 로컬 ρ 0.73462와 같다**(차이 0.0006).
      즉 순위 능력은 그대로 옮겨 갔고 **격차는 전부 위치 보정 오차**다.
      b = 0.299 [0.275, 0.319]. 근거: DISTRIBUTION_SHIFT_DIAGNOSIS_20260820.md §9.

    그런데 공개 validation 400편은 학습에서 **본** 9문항의 새 에세이라 이 격차를
    측정하지 못한다. 그래서 Q8·Q9를 학습에서 통째로 뺀 arm의 **Q8+Q9 98편 성능**을
    1차 기준으로 쓴다. 배포 c02는 그 문항을 봤으므로 두 숫자의 차이가
    "문항을 못 봤을 때의 손해"다.

    `official_matched_rmse_shifted`는 판정에 쓰지 않는다. σ_g 확대 가설이 보고서로
    반증됐다.

usage:
    python -m main_code.decide_submission                    # 기본 group 전부
    python -m main_code.decide_submission <group_dir> ...
"""

from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .official_metrics import official_pred_avg

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "main_code/results"
NP = RESULTS / "new_proposed"
BASELINE = (
    NP
    / "y6_final_combo_noext_s43_v1/c02_avg_mse025/ax4_light/a228dbe60ed8"
    / "eval_best_official_rmse/score_predictions.jsonl"
)
DEFAULT_GROUPS = (
    # 2026-08-21 데이터 분할(LOPO) arm은 결과 트리를 옮겼다. 내부/외부를 나눠 재는
    # 설계라 new_proposed와 섞으면 같은 컬럼이 arm마다 다른 뜻이 된다.
    RESULTS / "final_proposed_datasplited",
    NP / "c02_scoring_v2",
)
DRAWS = 4000
SEED = 43
# 배포 c02가 재현하는 리더보드 실적. 후보가 이걸 못 넘기면 c02를 그대로 낸다.
DEPLOYED_LEADERBOARD = (0.5191, 0.7340)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def gold_map() -> dict[str, float]:
    source = glob.glob(str(ROOT / "datasets/*2026_validation.jsonl"))[0]
    return {
        row["id"]: float(row["score"]["average"]) for row in _rows(Path(source))
    }


def prompt_map() -> dict[str, str]:
    source = glob.glob(str(ROOT / "datasets/*2026_validation.jsonl"))[0]
    return {row["id"]: str(row["prompt_num"]) for row in _rows(Path(source))}


def load(path: Path) -> dict[str, float]:
    """essay_id -> 공식 정의(영역별 반올림 후 평균)의 예측값."""

    return {
        row["essay_id"]: float(official_pred_avg(row["submitted_scores"]))
        for row in _rows(path)
    }


def arm_holdout(arm_dir: Path) -> tuple[str, ...]:
    """이 arm이 학습에서 뺀 문항. resolved_config.json이 정본이다.

    임시 `--train-file` 필터로 문항을 빼던 시절에는 이 정보가 어디에도 남지 않아
    run 사이 비교가 깨졌다. `unseen_prompt_holdout`은 config에 남으므로 판정이
    arm마다 올바른 부분집합을 자동으로 고른다.
    """

    resolved = arm_dir / "resolved_config.json"
    if not resolved.is_file():
        return ()
    value = json.loads(resolved.read_text(encoding="utf-8")).get(
        "unseen_prompt_holdout", ""
    )
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return math.sqrt(float(np.mean((prediction - truth) ** 2)))


def describe(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(prediction, truth),
        "spearman": float(spearmanr(prediction, truth)[0]),
        "bias": float(np.mean(prediction - truth)),
        "sd": float(prediction.std()),
    }


def by_prompt(
    prediction: np.ndarray, truth: np.ndarray, groups: np.ndarray
) -> dict[str, float]:
    """문항 매크로 / 최악 문항 / 문항별 편향 폭.

    왜 이걸 함께 보는가
        공개 validation 400편의 문항 구성은 우리 학습 편수에 비례한다(Q1 25편 …
        Q5 51편). 그런데 2025 채점 데이터 4,000편은 Q11 1,512 / Q12 1,331편으로
        **완전히 다른 구성**이다(신규 두 문항이 71.1%). essay micro 평균으로 고르면
        우리 validation의 구성에만 맞춘 모델을 고르게 된다.

        문항 매크로는 구성에 불변이다. 최악 문항은 "약한 문항이 하나도 없는" 모델을
        고른다. 문항별 편향 폭(max−min)은 미학습 문항에서 터질 편향의 대리 지표다 —
        본 문항에서도 편향이 문항마다 흔들리는 모델은 새 문항에서 더 흔들린다.
    """

    rmses, spearmans, biases = [], [], []
    for key in sorted(set(groups.tolist())):
        mask = groups == key
        if mask.sum() < 5:
            continue
        rmses.append(rmse(prediction[mask], truth[mask]))
        biases.append(float(np.mean(prediction[mask] - truth[mask])))
        if len(set(prediction[mask].tolist())) >= 2:
            spearmans.append(float(spearmanr(prediction[mask], truth[mask])[0]))
    return {
        "prompt_macro_rmse": float(np.mean(rmses)) if rmses else float("nan"),
        "worst_prompt_rmse": float(np.max(rmses)) if rmses else float("nan"),
        "prompt_macro_spearman": (
            float(np.mean(spearmans)) if spearmans else float("nan")
        ),
        "bias_spread": float(np.max(biases) - np.min(biases)) if biases else float("nan"),
        "prompts": len(rmses),
    }


def paired(candidate: np.ndarray, baseline: np.ndarray, truth: np.ndarray) -> dict:
    """후보 − 기준선. RMSE는 음수가 개선, Spearman은 양수가 개선."""

    result = {
        "d_rmse": rmse(candidate, truth) - rmse(baseline, truth),
        "d_spearman": float(spearmanr(candidate, truth)[0])
        - float(spearmanr(baseline, truth)[0]),
    }
    generator = np.random.default_rng(SEED)
    count = len(truth)
    draws: dict[str, list[float]] = {"d_rmse": [], "d_spearman": []}
    for _ in range(DRAWS):
        index = generator.integers(0, count, count)
        c, b, t = candidate[index], baseline[index], truth[index]
        draws["d_rmse"].append(rmse(c, t) - rmse(b, t))
        first, second = spearmanr(c, t)[0], spearmanr(b, t)[0]
        if not (math.isnan(first) or math.isnan(second)):
            draws["d_spearman"].append(float(first - second))
    for key, values in draws.items():
        result[f"{key}_se"] = float(np.std(values)) if values else float("nan")
    return result


def main() -> int:
    groups = [Path(a) for a in sys.argv[1:]] or list(DEFAULT_GROUPS)
    gold, prompts = gold_map(), prompt_map()
    base_all = load(BASELINE)

    unseen = {i for i, q in prompts.items() if q in {"Q8", "Q9"}}
    print(
        "배포 c02 기준선 (문항을 모두 학습에서 본 모델). "
        f"리더보드 재현 실적 RMSE {DEPLOYED_LEADERBOARD[0]} / "
        f"Spearman {DEPLOYED_LEADERBOARD[1]}"
    )
    for tag, keys in (
        ("전체 400", set(base_all)),
        ("Q8+Q9 (미학습 대상 98편)", unseen),
        ("Q1~Q7 (302편)", set(base_all) - unseen),
    ):
        ids = sorted(keys)
        p = np.array([base_all[i] for i in ids])
        g = np.array([gold[i] for i in ids])
        d = describe(p, g)
        m = by_prompt(p, g, np.array([prompts[i] for i in ids]))
        print(
            f"  {tag:26s} n={len(ids):3d}  RMSE {d['rmse']:.5f}  "
            f"Spearman {d['spearman']:.4f}  편향 {d['bias']:+.4f}  예측SD {d['sd']:.4f}"
        )
        print(
            f"  {'':26s} 문항 {m['prompts']}개 | 매크로 RMSE {m['prompt_macro_rmse']:.5f}  "
            f"최악 문항 {m['worst_prompt_rmse']:.5f}  매크로 Spearman "
            f"{m['prompt_macro_spearman']:.4f}  편향 폭 {m['bias_spread']:.4f}"
        )
    print()

    found: list[Path] = []
    for group in groups:
        found.extend(sorted(group.glob("*/eval_*/score_predictions.jsonl")))
    if not found:
        print(f"평가 결과가 아직 없습니다: {[str(g) for g in groups]}")
        return 1

    header = (
        f"{'arm / 평가집합':44s} {'n':>4s} {'RMSE':>8s} {'Spear':>7s} {'편향':>7s} "
        f"{'예측SD':>7s} | {'ΔRMSE':>9s} {'(SE)':>8s} {'ΔSpear':>8s} {'(SE)':>8s}"
    )
    print(header)
    print("-" * len(header))
    verdicts: dict[str, dict[str, dict]] = {}
    for path in found:
        candidate = load(path)
        ids = sorted(set(candidate) & set(base_all))
        if len(ids) < 30:
            continue
        c = np.array([candidate[i] for i in ids])
        b = np.array([base_all[i] for i in ids])
        g = np.array([gold[i] for i in ids])
        d = describe(c, g)
        delta = paired(c, b, g)
        groups = np.array([prompts[i] for i in ids])
        macro = by_prompt(c, g, groups)
        base_macro = by_prompt(b, g, groups)
        arm = path.parent.parent.name
        subset = path.parent.name.removeprefix("eval_")
        holdout = arm_holdout(path.parent.parent)
        verdicts.setdefault(arm, {})[subset] = {
            **d, **delta, **macro, "n": len(ids),
            "d_prompt_macro": macro["prompt_macro_rmse"]
            - base_macro["prompt_macro_rmse"],
            "d_worst_prompt": macro["worst_prompt_rmse"]
            - base_macro["worst_prompt_rmse"],
            "d_bias_spread": macro["bias_spread"] - base_macro["bias_spread"],
        }
        print(
            f"{f'{arm} / {subset}'[:44]:44s} {len(ids):4d} {d['rmse']:8.5f} "
            f"{d['spearman']:7.4f} {d['bias']:+7.4f} {d['sd']:7.4f} | "
            f"{delta['d_rmse']:+9.5f} {delta['d_rmse_se']:8.5f} "
            f"{delta['d_spearman']:+8.4f} {delta['d_spearman_se']:8.4f}"
        )
        if macro["prompts"] >= 2:
            print(
                f"{'':44s} {'':4s} 문항매크로 {macro['prompt_macro_rmse']:.5f} "
                f"(Δ{macro['prompt_macro_rmse']-base_macro['prompt_macro_rmse']:+.5f})  "
                f"최악문항 {macro['worst_prompt_rmse']:.5f} "
                f"(Δ{macro['worst_prompt_rmse']-base_macro['worst_prompt_rmse']:+.5f})  "
                f"편향폭 {macro['bias_spread']:.4f} "
                f"(Δ{macro['bias_spread']-base_macro['bias_spread']:+.4f})"
            )
        # 보류 문항이 있으면 그 부분집합을 따로 잰다. 이게 1차 판정 기준이다.
        # 배포 c02는 그 문항을 **봤으므로** 두 값의 차이가 "문항을 못 봤을 때의 손해"다.
        if holdout:
            held = np.array([prompts[i] in set(holdout) for i in ids])
            if int(held.sum()) >= 20:
                held_delta = paired(c[held], b[held], g[held])
                seen_delta = paired(c[~held], b[~held], g[~held])
                verdicts[arm][subset]["held"] = {**held_delta, "n": int(held.sum())}
                print(
                    f"{'':44s} {'':4s} 보류({'+'.join(holdout)}) n={int(held.sum()):3d}  "
                    f"RMSE {rmse(c[held], g[held]):.5f} "
                    f"(Δ{held_delta['d_rmse']:+.5f}, SE {held_delta['d_rmse_se']:.5f})  "
                    f"편향 {float(np.mean(c[held]-g[held])):+.4f}  "
                    f"| 본 문항 Δ{seen_delta['d_rmse']:+.5f}  "
                    f"| δ(보류−본) 편향차 "
                    f"{float(np.mean(c[held]-g[held]) - np.mean(c[~held]-g[~held])):+.4f}"
                )

    print()
    print("=== 사전 등록 판정 ===")
    print(
        "  LOPO arm: Q89 ΔRMSE <= -2SE (1차)  +  전체 400 ΔSpearman >= -2SE (방어선)\n"
        "  그 외 arm: 전체 400 ΔRMSE <= -2SE  +  ΔSpearman >= -2SE\n"
        "  공통 방어선: 문항 매크로 RMSE가 2SE 이상 악화되면 탈락.\n"
        "    (숨은 평가셋의 문항 구성이 우리 validation과 전혀 다르므로 essay micro만\n"
        "     좋아진 모델은 그 구성에만 맞춘 것일 수 있다.)\n"
        "  하나라도 어기면 c02를 그대로 제출한다."
    )
    print()
    for arm, subsets in sorted(verdicts.items()):
        reasons: list[str] = []
        # 우선순위: (1) 보류 문항 부분집합, (2) 과거 eval_Q89 디렉터리, (3) 전체 400
        held_source = next(
            (item["held"] for item in subsets.values() if "held" in item), None
        )
        if held_source is not None:
            primary, label = held_source, "보류 문항"
        elif "Q89" in subsets:
            primary, label = subsets["Q89"], "Q89"
        else:
            primary = (
                subsets.get("best")
                or subsets.get("official_matched_rmse")
                or subsets.get("full400")
            )
            label = "전체 400"
            if primary is None:
                print(f"  {arm:34s} 판정 불가 (1차 지표 평가 없음)")
                continue
        if primary["d_rmse"] > -2 * primary["d_rmse_se"]:
            reasons.append(
                f"{label} RMSE 개선 {-primary['d_rmse']:+.5f} < 2SE({2*primary['d_rmse_se']:.5f})"
            )
        guard = (
            subsets.get("best")
            or subsets.get("full400")
            or subsets.get("official_matched_rmse")
            or primary
        )
        if guard["d_spearman"] < -2 * guard["d_spearman_se"]:
            reasons.append(
                f"Spearman {guard['d_spearman']:+.4f} < -2SE({-2*guard['d_spearman_se']:.4f})"
            )
        # 문항 구성 불변 방어선. 숨은 평가셋의 문항 구성이 우리 validation과
        # 전혀 다르므로(신규 두 문항이 71%) essay micro만 좋아진 모델은 걸러낸다.
        full = (
            subsets.get("best")
            or subsets.get("full400")
            or subsets.get("official_matched_rmse")
        )
        if full is not None and full.get("d_prompt_macro") is not None:
            if full["d_prompt_macro"] > 2 * full["d_rmse_se"]:
                reasons.append(
                    f"문항 매크로 RMSE {full['d_prompt_macro']:+.5f} > 2SE"
                    f"({2*full['d_rmse_se']:.5f})"
                )
        state = "제출 후보" if not reasons else "탈락"
        print(f"  {arm:34s} {state}: {', '.join(reasons) if reasons else '두 조건 통과'}")
    print()
    print(
        "참고: LOPO arm의 Q89 성능을 c02의 Q89 성능(0.40400)과 비교한 차이가 "
        "'문항을 못 봤을 때의 손해'다. 이 값이 크면 미학습 문항이 주요 원인이라는\n"
        "     진단이 확인되고, x02(지문 제거)가 x01보다 작으면 지문 제거가 그 손해를 "
        "줄인다는 뜻이다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
