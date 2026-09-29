"""데이터 분할(LOPO) 실험을 **내부/외부로 나눠** 집계한다.

왜 별도 집계기가 필요한가
------------------------
기존 `aggregate_results.py` / `aggregate_results_all.py`는 공식 validation 400편 전체에
대한 지표 하나를 낸다. 그 400편에는 9문항이 다 들어 있으므로:

  * 홀드아웃이 **없는** arm(c02 등)  → 400편 전부가 학습에서 본 문항 = 전부 내부
  * 홀드아웃이 **있는** arm(g1/g2b)  → 학습에서 뺀 문항이 섞여 들어간다

즉 **같은 컬럼인데 arm마다 뜻이 다르다.** 그 값으로 c02와 홀드아웃 arm을 줄 세우면
안 된다. 이 집계기는 문항을 보류 집합과 나머지로 갈라 따로 재고, c02와 **같은 문항
부분집합**에서 비교해 학습량 효과와 미학습 효과를 분리한다.

무엇을 분리하는가
----------------
홀드아웃 arm은 c02와 네 가지가 다르다 — 보류 문항, 학습 행수(-23~27%), step 수,
checkpoint 선택 지표. 그래서 "외부가 c02보다 나쁘다"를 그대로 미학습 효과로 읽으면
안 된다. 같은 arm의 **내부** 문항도 c02보다 나쁘고, 그 차이가 보류와 무관한 부분이다.

    외부 페널티 = arm(보류 문항)      - c02(같은 문항)
    내부 페널티 = arm(나머지 문항)    - c02(같은 문항)   <- 학습량/step/선택 효과
    순 미학습 효과 = 외부 페널티 - 내부 페널티            <- 이것이 우리가 알고 싶은 값

편향 δ도 같이 낸다: δ = (외부 편향) - (내부 편향). 편향은 `gold - 예측`이라 양수면
낮게 예측한 것이다.

gold는 기존 집계기와 같이 원본 데이터의 `score.average` **필드 그대로**다. 세 영역
라벨의 산술평균이 아니다((3.5+3.25+4.0)/3 = 3.5833인데 필드값은 3.58).

쓰는 법
------
    python -m main_code.aggregate_datasplit_results
    python -m main_code.aggregate_datasplit_results --root <다른 결과 폴더> --output <경로>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "main_code/results/final_proposed_datasplited"
# 전 문항을 학습한 배포 모델. 같은 문항 부분집합에서의 기준선으로 쓴다.
DEFAULT_BASELINE = (
    ROOT
    / "main_code/results/new_proposed/c02_soup_v1/base_c02/score_predictions.jsonl"
)
BASELINE_LABEL = "c02 (전 문항 학습, 11,600행)"

TRAITS = ("content", "organization", "expression")
SKIP_DIRS = {"_abandoned", "_logs"}
# 제출 표면. 공식 지표는 이 표면으로 읽는다.
SUBMITTED_SURFACE = "average_matched"


# --------------------------------------------------------------------- 지표


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


def spearman(prediction: np.ndarray, truth: np.ndarray) -> float | None:
    if prediction.size < 2:
        return None
    if np.unique(prediction).size < 2 or np.unique(truth).size < 2:
        return None
    value = float(np.corrcoef(average_ranks(prediction), average_ranks(truth))[0, 1])
    return None if math.isnan(value) else value


def rmse(prediction: np.ndarray, truth: np.ndarray) -> float | None:
    if prediction.size == 0:
        return None
    return math.sqrt(float(((prediction - truth) ** 2).mean()))


def bias(prediction: np.ndarray, truth: np.ndarray) -> float | None:
    """gold - 예측. 양수면 낮게 예측한 것이다."""

    if prediction.size == 0:
        return None
    return float((truth - prediction).mean())


def round_half_up(values: np.ndarray) -> np.ndarray:
    return np.floor(np.asarray(values, dtype=np.float64) + 0.5)


def average_matched_mean(continuous: np.ndarray) -> np.ndarray:
    """세 정수의 합을 연속 합에 맞춘 뒤의 essay별 평균 (= 공식 예측값)."""

    total = np.clip(round_half_up(continuous.sum(axis=1)), 3.0, 15.0)
    return total / 3.0


# ----------------------------------------------------------------- 자료 읽기


def load_official_gold(validation: Path) -> dict[str, float]:
    gold: dict[str, float] = {}
    for line in validation.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        score = row.get("score") or {}
        if "average" not in score:
            raise SystemExit(f"{validation}: score.average가 없습니다 ({row.get('id')})")
        gold[str(row["id"])] = float(score["average"])
    return gold


def find_validation(explicit: Path | None) -> Path:
    """한글 파일명을 소스에 적지 않는다 — 디스크는 NFD, 소스는 NFC라 안 맞는다."""

    if explicit is not None:
        return explicit
    matches = sorted((ROOT / "datasets").glob("*2026_validation.jsonl"))
    if not matches:
        raise SystemExit("공식 validation 파일을 찾지 못했습니다")
    return matches[0]


class Predictions:
    """한 평가 산출물의 연속·제출 표면과 문항 표시."""

    def __init__(self, path: Path, gold: Mapping[str, float]) -> None:
        ids: list[str] = []
        rows: list[list[float]] = []
        prompts: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = str(record.get("essay_id") or "")
            scores = record.get("scores")
            if key not in gold or not isinstance(scores, Mapping):
                raise SystemExit(f"{path}: gold에 없거나 scores가 없는 행 ({key})")
            ids.append(key)
            rows.append([float(scores[trait]) for trait in TRAITS])
            prompts.append(str(record.get("prompt_num") or ""))
        if not ids:
            raise SystemExit(f"{path}: 행이 없습니다")
        order = np.argsort(np.asarray(ids))
        self.path = path
        self.ids = np.asarray(ids)[order]
        self.continuous_traits = np.asarray(rows, dtype=np.float64)[order]
        self.prompts = np.asarray(prompts)[order]
        self.truth = np.asarray([gold[key] for key in self.ids], dtype=np.float64)
        self.continuous = self.continuous_traits.mean(axis=1)
        self.submitted = average_matched_mean(self.continuous_traits)

    def subset(self, mask: np.ndarray) -> dict[str, Any]:
        return {
            "count": int(mask.sum()),
            "submitted_rmse": rmse(self.submitted[mask], self.truth[mask]),
            "submitted_spearman": spearman(self.submitted[mask], self.truth[mask]),
            "submitted_bias": bias(self.submitted[mask], self.truth[mask]),
            "continuous_rmse": rmse(self.continuous[mask], self.truth[mask]),
            "continuous_spearman": spearman(self.continuous[mask], self.truth[mask]),
        }

    def mask_for(self, prompt_set: frozenset[str]) -> np.ndarray:
        return np.isin(self.prompts, sorted(prompt_set))

    def prompt_names(self) -> list[str]:
        return sorted(set(self.prompts.tolist()))


# ------------------------------------------------------------------- 수집


def parse_holdout(raw: Any) -> frozenset[str]:
    if not raw:
        return frozenset()
    if isinstance(raw, (list, tuple)):
        items = [str(item).strip() for item in raw]
    else:
        items = [part.strip() for part in str(raw).split(",")]
    return frozenset(item for item in items if item)


def discover_arms(root: Path) -> list[Path]:
    arms: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name in SKIP_DIRS:
            continue
        if not (path / "resolved_config.json").is_file():
            continue
        if not any(path.glob("eval_*/score_predictions.jsonl")):
            continue
        arms.append(path)
    return arms


def collect(root: Path, gold: Mapping[str, float], baseline: Predictions | None):
    records: list[dict[str, Any]] = []
    for arm_dir in discover_arms(root):
        config = json.loads((arm_dir / "resolved_config.json").read_text("utf-8"))
        run: dict[str, Any] = {}
        run_path = arm_dir / "run.json"
        if run_path.is_file():
            run = json.loads(run_path.read_text("utf-8"))
        held = parse_holdout(config.get("unseen_prompt_holdout"))
        for pred_path in sorted(arm_dir.glob("eval_*/score_predictions.jsonl")):
            predictions = Predictions(pred_path, gold)
            held_mask = predictions.mask_for(held)
            seen_mask = ~held_mask
            record: dict[str, Any] = {
                "arm": arm_dir.name,
                "evaluation": pred_path.parent.name.removeprefix("eval_"),
                "holdout": ",".join(sorted(held)) if held else "-",
                "training_rows": run.get("training_rows"),
                "max_train_steps": config.get("max_train_steps"),
                "best_checkpoint_metric": config.get("best_checkpoint_metric"),
                "input_format": config.get("input_format"),
                "count": int(predictions.ids.size),
                "all": predictions.subset(np.ones_like(held_mask, dtype=bool)),
                "external": predictions.subset(held_mask) if held.__len__() else None,
                "internal": predictions.subset(seen_mask),
                "per_prompt": {
                    name: predictions.subset(predictions.prompts == name)
                    for name in predictions.prompt_names()
                },
                "held_set": held,
            }
            record["delta_bias"] = _delta(record)
            record["baseline"] = (
                _baseline_comparison(record, predictions, baseline, held)
                if baseline is not None
                else None
            )
            record["prompt_macro_rmse"], record["worst_prompt_rmse"] = _macro_worst(
                record["per_prompt"]
            )
            records.append(record)
    return records


def _delta(record: Mapping[str, Any]) -> float | None:
    external, internal = record.get("external"), record.get("internal")
    if not external or not internal:
        return None
    if external["submitted_bias"] is None or internal["submitted_bias"] is None:
        return None
    return external["submitted_bias"] - internal["submitted_bias"]


def _macro_worst(per_prompt: Mapping[str, Mapping[str, Any]]):
    values = [
        stats["submitted_rmse"]
        for stats in per_prompt.values()
        if stats["submitted_rmse"] is not None and stats["count"] >= 5
    ]
    if not values:
        return None, None
    return float(np.mean(values)), float(max(values))


def _baseline_comparison(
    record: Mapping[str, Any],
    predictions: Predictions,
    baseline: Predictions,
    held: frozenset[str],
) -> dict[str, Any] | None:
    """**같은 문항 부분집합**에서 c02와 비교해 학습량 효과와 미학습 효과를 분리한다."""

    if not held:
        return None
    arm_held = record["external"]["submitted_rmse"]
    arm_seen = record["internal"]["submitted_rmse"]
    base_held_mask = baseline.mask_for(held)
    base_seen_mask = ~base_held_mask
    base_held = rmse(baseline.submitted[base_held_mask], baseline.truth[base_held_mask])
    base_seen = rmse(baseline.submitted[base_seen_mask], baseline.truth[base_seen_mask])
    if None in (arm_held, arm_seen, base_held, base_seen):
        return None
    external_penalty = arm_held - base_held
    internal_penalty = arm_seen - base_seen
    return {
        "baseline_external_rmse": base_held,
        "baseline_internal_rmse": base_seen,
        "external_penalty": external_penalty,
        "internal_penalty": internal_penalty,
        # 보류와 무관한 부분(학습량/step/선택)을 뺀 나머지가 순 미학습 효과다.
        "net_unseen_effect": external_penalty - internal_penalty,
    }


# ---------------------------------------------------------------- 표 렌더링


def fmt(value: Any, digits: int = 5) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def signed(value: Any, digits: int = 5) -> str:
    if value is None:
        return "-"
    return f"{value:+.{digits}f}"


def table(header: Sequence[str], align: Sequence[str], rows: Iterable[Sequence[str]]):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def sort_key(record: Mapping[str, Any]):
    return (record["arm"], record["evaluation"])


def render(records: Sequence[dict[str, Any]], root: Path, baseline_label: str):
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    out: list[str] = [
        "# 데이터 분할(LOPO) 실험 결과 — 내부/외부 분리",
        "",
        f"- 생성 시각: {stamp}",
        f"- 결과 root: `{root.relative_to(ROOT)}`",
        f"- 기준선: {baseline_label}",
        "- gold는 원본 데이터의 `score.average` **필드 그대로**다. 세 영역 라벨의 "
        "산술평균이 아니다((3.5+3.25+4.0)/3 = 3.5833인데 필드값은 3.58).",
        "- 지표는 제출 표면(`average_matched`)이 기본이고, 판정용 연속 표면도 같이 낸다.",
        "- 편향은 `gold - 예측`이다. **양수면 낮게 예측**한 것이다.",
        "",
        "> **전체 컬럼으로 c02와 줄 세우면 안 된다.** 공식 validation 400편에는 9문항이",
        "> 다 들어 있어서, 홀드아웃이 없는 arm은 400편 전부가 내부인데 홀드아웃 arm은",
        "> 외부가 섞여 들어간다. 같은 컬럼인데 arm마다 뜻이 다르다. 비교는 아래",
        "> 「같은 문항에서의 c02 대비」 절을 쓴다.",
        "",
        "## 1. 내부 / 외부 분리 (제출 표면 average_matched)",
        "",
    ]

    out += table(
        ["arm", "eval", "보류", "학습행", "N",
         "전체 RMSE", "전체 ρ",
         "내부 RMSE", "내부 ρ", "내부 편향",
         "외부 RMSE", "외부 ρ", "외부 편향", "δ"],
        ["---", "---", "---", "---:", "---:",
         "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
        (
            [
                f"`{r['arm']}`",
                r["evaluation"],
                r["holdout"],
                fmt(r["training_rows"]),
                str(r["count"]),
                fmt(r["all"]["submitted_rmse"]),
                fmt(r["all"]["submitted_spearman"]),
                fmt(r["internal"]["submitted_rmse"]),
                fmt(r["internal"]["submitted_spearman"]),
                signed(r["internal"]["submitted_bias"], 4),
                fmt(r["external"]["submitted_rmse"]) if r["external"] else "-",
                fmt(r["external"]["submitted_spearman"]) if r["external"] else "-",
                signed(r["external"]["submitted_bias"], 4) if r["external"] else "-",
                signed(r["delta_bias"], 4),
            ]
            for r in sorted(records, key=sort_key)
        ),
    )

    out += [
        "",
        "`δ = 외부 편향 − 내부 편향`이다. 학습에서 못 본 문항을 **얼마나 더 낮게** "
        "예측하는지를 재고, 문항 난이도 차이는 편향 차분에서 대부분 상쇄된다.",
        "",
        "## 2. 같은 문항에서의 c02 대비 — 학습량 효과와 미학습 효과 분리",
        "",
        "홀드아웃 arm은 c02와 네 가지가 다르다: 보류 문항, 학습 행수(−23~27%), "
        "step 수, checkpoint 선택 지표. 그래서 외부 성능 차이를 그대로 미학습 효과로 "
        "읽으면 안 된다. **같은 arm의 내부 문항도 c02보다 나쁘고, 그 몫이 보류와 무관한 "
        "부분**이다. 그것을 빼야 순 효과가 나온다.",
        "",
    ]

    comparable = [r for r in sorted(records, key=sort_key) if r.get("baseline")]
    if comparable:
        out += table(
            ["arm", "eval", "보류",
             "외부: arm", "외부: c02", "외부 페널티",
             "내부: arm", "내부: c02", "내부 페널티",
             "**순 미학습 효과**"],
            ["---", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
            (
                [
                    f"`{r['arm']}`",
                    r["evaluation"],
                    r["holdout"],
                    fmt(r["external"]["submitted_rmse"]),
                    fmt(r["baseline"]["baseline_external_rmse"]),
                    signed(r["baseline"]["external_penalty"]),
                    fmt(r["internal"]["submitted_rmse"]),
                    fmt(r["baseline"]["baseline_internal_rmse"]),
                    signed(r["baseline"]["internal_penalty"]),
                    f"**{signed(r['baseline']['net_unseen_effect'])}**",
                ]
                for r in comparable
            ),
        )
        out += [
            "",
            "순 미학습 효과가 0에 가깝다는 것은 **미학습 문항의 손해가 RMSE로는 거의 "
            "나타나지 않는다**는 뜻이다. 손해는 위 표의 δ(편향)와 외부 ρ로 나타난다.",
            "",
        ]
    else:
        out += ["비교 가능한 홀드아웃 arm이 없다.", ""]

    out += ["## 3. 판정용 연속 표면", "",
            "checkpoint 선택 잡음은 `average_matched`의 1/3 격자 스냅에서 일부 온다. "
            "arm 사이 판정은 연속 표면으로 한다.", ""]
    out += table(
        ["arm", "eval", "전체 RMSE", "전체 ρ", "내부 RMSE", "외부 RMSE",
         "문항 macro RMSE", "최악 문항 RMSE"],
        ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:"],
        (
            [
                f"`{r['arm']}`",
                r["evaluation"],
                fmt(r["all"]["continuous_rmse"]),
                fmt(r["all"]["continuous_spearman"]),
                fmt(r["internal"]["continuous_rmse"]),
                fmt(r["external"]["continuous_rmse"]) if r["external"] else "-",
                fmt(r["prompt_macro_rmse"]),
                fmt(r["worst_prompt_rmse"]),
            ]
            for r in sorted(records, key=sort_key)
        ),
    )

    out += ["", "## 4. 문항별 (제출 표면) — `*`는 그 arm이 학습에서 뺀 문항", ""]
    prompts = sorted({name for r in records for name in r["per_prompt"]})
    out += table(
        ["arm", "eval"] + prompts,
        ["---", "---"] + ["---:"] * len(prompts),
        (
            [f"`{r['arm']}`", r["evaluation"]]
            + [
                (
                    fmt(r["per_prompt"][name]["submitted_rmse"])
                    + ("*" if name in r["held_set"] else "")
                    if name in r["per_prompt"]
                    else "-"
                )
                for name in prompts
            ]
            for r in sorted(records, key=sort_key)
        ),
    )

    out += ["", "## 5. 설정", ""]
    seen: set[str] = set()
    out += table(
        ["arm", "보류", "학습행", "step", "선택 지표", "입력 형식"],
        ["---", "---", "---:", "---:", "---", "---"],
        (
            [
                f"`{r['arm']}`",
                r["holdout"],
                fmt(r["training_rows"]),
                fmt(r["max_train_steps"]),
                f"`{r['best_checkpoint_metric']}`",
                f"`{r['input_format']}`",
            ]
            for r in sorted(records, key=sort_key)
            if not (r["arm"] in seen or seen.add(r["arm"]))
        ),
    )
    out.append("")
    return "\n".join(out)


def render_csv(records: Sequence[dict[str, Any]], path: Path) -> None:
    fields = [
        "arm", "evaluation", "holdout", "training_rows", "max_train_steps",
        "best_checkpoint_metric", "input_format", "count",
    ]
    for scope in ("all", "internal", "external"):
        for metric in (
            "count", "submitted_rmse", "submitted_spearman", "submitted_bias",
            "continuous_rmse", "continuous_spearman",
        ):
            fields.append(f"{scope}_{metric}")
    fields += [
        "delta_bias", "prompt_macro_rmse", "worst_prompt_rmse",
        "baseline_external_rmse", "baseline_internal_rmse",
        "external_penalty", "internal_penalty", "net_unseen_effect",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=sort_key):
            row = {key: record.get(key) for key in fields if key in record}
            for scope in ("all", "internal", "external"):
                stats = record.get(scope) or {}
                for metric, value in stats.items():
                    row[f"{scope}_{metric}"] = value
            for key, value in (record.get("baseline") or {}).items():
                row[key] = value
            writer.writerow({key: row.get(key) for key in fields})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--validation", type=Path, default=None)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="전 문항을 학습한 기준선의 score_predictions.jsonl",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        raise SystemExit(f"결과 폴더가 없습니다: {args.root}")
    gold = load_official_gold(find_validation(args.validation))
    baseline = (
        Predictions(args.baseline, gold) if args.baseline.is_file() else None
    )
    if baseline is None:
        print(f"경고: 기준선 파일이 없습니다 ({args.baseline}) — c02 대비 절을 생략합니다")

    records = collect(args.root, gold, baseline)
    if not records:
        raise SystemExit(f"집계할 평가 산출물이 없습니다: {args.root}")

    markdown = args.output or (args.root / "results.md")
    markdown.write_text(
        render(records, args.root, BASELINE_LABEL if baseline else "(없음)"),
        encoding="utf-8",
    )
    csv_path = markdown.with_suffix(".csv")
    render_csv(records, csv_path)
    print(f"arm {len({r['arm'] for r in records})}개 / 평가 {len(records)}개")
    print(f"  {markdown.relative_to(ROOT)}")
    print(f"  {csv_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
