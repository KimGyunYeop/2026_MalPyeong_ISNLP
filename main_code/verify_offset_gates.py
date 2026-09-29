"""정수 총점 offset의 사전 등록 게이트를 제출 전에 확인한다 (2026-08-21).

배경
----
리더보드 RMSE 0.5191과 로컬 0.4168의 격차는 대부분 **평균 편향**이다. 관측
(RMSE 0.5191, Spearman 0.7340)을 열화 모형 6종(가우시안·두꺼운 꼬리 10/20/30%·
감쇠·gold 비례 계통)에서 동시에 맞추면 편향이 0.273~0.285로 모인다. 편향을 손익분기
1/6로 **강제**하면 Spearman이 0.645~0.704까지 내려가 관측 0.7340과 양립하지 않는다.
그래서 예측을 +1/3 올리면 RMSE만 움직인다.

핵심은 **1/3의 배수여야 한다**는 것이다. 제출값은 T/3 격자 위에 있어서, 1/3의 배수가
아닌 이동은 essay마다 T를 다르게 움직여 동점 구조를 바꾸고 Spearman을 깎는다. 정수 T를
같이 옮기면 순위 벡터가 비트 단위로 보존된다. 이 스크립트가 그것을 실제로 확인한다.

쓰는 법
------
    python -m main_code.verify_offset_gates --offset 1
    python -m main_code.verify_offset_gates --offset 1 \
        --docker-submission <docker가 낸 submission.json>

게이트를 하나라도 어기면 종료 코드가 1이다. 제출 전에 반드시 통과해야 한다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from .postprocess import ScorePostprocessor
from .utils import TRAITS

ROOT = Path(__file__).resolve().parents[1]

def _default_validation() -> Path | None:
    """공식 validation 파일을 glob으로 찾는다.

    한글 파일명을 소스에 적으면 안 된다 — 디스크의 이름은 NFD 정규화이고 이 파일은
    NFC로 저장되므로 하드코딩한 경로는 FileNotFoundError가 된다. 큐 스크립트도 같은
    이유로 `ls datasets/*2026_validation.jsonl`을 쓴다.
    """

    matches = sorted((ROOT / "datasets").glob("*2026_validation.jsonl"))
    return matches[0] if matches else None


DEFAULT_VALIDATION = _default_validation()
DEFAULT_PREDICTIONS = (
    ROOT
    / "main_code/results/new_proposed/c02_soup_v1/base_c02/score_predictions.jsonl"
)

# 리더베드 실측. 같은 image를 두 번 내서 동일하게 재현된 값이다.
LEADERBOARD_RMSE = 0.5191
LEADERBOARD_SPEARMAN = 0.7340
# 위 두 관측을 열화 모형 6종에서 동시에 맞춰 얻은 편향 범위. 손익분기는 1/6.
BIAS_LOW, BIAS_HIGH = 0.273, 0.285
BREAK_EVEN_BIAS = 1.0 / 6.0


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
    return float(np.corrcoef(average_ranks(a), average_ranks(b))[0, 1])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return math.sqrt(float(((a - b) ** 2).mean()))


def load_official_gold(path: Path) -> dict[str, float]:
    """공식 gold는 `score.average`다.

    세 trait 라벨의 산술평균이 **아니다** — 소수 2자리로 반올림된 별도 필드다
    (예: (3.5+3.25+4.0)/3 = 3.5833이지만 score.average는 3.58). 운영진의
    2026-07-20 답변이 이 필드를 쓴다고 확인했으므로 지표도 이것으로 계산한다.
    """

    gold: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        score = row.get("score") or {}
        if "average" not in score:
            raise SystemExit(f"score.average가 없습니다: {row.get('id')}")
        gold[str(row["id"])] = float(score["average"])
    return gold


def load_predictions(path: Path) -> tuple[list[str], np.ndarray]:
    ids: list[str] = []
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        ids.append(str(record["essay_id"]))
        rows.append([float(record["scores"][trait]) for trait in TRAITS])
    return ids, np.asarray(rows, dtype=np.float64)


def load_docker_submission(path: Path) -> dict[str, dict[str, float]]:
    """docker HTTP 결과를 essay_id -> 정수 삼중으로 읽는다."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("results", payload)
    if isinstance(records, dict):
        records = list(records.values())
    out: dict[str, dict[str, float]] = {}
    for record in records:
        key = str(record.get("id") or record.get("essay_id"))
        scores = record.get("submitted_scores") or record.get("scores") or record
        out[key] = {trait: float(scores[trait]) for trait in TRAITS}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offset", type=int, required=True, help="정수 총점 offset")
    parser.add_argument(
        "--validation",
        type=Path,
        default=DEFAULT_VALIDATION,
        required=DEFAULT_VALIDATION is None,
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument(
        "--docker-submission",
        type=Path,
        default=None,
        help="docker HTTP가 낸 submission.json (G1을 실제 컨테이너로 확인)",
    )
    args = parser.parse_args(argv)

    gold_map = load_official_gold(args.validation)
    ids, continuous = load_predictions(args.predictions)
    missing = [key for key in ids if key not in gold_map]
    if missing:
        raise SystemExit(f"gold에 없는 essay_id {len(missing)}개: {missing[:3]}")
    gold = np.asarray([gold_map[key] for key in ids], dtype=np.float64)

    base_int = ScorePostprocessor("average_matched", 0).apply(continuous)
    moved_int = ScorePostprocessor("average_matched", args.offset).apply(continuous)
    base_total, moved_total = base_int.sum(axis=1), moved_int.sum(axis=1)
    base_pred, moved_pred = base_total / 3.0, moved_total / 3.0

    failures: list[str] = []

    def gate(name: str, ok: bool, detail: str) -> None:
        mark = "통과" if ok else "실패"
        print(f"  [{mark}] {name}: {detail}")
        if not ok:
            failures.append(name)

    print(f"n = {len(ids)}   offset = {args.offset:+d}  (예측 {args.offset/3:+.4f})\n")
    print("### 사전 등록 게이트 ###")

    unmoved = int((moved_total != base_total + args.offset).sum())
    gate(
        "G1a 모든 T가 정확히 offset만큼 이동",
        unmoved == 0,
        f"어긋난 편 {unmoved}/{len(ids)}",
    )
    clipped = int((base_total + args.offset > 15).sum() + (base_total + args.offset < 3).sum())
    gate(
        "G1b clip에 걸린 편 없음",
        clipped == 0,
        f"clip {clipped}편, T 최대 {int(base_total.max())} -> {int(moved_total.max())}",
    )

    rho_base, rho_moved = spearman(base_pred, gold), spearman(moved_pred, gold)
    gate(
        "G2 Spearman 비트 일치",
        rho_moved == rho_base,
        f"{rho_base!r} -> {rho_moved!r}  (차이 {rho_moved - rho_base:.3e})",
    )
    gate(
        "G2b 순위 벡터 동일",
        np.array_equal(
            np.argsort(np.argsort(base_pred)), np.argsort(np.argsort(moved_pred))
        ),
        "argsort 일치",
    )

    in_range = bool(moved_int.min() >= 1.0 and moved_int.max() <= 5.0)
    gate(
        "G3a 영역 점수 1~5",
        in_range,
        f"[{moved_int.min():.0f}, {moved_int.max():.0f}]",
    )
    integral = bool(np.array_equal(moved_int, np.round(moved_int)))
    gate("G3b 정수", integral, "모두 정수")
    gate(
        "G3c 세 영역 합 == 목표 총점",
        bool(np.array_equal(moved_int.sum(axis=1), moved_total)),
        "일치",
    )

    if args.docker_submission is not None:
        docker = load_docker_submission(args.docker_submission)
        mismatch = [
            key
            for index, key in enumerate(ids)
            if key not in docker
            or any(
                abs(docker[key][trait] - moved_int[index][position]) > 1e-9
                for position, trait in enumerate(TRAITS)
            )
        ]
        gate(
            "G1c docker HTTP 출력이 연구 추론과 exact 일치",
            not mismatch,
            f"불일치 {len(mismatch)}편" + (f" 예: {mismatch[:3]}" if mismatch else ""),
        )
    else:
        print("  [건너뜀] G1c docker 대조: --docker-submission 미지정")

    print("\n### 로컬 400편 (참고 — 판정 기준이 아니다) ###")
    r_base, r_moved = rmse(base_pred, gold), rmse(moved_pred, gold)
    print(f"  RMSE     {r_base:.6f} -> {r_moved:.6f}  ({r_moved - r_base:+.6f})")
    print(f"  Spearman {rho_base:.6f} -> {rho_moved:.6f}")
    print(f"  예측 평균 {base_pred.mean():.4f} -> {moved_pred.mean():.4f}   "
          f"gold 평균 {gold.mean():.4f}")
    print("  로컬 편향은 0에 가까우므로 **로컬 RMSE는 반드시 나빠진다**.")
    print("  이 축은 리더보드 편향을 겨냥하므로 로컬 RMSE로 판정하면 안 된다.")

    print("\n### 리더보드 예상 (편향 추정 구간) ###")
    delta = args.offset / 3.0
    print(f"  {'편향 b':>8} {'현행 RMSE':>10} {f'offset {args.offset:+d}':>14} {'변화':>9}")
    for bias in (BREAK_EVEN_BIAS, BIAS_LOW, 0.279, BIAS_HIGH, 0.31):
        new = math.sqrt(max(LEADERBOARD_RMSE**2 - 2 * bias * delta + delta**2, 0.0))
        tag = "  <- 손익분기" if abs(bias - BREAK_EVEN_BIAS) < 1e-9 else ""
        print(f"  {bias:8.4f} {LEADERBOARD_RMSE:10.4f} {new:14.4f} "
              f"{new - LEADERBOARD_RMSE:+9.4f}{tag}")
    print(f"\n  손익분기 편향 = offset/6 = {args.offset / 6.0:.4f}")
    print("  보정 프로브: 제출 후 b = (δ² - R₁² + R₀²) / (2δ)로 참 편향이 풀린다.")
    print(f"           R₀ = {LEADERBOARD_RMSE}, δ = {delta:.6f}")

    print()
    if failures:
        print(f"게이트 {len(failures)}개 실패: {', '.join(failures)}")
        return 1
    print("모든 게이트 통과.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
