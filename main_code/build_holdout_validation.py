"""학습 풀 holdout을 추론 입력 파일로 굽는다.

`--validation-holdout-size`는 학습 중 checkpoint 선택에 쓰는 평가 집합만 넓힌다.
최종 지표는 `infer`가 `--input` 파일로 다시 계산하므로, arm 사이 비교의 표준오차를
실제로 줄이려면 그 입력 파일도 같은 holdout을 담아야 한다. 이 스크립트가 그 파일을 만든다.

`main_code/train.py`와 **같은** `split_validation_holdout`을 호출하므로 두 곳이 갈라질 수
없다. essay는 processed row의 `official_raw` surface를 그대로 쓴다(공식 validation 파일과
byte 단위로 같은 표면임을 확인했다).

사용:
    PYTHONPATH=$PWD python3 -m main_code.build_holdout_validation --size 1600
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PACKAGE_ROOT, TRAITS
from .datasets import split_validation_holdout

PROCESSED = PACKAGE_ROOT / "datasets" / "processed_dataset"
OFFICIAL_KEYS = ("id", "document_id", "prompt_num", "prompt", "essay", "score")


def _read_jsonl(path: Path) -> list[dict]:
    # str.splitlines()는  / /\x0b 같은 문자에서도 쪼갠다. official_raw는 그런
    # 문자를 보존하므로 essay 안에서 줄이 갈라져 JSON이 깨진다. \n으로만 나눠야 한다.
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _official_shape(row: dict) -> dict:
    """processed row를 공식 validation 파일과 같은 6개 key로 줄인다."""

    surface = (row.get("essay_surfaces") or {}).get("official_raw")
    if not isinstance(surface, str) or not surface:
        raise ValueError(f"{row.get('id')}에 official_raw surface가 없습니다")
    scores = {trait: float(row["score"][trait]) for trait in TRAITS}
    # 공식 파일의 average는 소수 둘째 자리에서 잘려 있어 정확한 trait 평균과 최대
    # .0067 다르다. holdout에는 정확한 값을 쓴다. 두 집합은 by_source_split으로 나뉘어
    # 기록되고, 이 차이가 RMSE에 주는 기여는 .0067^2 = MSE의 0.03%다.
    scores["average"] = sum(scores.values()) / len(TRAITS)
    return {
        "id": row["id"],
        "document_id": row.get("document_id", row["id"]),
        "prompt_num": row["prompt_num"],
        "prompt": row["prompt"],
        "essay": surface,
        "score": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, required=True, help="학습 풀에서 뗄 편 수")
    parser.add_argument(
        "--official-validation",
        default="",
        help="공식 400편 파일. 생략하면 datasets/ 아래에서 하나를 찾는다",
    )
    parser.add_argument("--out-dir", default=str(PACKAGE_ROOT / "datasets"))
    args = parser.parse_args()

    train_rows = _read_jsonl(PROCESSED / "train.jsonl")
    _, holdout = split_validation_holdout(train_rows, args.size)

    if args.official_validation:
        official_path = Path(args.official_validation)
    else:
        candidates = sorted(
            path
            for path in (PACKAGE_ROOT.parent / "datasets").glob("*.jsonl")
            if "validation" in path.name
        )
        if len(candidates) != 1:
            raise SystemExit(f"공식 validation 파일을 하나로 특정하지 못했습니다: {candidates}")
        official_path = candidates[0]
    official = _read_jsonl(official_path)

    processed_validation = _read_jsonl(PROCESSED / "validation.jsonl")
    holdout_ids = {row["id"] for row in holdout}
    if holdout_ids & {row["id"] for row in official}:
        raise SystemExit("holdout이 공식 validation과 겹칩니다")

    out_dir = Path(args.out_dir)
    combined = official + [_official_shape(row) for row in holdout]
    combined_path = out_dir / f"official_plus_holdout{args.size}_validation.jsonl"
    detail_path = PROCESSED / f"official_plus_holdout{args.size}_validation.jsonl"
    for path, rows in (
        (combined_path, combined),
        (detail_path, processed_validation + holdout),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        print(f"{len(rows):5d}편 -> {path}")
    missing = {row["id"] for row in combined} - {
        row["id"] for row in processed_validation + holdout
    }
    if missing:
        raise SystemExit(f"detail label이 없는 essay가 있습니다: {sorted(missing)[:3]}")
    print(f"공식 {len(official)}편 + holdout {len(holdout)}편")


if __name__ == "__main__":
    main()
