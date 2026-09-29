"""`results.md`가 모든 행을 같은 지표 정의로 채우는지 검증한다.

공식 지표가 확정되기 전 artifact의 `metrics.json`에는 `trait_average` key가 없다. 그때
`results.md`를 `-`로 비워 두면 정의 A로 정렬된 과거 행과 정의 C를 가진 최근 행이 한 표에
섞여 순위 자체가 무의미해진다. 예측 파일이 남아 있으므로 다시 계산해야 한다.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from main_code.aggregate_results import (
    _spearman,
    official_metrics_from_predictions,
    summarize_metrics,
    surface_class,
    write_markdown,
)
from main_code.utils import average_matched_integer_scores

TRAITS = ("content", "organization", "expression")


def _write_predictions(eval_dir: Path, rows: list[tuple[str, tuple[float, float, float]]]) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    with (eval_dir / "score_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for essay_id, scores in rows:
            handle.write(
                json.dumps(
                    {
                        "essay_id": essay_id,
                        "scores": dict(zip(TRAITS, scores, strict=True)),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _write_gold_manifest(eval_dir: Path, gold: dict[str, float]) -> None:
    """평가 당시 실제 입력과 manifest를 함께 만든다."""

    eval_dir.mkdir(parents=True, exist_ok=True)
    input_path = eval_dir.parent / f"{eval_dir.name}_official_validation.jsonl"
    with input_path.open("w", encoding="utf-8") as handle:
        for essay_id, average in gold.items():
            handle.write(
                json.dumps(
                    {
                        "id": essay_id,
                        # 실제 holdout 에세이에 있는 Unicode line separator. JSONL은 LF만
                        # 행 경계이므로 이 문자가 record를 둘로 자르면 안 된다.
                        "essay": "앞 문장\u2028뒤 문장",
                        "score": {"average": average},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    (eval_dir / "inference_manifest.json").write_text(
        json.dumps({"input": str(input_path.resolve())}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_recompute_matches_hand_calculation(tmp_path: Path) -> None:
    """공식 정의대로 세 trait을 평균해 score.average와 한 번 비교해야 한다."""

    gold = {"a": 3.0, "b": 4.0, "c": 2.0}
    _write_predictions(
        tmp_path / "eval",
        [
            ("a", (3.0, 3.0, 3.0)),   # 평균 3.0, 오차 0
            ("b", (3.0, 3.0, 3.0)),   # 평균 3.0, 오차 -1
            ("c", (3.0, 3.0, 3.0)),   # 평균 3.0, 오차 +1
        ],
    )
    _write_gold_manifest(tmp_path / "eval", gold)
    result = official_metrics_from_predictions(tmp_path / "eval", [])
    assert result is not None
    assert result["count"] == 3
    assert result["rmse"] == pytest.approx(math.sqrt(2 / 3))
    assert result["raw_continuous"]["rmse"] == pytest.approx(math.sqrt(2 / 3))
    assert result["average_matched"]["rmse"] == pytest.approx(math.sqrt(2 / 3))
    assert result["gold_source"] == "score_average_recomputed"


def test_recompute_uses_trait_mean_not_trait_metrics(tmp_path: Path) -> None:
    """trait별 오차가 커도 평균이 맞으면 공식 RMSE는 0이다.

    정의 A와 정의 C가 실제로 다른 값을 낸다는 것을 고정한다. 두 정의를 한 표에 섞으면
    안 되는 이유가 이것이다.
    """

    gold = {"a": 3.0, "b": 3.0}
    _write_predictions(
        tmp_path / "eval",
        [("a", (1.0, 3.0, 5.0)), ("b", (5.0, 3.0, 1.0))],
    )
    _write_gold_manifest(tmp_path / "eval", gold)
    result = official_metrics_from_predictions(tmp_path / "eval", [])
    assert result is not None
    # trait별 RMSE는 각각 2 이상이지만 세 trait 평균은 정확히 3.0이다.
    assert result["rmse"] == pytest.approx(0.0)


def test_missing_predictions_returns_none(tmp_path: Path) -> None:
    assert official_metrics_from_predictions(tmp_path / "eval", []) is None


def test_recompute_prefers_new_surface_schema_and_reports_both_surfaces(
    tmp_path: Path,
) -> None:
    gold = {"a": 3.0, "b": 4.0}
    raw_rows = {
        "a": (3.49, 3.49, 3.49),
        "b": (3.51, 4.49, 4.49),
    }
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    with (eval_dir / "score_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for essay_id, scores in raw_rows.items():
            raw = dict(zip(TRAITS, scores, strict=True))
            matched_array = average_matched_integer_scores(
                np.asarray([scores], dtype=np.float64)
            )[0]
            matched = dict(zip(TRAITS, matched_array.tolist(), strict=True))
            handle.write(
                json.dumps(
                    {
                        "essay_id": essay_id,
                        # 잘못된 legacy alias보다 canonical block을 우선해야 한다.
                        "scores": {trait: 1.0 for trait in TRAITS},
                        "score_surfaces": {
                            "raw_continuous": {
                                "scores": raw,
                                "average": sum(raw.values()) / 3,
                            },
                            "average_matched": {
                                "scores": matched,
                                "average": sum(matched.values()) / 3,
                            },
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    _write_gold_manifest(eval_dir, gold)
    result = official_metrics_from_predictions(eval_dir, [])
    assert result is not None
    raw_average = np.asarray([sum(scores) / 3 for scores in raw_rows.values()])
    raw_gold = np.asarray(list(gold.values()))
    assert result["raw_continuous"]["rmse"] == pytest.approx(
        np.sqrt(np.mean(np.square(raw_average - raw_gold)))
    )
    assert result["average_matched"]["rmse"] != result["raw_continuous"]["rmse"]


def test_each_eval_uses_its_manifest_input_instead_of_a_global_gold(
    tmp_path: Path,
) -> None:
    """공식 400편과 holdout 입력을 한 process에서 집계해도 gold가 섞이지 않는다."""

    rows = [("a", (3.0, 3.0, 3.0)), ("b", (3.0, 3.0, 3.0))]
    first = tmp_path / "first" / "eval"
    second = tmp_path / "second" / "eval"
    for eval_dir in (first, second):
        _write_predictions(eval_dir, rows)
    _write_gold_manifest(first, {"a": 3.0, "b": 3.0})
    _write_gold_manifest(second, {"a": 4.0, "b": 4.0})

    first_result = official_metrics_from_predictions(first, [])
    second_result = official_metrics_from_predictions(second, [])
    assert first_result is not None and second_result is not None
    assert first_result["raw_continuous"]["rmse"] == pytest.approx(0.0)
    assert second_result["raw_continuous"]["rmse"] == pytest.approx(1.0)


def test_metric_summary_migrates_v2_and_legacy_artifacts() -> None:
    v2_row: dict[str, object] = {}
    summarize_metrics(
        v2_row,
        {
            "count": 2,
            "official": {
                "gold_source": "score_average",
                "raw_continuous": {"rmse": 0.41, "spearman": 0.72},
                "average_matched": {"rmse": 0.44, "spearman": 0.69},
            },
        },
    )
    assert v2_row["raw_average_rmse"] == 0.41
    assert v2_row["submitted_average_rmse"] == 0.44
    assert v2_row["official_rmse"] == 0.41  # legacy alias
    assert v2_row["candidate_rmse"] == 0.44  # legacy alias

    legacy_row: dict[str, object] = {}
    summarize_metrics(
        legacy_row,
        {
            "trait_average": {
                "rmse": 0.42,
                "spearman": 0.71,
                "gold_source": "score_average",
            },
            "trait_average_rounded": {
                "average_matched_integer": {"rmse": 0.45, "spearman": 0.68}
            },
        },
    )
    assert legacy_row["raw_average_rmse"] == 0.42
    assert legacy_row["submitted_average_rmse"] == 0.45
    assert legacy_row["official_gold_source"] == "score_average"


def test_markdown_prioritizes_only_trait_raw_and_submitted_metrics(
    tmp_path: Path,
) -> None:
    row = {
        "rank": 1,
        "status": "complete",
        "config.essay_surface": "official_raw",
        "inference_essay_surface": "official_raw",
        "submitted_average_rmse": 0.44,
        "submitted_average_spearman": 0.69,
        "raw_average_rmse": 0.41,
        "raw_average_spearman": 0.72,
        "traits": {},
        "content_rmse": 0.50,
        "content_spearman": 0.60,
        "organization_rmse": 0.51,
        "organization_spearman": 0.61,
        "expression_rmse": 0.52,
        "expression_spearman": 0.62,
        "case_name": "case",
        "base_name": "base",
    }
    path = tmp_path / "results.md"
    write_markdown(path, [row], tmp_path / "results.csv", tmp_path)
    text = path.read_text(encoding="utf-8")

    assert "제출 평균 RMSE" in text
    assert "Raw 평균 RMSE" in text
    assert "C RMSE" in text
    assert "C RMSE/ρ" not in text
    assert "0.5000/0.6000" not in text
    assert "trait Spearman" in text
    assert "예상RMSE" not in text
    assert "후보RMSE" not in text
    assert "both_per_trait_rounded" not in text
    full_table = text.split("## 전체 표", 1)[1]
    table_lines = [line for line in full_table.splitlines() if line.startswith("|")]
    assert all(line.count("|") == table_lines[0].count("|") for line in table_lines)


def test_spearman_uses_average_ranks_for_ties() -> None:
    # score.average는 소수 2자리라 동점이 생긴다. 운영측은 scipy 기본 average rank를 쓴다.
    assert _spearman([1.0, 1.0, 2.0], [1.0, 1.0, 2.0]) == pytest.approx(1.0)
    assert _spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_surface_class_flags_non_deployable_runs() -> None:
    contract = {
        "config.essay_surface": "official_raw",
        "inference_essay_surface": "official_raw",
    }
    assert surface_class(contract) == "contract"

    # 폴더 rename이 config보다 우선한다. base_name은 experiment.json의 옛 이름이다.
    renamed = {
        "base_name": "tri7_b32ga1_full_baseline_lock_gc1_v1",
        "relative_run_dir": "proposed/NOT_DEPLOYABLE_canonical__tri7_b32ga1_full_baseline_lock_gc1_v1/p21/tri_7b/abc",
    }
    assert surface_class(renamed) == "leaked"

    canonical = {"config.essay_surface": "canonical", "inference_essay_surface": "canonical"}
    assert surface_class(canonical) == "leaked"

    derived = {"config.essay_surface": "flat", "inference_essay_surface": "flat"}
    assert surface_class(derived) == "derived"

    # validation을 학습에 넣은 진단 run은 표에서 가장 위험한 행이다.
    label_leak = {
        "config.essay_surface": "official_raw",
        "inference_essay_surface": "official_raw",
        "config.primary_data_profile": "validation_leaked",
    }
    assert surface_class(label_leak) == "label_leak"
