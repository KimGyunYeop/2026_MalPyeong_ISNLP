from __future__ import annotations

import copy
import re
from pathlib import Path

import numpy as np
import pytest

from main_code.aggregate_results_all import (
    SURFACES,
    assign_ranks,
    compute_metric_matrix,
    render_markdown,
    write_scope_report,
)


RAW = np.asarray(
    [
        [1.49, 2.49, 4.49],
        [1.51, 2.51, 4.51],
        [2.60, 2.60, 2.10],
        [2.40, 2.40, 2.40],
        [4.49, 3.49, 1.49],
    ],
    dtype=np.float64,
)
GOLD_TRAITS = np.asarray(
    [
        [1.20, 2.30, 4.40],
        [2.30, 2.70, 4.80],
        [2.40, 3.20, 2.10],
        [2.00, 2.20, 2.50],
        [4.20, 3.70, 1.30],
    ],
    dtype=np.float64,
)
# These are the stored score.average values.  In particular, the metric code
# must neither round them nor silently replace them with GOLD_TRAITS.mean(1).
GOLD_AVERAGE = np.asarray([2.63, 3.27, 2.57, 2.23, 3.07], dtype=np.float64)


@pytest.fixture()
def golden_matrix() -> tuple[
    dict[str, np.ndarray], dict[str, dict[str, float | None]]
]:
    return compute_metric_matrix(RAW, GOLD_TRAITS, GOLD_AVERAGE)


def test_complete_surface_matrix_matches_hand_checked_golden(
    golden_matrix: tuple[
        dict[str, np.ndarray], dict[str, dict[str, float | None]]
    ],
) -> None:
    surfaces, matrix = golden_matrix

    np.testing.assert_array_equal(
        surfaces["average_matched"],
        np.asarray(
            [
                [1, 3, 4],
                [2, 2, 5],
                [2, 3, 2],
                [3, 2, 2],
                [5, 3, 1],
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_array_equal(
        surfaces["independent_half_up"],
        np.asarray(
            [
                [1, 2, 4],
                [2, 3, 5],
                [3, 3, 2],
                [2, 2, 2],
                [4, 3, 1],
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_array_equal(surfaces["raw_continuous"], RAW)

    expected = {
        "average_matched": {
            "official_rmse": 0.17078251276599324,
            "official_spearman": 0.9486832980505137,
            "content_rmse": 0.6212889826803626,
            "content_spearman": 0.6668859288553503,
            "organization_rmse": 0.5567764362830023,
            "organization_spearman": 0.5773502691896257,
            "expression_rmse": 0.3316624790355401,
            "expression_spearman": 0.9746794344808963,
            "trait_macro_rmse": 0.5032426326663016,
            "trait_macro_spearman": 0.7396385441752907,
            "mean_first_rmse": 0.17078251276599324,
            "mean_first_spearman": 0.9486832980505137,
            "pooled_rmse": 0.5183306537980044,
            "pooled_spearman": 0.7749675364697716,
        },
        "independent_half_up": {
            "official_rmse": 0.25177150134375587,
            "official_spearman": 0.8207826816681234,
            "content_rmse": 0.32557641192199416,
            "content_spearman": 0.9746794344808963,
            "organization_rmse": 0.38729833462074176,
            "organization_spearman": 0.8660254037844386,
            "expression_rmse": 0.3316624790355401,
            "expression_spearman": 0.9746794344808963,
            "trait_macro_rmse": 0.34817907519275865,
            "trait_macro_spearman": 0.9384614242487438,
            "mean_first_rmse": 0.25177150134375587,
            "mean_first_spearman": 0.8207826816681234,
            "pooled_rmse": 0.3492849839314597,
            "pooled_spearman": 0.9392980020344459,
        },
        "raw_continuous": {
            "official_rmse": 0.25177150134375587,
            "official_spearman": 0.8207826816681234,
            "content_rmse": 0.44548849592329537,
            "content_spearman": 0.8999999999999998,
            "organization_rmse": 0.32134094043554434,
            "organization_spearman": 0.9999999999999999,
            "expression_rmse": 0.16631295800387894,
            "expression_spearman": 0.9999999999999999,
            "trait_macro_rmse": 0.3110474647875729,
            "trait_macro_spearman": 0.9666666666666665,
            "mean_first_rmse": 0.23431223233587753,
            "mean_first_spearman": 0.8999999999999998,
            "pooled_rmse": 0.33135077083558045,
            "pooled_spearman": 0.9244738890416517,
        },
    }
    # 공식 평가자는 우리가 보낸 값에 영역별 round_half_up을 적용한 뒤 평균한다.
    # 따라서 연속값을 제출하면 독립 반올림과 **완전히 같은 점수**가 되고,
    # average_matched의 합 맞춤 조정은 통째로 사라진다.
    assert (
        expected["raw_continuous"]["official_rmse"]
        == expected["independent_half_up"]["official_rmse"]
    )
    assert (
        expected["average_matched"]["official_rmse"]
        != expected["independent_half_up"]["official_rmse"]
    )
    assert set(matrix) == set(expected) == set(SURFACES)
    for surface, expected_metrics in expected.items():
        assert set(matrix[surface]) == set(expected_metrics)
        for name, value in expected_metrics.items():
            assert matrix[surface][name] == pytest.approx(value, abs=1e-12)

    # These explicit non-equalities protect the three commonly confused axes.
    assert matrix["average_matched"]["trait_macro_rmse"] != pytest.approx(
        matrix["average_matched"]["pooled_rmse"]
    )
    rounded_gold_pooled = float(
        np.sqrt(
            np.mean(
                (
                    surfaces["average_matched"]
                    - np.floor(GOLD_TRAITS + 0.5)
                )
                ** 2
            )
        )
    )
    assert rounded_gold_pooled == pytest.approx(0.6324555320336759)
    assert matrix["average_matched"]["pooled_rmse"] != pytest.approx(
        rounded_gold_pooled
    )


def _report_row(
    scope: Path,
    matrix: dict[str, dict[str, float | None]],
    *,
    case_name: str = "pipe|case\nline",
    record_type: str = "primary",
) -> dict[str, object]:
    run_dir = scope / "suite" / "base" / "case" / "model" / "config"
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "record_type": record_type,
        "suite": "suite",
        "physical_base_name": "base",
        "logical_base_name": "base",
        "case_name": case_name,
        "model_slug": "model",
        "config_id": "config",
        "run_status": "complete",
        "analysis_status": "complete",
        "surface_class": "contract",
        "evaluation_name": "eval" if record_type == "primary" else "eval_best_rmse",
        "checkpoint": "best_checkpoint",
        "same_raw_as_primary": None,
        "count": len(RAW),
        "input_path": str(scope / "validation.jsonl"),
        "input_name": "validation.jsonl",
        "input_sha256": "a" * 64,
        "prediction_path": str(run_dir / "eval" / "score_predictions.jsonl"),
        "prediction_sha256": "b" * 64,
        "raw_stream_sha256": "c" * 64,
        "relative_run_dir": "suite/base/case/model/config",
        "relative_eval_dir": "suite/base/case/model/config/eval",
        "run_dir": str(run_dir),
        "eval_dir": str(run_dir / "eval"),
        "warnings": "",
        "metrics": copy.deepcopy(matrix),
        "ranks": {},
    }


def test_ranks_each_surface_by_the_official_metric_not_trait_macro(
    tmp_path: Path,
    golden_matrix: tuple[
        dict[str, np.ndarray], dict[str, dict[str, float | None]]
    ],
) -> None:
    _, matrix = golden_matrix
    first = _report_row(tmp_path, matrix, case_name="first")
    second = _report_row(tmp_path, matrix, case_name="second")
    variant = _report_row(tmp_path, matrix, case_name="variant", record_type="variant")

    first_metrics = first["metrics"]
    second_metrics = second["metrics"]
    variant_metrics = variant["metrics"]
    assert isinstance(first_metrics, dict)
    assert isinstance(second_metrics, dict)
    assert isinstance(variant_metrics, dict)

    # The preferred row differs by surface.  Deliberately make mean-first and
    # pooled values point the other way so they cannot accidentally drive rank.
    first_metrics["average_matched"]["official_rmse"] = 0.40
    first_metrics["average_matched"]["trait_macro_rmse"] = 0.90
    first_metrics["average_matched"]["mean_first_rmse"] = 0.90
    first_metrics["average_matched"]["pooled_rmse"] = 0.90
    # 2026-08-19 운영측 답변으로 산출식이 확정됐다. 순위는 공식 지표를 따라야 하며,
    # 예전 기본값이던 trait_macro_rmse가 반대로 정렬돼 있어도 결과가 뒤집히면 안 된다.
    second_metrics["average_matched"]["official_rmse"] = 0.50
    second_metrics["average_matched"]["trait_macro_rmse"] = 0.10
    second_metrics["average_matched"]["mean_first_rmse"] = 0.10
    second_metrics["average_matched"]["pooled_rmse"] = 0.10

    first_metrics["raw_continuous"]["official_rmse"] = 0.60
    second_metrics["raw_continuous"]["official_rmse"] = 0.30
    variant_metrics["average_matched"]["official_rmse"] = 0.01

    rows = [first, second, variant]
    assign_ranks(rows)  # type: ignore[arg-type]

    assert first["ranks"]["average_matched"] == 1  # type: ignore[index]
    assert second["ranks"]["average_matched"] == 2  # type: ignore[index]
    assert second["ranks"]["raw_continuous"] == 1  # type: ignore[index]
    assert first["ranks"]["raw_continuous"] == 2  # type: ignore[index]
    assert variant["ranks"] == {}


def test_markdown_has_three_readable_surface_sections_and_valid_tables(
    tmp_path: Path,
    golden_matrix: tuple[
        dict[str, np.ndarray], dict[str, dict[str, float | None]]
    ],
) -> None:
    _, matrix = golden_matrix
    row = _report_row(tmp_path, matrix)
    assign_ranks([row])  # type: ignore[list-item]

    text = render_markdown(
        [row],  # type: ignore[list-item]
        [],
        results_root=tmp_path,
        scope_dir=tmp_path,
        csv_name="result_all.csv",
    )
    primary_text = text.split("## Checkpoint/diagnostic variant 부록", 1)[0]

    for heading in (
        "### Average-matched 정수",
        "### 일반 사사오입 정수",
        "### Raw continuous",
    ):
        assert primary_text.count(heading) == 1
    assert "Gold C/O/E와 저장된 `score.average`는 반올림하지 않는다" in text
    assert "`영역 RMSE 평균` 오름차순" in text
    assert "3N pooled" in text
    assert "pipe\\|case line" in text
    assert "0.5032426" in text
    assert "0.1707825" in text
    assert "0.5183307" in text
    assert "nan" not in text.lower()
    assert "none" not in text.lower()

    # Every contiguous Markdown table must have a stable column count.
    table: list[str] = []
    for line in [*text.splitlines(), ""]:
        if line.startswith("|"):
            table.append(line)
            continue
        if table:
            assert len(table) >= 2
            expected_pipes = len(re.findall(r"(?<!\\)\|", table[0]))
            assert all(
                len(re.findall(r"(?<!\\)\|", row_line)) == expected_pipes
                for row_line in table
            )
            table = []

    # A Markdown table has exactly one alignment row.  A second one renders as
    # a bogus data record and makes the long report materially harder to read.
    summary_alignment = (
        "|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    assert primary_text.count(summary_alignment) == len(SURFACES)


def test_markdown_audits_fixed_varied_and_schema_drift_config(
    tmp_path: Path,
    golden_matrix: tuple[
        dict[str, np.ndarray], dict[str, dict[str, float | None]]
    ],
) -> None:
    _, matrix = golden_matrix
    first = _report_row(tmp_path, matrix, case_name="control")
    second = _report_row(tmp_path, matrix, case_name="treatment")
    first.update(
        description="기준 설명",
        method_config={"lora_r": 32, "listwise_loss_weight": 0.2, "old_only": 1},
        explicit_overrides=["--lora-r", "32"],
    )
    second.update(
        description="변경 설명",
        method_config={"lora_r": 32, "listwise_loss_weight": 0.1},
        explicit_overrides=["--listwise-loss-weight", "0.1"],
    )
    text = render_markdown(
        [first, second],  # type: ignore[list-item]
        [],
        results_root=tmp_path,
        scope_dir=tmp_path,
        csv_name="result_all.csv",
    )
    assert "## 그룹별 방법론 설정" in text
    assert "고정 방법론 설정 전체 1개" in text
    assert "`lora_r`" in text
    assert "`listwise_loss_weight`" in text
    assert "#### Schema drift" in text
    assert "`old_only`" in text
    assert "기준 설명" in text
    assert "변경 설명" in text


def test_unmeasured_markdown_keeps_experiment_description(
    tmp_path: Path,
    golden_matrix: tuple[
        dict[str, np.ndarray], dict[str, dict[str, float | None]]
    ],
) -> None:
    _, matrix = golden_matrix
    row = _report_row(tmp_path, matrix, case_name="not_scored")
    row.update(
        record_type="unmeasured",
        analysis_status="missing_primary_predictions",
        description="학습 전에 실패한 실험의 설명",
        method_config={},
        metrics={},
    )
    text = render_markdown(
        [],
        [row],  # type: ignore[list-item]
        results_root=tmp_path,
        scope_dir=tmp_path,
        csv_name="result_all.csv",
    )
    assert "| Suite/Base | Case | 설명 |" in text
    assert "학습 전에 실패한 실험의 설명" in text


def test_write_scope_report_never_overwrites_historical_results(
    tmp_path: Path,
    golden_matrix: tuple[
        dict[str, np.ndarray], dict[str, dict[str, float | None]]
    ],
) -> None:
    _, matrix = golden_matrix
    results_root = tmp_path / "results"
    scope = results_root / "suite" / "base"
    scope.mkdir(parents=True)
    historical_md = scope / "results.md"
    historical_csv = scope / "results.csv"
    md_sentinel = b"historical markdown must remain byte-identical\n"
    csv_sentinel = b"historical,csv,must,remain\n"
    historical_md.write_bytes(md_sentinel)
    historical_csv.write_bytes(csv_sentinel)
    row = _report_row(results_root, matrix, case_name="case")
    assign_ranks([row])  # type: ignore[list-item]

    markdown_path, csv_path, row_count, unmeasured_count = write_scope_report(
        scope,
        [row],  # type: ignore[list-item]
        [],
        results_root=results_root,
    )

    assert markdown_path == (scope / "result_all.md").resolve()
    assert csv_path == (scope / "result_all.csv").resolve()
    assert row_count == 1
    assert unmeasured_count == 0
    assert markdown_path.is_file()
    assert csv_path.is_file()
    assert historical_md.read_bytes() == md_sentinel
    assert historical_csv.read_bytes() == csv_sentinel

    singular_historical_md = scope / "result.md"
    singular_historical_md.write_bytes(md_sentinel)
    with pytest.raises(ValueError, match="기존 보고서 덮어쓰기 금지"):
        write_scope_report(
            scope,
            [row],  # type: ignore[list-item]
            [],
            results_root=results_root,
            markdown_path=singular_historical_md,
            csv_path=scope / "another_result_all.csv",
        )
    assert singular_historical_md.read_bytes() == md_sentinel

    with pytest.raises(ValueError, match="기존 보고서 덮어쓰기 금지"):
        write_scope_report(
            scope,
            [row],  # type: ignore[list-item]
            [],
            results_root=results_root,
            markdown_path=historical_md,
            csv_path=scope / "another_result_all.csv",
        )
    assert historical_md.read_bytes() == md_sentinel
    assert historical_csv.read_bytes() == csv_sentinel
