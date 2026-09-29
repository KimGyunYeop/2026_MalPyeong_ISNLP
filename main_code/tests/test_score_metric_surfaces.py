from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from main_code.config import TRAITS
from main_code.infer import (
    ScoringOutput,
    attach_score_surfaces,
    evaluate_predictions,
    score_surface_payload,
)
from main_code.postprocess import ScorePostprocessor
from main_code.train import compute_trainer_metrics
from main_code.utils import average_matched_integer_scores, regression_metrics


def test_metrics_expose_only_the_two_official_score_surfaces() -> None:
    labels = np.asarray(
        [[2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [1.0, 2.0, 3.0]],
        dtype=np.float64,
    )
    predictions = np.asarray(
        [[2.49, 3.49, 4.49], [3.51, 3.51, 4.51], [1.49, 2.51, 3.49]],
        dtype=np.float64,
    )
    # 공식 gold는 trait label 평균과 별개로 보존되는 데이터셋 score.average다.
    average_labels = np.asarray([3.1, 3.9, 2.1], dtype=np.float64)

    metrics = regression_metrics(labels, predictions, average_labels)
    official = metrics["official"]
    assert metrics["schema_version"] == 2
    assert set(official) == {
        "gold_source",
        "raw_continuous",
        "average_matched",
    }
    assert official["gold_source"] == "score_average"

    raw_average = predictions.mean(axis=1)
    submitted_average = average_matched_integer_scores(predictions).mean(axis=1)
    assert official["raw_continuous"]["rmse"] == pytest.approx(
        np.sqrt(np.mean(np.square(raw_average - average_labels)))
    )
    assert official["average_matched"]["rmse"] == pytest.approx(
        np.sqrt(np.mean(np.square(submitted_average - average_labels)))
    )
    assert official["raw_continuous"]["spearman"] is not None
    assert official["average_matched"]["spearman"] is not None

    # checkpoint/과거 reader 호환 alias는 두 canonical surface만 가리킨다.
    assert metrics["trait_average"]["rmse"] == official["raw_continuous"]["rmse"]
    assert set(metrics["trait_average_rounded"]) == {"average_matched_integer"}
    assert (
        metrics["trait_average_rounded"]["average_matched_integer"]
        == official["average_matched"]
    )
    removed_variants = {
        "as_submitted_float",
        "both_per_trait_rounded",
        "gold_average_rounded",
        "prediction_average_rounded",
        "per_trait_then_average_rounded",
        "continuous",
        "definition_a_rounded",
    }
    assert removed_variants.isdisjoint(metrics["trait_average_rounded"])


def test_inference_uses_stored_score_average_without_recomputing_traits() -> None:
    """공식 gold는 trait 평균과 달라도 JSON의 score.average를 그대로 보존한다."""

    official_averages = [2.0, 3.0, 4.0]
    rows = [
        {
            "essay_id": f"e{index}",
            "prompt_num": "Q1",
            "prompt_text": "prompt",
            "essay_text": "essay",
            # 의도적으로 trait 평균(3.0)과 다르게 둬 재계산 회귀를 잡는다.
            "score": {
                "content": 3.0,
                "organization": 3.0,
                "expression": 3.0,
                "average": average,
            },
        }
        for index, average in enumerate(official_averages)
    ]
    labels = [{trait: float(row["score"][trait]) for trait in TRAITS} for row in rows]
    predictions = np.asarray(
        [[average, average, average] for average in official_averages],
        dtype=np.float64,
    )

    metrics, _ = evaluate_predictions(
        rows,
        labels,
        [predictions],
        prompt_lookup={},
        prompt_routing_active=False,
    )

    assert metrics["official"]["gold_source"] == "score_average"
    assert metrics["official"]["raw_continuous"]["rmse"] == pytest.approx(0.0)
    assert metrics["traits"]["content"]["rmse"] > 0.0


def _scoring(scores: dict[str, float], rule: str) -> ScoringOutput:
    output = ScoringOutput(
        records=[{"essay_id": "essay-1", "judge": {}}],
        score_records=[{"essay_id": "essay-1", "scores": dict(scores)}],
        prediction_arrays=[],
        detail_prediction_arrays=[],
        detail_probability_arrays=[],
        unknown_prompt_count=0,
    )
    ScorePostprocessor(rule).apply_rows(output.score_records)
    attach_score_surfaces(output)
    return output


def test_both_prediction_jsonl_rows_get_raw_and_average_matched_scores_and_means() -> (
    None
):
    scores = {"content": 3.257, "organization": 3.584, "expression": 3.805}
    expected = score_surface_payload(scores)
    output = _scoring(scores, "average_matched")

    for record in (*output.records, *output.score_records):
        assert set(record["score_surfaces"]) == {
            "raw_continuous",
            "average_matched",
        }
        assert record["score_surfaces"] == expected
        assert record["score_surfaces"]["raw_continuous"]["scores"] == scores
        assert record["score_surfaces"]["raw_continuous"]["average"] == pytest.approx(
            sum(scores.values()) / len(TRAITS)
        )
        assert record["score_surfaces"]["average_matched"]["average"] == pytest.approx(
            sum(record["score_surfaces"]["average_matched"]["scores"].values())
            / len(TRAITS)
        )
        assert record["submission_matches_average_matched"] is True

    # default inference가 submission.json에 쓰는 submitted_scores와 canonical surface가 같다.
    assert (
        output.score_records[0]["submitted_scores"]
        == expected["average_matched"]["scores"]
    )
    # 한 JSONL row를 후처리해도 다른 파일의 중첩 payload가 변하지 않는다.
    output.records[0]["score_surfaces"]["raw_continuous"]["scores"]["content"] = 1.0
    assert output.score_records[0]["score_surfaces"] == expected


def test_non_default_submission_rule_is_explicitly_marked_as_noncanonical() -> None:
    # 이 값에서는 per-trait 사사오입 [3,3,3]과 average-matched [4,3,3]이 다르다.
    scores = {trait: 3.49 for trait in TRAITS}
    output = _scoring(scores, "per_trait_round")

    assert (
        output.score_records[0]["submitted_scores"]
        != output.score_records[0]["score_surfaces"]["average_matched"]["scores"]
    )
    for record in (*output.records, *output.score_records):
        assert record["score_postprocess"] == "per_trait_round"
        assert record["submission_matches_average_matched"] is False


def test_trainer_keeps_checkpoint_keys_but_drops_rounding_variant_key() -> None:
    labels = np.asarray(
        [[2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [1.0, 2.0, 3.0]],
        dtype=np.float64,
    )
    predictions = labels + 0.26
    nested = regression_metrics(labels, predictions)
    flat = compute_trainer_metrics(
        SimpleNamespace(label_ids=labels, predictions=predictions)
    )

    assert flat["official_rmse"] == nested["official"]["raw_continuous"]["rmse"]
    assert (
        flat["official_matched_rmse"] == nested["official"]["average_matched"]["rmse"]
    )
    matched = average_matched_integer_scores(predictions)
    expected_macro = np.mean(
        [
            np.sqrt(np.mean((matched[:, index] - labels[:, index]) ** 2))
            for index in range(3)
        ]
    )
    assert flat["submitted_trait_macro_rmse"] == pytest.approx(expected_macro)
    assert "official_float_rounded_rmse" not in flat


def test_trainer_uses_stored_average_label_for_matched_spearman() -> None:
    labels = np.asarray(
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]],
        dtype=np.float64,
    )
    predictions = labels.copy()
    # Reverse the stored average ordering.  Recomputing labels.mean(1) would
    # produce +1, so this fixture detects accidental fallback exactly.
    stored_average = np.asarray([4.0, 3.0, 2.0, 1.0], dtype=np.float64)
    flat = compute_trainer_metrics(
        SimpleNamespace(
            label_ids=(labels, stored_average),
            predictions=predictions,
        )
    )
    assert flat["official_matched_spearman"] == pytest.approx(-1.0)
