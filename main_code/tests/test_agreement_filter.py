from __future__ import annotations

import pytest

from main_code.config import RegressionConfig
from main_code.datasets import (
    DETAIL_CRITERIA,
    filter_origin_extra_by_rater_agreement,
    primary_rater_disagreement_sum,
)
from main_code.train import competition_selection_description


def row(row_id: str, source_split: str, differences: list[int]) -> dict:
    traits = {}
    offset = 0
    for trait, count in (("content", 5), ("organization", 2), ("expression", 2)):
        criteria = {}
        for index in range(1, count + 1):
            difference = differences[offset]
            offset += 1
            criteria[f"{trait}_{index}"] = {
                "rater_scores": {"evaluator1": 2, "evaluator2": 2 + difference}
            }
        traits[trait] = {"criteria": criteria}
    return {
        "id": row_id,
        "source_split": source_split,
        "score_details": {
            "primary_raters": ["evaluator1", "evaluator2"],
            "traits": traits,
        },
    }


def test_filter_keeps_official_rows_and_filters_only_origin_extra() -> None:
    official = row("official", "official_train", [3] * 9)
    low = row("low", "origin_pool_extra", [1, 1, 1, 1, 1, 1, 0, 0, 0])
    high = row("high", "origin_pool_extra", [1] * 9)

    filtered = filter_origin_extra_by_rater_agreement(
        [official, low, high], max_disagreement=6
    )

    assert [item["id"] for item in filtered] == ["official", "low"]
    assert filter_origin_extra_by_rater_agreement([high], -1) == [high]


def test_missing_primary_slot_uses_next_numbered_original_rater() -> None:
    example = row("fallback", "origin_pool_extra", [0] * 9)
    for criterion in DETAIL_CRITERIA:
        trait = criterion.rsplit("_", 1)[0]
        scores = example["score_details"]["traits"][trait]["criteria"][criterion][
            "rater_scores"
        ]
        scores["evaluator2"] = None
        scores["evaluator3"] = 4
        scores["re-evaluator"] = 1

    assert primary_rater_disagreement_sum(example) == 18


def test_invalid_extra_supervision_fails_instead_of_silent_selection() -> None:
    example = row("broken", "origin_pool_extra", [0] * 9)
    del example["score_details"]["traits"]["content"]["criteria"]["content_1"]
    with pytest.raises(ValueError, match="content_1"):
        filter_origin_extra_by_rater_agreement([example], 8)


def test_config_default_is_off_and_threshold_must_be_valid() -> None:
    assert RegressionConfig().validate().origin_extra_max_rater_disagreement == -1
    assert (
        RegressionConfig(origin_extra_max_rater_disagreement=8).validate()
        .origin_extra_max_rater_disagreement
        == 8
    )
    for invalid in (-2, -0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="origin_extra_max_rater_disagreement"):
            RegressionConfig(origin_extra_max_rater_disagreement=invalid).validate()


def test_selection_description_does_not_mislabel_legacy_subsets() -> None:
    assert competition_selection_description(RegressionConfig()) == "full"
    assert (
        competition_selection_description(RegressionConfig(competition_train_limit=2000))
        == "prompt-proportional deterministic subset"
    )
    assert "official rows retained" in competition_selection_description(
        RegressionConfig(origin_extra_max_rater_disagreement=8)
    )
    assert competition_selection_description(
        RegressionConfig(
            competition_train_limit=2000,
            origin_extra_max_rater_disagreement=8,
        )
    ).startswith("agreement filter")
