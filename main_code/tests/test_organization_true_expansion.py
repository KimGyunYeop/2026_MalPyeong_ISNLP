from __future__ import annotations

from copy import deepcopy
from itertools import permutations
from types import SimpleNamespace

import pytest

import main_code.organization_data as organization_data
from main_code.datasets import MultiSourceStepSampler
from main_code.organization_data import (
    apply_external_organization_policy,
    build_paragraph_order_rows,
    build_sentence_order_rows,
)
from main_code.tests.config_helpers import legacy_config


def _aihub24_row(
    row_id: str,
    *,
    dataset_name: str = "aihub24_essay",
    grade: str = "고등_2",
    essay_weight: float = 1.0,
    paragraph_weight: float = 1.0,
    coherence_weight: float = 1.0,
) -> dict:
    return {
        "id": row_id,
        "_dataset_name": dataset_name,
        "metadata": {"grade": grade},
        # The stored organization value is deliberately different from the
        # equal-weight structural target expected after policy application.
        "score": {
            "content": 4.0,
            "organization": 4.75,
            "expression": 2.0,
            "average": 3.5833333333333335,
        },
        "score_details": {
            "traits": {
                "organization": {
                    "criteria": {
                        "org_essay": {
                            "official_score": 4.5,
                            "weight": essay_weight,
                        },
                        "org_paragraph": {
                            "official_score": 2.5,
                            "weight": paragraph_weight,
                        },
                        "org_coherence": {
                            "official_score": 5.0,
                            "weight": coherence_weight,
                        },
                    }
                }
            }
        },
    }


def _competition_row() -> dict:
    return {
        "id": "source-1",
        "document_id": "source-1",
        "prompt_num": "Q1",
        "prompt": "네 문단으로 논증하시오.",
        "essay": "A\n\nB\n\nC\n\nD",
        "essay_surfaces": {"official_raw": "A B C D"},
        "metadata": {"paragraph_count": 4},
        "score_details": {
            "label_policy": "mean_of_two_raters",
            "traits": {
                "organization": {
                    "criteria": {
                        "organization_1": {
                            "rater_scores": {"rater-a": 5, "rater-b": 4},
                            "selected_for_target": {
                                "rater-a": True,
                                "rater-b": True,
                            },
                        },
                        "organization_2": {
                            "rater_scores": {"rater-a": 4, "rater-b": 5},
                            "selected_for_target": {
                                "rater-a": True,
                                "rater-b": True,
                            },
                        },
                    }
                }
            },
        },
    }


def _sampler_row(row_id: str, source: str) -> dict:
    return {
        "id": row_id,
        "_dataset_name": source,
        "prompt_num": "Q1",
        "prompt": f"{source} prompt",
        "essay": f"{source} essay {row_id}",
        "score": {"content": 3.0, "organization": 3.0, "expression": 3.0},
    }


def _aihub26_policy_row(row_id: str, purpose: str) -> dict:
    return {
        "id": row_id,
        "_dataset_name": "aihub26_essay",
        "metadata": {"purpose": purpose},
        "score": {
            "content": 3.0,
            "organization": 3.25,
            "expression": 3.5,
            "average": 3.25,
        },
    }


def test_aihub24_structural_policy_filters_and_relabels_without_mutation() -> None:
    eligible = _aihub24_row("eligible")
    original = deepcopy(eligible)
    other_source = _aihub24_row("other-source", dataset_name="nikl24_summary")
    rows = [
        eligible,
        _aihub24_row("wrong-grade", grade="초등_6"),
        other_source,
        _aihub24_row("no-essay", essay_weight=0.0),
        _aihub24_row("no-paragraph", paragraph_weight=0.0),
        _aihub24_row("no-coherence", coherence_weight=0.0),
    ]

    selected = apply_external_organization_policy(
        rows, "aihub24_highschool_structural_v1"
    )

    assert [row["id"] for row in selected] == ["eligible", "other-source"]
    assert selected[0] is not eligible
    assert selected[1] is other_source
    assert selected[0]["score"] == {
        "content": 4.0,
        "organization": 3.5,
        "expression": 2.0,
        "average": pytest.approx((4.0 + 3.5 + 2.0) / 3.0),
    }
    assert (
        selected[0]["organization_label_policy"] == "aihub24_highschool_structural_v1"
    )
    assert eligible == original


def test_external_organization_policy_and_config_fail_closed() -> None:
    row = _aihub24_row("eligible")
    source_rows = [row]
    official = apply_external_organization_policy(source_rows, "official")
    assert official == [row]
    assert official is not source_rows

    with pytest.raises(ValueError, match="지원하지 않는 external organization policy"):
        apply_external_organization_policy([row], "unknown")

    config = legacy_config(
        extended_datasets="aihub24_essay nikl24_summary",
        dataset_schedule="external_only",
        max_train_steps=4,
        external_organization_label_policy="aihub24_highschool_structural_v1",
    )
    assert config.validate() is config

    with pytest.raises(ValueError, match="extended_datasets에 aihub24_essay"):
        legacy_config(
            extended_datasets="nikl24_summary",
            dataset_schedule="external_only",
            max_train_steps=4,
            external_organization_label_policy="aihub24_highschool_structural_v1",
        ).validate()


def test_a24_a26_pack_policy_is_source_local_and_fail_closed() -> None:
    a24 = _aihub24_row("a24")
    a26_persuasion = _aihub26_policy_row("a26-persuasion", "설득")
    a26_explanation = _aihub26_policy_row("a26-explanation", "설명")
    selected = apply_external_organization_policy(
        [a24, a26_persuasion, a26_explanation],
        "aihub24_structural_aihub26_persuasion_v1",
    )

    assert [row["id"] for row in selected] == ["a24", "a26-persuasion"]
    assert selected[0]["score"]["organization"] == pytest.approx(3.5)
    assert selected[1] is a26_persuasion
    assert selected[1]["score"]["organization"] == pytest.approx(3.25)

    config = legacy_config(
        extended_datasets="aihub24_essay,aihub26_essay",
        dataset_schedule="external_only",
        max_train_steps=4,
        external_organization_label_policy=(
            "aihub24_structural_aihub26_persuasion_v1"
        ),
    )
    assert config.validate() is config

    for datasets in (
        "aihub24_essay",
        "aihub26_essay",
        "aihub24_essay,aihub26_essay,nikl24_summary",
    ):
        with pytest.raises(
            ValueError,
            match="extended_datasets=aihub24_essay,aihub26_essay",
        ):
            legacy_config(
                extended_datasets=datasets,
                dataset_schedule="external_only",
                max_train_steps=4,
                external_organization_label_policy=(
                    "aihub24_structural_aihub26_persuasion_v1"
                ),
            ).validate()


def test_paragraph_order_rows_are_deterministic_exact_span_pairs() -> None:
    source = _competition_row()
    original_source = deepcopy(source)

    rows = build_paragraph_order_rows([source], seed=42)

    assert rows == build_paragraph_order_rows([source], seed=42)
    assert source == original_source
    assert len(rows) == 4
    assert len({row["id"] for row in rows}) == 4
    assert {row["_dataset_name"] for row in rows} == {
        "paragraph_order_high_confidence_v1"
    }

    exact_spans = ("A", " B", " C", " D")
    exact_span_permutations = {"".join(order) for order in permutations(exact_spans)}
    by_kind: dict[str, list[dict]] = {}
    for row in rows:
        by_kind.setdefault(row["metadata"]["corruption"], []).append(row)
        assert row["prompt"] == source["prompt"]
        assert row["essay"] in exact_span_permutations
        assert "\n" not in row["essay"]
        assert len(row["essay"]) == len(source["essay_surfaces"]["official_raw"])
        score = row["score"]
        assert score["content"] == 3.0
        assert score["expression"] == 3.0
        assert score["average"] == pytest.approx(
            (score["content"] + score["organization"] + score["expression"]) / 3.0
        )

    assert set(by_kind) == {"moderate", "severe"}
    for kind, pair in by_kind.items():
        assert {row["metadata"]["role"] for row in pair} == {
            "original",
            "corrupted",
        }
        assert len({row["prompt_num"] for row in pair}) == 1
        original = next(row for row in pair if row["metadata"]["role"] == "original")
        corrupted = next(row for row in pair if row["metadata"]["role"] == "corrupted")
        assert original["essay"] == source["essay_surfaces"]["official_raw"]
        assert corrupted["essay"] != original["essay"]
        assert original["score"]["organization"] > corrupted["score"]["organization"]
        expected_low = 2.0 if kind == "moderate" else 1.0
        assert corrupted["score"]["organization"] == expected_low

    assert len({pair[0]["prompt_num"] for pair in by_kind.values()}) == 2


def test_paragraph_order_requires_four_high_confidence_ratings_and_four_paragraphs() -> (
    None
):
    too_short = _competition_row()
    too_short["metadata"]["paragraph_count"] = 3
    assert build_paragraph_order_rows([too_short], seed=42) == []

    low_confidence = _competition_row()
    low_confidence["score_details"]["traits"]["organization"]["criteria"][
        "organization_2"
    ]["rater_scores"]["rater-b"] = 3
    assert build_paragraph_order_rows([low_confidence], seed=42) == []

    wrong_label_policy = _competition_row()
    wrong_label_policy["score_details"]["label_policy"] = "single_re_evaluator"
    assert build_paragraph_order_rows([wrong_label_policy], seed=42) == []


def test_sentence_order_rows_are_deterministic_exact_span_pairs(monkeypatch) -> None:
    source = _competition_row()
    source["essay_surfaces"]["official_raw"] = "A B C D E F G H"
    sentence_spans = tuple((index, index + 1) for index in range(0, 15, 2))
    monkeypatch.setattr(
        organization_data,
        "kiwi_sentence_spans",
        lambda official_raw: sentence_spans,
    )

    rows = build_sentence_order_rows([source], seed=42)

    assert rows == build_sentence_order_rows([source], seed=42)
    assert len(rows) == 4
    assert {row["_dataset_name"] for row in rows} == {
        "sentence_order_high_confidence_v1"
    }
    exact_spans = ("A", " B", " C", " D", " E", " F", " G", " H")
    exact_span_permutations = {"".join(order) for order in permutations(exact_spans)}
    by_kind: dict[str, list[dict]] = {}
    for row in rows:
        by_kind.setdefault(row["metadata"]["corruption"], []).append(row)
        assert row["essay"] in exact_span_permutations
        assert len(row["essay"]) == len(source["essay_surfaces"]["official_raw"])
        assert "\n" not in row["essay"]

    assert set(by_kind) == {"moderate", "severe"}
    for pair in by_kind.values():
        original = next(row for row in pair if row["metadata"]["role"] == "original")
        corrupted = next(row for row in pair if row["metadata"]["role"] == "corrupted")
        assert original["essay"] == source["essay_surfaces"]["official_raw"]
        assert corrupted["essay"] != original["essay"]
        assert original["score"]["organization"] > corrupted["score"]["organization"]
        assert len({row["prompt_num"] for row in pair}) == 1


def test_sentence_order_requires_eight_sentences_and_high_confidence(
    monkeypatch,
) -> None:
    source = _competition_row()
    source["essay_surfaces"]["official_raw"] = "A B C D E F G"
    monkeypatch.setattr(
        organization_data,
        "kiwi_sentence_spans",
        lambda official_raw: tuple((index, index + 1) for index in range(0, 13, 2)),
    )
    assert build_sentence_order_rows([source], seed=42) == []

    low_confidence = _competition_row()
    low_confidence["score_details"]["traits"]["organization"]["criteria"][
        "organization_1"
    ]["rater_scores"]["rater-a"] = 3
    assert build_sentence_order_rows([low_confidence], seed=42) == []


def test_external_only_paragraph_sampler_keeps_each_physical_batch_an_exact_pair() -> (
    None
):
    competition = _competition_row()
    augmentation = build_paragraph_order_rows([competition], seed=42)
    rows = [competition, *augmentation]
    sampler = MultiSourceStepSampler(
        SimpleNamespace(rows=rows),
        batch_size=2,
        gradient_accumulation=1,
        main_steps=2,
        final_competition_steps=0,
        schedule="external_only",
        competition_mix_ratio=0.5,
        competition_every_n_steps=2,
        extended_source_sampling="uniform",
        extended_source_weights="",
        inbatch_sampling="same_question",
        seed=42,
    )

    assert sampler.source_plan() == ["paragraph_order_high_confidence_v1"] * 2
    sampled = list(iter(sampler))
    assert len(sampled) == 4
    observed_kinds = set()
    for start in range(0, len(sampled), 2):
        pair = [rows[index] for index in sampled[start : start + 2]]
        assert {row["metadata"]["role"] for row in pair} == {
            "original",
            "corrupted",
        }
        assert len({row["prompt_num"] for row in pair}) == 1
        observed_kinds.add(pair[0]["metadata"]["corruption"])
    assert observed_kinds == {"moderate", "severe"}


def test_external_only_weighted_plan_has_no_competition_steps() -> None:
    rows = [
        _sampler_row("competition-1", "competition"),
        _sampler_row("competition-2", "competition"),
        *[_sampler_row(f"a-{index}", "ext-a") for index in range(4)],
        *[_sampler_row(f"b-{index}", "ext-b") for index in range(4)],
    ]
    sampler = MultiSourceStepSampler(
        SimpleNamespace(rows=rows),
        batch_size=2,
        gradient_accumulation=2,
        main_steps=8,
        final_competition_steps=0,
        schedule="external_only",
        competition_mix_ratio=0.5,
        competition_every_n_steps=2,
        extended_source_sampling="uniform",
        extended_source_weights="ext-a=3 ext-b=1",
        inbatch_sampling="random",
        seed=7,
    )

    plan = sampler.source_plan()
    assert plan.count("ext-a") == 6
    assert plan.count("ext-b") == 2
    assert "competition" not in plan
    assert sampler.planned_summary() == {
        "main_steps": 8,
        "final_competition_steps": 0,
        "total_steps": 8,
        "source_steps": {"ext-a": 6, "ext-b": 2},
        "source_examples": {"ext-a": 24, "ext-b": 8},
    }

    sampled = iter(sampler)
    for source in plan:
        batch = [rows[next(sampled)] for _ in range(4)]
        assert {row["_dataset_name"] for row in batch} == {source}
    with pytest.raises(StopIteration):
        next(sampled)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"dataset_schedule": "external_only", "max_train_steps": 4},
            "확장 데이터 schedule",
        ),
        (
            {
                "extended_datasets": "nikl24_summary",
                "dataset_schedule": "external_only",
                "max_train_steps": 0,
            },
            "max_train_steps >= 1",
        ),
        (
            {
                "extended_datasets": "nikl24_summary",
                "dataset_schedule": "external_only",
                "max_train_steps": 4,
                "final_competition_epochs": 1,
            },
            "external_only",
        ),
        (
            {
                "primary_data_profile": "full",
                "dataset_schedule": "external_only",
                "split_primary_sources": True,
                "max_train_steps": 4,
            },
            "external_only",
        ),
    ],
)
def test_external_only_invalid_configurations_are_rejected(
    updates: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        legacy_config(**updates).validate()


@pytest.mark.parametrize(
    "augmentation",
    [
        "paragraph_order_high_confidence_v1",
        "sentence_order_high_confidence_v1",
    ],
)
def test_external_only_accepts_online_organization_augmentation_as_its_source(
    augmentation: str,
) -> None:
    config = legacy_config(
        dataset_schedule="external_only",
        max_train_steps=2,
        organization_augmentation=augmentation,
    )
    assert config.validate() is config
