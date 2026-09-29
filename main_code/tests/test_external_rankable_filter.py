from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from main_code.config import DATA_ROOT
from main_code.datasets import (
    MultiSourceStepSampler,
    human_scores,
    load_prepared_extended_rows,
    normalized_essay_hash,
    question_key,
)
from main_code.train import filter_external_rankable_groups
from main_code.tests.config_helpers import legacy_config
import pathlib

# 대회 데이터는 배포가 제한되어 저장소에 없다. 데이터가 있는 환경에서만 돈다.
_DATA_ROOT = pathlib.Path(__file__).resolve().parents[2] / "main_code/datasets/processed_dataset"
pytestmark = pytest.mark.skipif(
    not (_DATA_ROOT / "train.jsonl").is_file(),
    reason="대회 데이터(main_code/datasets/processed_dataset)가 없는 환경",
)



SUMMARY_ALIASES = ("nikl24_summary", "nikl25_argumentative_summary")


def _competition_row(index: int) -> dict:
    return {
        "id": f"competition-{index}",
        "_dataset_name": "competition",
        "prompt_num": "Q1",
        "prompt": "target prompt",
        "essay": f"target essay {index}",
        "score": {
            "content": 3.0,
            "organization": float(2 + index),
            "expression": 3.0,
        },
    }


def _filtered_summary_rows() -> tuple[list[dict], dict[str, int]]:
    rows, _ = load_prepared_extended_rows(
        DATA_ROOT,
        SUMMARY_ALIASES,
        validate_manifest=False,
    )
    filtered = filter_external_rankable_groups(rows, "organization")
    counts = Counter(str(row["_dataset_name"]) for row in filtered)
    return filtered, dict(counts)


def test_external_rankable_filter_is_explicit_and_preserves_source_boundaries() -> None:
    rows = [
        {
            "id": "a1",
            "_dataset_name": "a",
            "prompt_num": "Q1",
            "prompt": "same prompt",
            "essay": "same essay",
            "score": {"content": 3, "organization": 2, "expression": 3},
        },
        {
            "id": "a2",
            "_dataset_name": "a",
            "prompt_num": "Q1",
            "prompt": "same prompt",
            "essay": "same essay",
            "score": {"content": 3, "organization": 4, "expression": 3},
        },
        {
            "id": "b1",
            "_dataset_name": "b",
            "prompt_num": "Q1",
            "prompt": "same prompt",
            "essay": "first distinct essay",
            "score": {"content": 3, "organization": 2, "expression": 3},
        },
        {
            "id": "b2",
            "_dataset_name": "b",
            "prompt_num": "Q1",
            "prompt": "same prompt",
            "essay": "second distinct essay",
            "score": {"content": 3, "organization": 2, "expression": 3},
        },
    ]

    assert filter_external_rankable_groups(rows, "") is rows
    assert filter_external_rankable_groups(rows, "organization") == []

    with pytest.raises(ValueError, match="extended_datasets"):
        legacy_config(external_rankable_trait="organization").validate()
    with pytest.raises(ValueError, match="external_train_limit"):
        legacy_config(
            extended_datasets="nikl24_summary",
            dataset_schedule="mixed",
            max_train_steps=2,
            external_rankable_trait="organization",
            external_train_limit=2,
        ).validate()


def test_real_summary_rankable_counts_and_b2ga16_source_plan() -> None:
    external_rows, counts = _filtered_summary_rows()
    assert counts == {
        "nikl24_summary": 5_866,
        "nikl25_argumentative_summary": 1_668,
    }
    assert len(external_rows) == 7_534

    grouped: dict[tuple[str, tuple[str, str]], list[dict]] = {}
    for row in external_rows:
        key = str(row["_dataset_name"]), question_key(row)
        grouped.setdefault(key, []).append(row)
    assert len(grouped) == 3_767
    for group in grouped.values():
        assert len(group) == 2
        assert len({normalized_essay_hash(row) for row in group}) == 2
        assert len({human_scores(row)["organization"] for row in group}) == 2

    rows = [_competition_row(0), _competition_row(1), *external_rows]
    sampler = MultiSourceStepSampler(
        SimpleNamespace(rows=rows),
        batch_size=2,
        gradient_accumulation=16,
        main_steps=1_461,
        final_competition_steps=0,
        schedule="mixed",
        competition_mix_ratio=0.6242299794661191,
        competition_every_n_steps=2,
        extended_source_sampling="proportional",
        extended_source_weights="",
        inbatch_sampling="same_question",
        seed=42,
    )
    summary = sampler.planned_summary()
    assert summary["source_steps"] == {
        "competition": 912,
        "nikl24_summary": 427,
        "nikl25_argumentative_summary": 122,
    }
    assert summary["source_examples"] == {
        "competition": 29_184,
        "nikl24_summary": 13_664,
        "nikl25_argumentative_summary": 3_904,
    }
    assert 912 * 16 == 14_592
    assert (427 + 122) * 16 == 8_784
    assert 1_461 * 16 == 23_376

    sampled_indices = iter(sampler)
    for source in sampler.source_plan():
        for _ in range(16):
            pair = [rows[next(sampled_indices)], rows[next(sampled_indices)]]
            assert all(row["_dataset_name"] == source for row in pair)
            assert question_key(pair[0]) == question_key(pair[1])
            if source != "competition":
                assert normalized_essay_hash(pair[0]) != normalized_essay_hash(pair[1])
                assert (
                    human_scores(pair[0])["organization"]
                    != human_scores(pair[1])["organization"]
                )
    with pytest.raises(StopIteration):
        next(sampled_indices)
