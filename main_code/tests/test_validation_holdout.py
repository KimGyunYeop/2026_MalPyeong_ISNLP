from __future__ import annotations

import pytest

from main_code.config import RegressionConfig
from main_code.datasets import split_validation_holdout


def row(row_id: str, source_split: str, prompt: str) -> dict:
    return {
        "id": row_id,
        "source_split": source_split,
        "essay_hash": f"hash-{row_id}",
        "prompt_num": prompt,
        "prompt": f"prompt text {prompt}",
    }


def pool() -> list[dict]:
    rows = [row(f"official-{index}", "official_train", "Q1") for index in range(10)]
    rows += [
        row(f"extra-{index:03d}", "origin_pool_extra", "Q1" if index < 60 else "Q2")
        for index in range(100)
    ]
    return rows


def test_zero_is_a_no_op() -> None:
    rows = pool()
    remaining, holdout = split_validation_holdout(rows, 0)
    assert holdout == []
    assert [item["id"] for item in remaining] == [item["id"] for item in rows]


def test_holdout_never_takes_official_train_rows() -> None:
    _, holdout = split_validation_holdout(pool(), 20)
    assert len(holdout) == 20
    assert {item["source_split"] for item in holdout} == {"origin_pool_extra"}


def test_holdout_and_training_are_disjoint() -> None:
    remaining, holdout = split_validation_holdout(pool(), 20)
    assert len(remaining) == 90
    assert not {item["id"] for item in remaining} & {item["id"] for item in holdout}


def test_holdout_keeps_the_prompt_mix() -> None:
    _, holdout = split_validation_holdout(pool(), 20)
    counts = {"Q1": 0, "Q2": 0}
    for item in holdout:
        counts[item["prompt_num"]] += 1
    # origin_pool_extra는 Q1 60 / Q2 40이므로 20편이면 12 / 8이 되어야 한다.
    assert counts == {"Q1": 12, "Q2": 8}


def test_selection_is_deterministic_across_calls() -> None:
    first = split_validation_holdout(pool(), 20)[1]
    second = split_validation_holdout(list(reversed(pool())), 20)[1]
    assert {item["id"] for item in first} == {item["id"] for item in second}


def test_too_large_a_holdout_is_an_error() -> None:
    with pytest.raises(ValueError, match="holdout"):
        split_validation_holdout(pool(), 500)


def test_config_rejects_a_holdout_without_the_extra_pool() -> None:
    with pytest.raises(ValueError, match="full profile"):
        RegressionConfig(
            model_id="skt/A.X-4.0-Light",
            training_mode="lora_only",
            primary_data_profile="official",
            validation_holdout_size=100,
        ).validate()


def test_config_default_keeps_the_current_behaviour() -> None:
    config = RegressionConfig(
        model_id="skt/A.X-4.0-Light", training_mode="lora_only"
    ).validate()
    assert config.validation_holdout_size == 0
