from __future__ import annotations

from types import SimpleNamespace

import pytest

from main_code.config import RegressionConfig
from main_code.datasets import MultiSourceStepSampler, split_primary_rows_by_source
from main_code.train import resolve_step_schedule
from main_code.tests.config_helpers import legacy_config


def _row(index: int, source_split: str) -> dict:
    return {
        "id": f"row-{index}",
        "prompt_num": "Q1",
        "prompt": "prompt",
        "essay": f"essay {index}",
        "source_split": source_split,
        "score": {
            "content": 3.0,
            "organization": 3.0,
            "expression": 3.0,
        },
    }


def _source_config(**updates) -> RegressionConfig:
    values = {
        "primary_data_profile": "full",
        "dataset_schedule": "pretrain_then_competition",
        "split_primary_sources": True,
        "max_train_steps": 6,
        "final_competition_epochs": 2,
        "batch_size": 2,
        "gradient_accumulation": 1,
        "inbatch_sampling": "random",
    }
    values.update(updates)
    return legacy_config(**values)


def test_split_primary_sources_is_a_valid_internal_schedule() -> None:
    config = _source_config()
    assert config.validate() is config

    with pytest.raises(ValueError, match="extended_datasets"):
        legacy_config(
            primary_data_profile="full",
            dataset_schedule="pretrain_then_competition",
            max_train_steps=6,
            final_competition_epochs=2,
        ).validate()


def test_primary_rows_are_partitioned_without_changing_source_metadata() -> None:
    rows = split_primary_rows_by_source(
        [_row(0, "official_train"), _row(1, "origin_pool_extra")]
    )
    assert [row["_dataset_name"] for row in rows] == [
        "competition",
        "origin_pool_extra",
    ]
    assert [row["source_split"] for row in rows] == [
        "official_train",
        "origin_pool_extra",
    ]


def test_final_epoch_size_uses_only_official_rows() -> None:
    rows = split_primary_rows_by_source(
        [_row(index, "official_train") for index in range(4)]
        + [_row(index + 4, "origin_pool_extra") for index in range(8)]
    )
    _, steps_per_epoch, final_steps = resolve_step_schedule(_source_config(), rows)
    assert steps_per_epoch == 2
    assert final_steps == 4


def test_pretrain_plan_places_origin_extra_before_official_rows() -> None:
    rows = split_primary_rows_by_source(
        [_row(index, "official_train") for index in range(4)]
        + [_row(index + 4, "origin_pool_extra") for index in range(8)]
    )
    sampler = MultiSourceStepSampler(
        SimpleNamespace(rows=rows),
        batch_size=2,
        gradient_accumulation=1,
        main_steps=6,
        final_competition_steps=4,
        schedule="pretrain_then_competition",
        competition_mix_ratio=0.5,
        competition_every_n_steps=2,
        extended_source_sampling="uniform",
        extended_source_weights="",
        inbatch_sampling="random",
        seed=42,
    )
    assert sampler.source_plan() == ["origin_pool_extra"] * 6 + ["competition"] * 4
