from __future__ import annotations

import pytest

from main_code.train import resolve_step_schedule
from main_code.tests.config_helpers import legacy_config


def _row(index: int) -> dict:
    return {
        "id": f"row-{index}",
        "prompt_num": "Q1",
        "prompt": "prompt",
        "essay": f"essay {index}",
        "score": {
            "content": 3.0,
            "organization": 3.0,
            "expression": 3.0,
        },
    }


@pytest.mark.parametrize("training_mode", ("lora_only", "head_only"))
def test_single_stage_step_schedule_records_zero_head_warmup(training_mode: str) -> None:
    config = legacy_config(
        training_mode=training_mode,
        batch_size=2,
        gradient_accumulation=1,
        max_train_steps=10,
        head_warmup_steps=0,
        eval_steps=0,
    )
    resolved, steps_per_epoch, _ = resolve_step_schedule(
        config, [_row(index) for index in range(4)]
    )
    assert steps_per_epoch == 2
    assert resolved.head_warmup_steps == 0
    assert resolved.eval_steps == 2


def test_two_stage_step_schedule_still_resolves_automatic_head_boundary() -> None:
    config = legacy_config(
        training_mode="two_stage",
        batch_size=2,
        gradient_accumulation=1,
        max_train_steps=10,
        head_epochs=2,
        head_warmup_steps=0,
        eval_steps=0,
    )
    resolved, steps_per_epoch, _ = resolve_step_schedule(
        config, [_row(index) for index in range(4)]
    )
    assert steps_per_epoch == 2
    assert resolved.head_warmup_steps == 4
    assert resolved.eval_steps == 2
