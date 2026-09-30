"""Test-only config bases with explicit recipe semantics."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from main_code.config import RegressionConfig, legacy_baseline_config


# 단위 테스트의 공통 기반은 과거 pre-y1 대조군(Gemma head-only)이다. configs/baseline.json은
# 기술서 기준 채점 모델로 바뀌었으므로, 그 위에 pre-y1 값만 되돌려 테스트 의미를 유지한다.
_PRE_Y1_OVERRIDES: dict[str, Any] = {
    "model_id": "google/gemma-4-12B-it",
    "model_slug": "gemma12",
    "model_revision": "main",
    "trust_remote_code": False,
    "training_mode": "head_only",
    "essay_surface": "canonical",
    "distribution_loss_weight": 1.0,
    "batch_size": 1,
    "gradient_accumulation": 16,
    "lora_learning_rate": 2e-05,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_include_mlp": False,
    "max_train_steps": 0,
    "eval_steps": 0,
    "best_checkpoint_metric": "rmse",
}
_LEGACY_BASELINE = replace(legacy_baseline_config(), **_PRE_Y1_OVERRIDES)


def legacy_config(**updates: Any) -> RegressionConfig:
    """Return the frozen pre-y1 baseline with unvalidated field updates."""

    return replace(_LEGACY_BASELINE, **updates)
