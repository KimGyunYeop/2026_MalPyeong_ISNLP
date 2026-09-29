"""Test-only config bases with explicit recipe semantics."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from main_code.config import RegressionConfig, legacy_baseline_config


_LEGACY_BASELINE = legacy_baseline_config()


def legacy_config(**updates: Any) -> RegressionConfig:
    """Return the frozen pre-y1 baseline with unvalidated field updates."""

    return replace(_LEGACY_BASELINE, **updates)
