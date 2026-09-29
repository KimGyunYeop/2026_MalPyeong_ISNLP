"""Self-contained, editable regression baseline for the MalPyeong task.

The package initializer intentionally does not import the large model stack.
Data preparation and config inspection therefore work without transformers or
GPU-only dependencies being imported as a side effect.
"""

from .config import RegressionConfig

__all__ = ["RegressionConfig", "RegressionScorer"]


def __getattr__(name: str):
    """Keep the old package export without importing the model stack eagerly."""

    if name == "RegressionScorer":
        from .models import RegressionScorer

        return RegressionScorer
    raise AttributeError(name)
