"""Fixed-score rationale generation baseline for the MalPyeong competition."""

TRAITS = ("content", "organization", "expression")

# The prepared score-training pool has exactly these two train-only sources.
TRAIN_SOURCE_SPLITS = frozenset({"official_train", "origin_pool_extra"})
