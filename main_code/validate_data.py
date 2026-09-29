"""Strict integrity checks for datasets produced by :mod:`prepare_data`.

The validator is intentionally independent from the training stack.  It checks
the files in one pass, verifies their manifests, reconstructs every detailed
score, and then checks the clean/leaked/public split relationships.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


# Declared profile contracts --------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PACKAGE_ROOT / "datasets"
DEFAULT_OFFICIAL_ROOT = DEFAULT_DATA_ROOT / "raw_dataset" / "official_competition"
SCHEMA_VERSION = 1
TRAITS = ("content", "organization", "expression")

PROFILES: dict[str, dict[str, Any]] = {
    "competition": {
        "directory": "processed_dataset",
        "group": "competition",
        "source": "nikl_competition",
        "train_rows": 11600,
        "validation_rows": 400,
        "contains_validation_labels": False,
        "safe_for_model_selection": True,
    },
    "competition_leaked": {
        "directory": "processed_dataset_validation_leaked",
        "group": "competition",
        "source": "nikl_competition",
        "train_rows": 12000,
        "validation_rows": 400,
        "contains_validation_labels": True,
        "safe_for_model_selection": False,
    },
    "official_only": {
        "directory": "processed_dataset_only_official_competition",
        "group": "competition",
        "source": "nikl_competition",
        "train_rows": 2000,
        "validation_rows": 400,
        "contains_validation_labels": False,
        "safe_for_model_selection": True,
    },
    "aihub24_essay": {
        "directory": "processed_dataset_aihub_external_에세이",
        "group": "external",
        "source": "aihub24_essay",
        "train_rows": 45484,
        "raw_rows": 45497,
        "exact_duplicates": 13,
        "source_splits": {
            "aihub_training": {"raw_rows": 39591, "written_rows": 39584},
            "aihub_validation": {"raw_rows": 5906, "written_rows": 5900},
        },
    },
    "aihub25_descriptive": {
        "directory": "processed_dataset_aihub_external_서술",
        "group": "external",
        "source": "aihub25_descriptive",
        "train_rows": 36006,
        "raw_rows": 36006,
        "exact_duplicates": 0,
        "source_splits": {
            "aihub_training": {"raw_rows": 32006, "written_rows": 32006},
            "aihub_validation": {"raw_rows": 4000, "written_rows": 4000},
        },
    },
    "aihub26_essay": {
        "directory": "processed_dataset_aihub_external_논술",
        "group": "external",
        "source": "aihub26_essay",
        "train_rows": 18010,
        "raw_rows": 18010,
        "exact_duplicates": 0,
        "source_splits": {
            "aihub_training": {"raw_rows": 16010, "written_rows": 16010},
            "aihub_validation": {"raw_rows": 2000, "written_rows": 2000},
        },
    },
    "aihub27_topic": {
        "directory": "processed_dataset_aihub_external_주제별",
        "group": "external",
        "source": "aihub27_topic",
        "train_rows": 18001,
        "raw_rows": 18001,
        "exact_duplicates": 0,
        "source_splits": {
            "aihub_training": {"raw_rows": 16001, "written_rows": 16001},
            "aihub_validation": {"raw_rows": 2000, "written_rows": 2000},
        },
    },
    "nikl24_summary": {
        "directory": "processed_dataset_nikl_external_summary_2024",
        "group": "external",
        "source": "nikl24_summary",
        "train_rows": 7712,
        "raw_rows": 7712,
        "exact_duplicates": 0,
        "source_splits": {
            "nikl24_summary": {"raw_rows": 7712, "written_rows": 7712},
        },
    },
    "nikl25_argumentative_summary": {
        "directory": "processed_dataset_nikl_external_argumentative_summary_2025",
        "group": "external",
        "source": "nikl25_argumentative_summary",
        "train_rows": 2016,
        "raw_rows": 2016,
        "exact_duplicates": 0,
        "source_splits": {
            "nikl25_argumentative_summary": {
                "raw_rows": 2016,
                "written_rows": 2016,
            },
        },
    },
    "nikl25_cooperative_summary": {
        "directory": "processed_dataset_nikl_external_cooperative_summary_2025",
        "group": "external",
        "source": "nikl25_cooperative_summary",
        "train_rows": 1020,
        "raw_rows": 1020,
        "exact_duplicates": 0,
        "source_splits": {
            "nikl25_cooperative_summary": {
                "raw_rows": 1020,
                "written_rows": 1020,
            },
        },
    },
}

EXPECTED_CRITERIA = {
    "nikl_competition": {"content": 5, "organization": 2, "expression": 2},
    "aihub24_essay": {"content": 4, "organization": 4, "expression": 3},
    "aihub25_descriptive": {"content": 3, "organization": 2, "expression": 2},
    "aihub26_essay": {"content": 3, "organization": 2, "expression": 2},
    "aihub27_topic": {"content": 3, "organization": 2, "expression": 2},
    "nikl24_summary": {"content": 3, "organization": 1, "expression": 1},
    "nikl25_argumentative_summary": {
        "content": 3,
        "organization": 1,
        "expression": 1,
    },
    "nikl25_cooperative_summary": {
        "content": 3,
        "organization": 1,
        "expression": 1,
    },
}


# Shared validation primitives -----------------------------------------------
class ValidationError(ValueError):
    """A processed dataset violates its declared contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def mapping(value: Any, context: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{context}: object required")
    return value


def sequence(value: Any, context: str) -> list[Any]:
    require(isinstance(value, list), f"{context}: list required")
    return value


def finite(value: Any, context: str, low: float, high: float) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{context}: number required",
    )
    result = float(value)
    require(
        math.isfinite(result) and low <= result <= high,
        f"{context}: out of range {result}",
    )
    return result


def close(
    actual: Any, expected: float, context: str, *, tolerance: float = 1e-8
) -> None:
    value = finite(actual, context, -1.0e12, 1.0e12)
    require(
        math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance),
        f"{context}: {value} != {expected}",
    )


def average(values: Iterable[float], context: str) -> float:
    items = list(values)
    require(bool(items), f"{context}: cannot average an empty list")
    return sum(items) / len(items)


def weighted_average(values: list[float], weights: list[float], context: str) -> float:
    require(
        len(values) == len(weights) and bool(values),
        f"{context}: invalid weighted values",
    )
    denominator = sum(weights)
    require(denominator > 0.0, f"{context}: weights must have a positive sum")
    return sum(value * weight for value, weight in zip(values, weights)) / denominator


def normalized_text_hash(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_compact_rating(value: Any, context: str) -> None:
    block = mapping(value, context)
    scores = mapping(block.get("rater_scores"), f"{context}.rater_scores")
    require(bool(scores), f"{context}.rater_scores: empty")
    numbers = [
        finite(score, f"{context}.rater_scores.{name}", 0.0, 5.0)
        for name, score in scores.items()
    ]
    close(block.get("mean_score"), average(numbers, context), f"{context}.mean_score")


def validate_score_details(row: dict[str, Any], context: str) -> None:
    source = str(row["source_dataset"])
    details = mapping(row.get("score_details"), f"{context}.score_details")
    policy = details.get("label_policy")
    require(
        isinstance(policy, str) and bool(policy),
        f"{context}.score_details.label_policy missing",
    )
    primary_raters = sequence(
        details.get("primary_raters"), f"{context}.primary_raters"
    )
    official_raters = sequence(
        details.get("official_raters"), f"{context}.official_raters"
    )
    require(
        bool(official_raters) and len(set(official_raters)) == len(official_raters),
        f"{context}: invalid official raters",
    )
    top_rater_traits = mapping(
        details.get("rater_trait_scores"), f"{context}.rater_trait_scores"
    )
    traits = mapping(details.get("traits"), f"{context}.traits")
    require(set(traits) == set(TRAITS), f"{context}.traits: exactly C/O/E required")

    native_traits_by_rater: dict[str, dict[str, float]] = {}
    criterion_counts = EXPECTED_CRITERIA[source]
    for trait in TRAITS:
        trait_context = f"{context}.score_details.traits.{trait}"
        trait_block = mapping(traits[trait], trait_context)
        order = sequence(
            trait_block.get("criteria_order"), f"{trait_context}.criteria_order"
        )
        require(
            len(order) == criterion_counts[trait] and len(set(order)) == len(order),
            f"{trait_context}: expected {criterion_counts[trait]} unique criteria",
        )
        criteria = mapping(trait_block.get("criteria"), f"{trait_context}.criteria")
        require(
            set(criteria) == set(order), f"{trait_context}: criteria/order mismatch"
        )
        trait_rater_scores = mapping(
            trait_block.get("rater_scores"), f"{trait_context}.rater_scores"
        )
        require(
            set(official_raters) <= set(trait_rater_scores),
            f"{trait_context}: official rater missing",
        )

        values_by_rater: dict[str, list[float | None]] = {
            str(rater): [] for rater in trait_rater_scores
        }
        weights: list[float] = []
        has_weights: bool | None = None
        for criterion_id in order:
            criterion_context = f"{trait_context}.criteria.{criterion_id}"
            criterion = mapping(criteria[criterion_id], criterion_context)
            rater_scores = mapping(
                criterion.get("rater_scores"), f"{criterion_context}.rater_scores"
            )
            require(
                set(rater_scores) == set(trait_rater_scores),
                f"{criterion_context}: rater keys mismatch",
            )
            for rater, raw_value in rater_scores.items():
                value = (
                    None
                    if raw_value is None
                    else finite(raw_value, f"{criterion_context}.{rater}", 1.0, 5.0)
                )
                values_by_rater[str(rater)].append(value)

            official_values = [rater_scores[rater] for rater in official_raters]
            require(
                all(value is not None for value in official_values),
                f"{criterion_context}: official value missing",
            )
            expected_official = average(
                (float(value) for value in official_values), criterion_context
            )
            close(
                criterion.get("official_score"),
                expected_official,
                f"{criterion_context}.official_score",
            )

            primary_values = [rater_scores.get(rater) for rater in primary_raters]
            expected_primary = (
                None
                if any(value is None for value in primary_values)
                else average(
                    (float(value) for value in primary_values), criterion_context
                )
            )
            if expected_primary is None:
                require(
                    criterion.get("primary_rater_mean") is None,
                    f"{criterion_context}.primary_rater_mean must be null",
                )
            else:
                close(
                    criterion.get("primary_rater_mean"),
                    expected_primary,
                    f"{criterion_context}.primary_rater_mean",
                )

            weighted = "weight" in criterion
            if has_weights is None:
                has_weights = weighted
            require(
                has_weights == weighted, f"{trait_context}: partial criterion weights"
            )
            if weighted:
                weights.append(
                    finite(
                        criterion["weight"], f"{criterion_context}.weight", 0.0, 100.0
                    )
                )

            if "native_rater_scores" in criterion:
                native_scores = mapping(
                    criterion["native_rater_scores"],
                    f"{criterion_context}.native_rater_scores",
                )
                require(
                    set(native_scores) == set(rater_scores),
                    f"{criterion_context}: native rater keys mismatch",
                )
                for rater, native_raw in native_scores.items():
                    native = finite(
                        native_raw, f"{criterion_context}.native.{rater}", 0.0, 3.0
                    )
                    close(
                        rater_scores[rater],
                        1.0 + 4.0 * native / 3.0,
                        f"{criterion_context}.canonical.{rater}",
                    )

        for rater, values in values_by_rater.items():
            trait_value = trait_rater_scores[rater]
            if all(value is None for value in values):
                require(
                    trait_value is None,
                    f"{trait_context}.{rater}: null criteria require null trait",
                )
                continue
            require(
                all(value is not None for value in values),
                f"{trait_context}.{rater}: partial criterion ratings",
            )
            numbers = [float(value) for value in values]
            expected_trait = (
                weighted_average(numbers, weights, trait_context)
                if has_weights
                else average(numbers, trait_context)
            )
            close(trait_value, expected_trait, f"{trait_context}.rater_scores.{rater}")

        official_trait_values = [trait_rater_scores[rater] for rater in official_raters]
        require(
            all(value is not None for value in official_trait_values),
            f"{trait_context}: official trait missing",
        )
        expected_trait_score = average(
            (float(value) for value in official_trait_values), trait_context
        )
        close(
            trait_block.get("official_score"),
            expected_trait_score,
            f"{trait_context}.official_score",
        )
        close(row["score"][trait], expected_trait_score, f"{context}.score.{trait}")

        primary_trait_values = [
            trait_rater_scores.get(rater) for rater in primary_raters
        ]
        if any(value is None for value in primary_trait_values):
            require(
                trait_block.get("primary_rater_mean") is None,
                f"{trait_context}.primary_rater_mean must be null",
            )
        else:
            close(
                trait_block.get("primary_rater_mean"),
                average(
                    (float(value) for value in primary_trait_values), trait_context
                ),
                f"{trait_context}.primary_rater_mean",
            )
        if "all_available_rater_mean" in trait_block:
            available = [
                float(value)
                for value in trait_rater_scores.values()
                if value is not None
            ]
            close(
                trait_block["all_available_rater_mean"],
                average(available, trait_context),
                f"{trait_context}.all_available_rater_mean",
            )

        for rater, value in trait_rater_scores.items():
            top = mapping(
                top_rater_traits.get(rater), f"{context}.rater_trait_scores.{rater}"
            )
            top_value = top.get(trait)
            if value is None:
                require(
                    top_value is None,
                    f"{context}: top/trait null mismatch for {rater}.{trait}",
                )
            else:
                close(
                    top_value,
                    float(value),
                    f"{context}: top/trait mismatch for {rater}.{trait}",
                )

        if "native_rater_scores" in trait_block:
            native = mapping(
                trait_block["native_rater_scores"],
                f"{trait_context}.native_rater_scores",
            )
            require(
                set(native) == set(trait_rater_scores),
                f"{trait_context}: native trait raters mismatch",
            )
            for rater, native_raw in native.items():
                native_value = finite(
                    native_raw, f"{trait_context}.native.{rater}", 0.0, 3.0
                )
                native_traits_by_rater.setdefault(str(rater), {})[trait] = native_value
                close(
                    trait_rater_scores[rater],
                    1.0 + 4.0 * native_value / 3.0,
                    f"{trait_context}.canonical.{rater}",
                )

    for rater, trait_values in top_rater_traits.items():
        block = mapping(trait_values, f"{context}.rater_trait_scores.{rater}")
        values = [block.get(trait) for trait in TRAITS]
        if any(value is None for value in values):
            require(
                block.get("average") is None, f"{context}.{rater}.average must be null"
            )
        else:
            close(
                block.get("average"),
                average((float(value) for value in values), context),
                f"{context}.{rater}.average",
            )

    if "stored_rater_total_scores" in details:
        stored = mapping(
            details["stored_rater_total_scores"], f"{context}.stored_rater_total_scores"
        )
        for rater, stored_value in stored.items():
            trait_row = mapping(
                top_rater_traits.get(rater), f"{context}.rater_trait_scores.{rater}"
            )
            values = [trait_row.get(trait) for trait in TRAITS]
            if stored_value is None or any(value is None for value in values):
                continue
            expected_total = sum(
                float(trait_row[trait]) * criterion_counts[trait] for trait in TRAITS
            )
            close(
                stored_value,
                expected_total,
                f"{context}.stored_rater_total_scores.{rater}",
            )

    if native_traits_by_rater:
        native_top = mapping(
            details.get("native_rater_trait_scores"),
            f"{context}.native_rater_trait_scores",
        )
        trait_weights_block = mapping(
            details.get("trait_weights"), f"{context}.trait_weights"
        )
        trait_weights = [
            finite(
                trait_weights_block[trait],
                f"{context}.trait_weights.{trait}",
                0.0,
                100.0,
            )
            for trait in TRAITS
        ]
        stored_overall = mapping(
            details.get("stored_overall_native_0_30"),
            f"{context}.stored_overall_native_0_30",
        )
        for rater, native_traits in native_traits_by_rater.items():
            require(
                set(native_traits) == set(TRAITS),
                f"{context}: incomplete native traits for {rater}",
            )
            top_native = mapping(
                native_top.get(rater), f"{context}.native_rater_trait_scores.{rater}"
            )
            for trait in TRAITS:
                close(
                    top_native.get(trait),
                    native_traits[trait],
                    f"{context}.native_rater_trait_scores.{rater}.{trait}",
                )
            expected_overall = 10.0 * weighted_average(
                [native_traits[trait] for trait in TRAITS], trait_weights, context
            )
            close(
                stored_overall.get(rater),
                expected_overall,
                f"{context}.stored_overall_native_0_30.{rater}",
                tolerance=1e-3,
            )

        paragraph_rows = sequence(
            details.get("paragraph_expression_scores"),
            f"{context}.paragraph_expression_scores",
        )
        expression_order = traits["expression"]["criteria_order"]
        expression_weights = [
            float(traits["expression"]["criteria"][name]["weight"])
            for name in expression_order
        ]
        for index, paragraph in enumerate(paragraph_rows):
            paragraph_context = f"{context}.paragraph_expression_scores[{index}]"
            paragraph = mapping(paragraph, paragraph_context)
            native_criteria = mapping(
                paragraph.get("native_expression_criteria"),
                f"{paragraph_context}.native_expression_criteria",
            )
            canonical_criteria = mapping(
                paragraph.get("canonical_expression_criteria"),
                f"{paragraph_context}.canonical_expression_criteria",
            )
            totals = mapping(
                paragraph.get("native_rater_scores"),
                f"{paragraph_context}.native_rater_scores",
            )
            require(
                set(native_criteria) == set(canonical_criteria) == set(totals),
                f"{paragraph_context}: rater mismatch",
            )
            for rater in totals:
                native_values = sequence(
                    native_criteria[rater], f"{paragraph_context}.native.{rater}"
                )
                canonical_values = sequence(
                    canonical_criteria[rater], f"{paragraph_context}.canonical.{rater}"
                )
                require(
                    len(native_values)
                    == len(expression_weights)
                    == len(canonical_values),
                    f"{paragraph_context}.{rater}: criterion count mismatch",
                )
                checked_native = [
                    finite(value, f"{paragraph_context}.native.{rater}", 0.0, 3.0)
                    for value in native_values
                ]
                for native_value, canonical_value in zip(
                    checked_native, canonical_values
                ):
                    close(
                        canonical_value,
                        1.0 + 4.0 * native_value / 3.0,
                        f"{paragraph_context}.canonical.{rater}",
                    )
                expected_total = weighted_average(
                    checked_native, expression_weights, paragraph_context
                )
                close(
                    totals[rater],
                    expected_total,
                    f"{paragraph_context}.native_rater_scores.{rater}",
                    tolerance=1e-3,
                )
            close(
                paragraph.get("native_mean"),
                average((float(value) for value in totals.values()), paragraph_context),
                f"{paragraph_context}.native_mean",
                tolerance=1e-3,
            )

    auxiliary = row.get("auxiliary_scores")
    if auxiliary is not None:
        for name, value in mapping(auxiliary, f"{context}.auxiliary_scores").items():
            validate_compact_rating(value, f"{context}.auxiliary_scores.{name}")


# Canonical row validation ----------------------------------------------------
def validate_row(row: dict[str, Any], spec: dict[str, Any], context: str) -> None:
    required = {
        "schema_version",
        "id",
        "document_id",
        "prompt_num",
        "prompt",
        "essay",
        "score",
        "score_details",
        "source_dataset",
        "dataset_group",
        "source_split",
        "prompt_hash",
        "essay_hash",
    }
    require(
        required <= set(row), f"{context}: fields missing {sorted(required - set(row))}"
    )
    require(
        row["schema_version"] == SCHEMA_VERSION,
        f"{context}: row schema version mismatch",
    )
    require(
        row["source_dataset"] == spec["source"], f"{context}: source_dataset mismatch"
    )
    require(row["dataset_group"] == spec["group"], f"{context}: dataset_group mismatch")
    for name in ("id", "document_id", "prompt_num", "prompt", "essay", "source_split"):
        require(
            isinstance(row[name], str) and bool(row[name].strip()),
            f"{context}.{name}: nonempty text required",
        )
    require(
        row["prompt_hash"] == normalized_text_hash(row["prompt"]),
        f"{context}: prompt hash mismatch",
    )
    require(
        row["essay_hash"] == normalized_text_hash(row["essay"]),
        f"{context}: essay hash mismatch",
    )
    if spec["group"] == "competition":
        surfaces = mapping(row.get("essay_surfaces"), f"{context}.essay_surfaces")
        official_raw = surfaces.get("official_raw")
        require(
            isinstance(official_raw, str) and bool(official_raw.strip()),
            f"{context}.essay_surfaces.official_raw: nonempty text required",
        )
        require(
            normalized_text_hash(official_raw) == row["essay_hash"],
            f"{context}.essay_surfaces.official_raw: text identity mismatch",
        )
    scores = mapping(row["score"], f"{context}.score")
    require(set(scores) == {*TRAITS, "average"}, f"{context}.score: unexpected fields")
    trait_scores = [
        finite(scores[trait], f"{context}.score.{trait}", 1.0, 5.0) for trait in TRAITS
    ]
    close(scores["average"], average(trait_scores, context), f"{context}.score.average")
    validate_score_details(row, context)


# Official-data reconstruction -----------------------------------------------
@dataclass
class OfficialRow:
    prompt: str
    essay: str
    prompt_hash: str
    essay_hash: str
    scores: dict[str, float]


def load_official(root: Path) -> tuple[dict[str, OfficialRow], dict[str, OfficialRow]]:
    def load(pattern: str, expected: int) -> dict[str, OfficialRow]:
        paths = sorted(root.glob(pattern))
        require(len(paths) == 1, f"{root}: exactly one {pattern} file required")
        result: dict[str, OfficialRow] = {}
        with paths[0].open(encoding="utf-8-sig") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                raw = mapping(json.loads(line), f"{paths[0]}:{line_number}")
                row_id = str(raw.get("id") or "").strip()
                require(
                    row_id and row_id not in result,
                    f"{paths[0]}:{line_number}: invalid/duplicate id",
                )
                score = mapping(raw.get("score"), f"{paths[0]}:{line_number}.score")
                prompt = str(raw.get("prompt", raw.get("prompt_text", "")) or "")
                essay = str(raw.get("essay", raw.get("essay_text", "")) or "")
                result[row_id] = OfficialRow(
                    prompt=prompt,
                    essay=essay,
                    prompt_hash=normalized_text_hash(prompt),
                    essay_hash=normalized_text_hash(essay),
                    scores={
                        trait: finite(
                            score.get(trait),
                            f"{paths[0]}:{line_number}.{trait}",
                            1.0,
                            5.0,
                        )
                        for trait in TRAITS
                    },
                )
        require(
            len(result) == expected,
            f"{paths[0]}: expected {expected} rows, found {len(result)}",
        )
        return result

    return load("*train.jsonl", 2000), load("*validation.jsonl", 400)


# JSONL and manifest validation ----------------------------------------------
@dataclass
class Snapshot:
    ids: set[str]
    document_ids: set[str]
    essay_hashes: set[str]
    public_rows: dict[str, tuple[str, str, str, str, dict[str, float]]]
    source_split_counts: dict[str, int]


def validate_jsonl(
    path: Path,
    manifest_record: dict[str, Any],
    expected_rows: int,
    spec: dict[str, Any],
    public_ids: set[str],
) -> Snapshot:
    require(path.is_file(), f"missing data file: {path}")
    require(
        manifest_record.get("rows") == expected_rows,
        f"{path}: manifest row count mismatch",
    )
    require(
        manifest_record.get("bytes") == path.stat().st_size,
        f"{path}: manifest byte count mismatch",
    )
    recorded_path = manifest_record.get("path")
    require(
        isinstance(recorded_path, str) and Path(recorded_path).name == path.name,
        f"{path}: invalid manifest path",
    )

    digest = hashlib.sha256()
    ids: set[str] = set()
    document_ids: set[str] = set()
    essay_hashes: set[str] = set()
    public_rows: dict[str, tuple[str, str, str, str, dict[str, float]]] = {}
    source_split_counts: dict[str, int] = {}
    rows = 0
    with path.open("rb") as file:
        for line_number, raw_line in enumerate(file, 1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            rows += 1
            context = f"{path}:{line_number}"
            try:
                row = mapping(json.loads(raw_line.decode("utf-8")), context)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValidationError(f"{context}: invalid UTF-8 JSON") from exc
            validate_row(row, spec, context)
            row_id = row["id"]
            document_id = row["document_id"]
            require(row_id not in ids, f"{context}: duplicate id {row_id}")
            require(
                document_id not in document_ids,
                f"{context}: duplicate document_id {document_id}",
            )
            ids.add(row_id)
            document_ids.add(document_id)
            essay_hashes.add(row["essay_hash"])
            source_split = str(row["source_split"])
            source_split_counts[source_split] = (
                source_split_counts.get(source_split, 0) + 1
            )
            if row_id in public_ids:
                public_rows[row_id] = (
                    row["prompt"],
                    row["essay_surfaces"]["official_raw"],
                    row["prompt_hash"],
                    row["essay_hash"],
                    {trait: float(row["score"][trait]) for trait in TRAITS},
                )
    require(
        rows == expected_rows, f"{path}: expected {expected_rows} rows, found {rows}"
    )
    require(
        digest.hexdigest() == manifest_record.get("sha256"),
        f"{path}: manifest SHA-256 mismatch",
    )
    return Snapshot(ids, document_ids, essay_hashes, public_rows, source_split_counts)


def validate_public_rows(
    snapshot: Snapshot, expected: dict[str, OfficialRow], context: str
) -> None:
    require(
        set(snapshot.public_rows) == set(expected),
        f"{context}: official ID membership mismatch",
    )
    for row_id, official in expected.items():
        prompt, official_raw, prompt_hash, essay_hash, scores = snapshot.public_rows[
            row_id
        ]
        require(
            prompt_hash == official.prompt_hash,
            f"{context}: official prompt mismatch {row_id}",
        )
        require(
            essay_hash == official.essay_hash,
            f"{context}: official essay mismatch {row_id}",
        )
        require(
            prompt == official.prompt,
            f"{context}: official prompt is not byte-exact {row_id}",
        )
        require(
            official_raw == official.essay,
            f"{context}: official raw essay is not byte-exact {row_id}",
        )
        for trait in TRAITS:
            close(
                scores[trait],
                official.scores[trait],
                f"{context}: official score mismatch {row_id}.{trait}",
            )


def load_manifest(
    directory: Path, profile: str, spec: dict[str, Any]
) -> dict[str, Any]:
    path = directory / "manifest.json"
    require(path.is_file(), f"missing manifest: {path}")
    try:
        manifest = mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{path}: invalid JSON") from exc
    require(
        manifest.get("schema_version") == SCHEMA_VERSION,
        f"{path}: schema version mismatch",
    )
    require(manifest.get("dataset") == profile, f"{path}: dataset/profile mismatch")
    require(
        manifest.get("dataset_group") == spec["group"],
        f"{path}: dataset group mismatch",
    )
    files = mapping(manifest.get("files"), f"{path}.files")
    expected_file_names = (
        {"train", "validation"} if "validation_rows" in spec else {"train"}
    )
    require(set(files) == expected_file_names, f"{path}: file set mismatch")

    if spec["group"] == "competition":
        require(
            manifest.get("contains_validation_labels")
            is spec["contains_validation_labels"],
            f"{path}: contains_validation_labels flag mismatch",
        )
        require(
            manifest.get("safe_for_model_selection")
            is spec["safe_for_model_selection"],
            f"{path}: safe_for_model_selection flag mismatch",
        )
        audit = mapping(manifest.get("audit"), f"{path}.audit")
        for key, expected in (
            ("origin_rows", 12000),
            ("public_rows_reconstructed", 2400),
            ("public_trait_scores_exact", 2400),
            ("public_prompt_and_essay_exact", 2400),
            ("public_official_prompt_byte_exact", 2400),
            ("public_official_raw_surface_byte_exact", 2400),
            ("public_baseline_v1_input_byte_exact", 2400),
            ("validation_ids_excluded", 400),
            ("validation_text_hashes_excluded", 400),
        ):
            require(
                audit.get(key) == expected, f"{path}.audit.{key}: expected {expected}"
            )
    else:
        require(
            manifest.get("raw_rows") == spec["raw_rows"],
            f"{path}: raw row count mismatch",
        )
        require(
            manifest.get("deduplicated_exact_examples") == spec["exact_duplicates"],
            f"{path}: exact duplicate count mismatch",
        )
        expected_policy_marker = (
            "Training+Validation"
            if str(spec["source"]).startswith("aihub")
            else "NIKL"
        )
        require(
            expected_policy_marker in str(manifest.get("policy")),
            f"{path}: external split policy mismatch",
        )
        require(
            manifest.get("excluded_official_validation_essay_hashes") == 0,
            f"{path}: competition validation exclusion count mismatch",
        )
        source_splits = mapping(manifest.get("source_splits"), f"{path}.source_splits")
        require(
            source_splits == spec["source_splits"],
            f"{path}: source split counts mismatch",
        )
    return manifest


# Cross-profile leakage and split invariants ---------------------------------
def validate_all(data_root: Path, official_root: Path) -> dict[str, Snapshot]:
    require(data_root.is_dir(), f"data root missing: {data_root}")
    official_train, official_validation = load_official(official_root)
    all_official = {**official_train, **official_validation}
    public_ids = set(all_official)

    schema_path = data_root / "schema.json"
    require(schema_path.is_file(), f"missing schema document: {schema_path}")
    schema = mapping(
        json.loads(schema_path.read_text(encoding="utf-8")), str(schema_path)
    )
    require(
        schema.get("schema_version") == SCHEMA_VERSION,
        f"{schema_path}: schema version mismatch",
    )

    snapshots: dict[str, Snapshot] = {}
    for profile, spec in PROFILES.items():
        directory = data_root / spec["directory"]
        manifest = load_manifest(directory, profile, spec)
        files = manifest["files"]
        snapshots[f"{profile}:train"] = validate_jsonl(
            directory / "train.jsonl",
            mapping(files["train"], f"{directory}/manifest.files.train"),
            spec["train_rows"],
            spec,
            public_ids,
        )
        if "validation_rows" in spec:
            snapshots[f"{profile}:validation"] = validate_jsonl(
                directory / "validation.jsonl",
                mapping(files["validation"], f"{directory}/manifest.files.validation"),
                spec["validation_rows"],
                spec,
                public_ids,
            )

    clean = snapshots["competition:train"]
    leaked = snapshots["competition_leaked:train"]
    official = snapshots["official_only:train"]
    validation = snapshots["competition:validation"]
    expected_train_ids = set(official_train)
    expected_validation_ids = set(official_validation)

    for name in ("competition", "competition_leaked", "official_only"):
        current = snapshots[f"{name}:validation"]
        require(
            current.ids == expected_validation_ids,
            f"{name}: validation IDs do not match public validation",
        )
        validate_public_rows(current, official_validation, f"{name}:validation")

    require(
        official.ids == expected_train_ids,
        "official-only train IDs do not match public train",
    )
    validate_public_rows(official, official_train, "official-only:train")
    require(
        expected_train_ids <= clean.ids,
        "clean pool does not contain all public train IDs",
    )
    validate_public_rows(
        Snapshot(
            clean.ids,
            clean.document_ids,
            clean.essay_hashes,
            {
                key: value
                for key, value in clean.public_rows.items()
                if key in expected_train_ids
            },
            clean.source_split_counts,
        ),
        official_train,
        "competition:train",
    )

    require(
        clean.ids.isdisjoint(expected_validation_ids),
        "clean train contains validation IDs",
    )
    require(
        clean.essay_hashes.isdisjoint(validation.essay_hashes),
        "clean train contains validation essay hashes",
    )
    require(
        leaked.ids & expected_validation_ids == expected_validation_ids,
        "leaked train must contain exactly all 400 validation IDs",
    )
    require(
        len(leaked.essay_hashes & validation.essay_hashes) == 400,
        "leaked train must overlap 400 validation essay hashes",
    )
    validate_public_rows(leaked, all_official, "competition-leaked:train")
    require(
        leaked.ids == clean.ids | validation.ids,
        "leaked IDs must equal clean + validation IDs",
    )
    require(
        len((clean.ids | validation.ids) & public_ids) == 2400,
        "clean train + validation must contain all official 2,400 rows",
    )
    for profile, spec in PROFILES.items():
        if spec["group"] != "external":
            continue
        external = snapshots[f"{profile}:train"]
        require(
            external.source_split_counts
            == {
                split: counts["written_rows"]
                for split, counts in spec["source_splits"].items()
            },
            f"{profile}: source_split row counts mismatch",
        )
        require(
            external.essay_hashes.isdisjoint(validation.essay_hashes),
            f"{profile}: contains official competition validation essays",
        )
    return snapshots


# CLI -------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshots = validate_all(args.data_root.resolve(), args.official_root.resolve())
    print(f"[VALID] data_root={args.data_root.resolve()}")
    for profile, spec in PROFILES.items():
        train = snapshots[f"{profile}:train"]
        suffix = (
            f", validation={spec['validation_rows']:,}"
            if "validation_rows" in spec
            else ""
        )
        print(f"  {profile}: train={len(train.ids):,}{suffix}")


if __name__ == "__main__":
    main()
