"""Build self-contained detailed AES datasets under ``main_code/datasets``.

The competition data is the full NIKL grading origin pool.  The default split
keeps the public validation essays out, while the explicit leaked split puts
them back for leakage diagnostics.  AIHub and NIKL summary-evaluation corpora
are written as separate external-data profiles.

Every row keeps two views of the labels:

* ``score``: the official/representative content, organization and expression
  float targets used by the existing regression models.
* ``score_details``: every available rater, every analytic criterion, rater
  trait means, criterion means, and the policy used to obtain ``score``.

This lets the current scorer continue unchanged while later multi-task models
can use the richer supervision without reparsing the raw corpora.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence


# Paths and dataset-specific schema constants --------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PACKAGE_ROOT / "datasets"
RAW_ROOT = DATA_ROOT / "raw_dataset"
OFFICIAL_ROOT = RAW_ROOT / "official_competition"


def output_directories(root: Path) -> dict[str, Path]:
    """Map logical dataset names to their fixed profile directories."""

    return {
        "competition": root / "processed_dataset",
        "competition_leaked": root / "processed_dataset_validation_leaked",
        "official_only": root / "processed_dataset_only_official_competition",
        "aihub24_essay": root / "processed_dataset_aihub_external_에세이",
        "aihub25_descriptive": root / "processed_dataset_aihub_external_서술",
        "aihub26_essay": root / "processed_dataset_aihub_external_논술",
        "aihub27_topic": root / "processed_dataset_aihub_external_주제별",
        "nikl24_summary": root / "processed_dataset_nikl_external_summary_2024",
        "nikl25_argumentative_summary": (
            root / "processed_dataset_nikl_external_argumentative_summary_2025"
        ),
        "nikl25_cooperative_summary": (
            root / "processed_dataset_nikl_external_cooperative_summary_2025"
        ),
    }


TRAITS = ("content", "organization", "expression")
SCHEMA_VERSION = 1

AIHUB_PREFIX = {
    "aihub24_essay": "AIHUB_24.",
    "aihub25_descriptive": "AIHUB_25.",
    "aihub26_essay": "AIHUB_26.",
    "aihub27_topic": "AIHUB_27.",
}

AIHUB24_CRITERIA = {
    "content": (
        "con_clearance",
        "con_novelty",
        "con_prompt",
        "con_description",
    ),
    "organization": (
        "org_essay",
        "org_paragraph",
        "org_coherence",
        "org_quantity",
    ),
    "expression": ("exp_grammar", "exp_vocab", "exp_style"),
}
AIHUB24_CRITERION_NAMES = {
    "con_clearance": "주제의 명료성",
    "con_novelty": "사고의 창의성",
    "con_prompt": "프롬프트 독해력",
    "con_description": "설명의 구체성",
    "org_essay": "문단 간 구조의 적절성",
    "org_paragraph": "문단 내 구조의 적절성",
    "org_coherence": "구조의 일관성",
    "org_quantity": "분량의 적절성",
    "exp_grammar": "문법의 정확성",
    "exp_vocab": "단어 사용의 적절성",
    "exp_style": "문장 표현의 적절성",
}
AIHUB24_DETAIL_KEYS = {
    "content": "essay_scoreT_cont",
    "organization": "essay_scoreT_org",
    "expression": "essay_scoreT_exp",
}
AIHUB24_WEIGHT_KEYS = {
    "content": "content_weight",
    "organization": "organization_weight",
    "expression": "expression_weight",
}
AIHUB24_CATEGORY_WEIGHTS = {
    "content": "con",
    "organization": "org",
    "expression": "exp",
}

NIKL_SUMMARY_RAW_FILES = {
    "nikl24_summary": (
        "NIKL_Summary Evaluation Corpus 2024",
        "NWSC2412503174.json",
    ),
    "nikl25_argumentative_summary": (
        "NIKL Argumentative Summary Evaluation Corpus 2025",
        "NWSC2512512240(평가).json",
    ),
    "nikl25_cooperative_summary": (
        "NIKL Cooperative Dialogue Summary Evaluation Corpus 2025",
        "SDSC2512512240(평가).json",
    ),
}

NIKL_SUMMARY_CRITERIA = {
    "document": {
        "content": (
            ("description", "문제 상황"),
            ("claims", "주장"),
            ("arguments", "논거·실천 방안"),
        ),
        "organization": (("completion", "글의 긴밀성·완결성"),),
        "expression": (("accuracy", "문장·어휘의 정확성"),),
    },
    "cooperative": {
        "content": (
            ("relevance", "전체 주제와 요약의 관련성"),
            ("clarity", "소주제와 중심 생각의 명료성"),
            ("objectivity", "대화 정보의 객관성·충실성"),
        ),
        "organization": (("completion", "글의 긴밀성·완결성"),),
        "expression": (("accuracy", "문장·어휘의 정확성"),),
    },
}


# Shared normalization and canonical row writer ------------------------------
class DataError(ValueError):
    pass


def clean_text(value: Any) -> str:
    text = str(value if value is not None else "")
    text = text.replace("#@문장구분#", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise DataError("empty text")
    return text


def clean_optional_text(value: Any) -> str:
    """Normalize optional paragraph text without rejecting an empty paragraph."""

    raw = str(value if value is not None else "")
    if not raw.replace("#@문장구분#", "").strip():
        return ""
    return clean_text(raw)


def text_hash(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value: Any, name: str, low: float, high: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DataError(f"invalid {name}: {value!r}") from exc
    if not math.isfinite(result) or not low <= result <= high:
        raise DataError(f"out-of-range {name}: {result}")
    return result


def mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise DataError("cannot average an empty sequence")
    return sum(items) / len(items)


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if len(values) != len(weights) or not values or sum(weights) <= 0:
        raise DataError("invalid weighted mean inputs")
    return sum(value * weight for value, weight in zip(values, weights)) / sum(weights)


def compact_rating(
    scores: Sequence[Any], *, low: float = 1.0, high: float = 5.0
) -> dict[str, Any]:
    values = [number(value, "rating", low, high) for value in scores]
    return {
        "rater_scores": {
            f"evaluator{index + 1}": value for index, value in enumerate(values)
        },
        "mean_score": mean(values),
    }


def canonical_row(
    *,
    row_id: str,
    document_id: str,
    prompt_num: str,
    prompt: str,
    essay: str,
    scores: dict[str, float],
    score_details: dict[str, Any],
    source_dataset: str,
    dataset_group: str,
    source_split: str,
    metadata: dict[str, Any] | None = None,
    auxiliary_scores: dict[str, Any] | None = None,
    essay_surfaces: dict[str, str] | None = None,
) -> dict[str, Any]:
    clean_prompt = clean_text(prompt)
    clean_essay = clean_text(essay)
    final_scores = {trait: number(scores[trait], trait, 1.0, 5.0) for trait in TRAITS}
    final_scores["average"] = mean(final_scores.values())
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "id": str(row_id),
        "document_id": str(document_id),
        "prompt_num": str(prompt_num),
        "prompt": clean_prompt,
        "essay": clean_essay,
        "score": final_scores,
        "score_details": score_details,
        "source_dataset": source_dataset,
        "dataset_group": dataset_group,
        "source_split": source_split,
        "prompt_hash": text_hash(clean_prompt),
        "essay_hash": text_hash(clean_essay),
    }
    if metadata:
        row["metadata"] = {
            key: value for key, value in metadata.items() if value not in (None, "")
        }
    if essay_surfaces:
        preserved_surfaces: dict[str, str] = {}
        for name, value in essay_surfaces.items():
            surface = str(value)
            if not surface or text_hash(surface) != text_hash(clean_essay):
                raise DataError(
                    f"essay surface {name!r} is empty or text-inconsistent"
                )
            preserved_surfaces[str(name)] = surface
        row["essay_surfaces"] = preserved_surfaces
    if auxiliary_scores:
        row["auxiliary_scores"] = auxiliary_scores
    return row


# Official competition and NIKL conversion -----------------------------------
def load_official_rows(
    root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    train_paths = sorted(root.glob("*train.jsonl"))
    validation_paths = sorted(root.glob("*validation.jsonl"))
    if len(train_paths) != 1 or len(validation_paths) != 1:
        raise FileNotFoundError(
            f"official train/validation JSONL이 각각 하나여야 합니다: {root}"
        )

    def read(path: Path) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        with path.open(encoding="utf-8-sig") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row_id = str(row.get("id") or row.get("essay_id") or "").strip()
                if not row_id or row_id in result:
                    raise DataError(f"{path}:{line_number}: invalid/duplicate id")
                result[row_id] = row
        return result

    return read(train_paths[0]), read(validation_paths[0])


def official_scores(row: dict[str, Any]) -> dict[str, float]:
    block = row.get("score")
    if not isinstance(block, dict):
        raise DataError("official row has no score object")
    return {trait: number(block[trait], trait, 1.0, 5.0) for trait in TRAITS}


def official_essay(row: dict[str, Any]) -> str:
    return clean_text(row.get("essay", row.get("essay_text")))


def official_prompt(row: dict[str, Any]) -> str:
    return clean_text(row.get("prompt", row.get("prompt_text")))


def nikl_rater_ids(evaluation: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in evaluation.get("evaluator", []):
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if key.endswith("_ID") and value not in (None, ""):
                rater = key.removesuffix("_ID")
                if rater != "final_evaluator":
                    result[rater] = str(value)
    return result


def rater_sort_key(name: str) -> tuple[int, int | str]:
    match = re.fullmatch(r"evaluator(\d+)", name)
    if match:
        return (0, int(match.group(1)))
    if name == "re-evaluator":
        return (1, 0)
    return (2, name)


def convert_nikl(raw: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    evaluation = document.get("evaluation")
    metadata = document.get("metadata")
    if not isinstance(evaluation, dict) or not isinstance(metadata, dict):
        raise DataError("NIKL metadata/evaluation missing")
    blocks = evaluation.get("evaluation_data")
    if not isinstance(blocks, dict):
        raise DataError("NIKL evaluation_data missing")

    has_re_evaluator = any(
        key.startswith("re-evaluator_score_")
        for block in blocks.values()
        if isinstance(block, dict)
        for key in block
    )
    primary_raters = ("evaluator1", "evaluator2")
    rater_ids = nikl_rater_ids(evaluation)
    score_raters = {
        match.group(1)
        for block in blocks.values()
        if isinstance(block, dict)
        for key in block
        if (match := re.match(r"^(.+)_score_(?:total_)?(?:con|org|exp)", key))
    }
    # The 2023 calibration subset has up to 68 evaluators.  Preserve all of
    # them, while keeping the official target policy independent of that fact.
    # evaluator1/evaluator2 are always represented in the canonical schema.
    # Some adjudicated raw rows omit one of those blocks and instead contain an
    # evaluator3 block; the missing canonical rater remains explicitly null.
    all_raters = tuple(
        sorted(set(rater_ids) | score_raters | set(primary_raters), key=rater_sort_key)
    )
    official_raters = ("re-evaluator",) if has_re_evaluator else primary_raters
    trait_spec = {
        "content": (
            "eva_score_con",
            "con",
            (
                "문제 상황 제시",
                "주장 제시",
                "이유·근거의 적절성",
                "이유·근거의 충분성",
                "반론 고려·대응",
            ),
        ),
        "organization": ("eva_score_org", "org", ("글 전체 조직", "내용 단위 내 조직")),
        "expression": ("eva_score_exp", "exp", ("문장·어휘", "어문 규범·관습")),
    }

    traits: dict[str, Any] = {}
    rater_trait_scores: dict[str, dict[str, float | None]] = {
        rater: {} for rater in all_raters
    }
    final_scores: dict[str, float] = {}
    for trait, (block_name, short_name, criterion_names) in trait_spec.items():
        block = blocks.get(block_name)
        if not isinstance(block, dict):
            raise DataError(f"NIKL {block_name} missing")
        criterion_count = len(criterion_names)
        criterion_ids = [f"{trait}_{index}" for index in range(1, criterion_count + 1)]
        raw_keys = [f"{short_name}{index}" for index in range(1, criterion_count + 1)]
        by_rater: dict[str, list[float] | None] = {}
        for rater in all_raters:
            raw_values = [block.get(f"{rater}_score_{raw_key}") for raw_key in raw_keys]
            raw_total = block.get(f"{rater}_score_total_{short_name}")
            present = [value not in (None, "") for value in (*raw_values, raw_total)]
            if not any(present):
                # 19 adjudicated rows omit one original evaluator block.
                by_rater[rater] = None
                rater_trait_scores[rater][trait] = None
                continue
            if not all(present):
                raise DataError(f"NIKL partial {rater}.{trait} score block")
            values = [
                number(value, f"{rater}.{raw_key}", 1.0, 5.0)
                for value, raw_key in zip(raw_values, raw_keys)
            ]
            total = number(
                raw_total,
                f"{rater}.total_{short_name}",
                float(criterion_count),
                float(criterion_count * 5),
            )
            if not math.isclose(sum(values), total, abs_tol=1e-8):
                raise DataError(f"NIKL {trait} criterion sum != stored total")
            by_rater[rater] = values
            rater_trait_scores[rater][trait] = total / criterion_count

        criteria: dict[str, Any] = {}
        for criterion_index, (criterion_id, criterion_name) in enumerate(
            zip(criterion_ids, criterion_names)
        ):
            rater_values = {
                rater: values[criterion_index] if values is not None else None
                for rater, values in by_rater.items()
            }
            available_values = [
                value for value in rater_values.values() if value is not None
            ]
            primary_values = [rater_values[rater] for rater in primary_raters]
            official_values = [rater_values[rater] for rater in official_raters]
            if any(value is None for value in official_values):
                raise DataError(f"NIKL official {trait}.{criterion_id} score missing")
            criteria[criterion_id] = {
                "name": criterion_name,
                "raw_key": f"{short_name}{criterion_index + 1}",
                "rater_scores": rater_values,
                "selected_for_target": {
                    rater: rater in official_raters for rater in all_raters
                },
                "all_available_rater_mean": mean(available_values),
                "primary_rater_mean": (
                    None
                    if any(value is None for value in primary_values)
                    else mean(primary_values)
                ),
                "official_score": mean(
                    value for value in official_values if value is not None
                ),
            }

        official_values = [
            rater_trait_scores[rater][trait] for rater in official_raters
        ]
        primary_values = [rater_trait_scores[rater][trait] for rater in primary_raters]
        available_values = [
            scores[trait]
            for scores in rater_trait_scores.values()
            if scores[trait] is not None
        ]
        if any(value is None for value in official_values):
            raise DataError(f"NIKL official {trait} score missing")
        official_score = mean(value for value in official_values if value is not None)
        primary_mean = (
            None
            if any(value is None for value in primary_values)
            else mean(value for value in primary_values if value is not None)
        )
        if not math.isclose(
            official_score,
            mean(criteria[name]["official_score"] for name in criterion_ids),
            abs_tol=1e-8,
        ):
            raise DataError(f"NIKL {trait} official criterion/trait mean mismatch")
        traits[trait] = {
            "criteria_order": criterion_ids,
            "criteria": criteria,
            "rater_scores": {
                rater: rater_trait_scores[rater][trait] for rater in all_raters
            },
            "all_available_rater_mean": mean(available_values),
            "primary_rater_mean": primary_mean,
            "official_score": official_score,
        }
        final_scores[trait] = official_score

    for scores in rater_trait_scores.values():
        trait_values = [scores[trait] for trait in TRAITS]
        scores["average"] = (
            None
            if any(value is None for value in trait_values)
            else mean(value for value in trait_values if value is not None)
        )
    stored_total_scores = {
        rater: (
            number(blocks[f"{rater}_total_score"], f"{rater}.total", 9.0, 45.0)
            if blocks.get(f"{rater}_total_score") not in (None, "")
            else None
        )
        for rater in all_raters
    }
    score_details = {
        "label_policy": (
            "official_re_evaluator" if has_re_evaluator else "mean_of_two_raters"
        ),
        "primary_raters": list(primary_raters),
        "official_raters": list(official_raters),
        "rater_ids": rater_ids,
        "rater_selected_for_target": {
            rater: rater in official_raters for rater in all_raters
        },
        "stored_rater_total_scores": stored_total_scores,
        "rater_trait_scores": rater_trait_scores,
        "traits": traits,
    }
    final_evaluator_id = next(
        (
            str(item["final_evaluator_ID"])
            for item in evaluation.get("evaluator", [])
            if isinstance(item, dict)
            and item.get("final_evaluator_ID") not in (None, "")
        ),
        None,
    )
    if final_evaluator_id is not None:
        # This is provenance only; it must never override the target policy.
        score_details["final_evaluator_id"] = final_evaluator_id

    prompt_block = metadata.get("prompt")
    paragraphs = document.get("paragraph")
    if not isinstance(prompt_block, dict) or not isinstance(paragraphs, list):
        raise DataError("NIKL prompt/paragraph missing")
    paragraph_forms = [
        str(paragraph.get("form", ""))
        for paragraph in paragraphs
        if isinstance(paragraph, dict)
    ]
    essay = "\n\n".join(paragraph_forms)
    # The official competition JSONL is not whitespace-normalized: for all
    # 2,400 public rows its essay is byte-identical to this direct concatenation.
    # Preserve it separately because canonical cleaning deliberately normalizes
    # paragraph layout, while legacy ``essay_surface=flat`` collapses every gap.
    official_raw_essay = "".join(paragraph_forms)
    author = metadata.get("author") if isinstance(metadata.get("author"), dict) else {}
    written = (
        metadata.get("written_stat")
        if isinstance(metadata.get("written_stat"), dict)
        else {}
    )
    row_id = str(raw.get("id") or document.get("id") or "").strip()
    document_id = str(document.get("id") or row_id)
    if not row_id:
        raise DataError("NIKL id missing")
    return canonical_row(
        row_id=row_id,
        document_id=document_id,
        prompt_num=str(prompt_block.get("prompt_num", "")),
        prompt=str(prompt_block.get("prompt_con", "")),
        essay=essay,
        scores=final_scores,
        score_details=score_details,
        source_dataset="nikl_competition",
        dataset_group="competition",
        source_split="origin_pool",
        essay_surfaces={"official_raw": official_raw_essay},
        metadata={
            "year": (
                raw.get("metadata", {}).get("year")
                if isinstance(raw.get("metadata"), dict)
                else None
            ),
            "grade": author.get("author_grade"),
            "age": author.get("author_age"),
            "sex": author.get("author_sex"),
            "written_length": written.get("written_length"),
            "paragraph_count": written.get("paragraph_num"),
            "sentence_count": written.get("sentence_num"),
        },
    )


def iter_nikl(root: Path) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    directories = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("NIKL_GRADING WRITING DATA")
    )
    if len(directories) != 3:
        raise FileNotFoundError(f"NIKL grading directories must be 3: {directories}")
    for path in sorted(
        file for directory in directories for file in directory.glob("*.json")
    ):
        with path.open(encoding="utf-8-sig") as file:
            raw = json.load(file)
        documents = raw.get("document")
        if not isinstance(documents, list) or not documents:
            raise DataError(f"{path}: document missing")
        for document in documents:
            if not isinstance(document, dict):
                raise DataError(f"{path}: invalid document")
            yield raw, document


# AIHub analytic-score conversion --------------------------------------------
def _rubric_info(raw: dict[str, Any], criterion: str) -> tuple[str, str | None]:
    rubric = raw.get("rubric")
    analytic = rubric.get("analytic") if isinstance(rubric, dict) else None
    block = analytic.get(criterion) if isinstance(analytic, dict) else None
    if not isinstance(block, dict):
        return criterion, None
    return str(block.get("name") or criterion), (
        str(block["rubric_key"]) if block.get("rubric_key") not in (None, "") else None
    )


def convert_aihub_common(raw: dict[str, Any], alias: str) -> dict[str, Any]:
    question = raw.get("essay_question")
    answer = raw.get("essay_answer")
    score_root = raw.get("score")
    personal = score_root.get("personal") if isinstance(score_root, dict) else None
    analytic = personal.get("analytic") if isinstance(personal, dict) else None
    if (
        not isinstance(question, dict)
        or not isinstance(answer, dict)
        or not isinstance(analytic, dict)
    ):
        raise DataError(f"{alias}: question/answer/analytic missing")

    rater_names = ("evaluator1", "evaluator2")
    traits: dict[str, Any] = {}
    rater_trait_scores = {rater: {} for rater in rater_names}
    final_scores: dict[str, float] = {}
    trait_criteria = {
        "content": ("content_1", "content_2", "content_3"),
        "organization": ("organization_1", "organization_2"),
        "expression": ("expression_1", "expression_2"),
    }
    for trait, criterion_ids in trait_criteria.items():
        criteria: dict[str, Any] = {}
        for criterion_id in criterion_ids:
            block = analytic.get(criterion_id)
            values = block.get("score") if isinstance(block, dict) else None
            if not isinstance(values, list) or len(values) != 2:
                raise DataError(f"{alias}: {criterion_id} needs two ratings")
            rater_values = {
                rater: number(value, f"{criterion_id}.{rater}", 1.0, 5.0)
                for rater, value in zip(rater_names, values)
            }
            name, rubric_key = _rubric_info(raw, criterion_id)
            criterion_row: dict[str, Any] = {
                "name": name,
                "rater_scores": rater_values,
                "primary_rater_mean": mean(rater_values.values()),
                "official_score": mean(rater_values.values()),
            }
            if rubric_key:
                criterion_row["rubric_key"] = rubric_key
            criteria[criterion_id] = criterion_row
        for rater in rater_names:
            rater_trait_scores[rater][trait] = mean(
                criteria[criterion]["rater_scores"][rater]
                for criterion in criterion_ids
            )
        official_score = mean(rater_trait_scores[rater][trait] for rater in rater_names)
        traits[trait] = {
            "criteria_order": list(criterion_ids),
            "criteria": criteria,
            "rater_scores": {
                rater: rater_trait_scores[rater][trait] for rater in rater_names
            },
            "primary_rater_mean": official_score,
            "official_score": official_score,
        }
        final_scores[trait] = official_score
    for scores in rater_trait_scores.values():
        scores["average"] = mean(scores.values())

    expert = raw.get("expert")
    expert_score = expert.get("score") if isinstance(expert, dict) else None
    rater_ids: dict[str, str] = {}
    if isinstance(expert_score, dict):
        for index, key in enumerate(("votes_1", "votes_2"), 1):
            item = expert_score.get(key)
            if isinstance(item, dict) and item.get("id") not in (None, ""):
                rater_ids[f"evaluator{index}"] = str(item["id"])
    score_details = {
        "label_policy": "mean_of_two_raters",
        "primary_raters": list(rater_names),
        "official_raters": list(rater_names),
        "rater_ids": rater_ids,
        "rater_trait_scores": rater_trait_scores,
        "traits": traits,
    }

    auxiliary: dict[str, Any] = {}
    task = analytic.get("task_1")
    if isinstance(task, dict) and isinstance(task.get("score"), list):
        auxiliary["task"] = compact_rating(task["score"])
        name, rubric_key = _rubric_info(raw, "task_1")
        auxiliary["task"]["name"] = name
        if rubric_key:
            auxiliary["task"]["rubric_key"] = rubric_key
    holistic = personal.get("holistic") if isinstance(personal, dict) else None
    if isinstance(holistic, dict) and isinstance(holistic.get("score"), list):
        auxiliary["holistic"] = compact_rating(holistic["score"], low=1.0, high=4.0)
        auxiliary["holistic"]["scale"] = {"minimum": 1.0, "maximum": 4.0}

    question_id = str(question.get("id") or "")
    prompt = str(question.get("prompt") or "")
    prompt_num = f"{alias}:{question_id}:{text_hash(clean_text(prompt))[:16]}"
    answer_id = str(answer.get("id") or "")
    return canonical_row(
        row_id=f"{alias}:{answer_id}",
        document_id=f"{alias}:{answer_id}",
        prompt_num=prompt_num,
        prompt=prompt,
        essay=str(answer.get("text") or ""),
        scores=final_scores,
        score_details=score_details,
        source_dataset=alias,
        dataset_group="external",
        source_split=aihub_source_split(raw),
        metadata={
            "source_prompt_id": question_id,
            "grade": question.get("grade"),
            "difficulty": question.get("level"),
            "subject": question.get("subject"),
            "topic": question.get("topic"),
            "purpose": question.get("purpose"),
            "essay_type": question.get("type"),
            "gender": answer.get("gender"),
            "region": answer.get("region"),
        },
        auxiliary_scores=auxiliary,
    )


def native_to_canonical(value: float) -> float:
    return 1.0 + (4.0 / 3.0) * value


def convert_aihub24(raw: dict[str, Any]) -> dict[str, Any]:
    rubric = raw.get("rubric")
    score_root = raw.get("score")
    details = (
        score_root.get("essay_scoreT_detail") if isinstance(score_root, dict) else None
    )
    info = raw.get("info")
    if (
        not isinstance(rubric, dict)
        or not isinstance(details, dict)
        or not isinstance(info, dict)
    ):
        raise DataError("aihub24 rubric/score/info missing")

    rater_names = ("evaluator1", "evaluator2", "evaluator3")
    traits: dict[str, Any] = {}
    rater_trait_scores = {rater: {} for rater in rater_names}
    native_trait_scores = {rater: {} for rater in rater_names}
    final_scores: dict[str, float] = {}
    for trait in TRAITS:
        criterion_ids = AIHUB24_CRITERIA[trait]
        detail_rows = details.get(AIHUB24_DETAIL_KEYS[trait])
        weight_block = rubric.get(AIHUB24_WEIGHT_KEYS[trait])
        if (
            not isinstance(detail_rows, list)
            or len(detail_rows) != 3
            or not isinstance(weight_block, dict)
        ):
            raise DataError(f"aihub24 invalid {trait} detail/weight")
        weights = [
            number(weight_block.get(criterion), f"weight.{criterion}", 0.0, 100.0)
            for criterion in criterion_ids
        ]
        if sum(weights) <= 0:
            raise DataError(f"aihub24 all-zero {trait} weights")
        criteria: dict[str, Any] = {}
        for criterion_index, criterion_id in enumerate(criterion_ids):
            native_values = {
                rater: number(
                    detail_rows[index][criterion_index], criterion_id, 0.0, 3.0
                )
                for index, rater in enumerate(rater_names)
            }
            canonical_values = {
                rater: native_to_canonical(value)
                for rater, value in native_values.items()
            }
            criteria[criterion_id] = {
                "name": AIHUB24_CRITERION_NAMES[criterion_id],
                "weight": weights[criterion_index],
                "native_rater_scores": native_values,
                "rater_scores": canonical_values,
                "primary_rater_mean": mean(canonical_values.values()),
                "official_score": mean(canonical_values.values()),
            }
        for rater_index, rater in enumerate(rater_names):
            native = weighted_mean(
                [
                    number(
                        detail_rows[rater_index][index], criterion_ids[index], 0.0, 3.0
                    )
                    for index in range(len(criterion_ids))
                ],
                weights,
            )
            native_trait_scores[rater][trait] = native
            rater_trait_scores[rater][trait] = native_to_canonical(native)
        official_score = mean(rater_trait_scores[rater][trait] for rater in rater_names)
        traits[trait] = {
            "criteria_order": list(criterion_ids),
            "criteria": criteria,
            "rater_scores": {
                rater: rater_trait_scores[rater][trait] for rater in rater_names
            },
            "native_rater_scores": {
                rater: native_trait_scores[rater][trait] for rater in rater_names
            },
            "primary_rater_mean": official_score,
            "official_score": official_score,
        }
        final_scores[trait] = official_score

    category_weights = {
        trait: number(
            rubric[AIHUB24_WEIGHT_KEYS[trait]].get(AIHUB24_CATEGORY_WEIGHTS[trait]),
            f"category_weight.{trait}",
            0.0,
            100.0,
        )
        for trait in TRAITS
    }
    stored_totals = (
        score_root.get("essay_scoreT") if isinstance(score_root, dict) else None
    )
    if not isinstance(stored_totals, list) or len(stored_totals) != 3:
        raise DataError("aihub24 essay_scoreT invalid")
    for index, rater in enumerate(rater_names):
        reconstructed = 10.0 * weighted_mean(
            [native_trait_scores[rater][trait] for trait in TRAITS],
            [category_weights[trait] for trait in TRAITS],
        )
        stored = number(stored_totals[index], "essay_scoreT", 0.0, 30.0)
        if not math.isclose(reconstructed, stored, abs_tol=1e-3):
            raise DataError(
                f"aihub24 total reconstruction mismatch: {reconstructed} != {stored}"
            )

    for scores in rater_trait_scores.values():
        scores["average"] = mean(scores.values())
    score_details = {
        "label_policy": "weighted_trait_mean_of_three_raters",
        "primary_raters": list(rater_names),
        "official_raters": list(rater_names),
        "native_scale": {"minimum": 0.0, "maximum": 3.0},
        "canonical_scale": {"minimum": 1.0, "maximum": 5.0},
        "trait_weights": category_weights,
        "native_rater_trait_scores": native_trait_scores,
        "stored_overall_native_0_30": {
            rater: number(stored_totals[index], "essay_scoreT", 0.0, 30.0)
            for index, rater in enumerate(rater_names)
        },
        "rater_trait_scores": rater_trait_scores,
        "traits": traits,
    }

    paragraphs = raw.get("paragraph")
    if not isinstance(paragraphs, list):
        raise DataError("aihub24 paragraphs missing")
    paragraph_scores = (
        score_root.get("paragraph_score") if isinstance(score_root, dict) else None
    )
    if not isinstance(paragraph_scores, list) or len(paragraph_scores) != len(
        paragraphs
    ):
        raise DataError("aihub24 paragraph text/score count mismatch")
    paragraph_details: list[dict[str, Any]] = []
    for paragraph, paragraph_score in zip(paragraphs, paragraph_scores):
        if not isinstance(paragraph, dict) or not isinstance(paragraph_score, dict):
            raise DataError("aihub24 invalid paragraph block")
        raw_detail = paragraph_score.get("paragraph_scoreT_detail", {}).get(
            "paragraph_scoreT_exp"
        )
        if not isinstance(raw_detail, list) or len(raw_detail) != 3:
            raise DataError("aihub24 paragraph expression detail missing")
        native_detail = {
            rater: [number(value, "paragraph.expression", 0.0, 3.0) for value in values]
            for rater, values in zip(rater_names, raw_detail)
        }
        canonical_detail = {
            rater: [native_to_canonical(value) for value in values]
            for rater, values in native_detail.items()
        }
        native_totals = paragraph_score.get("paragraph_scoreT")
        if not isinstance(native_totals, list) or len(native_totals) != 3:
            raise DataError("aihub24 paragraph total missing")
        paragraph_details.append(
            {
                "paragraph_id": str(paragraph.get("paragraph_id") or ""),
                "text": clean_optional_text(paragraph.get("paragraph_txt")),
                "native_expression_criteria": native_detail,
                "canonical_expression_criteria": canonical_detail,
                "native_rater_scores": {
                    rater: number(value, "paragraph.total", 0.0, 3.0)
                    for rater, value in zip(rater_names, native_totals)
                },
                "native_mean": number(
                    paragraph_score.get("paragraph_scoreT_avg"),
                    "paragraph.mean",
                    0.0,
                    3.0,
                ),
            }
        )
    score_details["paragraph_expression_scores"] = paragraph_details
    essay = "\n\n".join(
        str(paragraph.get("paragraph_txt", ""))
        for paragraph in paragraphs
        if isinstance(paragraph, dict)
    )
    student = raw.get("student") if isinstance(raw.get("student"), dict) else {}
    row_id = str(info.get("essay_id") or "")
    return canonical_row(
        row_id=f"aihub24_essay:{row_id}",
        document_id=f"aihub24_essay:{row_id}",
        prompt_num=f"aihub24_essay:{text_hash(clean_text(info.get('essay_prompt')))[:16]}",
        prompt=str(info.get("essay_prompt") or ""),
        essay=essay,
        scores=final_scores,
        score_details=score_details,
        source_dataset="aihub24_essay",
        dataset_group="external",
        source_split=aihub_source_split(raw),
        metadata={
            "grade": student.get("student_grade"),
            "difficulty": info.get("essay_level"),
            "essay_type": info.get("essay_type") or rubric.get("essay_type"),
            "subject": info.get("essay_main_subject")
            or rubric.get("essay_main_subject"),
        },
    )


# NIKL summary-evaluation conversion ----------------------------------------
def summary_1to7_to_canonical(value: Any, name: str) -> float:
    """Map the summary corpora's human 1--7 scale linearly onto 1--5."""

    native = number(value, name, 1.0, 7.0)
    return 1.0 + (native - 1.0) * (4.0 / 6.0)


def _load_nikl_summary_document(root: Path, alias: str) -> list[dict[str, Any]]:
    directory_name, file_name = NIKL_SUMMARY_RAW_FILES[alias]
    path = root / directory_name / file_name
    if not path.is_file():
        raise FileNotFoundError(f"{alias}: raw JSON missing: {path}")
    with path.open(encoding="utf-8-sig") as file:
        payload = json.load(file)
    documents = payload.get("document") if isinstance(payload, dict) else None
    if not isinstance(documents, list):
        raise DataError(f"{alias}: top-level document array missing")
    return documents


def iter_nikl_summary(root: Path, alias: str) -> Iterator[dict[str, Any]]:
    """Yield one scored candidate summary at a time.

    The 2024 delivery differs from its PDF: SC1/SC2 are arrays and some rows
    contain a second, feedback-derived rewrite.  The default external profile
    deliberately keeps only index 0.  This avoids silently multiplying nearly
    identical candidates and leaves the revised variants for an explicit later
    ablation.  One source document's index-0 candidates omit ``evaluation``;
    each has a later object with the exact same visible summary and a valid
    evaluation, so only those two labels are recovered without changing the
    model-visible input.
    """

    for document in _load_nikl_summary_document(root, alias):
        if not isinstance(document, dict):
            raise DataError(f"{alias}: invalid document block")
        for slot in ("SC1", "SC2"):
            raw_candidate = document.get(slot)
            candidates = raw_candidate if isinstance(raw_candidate, list) else [raw_candidate]
            if not candidates or not isinstance(candidates[0], dict):
                raise DataError(f"{alias}: {slot} candidate missing")
            candidate = candidates[0]
            selected_variant_index = 0
            if not isinstance(candidate.get("evaluation"), dict):
                original_summary = clean_text(
                    candidate.get("main_summary", candidate.get("summary"))
                )
                recovered = next(
                    (
                        (index, replacement)
                        for index, replacement in enumerate(candidates[1:], 1)
                        if isinstance(replacement, dict)
                        and isinstance(replacement.get("evaluation"), dict)
                        and clean_text(
                            replacement.get(
                                "main_summary", replacement.get("summary")
                            )
                        )
                        == original_summary
                    ),
                    None,
                )
                if recovered is None:
                    raise DataError(f"{alias}: labelled {slot} evaluation missing")
                selected_variant_index, candidate = recovered
            yield {
                "alias": alias,
                "document": document,
                "candidate": candidate,
                "candidate_slot": slot,
                "available_variant_count": len(candidates),
                "selected_variant_index": selected_variant_index,
            }


def _dialogue_source(document: dict[str, Any]) -> str:
    utterances = document.get("utterance")
    if not isinstance(utterances, list) or not utterances:
        raise DataError("cooperative summary utterances missing")
    turns: list[str] = []
    current_speaker = ""
    fragments: list[str] = []
    for utterance in utterances:
        if not isinstance(utterance, dict):
            continue
        speaker = str(
            utterance.get("sc_speaker_id") or utterance.get("speaker_id") or "화자"
        ).strip()
        form = str(utterance.get("form") or "").strip()
        if not form:
            continue
        if fragments and speaker != current_speaker:
            turns.append(f"{current_speaker}: {' '.join(fragments)}")
            fragments = []
        current_speaker = speaker
        fragments.append(form)
    if fragments:
        turns.append(f"{current_speaker}: {' '.join(fragments)}")
    if not turns:
        raise DataError("cooperative summary dialogue is empty")
    return "\n".join(turns)


def _nikl_summary_prompt(alias: str, document: dict[str, Any]) -> str:
    metadata = document.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    if alias == "nikl25_cooperative_summary":
        topic = clean_optional_text(metadata.get("topic"))
        topic_block = f"[대화 주제]\n{topic}\n\n" if topic else ""
        return f"{topic_block}[요약 평가 원문 대화]\n{_dialogue_source(document)}"

    blocks = document.get("paragraph") or document.get("sentence")
    if not isinstance(blocks, list) or not blocks:
        raise DataError(f"{alias}: source paragraph/sentence missing")
    source = "\n\n".join(
        str(block.get("form") or "") for block in blocks if isinstance(block, dict)
    )
    source_kind = "논증적 원문" if alias == "nikl25_argumentative_summary" else "원문"
    return f"[요약 평가 {source_kind}]\n{source}"


def _summary_comment(block: dict[str, Any], criterion_key: str) -> str | None:
    comments = block.get("comments")
    if isinstance(comments, dict):
        value = comments.get(criterion_key)
    else:
        value = block.get("comment")
    if value in (None, ""):
        return None
    return str(value)


def convert_nikl_summary(raw: dict[str, Any]) -> dict[str, Any]:
    alias = str(raw["alias"])
    document = raw["document"]
    candidate = raw["candidate"]
    slot = str(raw["candidate_slot"])
    evaluation = candidate.get("evaluation")
    evaluators = evaluation.get("evaluators") if isinstance(evaluation, dict) else None
    if not isinstance(evaluators, list) or len(evaluators) != 3:
        raise DataError(f"{alias}: exactly three evaluators are required")
    rater_names = [str(rater.get("id") or "").strip() for rater in evaluators]
    if any(not name for name in rater_names) or len(set(rater_names)) != 3:
        raise DataError(f"{alias}: invalid evaluator IDs")

    family = "cooperative" if alias == "nikl25_cooperative_summary" else "document"
    trait_spec = NIKL_SUMMARY_CRITERIA[family]
    traits: dict[str, Any] = {}
    canonical_rater_traits: dict[str, dict[str, float]] = {
        name: {} for name in rater_names
    }
    native_rater_traits: dict[str, dict[str, float]] = {
        name: {} for name in rater_names
    }
    final_scores: dict[str, float] = {}

    for trait, criterion_spec in trait_spec.items():
        criterion_ids = [
            f"{trait}_{index}" for index in range(1, len(criterion_spec) + 1)
        ]
        canonical_by_rater: dict[str, list[float]] = {
            name: [] for name in rater_names
        }
        native_by_rater: dict[str, list[float]] = {name: [] for name in rater_names}
        criteria: dict[str, Any] = {}
        for criterion_id, (raw_key, korean_name) in zip(
            criterion_ids, criterion_spec, strict=True
        ):
            native_scores: dict[str, float] = {}
            canonical_scores: dict[str, float] = {}
            rater_comments: dict[str, str] = {}
            for rater_name, evaluator in zip(rater_names, evaluators, strict=True):
                trait_block = evaluator.get(trait)
                if not isinstance(trait_block, dict):
                    raise DataError(f"{alias}: {rater_name}.{trait} block missing")
                native = number(
                    trait_block.get(raw_key),
                    f"{alias}.{rater_name}.{trait}.{raw_key}",
                    1.0,
                    7.0,
                )
                canonical = summary_1to7_to_canonical(
                    native, f"{alias}.{rater_name}.{trait}.{raw_key}"
                )
                native_scores[rater_name] = native
                canonical_scores[rater_name] = canonical
                native_by_rater[rater_name].append(native)
                canonical_by_rater[rater_name].append(canonical)
                comment = _summary_comment(trait_block, raw_key)
                if comment is not None:
                    rater_comments[rater_name] = comment
            criterion = {
                "name": korean_name,
                "raw_key": raw_key,
                "source_rater_scores_1to7": native_scores,
                "rater_scores": canonical_scores,
                "primary_rater_mean": mean(canonical_scores.values()),
                "official_score": mean(canonical_scores.values()),
            }
            if rater_comments:
                criterion["rater_comments"] = rater_comments
            criteria[criterion_id] = criterion

        trait_native_scores = {
            name: mean(native_by_rater[name]) for name in rater_names
        }
        trait_canonical_scores = {
            name: mean(canonical_by_rater[name]) for name in rater_names
        }
        for name in rater_names:
            native_rater_traits[name][trait] = trait_native_scores[name]
            canonical_rater_traits[name][trait] = trait_canonical_scores[name]
        official_score = mean(trait_canonical_scores.values())
        traits[trait] = {
            "criteria_order": criterion_ids,
            "criteria": criteria,
            "source_rater_scores_1to7": trait_native_scores,
            "rater_scores": trait_canonical_scores,
            "primary_rater_mean": official_score,
            "official_score": official_score,
        }
        final_scores[trait] = official_score

    for name in rater_names:
        canonical_rater_traits[name]["average"] = mean(
            canonical_rater_traits[name][trait] for trait in TRAITS
        )
        native_rater_traits[name]["average"] = mean(
            native_rater_traits[name][trait] for trait in TRAITS
        )
    all_native_criterion_scores = [
        float(evaluator[trait][raw_key])
        for evaluator in evaluators
        for trait, criterion_spec in trait_spec.items()
        for raw_key, _ in criterion_spec
    ]
    reconstructed_average = (
        mean(all_native_criterion_scores) - 1.0
    ) * (100.0 / 6.0)
    stored_average = number(
        evaluation.get("average_score"),
        f"{alias}.average_score",
        0.0,
        100.5,
    )
    score_details = {
        "label_policy": "mean_of_three_raters_mapped_1to7_to_1to5",
        "primary_raters": rater_names,
        "official_raters": rater_names,
        "source_scale": {"minimum": 1.0, "maximum": 7.0},
        "canonical_scale": {"minimum": 1.0, "maximum": 5.0},
        "source_rater_trait_scores_1to7": native_rater_traits,
        "rater_trait_scores": canonical_rater_traits,
        "traits": traits,
        "stored_average_score_0_100": stored_average,
        "reconstructed_equal_criterion_average_score_0_100": reconstructed_average,
    }

    document_id = str(document.get("id") or "").strip()
    if not document_id:
        raise DataError(f"{alias}: document id missing")
    summary = candidate.get("main_summary", candidate.get("summary"))
    row_id = f"{alias}:{document_id}:{slot}:0"
    metadata = document.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return canonical_row(
        row_id=row_id,
        # Two candidate summaries share one source document.  The canonical
        # row contract requires a unique document_id, so source provenance is
        # retained separately in metadata.
        document_id=row_id,
        prompt_num=f"{alias}:{document_id}",
        prompt=_nikl_summary_prompt(alias, document),
        essay=str(summary or ""),
        scores=final_scores,
        score_details=score_details,
        source_dataset=alias,
        dataset_group="external",
        source_split=alias,
        metadata={
            "source_document_id": document_id,
            "candidate_slot": slot,
            "available_variant_count": raw["available_variant_count"],
            "selected_variant_index": raw["selected_variant_index"],
            "title": metadata.get("title"),
            "topic": metadata.get("topic"),
            "date": metadata.get("date"),
        },
    )


# Raw AIHub discovery ---------------------------------------------------------
def find_unique_directory(root: Path, prefix: str) -> Path:
    matches = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{prefix!r} directory must be unique under {root}: {matches}"
        )
    return matches[0]


def aihub_source_split(raw: dict[str, Any]) -> str:
    """Return the provider split recorded by an AIHub raw iterator."""

    source_split = raw.get("__canonical_source_split")
    if source_split not in {"aihub_training", "aihub_validation"}:
        raise DataError(f"invalid/missing AIHub source split: {source_split!r}")
    return str(source_split)


def mark_aihub_split(raw: dict[str, Any], source_split: str) -> dict[str, Any]:
    """Attach private provenance without changing the original label schema."""

    if source_split not in {"aihub_training", "aihub_validation"}:
        raise DataError(f"invalid AIHub source split: {source_split}")
    raw["__canonical_source_split"] = source_split
    return raw


def iter_aihub_common(root: Path, alias: str) -> Iterator[dict[str, Any]]:
    dataset_root = find_unique_directory(root, AIHUB_PREFIX[alias])
    for provider_split, source_split in (
        ("Training", "aihub_training"),
        ("Validation", "aihub_validation"),
    ):
        label_roots = [
            path
            for path in dataset_root.rglob("02.라벨링데이터")
            if provider_split in path.parts
        ]
        archives = sorted(
            path for label_root in label_roots for path in label_root.glob("*.zip")
        )
        direct_json = sorted(
            path for label_root in label_roots for path in label_root.rglob("*.json")
        )
        if not archives and not direct_json:
            raise FileNotFoundError(
                f"{alias}: {provider_split} label JSON/archives missing"
            )
        # Some deliveries contain both extracted JSON and their original ZIP.
        # Read exactly one representation so each annotation is seen once.
        if direct_json:
            for path in direct_json:
                with path.open(encoding="utf-8-sig") as file:
                    yield mark_aihub_split(json.load(file), source_split)
        else:
            for archive_path in archives:
                with zipfile.ZipFile(archive_path) as archive:
                    for member in sorted(
                        name
                        for name in archive.namelist()
                        if name.lower().endswith(".json")
                    ):
                        raw = json.loads(archive.read(member).decode("utf-8-sig"))
                        yield mark_aihub_split(raw, source_split)


def iter_aihub24(root: Path) -> Iterator[dict[str, Any]]:
    dataset_root = find_unique_directory(root, AIHUB_PREFIX["aihub24_essay"])
    for provider_split, source_split in (
        ("1.Training", "aihub_training"),
        ("2.Validation", "aihub_validation"),
    ):
        label_roots = [
            path
            for path in dataset_root.rglob("라벨링데이터")
            if provider_split in path.parts
        ]
        paths = sorted(
            path for label_root in label_roots for path in label_root.rglob("*.json")
        )
        archives = sorted(
            path for label_root in label_roots for path in label_root.glob("*.zip")
        )
        if not paths and not archives:
            raise FileNotFoundError(
                f"aihub24 {provider_split} label JSON/archives missing"
            )
        if paths:
            for path in paths:
                with path.open(encoding="utf-8-sig") as file:
                    yield mark_aihub_split(json.load(file), source_split)
        else:
            for archive_path in archives:
                with zipfile.ZipFile(archive_path) as archive:
                    for member in sorted(
                        name
                        for name in archive.namelist()
                        if name.lower().endswith(".json")
                    ):
                        raw = json.loads(archive.read(member).decode("utf-8-sig"))
                        yield mark_aihub_split(raw, source_split)


# Atomic output and manifests -------------------------------------------------
def prepare_target(path: Path, *, force: bool) -> tuple[Path, Path]:
    path.mkdir(parents=True, exist_ok=True)
    train_path = path / "train.jsonl"
    temporary = path / "train.jsonl.tmp"
    if train_path.exists() and not force:
        raise FileExistsError(f"output exists; pass --force: {train_path}")
    if temporary.exists():
        temporary.unlink()
    return train_path, temporary


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            count += 1
    return count


def write_manifest(directory: Path, payload: dict[str, Any]) -> None:
    target = directory / "manifest.json"
    temporary = directory / "manifest.json.tmp"
    document = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(target)


def file_record(path: Path, rows: int) -> dict[str, Any]:
    return {
        # Relative paths remain correct after a validated staging build is
        # atomically promoted into the fixed package directory.
        "path": path.name,
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def duplicate_input_audit(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Count duplicate inputs without deleting differently annotated raw rows."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["prompt_hash"], row["essay_hash"]), []).append(row)
    duplicate_groups = [group for group in groups.values() if len(group) > 1]
    conflict_groups = [
        group
        for group in duplicate_groups
        if len({tuple(row["score"][trait] for trait in TRAITS) for row in group}) > 1
    ]
    return {
        "duplicate_input_groups": len(duplicate_groups),
        "duplicate_input_extra_rows": sum(len(group) - 1 for group in duplicate_groups),
        "duplicate_input_conflict_groups": len(conflict_groups),
        "conflict_ids": [[row["id"] for row in group] for group in conflict_groups],
    }


# Dataset builders ------------------------------------------------------------
def build_competition(
    raw_root: Path,
    official_root: Path,
    outputs: dict[str, Path],
    *,
    force: bool,
) -> dict[str, Any]:
    official_train, official_validation = load_official_rows(official_root)
    train_ids = set(official_train)
    validation_ids = set(official_validation)
    validation_hashes = {
        text_hash(official_essay(row)) for row in official_validation.values()
    }
    all_rows: list[dict[str, Any]] = []
    matched_ids: set[str] = set()
    score_mismatches: list[str] = []
    text_mismatches: list[str] = []
    prompt_surface_mismatches: list[str] = []
    raw_surface_mismatches: list[str] = []

    for raw, document in iter_nikl(raw_root):
        row = convert_nikl(raw, document)
        row_id = str(row["id"])
        if row_id in official_train or row_id in official_validation:
            official = official_train.get(row_id) or official_validation[row_id]
            matched_ids.add(row_id)
            if any(
                not math.isclose(
                    row["score"][trait], official_scores(official)[trait], abs_tol=1e-8
                )
                for trait in TRAITS
            ):
                score_mismatches.append(row_id)
            if row["essay_hash"] != text_hash(official_essay(official)) or row[
                "prompt_hash"
            ] != text_hash(official_prompt(official)):
                text_mismatches.append(row_id)
            official_raw_essay = str(
                official.get("essay", official.get("essay_text", ""))
            )
            official_raw_prompt = str(
                official.get("prompt", official.get("prompt_text", ""))
            )
            if row["prompt"] != official_raw_prompt:
                prompt_surface_mismatches.append(row_id)
            if row.get("essay_surfaces", {}).get("official_raw") != official_raw_essay:
                raw_surface_mismatches.append(row_id)
            row["source_split"] = (
                "official_train" if row_id in train_ids else "official_validation"
            )
        else:
            row["source_split"] = "origin_pool_extra"
        all_rows.append(row)

    public_ids = train_ids | validation_ids
    if (
        len(all_rows) != 12000
        or matched_ids != public_ids
        or score_mismatches
        or text_mismatches
        or prompt_surface_mismatches
        or raw_surface_mismatches
    ):
        raise DataError(
            "NIKL/public audit failed: "
            f"raw={len(all_rows)}, matched={len(matched_ids)}/{len(public_ids)}, "
            f"score_mismatch={score_mismatches[:5]}, "
            f"text_mismatch={text_mismatches[:5]}, "
            f"prompt_surface_mismatch={prompt_surface_mismatches[:5]}, "
            f"raw_surface_mismatch={raw_surface_mismatches[:5]}"
        )

    default_candidates = [
        row
        for row in all_rows
        if row["id"] not in validation_ids
        and row["essay_hash"] not in validation_hashes
    ]
    # Raw rows are the unit of supervision.  One NIKL input appears under two
    # IDs with genuinely different detailed annotations, so deduplicating by
    # essay would silently discard useful labels.
    default_rows = sorted(default_candidates, key=lambda row: str(row["id"]))
    leaked_rows = sorted(all_rows, key=lambda row: str(row["id"]))
    official_train_rows = sorted(
        (row for row in all_rows if row["id"] in train_ids), key=lambda row: row["id"]
    )
    validation_rows = sorted(
        (row for row in all_rows if row["id"] in validation_ids),
        key=lambda row: row["id"],
    )

    expected = {
        "default": 11600,
        "leaked": 12000,
        "official_train": 2000,
        "validation": 400,
    }
    actual = {
        "default": len(default_rows),
        "leaked": len(leaked_rows),
        "official_train": len(official_train_rows),
        "validation": len(validation_rows),
    }
    if actual != expected:
        raise DataError(f"unexpected competition row counts: {actual} != {expected}")

    common_audit = {
        "origin_rows": len(all_rows),
        "public_rows_reconstructed": len(matched_ids),
        "public_trait_scores_exact": len(matched_ids),
        "public_prompt_and_essay_exact": len(matched_ids),
        "public_official_prompt_byte_exact": len(matched_ids),
        "public_official_raw_surface_byte_exact": len(matched_ids),
        # baseline_v1 is a fixed literal template.  Exact prompt and essay
        # fields therefore imply exact formatted model input as well.
        "public_baseline_v1_input_byte_exact": len(matched_ids),
        "public_official_raw_rows_with_double_space": sum(
            "  " in row["essay_surfaces"]["official_raw"]
            for row in all_rows
            if row["id"] in public_ids
        ),
        "validation_ids_excluded": len(validation_ids),
        "validation_text_hashes_excluded": len(validation_hashes),
        "default_duplicate_inputs": duplicate_input_audit(default_rows),
        "leaked_duplicate_inputs": duplicate_input_audit(leaked_rows),
    }
    manifests: dict[str, Any] = {}
    for name, train_rows, include_validation, intentional_leak, policy in (
        (
            "competition",
            default_rows,
            True,
            False,
            "full_nikl_origin_minus_public_validation",
        ),
        (
            "competition_leaked",
            leaked_rows,
            True,
            True,
            "full_nikl_origin_including_public_validation",
        ),
        (
            "official_only",
            official_train_rows,
            True,
            False,
            "public_official_train_only",
        ),
    ):
        directory = outputs[name]
        train_path, temporary = prepare_target(directory, force=force)
        train_count = write_jsonl(temporary, train_rows)
        temporary.replace(train_path)
        files = {"train": file_record(train_path, train_count)}
        if include_validation:
            validation_path = directory / "validation.jsonl"
            validation_tmp = directory / "validation.jsonl.tmp"
            if validation_path.exists() and not force:
                raise FileExistsError(f"output exists; pass --force: {validation_path}")
            validation_count = write_jsonl(validation_tmp, validation_rows)
            validation_tmp.replace(validation_path)
            files["validation"] = file_record(validation_path, validation_count)
        manifest = {
            "dataset": name,
            "dataset_group": "competition",
            "policy": policy,
            "label_policy": "mean_of_two_raters_or_official_re_evaluator",
            "score_details": "all criterion and rater scores retained",
            "deduplication_policy": "retain_each_raw_NIKL_annotation_row",
            "contains_validation_labels": intentional_leak,
            "safe_for_model_selection": not intentional_leak,
            "audit": common_audit,
            "files": files,
        }
        write_manifest(directory, manifest)
        manifests[name] = manifest
    return manifests


def build_external(
    alias: str,
    raw_rows: Iterable[dict[str, Any]],
    converter: Callable[[dict[str, Any]], dict[str, Any]],
    outputs: dict[str, Path],
    official_validation_hashes: set[str],
    *,
    expected_counts: tuple[int, int],
    policy: str,
    deduplicate_exact_examples: bool = True,
    force: bool,
) -> dict[str, Any]:
    directory = outputs[alias]
    train_path, temporary = prepare_target(directory, force=force)
    raw_count = 0
    written = 0
    duplicate_count = 0
    validation_overlap_count = 0
    raw_split_counts: dict[str, int] = {}
    written_split_counts: dict[str, int] = {}
    seen_examples: set[tuple[str, str, str]] = set()
    seen_ids: set[str] = set()
    with temporary.open("w", encoding="utf-8") as file:
        for raw in raw_rows:
            raw_count += 1
            row = converter(raw)
            source_split = str(row["source_split"])
            raw_split_counts[source_split] = raw_split_counts.get(source_split, 0) + 1
            # Provider validation labels are legal external supervision, but
            # the competition validation essays must remain held out.
            if row["essay_hash"] in official_validation_hashes:
                validation_overlap_count += 1
                continue
            if row["id"] in seen_ids:
                raise DataError(f"{alias}: duplicate canonical id: {row['id']}")
            seen_ids.add(row["id"])
            # A duplicate means same input *and* same rich annotation.  This
            # deliberately retains AIHub26 rows whose C/O/E aggregates happen
            # to match while their criterion/rater labels differ.
            annotation = json.dumps(
                {
                    "score": row["score"],
                    "score_details": row["score_details"],
                    "auxiliary_scores": row.get("auxiliary_scores"),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            key = (
                row["prompt_hash"],
                row["essay_hash"],
                annotation,
            )
            if deduplicate_exact_examples and key in seen_examples:
                duplicate_count += 1
                continue
            seen_examples.add(key)
            file.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            written += 1
            written_split_counts[source_split] = (
                written_split_counts.get(source_split, 0) + 1
            )
    if validation_overlap_count != 0 or (raw_count, written) != expected_counts:
        raise DataError(
            f"{alias}: unexpected raw/written/competition-validation counts "
            f"{(raw_count, written, validation_overlap_count)} != "
            f"{(*expected_counts, 0)}"
        )
    temporary.replace(train_path)
    manifest = {
        "dataset": alias,
        "dataset_group": "external",
        "policy": policy,
        "score_details": "all analytic criterion and rater scores retained",
        "raw_rows": raw_count,
        "deduplicated_exact_examples": duplicate_count,
        "excluded_official_validation_essay_hashes": validation_overlap_count,
        "source_splits": {
            split: {
                "raw_rows": raw_split_counts[split],
                "written_rows": written_split_counts.get(split, 0),
            }
            for split in raw_split_counts
        },
        "files": {"train": file_record(train_path, written)},
    }
    write_manifest(directory, manifest)
    return manifest


def write_schema_document(data_root: Path) -> None:
    schema = {
        "schema_version": SCHEMA_VERSION,
        "required_fields": [
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
        ],
        "score": {
            "content": "official representative float in [1, 5]",
            "organization": "official representative float in [1, 5]",
            "expression": "official representative float in [1, 5]",
            "average": "mean of the three traits",
        },
        "score_details": {
            "label_policy": "how the representative score was selected",
            "primary_raters": "raters used before adjudication",
            "official_raters": "raters defining score",
            "rater_trait_scores": "per-rater trait means",
            "traits": "criterion order, per-rater criterion values and means",
        },
        "optional_fields": {
            "essay_surfaces.official_raw": (
                "byte-preserved direct concatenation of NIKL paragraph forms; "
                "matches the official public competition essay surface"
            )
        },
    }
    target = data_root / "schema.json"
    target.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_all(
    raw_root: Path,
    official_root: Path,
    output_root: Path,
    *,
    force: bool,
    selected: set[str],
) -> dict[str, Any]:
    outputs = output_directories(output_root)
    _, official_validation = load_official_rows(official_root)
    official_validation_hashes = {
        text_hash(official_essay(row)) for row in official_validation.values()
    }
    summary: dict[str, Any] = {}
    if "competition" in selected:
        summary.update(build_competition(raw_root, official_root, outputs, force=force))
    external_builders = {
        "aihub24_essay": lambda: build_external(
            "aihub24_essay",
            iter_aihub24(raw_root),
            convert_aihub24,
            outputs,
            official_validation_hashes,
            expected_counts=(45497, 45484),
            policy=(
                "AIHub provider Training+Validation labels used as external "
                "training; official competition validation essays excluded"
            ),
            force=force,
        ),
        "aihub25_descriptive": lambda: build_external(
            "aihub25_descriptive",
            iter_aihub_common(raw_root, "aihub25_descriptive"),
            lambda raw: convert_aihub_common(raw, "aihub25_descriptive"),
            outputs,
            official_validation_hashes,
            expected_counts=(36006, 36006),
            policy=(
                "AIHub provider Training+Validation labels used as external "
                "training; official competition validation essays excluded"
            ),
            force=force,
        ),
        "aihub26_essay": lambda: build_external(
            "aihub26_essay",
            iter_aihub_common(raw_root, "aihub26_essay"),
            lambda raw: convert_aihub_common(raw, "aihub26_essay"),
            outputs,
            official_validation_hashes,
            expected_counts=(18010, 18010),
            policy=(
                "AIHub provider Training+Validation labels used as external "
                "training; official competition validation essays excluded"
            ),
            force=force,
        ),
        "aihub27_topic": lambda: build_external(
            "aihub27_topic",
            iter_aihub_common(raw_root, "aihub27_topic"),
            lambda raw: convert_aihub_common(raw, "aihub27_topic"),
            outputs,
            official_validation_hashes,
            expected_counts=(18001, 18001),
            policy=(
                "AIHub provider Training+Validation labels used as external "
                "training; official competition validation essays excluded"
            ),
            force=force,
        ),
        "nikl24_summary": lambda: build_external(
            "nikl24_summary",
            iter_nikl_summary(raw_root, "nikl24_summary"),
            convert_nikl_summary,
            outputs,
            official_validation_hashes,
            expected_counts=(7712, 7712),
            policy=(
                "NIKL 2024 provider SC1/SC2 index-0 summaries only; two missing "
                "evaluations recovered from text-identical later objects; revised "
                "variants excluded; official competition validation essays excluded"
            ),
            # Text-identical SC1/SC2 rows are independent three-rater panels.
            # Retaining both is equivalent to weighting the pooled six ratings;
            # they must not become artificial ranking pairs in later methods.
            deduplicate_exact_examples=False,
            force=force,
        ),
        "nikl25_argumentative_summary": lambda: build_external(
            "nikl25_argumentative_summary",
            iter_nikl_summary(raw_root, "nikl25_argumentative_summary"),
            convert_nikl_summary,
            outputs,
            official_validation_hashes,
            expected_counts=(2016, 2016),
            policy=(
                "NIKL 2025 SC1/SC2 argumentative summaries with three human "
                "ratings; official competition validation essays excluded"
            ),
            deduplicate_exact_examples=False,
            force=force,
        ),
        "nikl25_cooperative_summary": lambda: build_external(
            "nikl25_cooperative_summary",
            iter_nikl_summary(raw_root, "nikl25_cooperative_summary"),
            convert_nikl_summary,
            outputs,
            official_validation_hashes,
            expected_counts=(1020, 1020),
            policy=(
                "NIKL 2025 SC1/SC2 cooperative-dialogue summaries with three "
                "human ratings; official competition validation essays excluded"
            ),
            deduplicate_exact_examples=False,
            force=force,
        ),
    }
    for alias, builder in external_builders.items():
        if alias in selected:
            summary[alias] = builder()
    write_schema_document(output_root)
    return summary


# CLI -------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--official-root", type=Path, default=OFFICIAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--datasets",
        default="all",
        help=(
            "comma list: competition,aihub24_essay,aihub25_descriptive,"
            "aihub26_essay,aihub27_topic,nikl24_summary,"
            "nikl25_argumentative_summary,nikl25_cooperative_summary or all"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    choices = {
        "competition",
        "aihub24_essay",
        "aihub25_descriptive",
        "aihub26_essay",
        "aihub27_topic",
        "nikl24_summary",
        "nikl25_argumentative_summary",
        "nikl25_cooperative_summary",
    }
    selected = (
        choices
        if args.datasets == "all"
        else {item.strip() for item in args.datasets.split(",") if item.strip()}
    )
    unknown = selected - choices
    if unknown or not selected:
        raise ValueError(
            f"invalid datasets={sorted(unknown)}; choices={sorted(choices)}"
        )
    summary = build_all(
        args.raw_root.resolve(),
        args.official_root.resolve(),
        args.output_root.resolve(),
        force=args.force,
        selected=selected,
    )
    for name, manifest in summary.items():
        train = manifest["files"]["train"]
        validation = manifest["files"].get("validation")
        suffix = f", validation={validation['rows']:,}" if validation else ""
        print(
            f"{name}: train={train['rows']:,}{suffix} -> "
            f"{output_directories(args.output_root.resolve())[name] / train['path']}"
        )


if __name__ == "__main__":
    main()
