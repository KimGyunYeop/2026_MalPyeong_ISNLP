from __future__ import annotations

import hashlib
import json
import math
import random
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from functools import lru_cache
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from .config import (
    EXTERNAL_DATASETS,
    ESSAY_SURFACES,
    RATER_SET_HEAD_MODES,
    TRAITS,
    RegressionConfig,
)


# Detailed NIKL rubric labels -------------------------------------------------
# Keep this order fixed across the collator, model heads and checkpoint config.
# The order is intentionally visible instead of being inferred from whichever
# keys happen to be present in one JSON row.
DETAIL_CRITERIA_BY_TRAIT = {
    "content": (
        "content_1",
        "content_2",
        "content_3",
        "content_4",
        "content_5",
    ),
    "organization": ("organization_1", "organization_2"),
    "expression": ("expression_1", "expression_2"),
}
DETAIL_CRITERIA = tuple(
    criterion for trait in TRAITS for criterion in DETAIL_CRITERIA_BY_TRAIT[trait]
)
DETAIL_SCORE_CLASSES = (1, 2, 3, 4, 5)
DETAIL_MISSING_CLASS = -100
DETAIL_PAD_RATER_ID = -1
DETAIL_OFFICIAL_RATER_SLOTS = 2


DetailRaterKey = tuple[str, str]


def detail_rater_key(row: dict[str, Any], evaluator_id: Any) -> DetailRaterKey:
    """Return the train-registry key for one actual evaluator.

    Evaluator number spaces are source-specific.  Keeping the source and raw
    ID as two strings avoids both sparse embedding indices and accidental ID
    collisions when an external dataset is added later.
    """

    return str(row.get("source_dataset", "")), str(evaluator_id)


def build_detail_rater_registry(
    rows: Sequence[dict[str, Any]],
) -> tuple[DetailRaterKey, ...]:
    """Build a deterministic evaluator registry from training rows only."""

    keys: set[DetailRaterKey] = set()
    for row in rows:
        details = row.get("score_details")
        if not isinstance(details, dict):
            continue
        rater_ids = details.get("rater_ids")
        if not isinstance(rater_ids, dict):
            continue
        for evaluator_id in rater_ids.values():
            if evaluator_id is None or not str(evaluator_id).strip():
                continue
            keys.add(detail_rater_key(row, evaluator_id))
    return tuple(sorted(keys))


def _detail_score(value: Any, *, require_integer: bool) -> float | None:
    """Read a finite 1--5 score without rounding or inventing a label."""

    if isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or not 1.0 <= score <= 5.0:
        return None
    if require_integer and not score.is_integer():
        return None
    return score


def _detail_criterion_record(
    row: dict[str, Any], criterion: str
) -> dict[str, Any] | None:
    trait = criterion.rsplit("_", 1)[0]
    details = row.get("score_details")
    if not isinstance(details, dict):
        return None
    traits = details.get("traits")
    if not isinstance(traits, dict):
        return None
    trait_details = traits.get(trait)
    if not isinstance(trait_details, dict):
        return None
    criteria = trait_details.get("criteria")
    if not isinstance(criteria, dict):
        return None
    record = criteria.get(criterion)
    return record if isinstance(record, dict) else None


def _rater_slot_sort_key(name: str) -> tuple[int, int | str]:
    match = re.fullmatch(r"evaluator(\d+)", name)
    if match:
        return (0, int(match.group(1)))
    return (1, name)


def primary_rater_disagreement_sum(row: dict[str, Any]) -> float:
    """Sum absolute differences between the two original ratings over 9 criteria.

    NIKL normally stores the original pair as evaluator1/evaluator2.  Nineteen
    adjudicated rows have one empty primary slot and the second original rating
    in the next non-empty evaluator slot, normally evaluator3.  We use that
    next numbered non-re-evaluator slot instead of dropping those rows or using
    the adjudicator's score.
    """

    details = row.get("score_details")
    primary_raters = (
        details.get("primary_raters", ()) if isinstance(details, dict) else ()
    )
    if not isinstance(primary_raters, (list, tuple)) or len(primary_raters) != 2:
        raise ValueError("두 primary_raters가 있는 NIKL score_details가 필요합니다")

    primary_names = tuple(str(name) for name in primary_raters)
    disagreement = 0.0
    for criterion in DETAIL_CRITERIA:
        record = _detail_criterion_record(row, criterion)
        rater_scores = record.get("rater_scores") if record is not None else None
        if not isinstance(rater_scores, dict):
            raise ValueError(f"{criterion}의 rater_scores가 필요합니다")

        pair = []
        for rater in primary_names:
            score = _detail_score(rater_scores.get(rater), require_integer=True)
            if score is not None:
                pair.append(score)
        if len(pair) < 2:
            fallback_raters = sorted(
                (
                    str(rater)
                    for rater in rater_scores
                    if str(rater) not in primary_names
                    and str(rater) != "re-evaluator"
                ),
                key=_rater_slot_sort_key,
            )
            for rater in fallback_raters:
                score = _detail_score(rater_scores.get(rater), require_integer=True)
                if score is not None:
                    pair.append(score)
                    break
        if len(pair) != 2:
            raise ValueError(f"{criterion}에서 두 원평정 점수를 찾지 못했습니다")
        disagreement += abs(pair[0] - pair[1])
    return disagreement


def filter_origin_extra_by_rater_agreement(
    rows: Sequence[dict[str, Any]], max_disagreement: float
) -> list[dict[str, Any]]:
    """Keep all non-extra rows and only agreement-matched origin-pool extras."""

    if max_disagreement < 0:
        return list(rows)
    return [
        row
        for row in rows
        if row.get("source_split") != "origin_pool_extra"
        or primary_rater_disagreement_sum(row) <= max_disagreement
    ]


def split_primary_rows_by_source(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose the official 2k and origin-extra 9.6k to the source sampler."""

    partitioned: list[dict[str, Any]] = []
    for row in rows:
        source_split = str(row.get("source_split", ""))
        if source_split == "official_train":
            dataset_name = "competition"
        elif source_split == "origin_pool_extra":
            dataset_name = "origin_pool_extra"
        else:
            raise ValueError(
                "split_primary_sources에는 official_train 또는 "
                f"origin_pool_extra row만 올 수 있습니다: {source_split!r}"
            )
        item = dict(row)
        item["_dataset_name"] = dataset_name
        partitioned.append(item)
    return partitioned


def _criterion_supervision(
    row: dict[str, Any],
) -> tuple[list[float], list[bool], list[list[float]], list[bool]]:
    """Extract official criterion scores and official-rater distributions."""

    details = row.get("score_details")
    official_raters = (
        details.get("official_raters", ()) if isinstance(details, dict) else ()
    )
    if not isinstance(official_raters, (list, tuple)):
        official_raters = ()

    criterion_scores: list[float] = []
    criterion_mask: list[bool] = []
    criterion_distributions: list[list[float]] = []
    criterion_distribution_mask: list[bool] = []

    for criterion in DETAIL_CRITERIA:
        record = _detail_criterion_record(row, criterion)
        official_score = _detail_score(
            record.get("official_score") if record is not None else None,
            require_integer=False,
        )
        criterion_scores.append(official_score if official_score is not None else 0.0)
        criterion_mask.append(official_score is not None)

        counts = [0] * len(DETAIL_SCORE_CLASSES)
        rater_scores = record.get("rater_scores") if record is not None else None
        if isinstance(rater_scores, dict):
            for rater_name in official_raters:
                score = _detail_score(
                    rater_scores.get(str(rater_name)), require_integer=True
                )
                if score is not None:
                    counts[int(score) - 1] += 1
        count = sum(counts)
        criterion_distributions.append(
            [value / count for value in counts]
            if count
            else [0.0] * len(DETAIL_SCORE_CLASSES)
        )
        criterion_distribution_mask.append(count > 0)

    return (
        criterion_scores,
        criterion_mask,
        criterion_distributions,
        criterion_distribution_mask,
    )


def _official_rater_set_supervision(
    row: dict[str, Any],
) -> tuple[list[list[int]], list[list[bool]]]:
    """Return the one or two official nine-score vectors without inventing a rater.

    The two branches of ``detail_head_mode='rater_set'`` are anonymous.  The
    model therefore matches these complete vectors permutation-invariantly
    instead of treating ``evaluator1`` and ``evaluator2`` as stable people.
    Adjudicated rows have one official re-evaluator; their second slot stays
    masked rather than duplicating an unobserved rating.
    """

    details = row.get("score_details")
    official_raters = (
        details.get("official_raters", ()) if isinstance(details, dict) else ()
    )
    if not isinstance(official_raters, (list, tuple)):
        official_raters = ()
    if len(official_raters) > DETAIL_OFFICIAL_RATER_SLOTS:
        raise ValueError(
            "rater_set은 공식 평가자 최대 2명을 가정합니다: "
            f"found={len(official_raters)}"
        )

    labels = [
        [DETAIL_MISSING_CLASS] * len(DETAIL_CRITERIA)
        for _ in range(DETAIL_OFFICIAL_RATER_SLOTS)
    ]
    masks = [[False] * len(DETAIL_CRITERIA) for _ in range(DETAIL_OFFICIAL_RATER_SLOTS)]
    for rater_slot, rater_name in enumerate(official_raters):
        for criterion_index, criterion in enumerate(DETAIL_CRITERIA):
            record = _detail_criterion_record(row, criterion)
            rater_scores = record.get("rater_scores") if record is not None else None
            score = (
                _detail_score(rater_scores.get(str(rater_name)), require_integer=True)
                if isinstance(rater_scores, dict)
                else None
            )
            if score is not None:
                labels[rater_slot][criterion_index] = int(score) - 1
                masks[rater_slot][criterion_index] = True
    return labels, masks


def _rater_supervision(
    row: dict[str, Any],
    registry_lookup: dict[DetailRaterKey, int],
) -> tuple[list[int], list[list[int]], list[list[bool]]]:
    """Extract every actual evaluator's nine hard labels for one essay."""

    details = row.get("score_details")
    rater_ids = details.get("rater_ids") if isinstance(details, dict) else None
    if not isinstance(rater_ids, dict):
        rater_ids = {}

    # In 19 re-evaluated NIKL rows one original nine-score block is null and
    # its evaluator ID is absent.  The null slot still appears in every
    # criterion's rater_scores.  Retain that slot as ID=-1/mask=False so the
    # missing block is explicit instead of disappearing from the batch.
    rater_names = list(rater_ids)
    seen_rater_names = set(rater_names)
    for criterion in DETAIL_CRITERIA:
        record = _detail_criterion_record(row, criterion)
        rater_scores = record.get("rater_scores") if record is not None else None
        if not isinstance(rater_scores, dict):
            continue
        for rater_name in rater_scores:
            if rater_name not in seen_rater_names:
                seen_rater_names.add(rater_name)
                rater_names.append(rater_name)

    ids: list[int] = []
    labels: list[list[int]] = []
    masks: list[list[bool]] = []
    for rater_name in rater_names:
        evaluator_id = rater_ids.get(rater_name)
        has_evaluator_id = evaluator_id is not None and bool(str(evaluator_id).strip())
        registry_id = (
            registry_lookup.get(detail_rater_key(row, evaluator_id))
            if has_evaluator_id
            else None
        )
        known_evaluator = registry_id is not None
        ids.append(registry_id if known_evaluator else DETAIL_PAD_RATER_ID)

        rater_labels: list[int] = []
        rater_masks: list[bool] = []
        for criterion in DETAIL_CRITERIA:
            record = _detail_criterion_record(row, criterion)
            rater_scores = record.get("rater_scores") if record is not None else None
            score = (
                _detail_score(rater_scores.get(str(rater_name)), require_integer=True)
                if isinstance(rater_scores, dict)
                else None
            )
            valid = known_evaluator and score is not None
            rater_labels.append(int(score) - 1 if valid else DETAIL_MISSING_CLASS)
            rater_masks.append(valid)
        labels.append(rater_labels)
        masks.append(rater_masks)
    return ids, labels, masks


def detail_supervision_summary(
    rows: Sequence[dict[str, Any]],
    registry: Sequence[DetailRaterKey],
    *,
    include_official_rater_set: bool = False,
) -> dict[str, int]:
    """Count the exact masked labels available to one training run.

    This uses the same extraction helpers as ``RegressionCollator`` so
    ``run.json`` records what the loss can actually see, including unknown-ID
    masking and essays with more than two historical raters.
    """

    registry_lookup = {key: index for index, key in enumerate(registry)}
    criterion_score_valid = 0
    criterion_distribution_valid = 0
    individual_rating_valid = 0
    official_rater_rating_valid = 0
    rows_with_one_official_rater = 0
    rows_with_two_official_raters = 0
    max_raters_per_essay = 0
    rows_with_criterion_scores = 0
    rows_with_individual_ratings = 0

    for row in rows:
        _, criterion_mask, _, distribution_mask = _criterion_supervision(row)
        score_count = sum(criterion_mask)
        criterion_score_valid += score_count
        criterion_distribution_valid += sum(distribution_mask)
        rows_with_criterion_scores += int(score_count > 0)

        if include_official_rater_set:
            _, official_rater_mask = _official_rater_set_supervision(row)
            official_rater_rating_valid += sum(
                sum(mask_row) for mask_row in official_rater_mask
            )
            official_rater_count = sum(
                any(mask_row) for mask_row in official_rater_mask
            )
            rows_with_one_official_rater += int(official_rater_count == 1)
            rows_with_two_official_raters += int(official_rater_count == 2)

        rater_ids, _, rater_mask = _rater_supervision(row, registry_lookup)
        max_raters_per_essay = max(max_raters_per_essay, len(rater_ids))
        row_rating_count = sum(sum(mask_row) for mask_row in rater_mask)
        individual_rating_valid += row_rating_count
        rows_with_individual_ratings += int(row_rating_count > 0)

    summary = {
        "rows": len(rows),
        "criterion_slots": len(rows) * len(DETAIL_CRITERIA),
        "criterion_score_valid": criterion_score_valid,
        "criterion_distribution_valid": criterion_distribution_valid,
        "rater_registry_count": len(registry),
        "individual_rating_valid": individual_rating_valid,
        "max_raters_per_essay": max_raters_per_essay,
        "rows_with_criterion_scores": rows_with_criterion_scores,
        "rows_with_individual_ratings": rows_with_individual_ratings,
    }
    if include_official_rater_set:
        summary.update(
            {
                "official_rater_rating_valid": official_rater_rating_valid,
                "rows_with_one_official_rater": rows_with_one_official_rater,
                "rows_with_two_official_raters": rows_with_two_official_raters,
            }
        )
    return summary


# Canonical row access --------------------------------------------------------
def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read canonical JSONL without depending on the repository-level package."""

    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: 잘못된 JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSON 객체가 필요합니다")
            rows.append(row)
    return rows


def essay_id(row: dict[str, Any]) -> str:
    value = row.get("essay_id") or row.get("id") or row.get("document_id")
    if value is None:
        raise KeyError("essay_id, id, document_id 중 하나가 필요합니다")
    return str(value)


def prompt_text(row: dict[str, Any]) -> str:
    value = row.get("prompt_text", row.get("prompt"))
    if value is None:
        raise KeyError("prompt_text 또는 prompt가 필요합니다")
    return str(value)


def essay_text(row: dict[str, Any]) -> str:
    value = row.get("essay_text", row.get("essay"))
    if value is None:
        raise KeyError("essay_text 또는 essay가 필요합니다")
    return str(value)


def human_scores(row: dict[str, Any]) -> dict[str, float] | None:
    value = row.get("human_score", row.get("score"))
    if not isinstance(value, dict) or not all(trait in value for trait in TRAITS):
        return None
    scores = {trait: float(value[trait]) for trait in TRAITS}
    # Competition labels are exact averages of integer rubric ratings.  The
    # generated JSON can nevertheless contain values such as
    # 1.2999999999999998 beside 1.3.  Canonicalizing only this known label
    # policy restores mathematical Spearman ties without rounding predictions
    # or assuming that external datasets use the same grid.
    details = row.get("score_details")
    label_policy = details.get("label_policy") if isinstance(details, dict) else None
    source = str(row.get("source_dataset", row.get("_dataset_name", "")))
    dataset_group = str(row.get("dataset_group", ""))
    uses_nikl_score_grid = dataset_group == "competition" or source in {
        "competition",
        "nikl_competition",
        "nikl_grading",
    }
    if uses_nikl_score_grid and label_policy in {
        "mean_of_two_raters",
        "official_re_evaluator",
    }:
        steps = {"content": 0.1, "organization": 0.25, "expression": 0.25}
        scores = {
            trait: round(scores[trait] / steps[trait]) * steps[trait]
            for trait in TRAITS
        }
    return scores


def official_average_score(row: dict[str, Any]) -> float | None:
    """공식 지표의 human_avg인 ``score.average``를 원본 그대로 읽는다.

    2026-07-20 운영진 답변에 따르면 공식 RMSE/Spearman의 정답은 세 trait label을
    다시 평균한 값이 아니라 데이터셋에 실려 있는 ``score.average``다. 실제 파일의
    ``average``는 세 trait 평균을 소수 2자리로 자른 값이라 재계산값과 최대 `.0067`
    다르므로 canonicalize하지 않고 실린 값을 그대로 쓴다. 없으면 None을 반환하고
    호출부가 세 trait 평균으로 대체한다.
    """

    value = row.get("human_score", row.get("score"))
    if not isinstance(value, dict):
        return None
    average = value.get("average")
    if average is None:
        return None
    return float(average)


# Input formatting and essay-token masks ------------------------------------
_KIWI_SENTENCE_SPLITTER: Any | None = None


def _get_kiwi_sentence_splitter() -> Any:
    """Load one CPU-only Kiwi splitter when its input surface is first used."""

    global _KIWI_SENTENCE_SPLITTER
    if _KIWI_SENTENCE_SPLITTER is None:
        try:
            from kiwipiepy import Kiwi
        except ImportError as exc:
            raise RuntimeError(
                "essay_surface='official_raw_kiwi_sentence_newline_v1'에는 "
                "kiwipiepy==0.23.2가 필요합니다. requirements-train.txt를 설치하세요."
            ) from exc
        _KIWI_SENTENCE_SPLITTER = Kiwi(num_workers=0)
    return _KIWI_SENTENCE_SPLITTER


@lru_cache(maxsize=None)
def _official_raw_with_kiwi_sentence_newlines(official_raw: str) -> str:
    """Insert sentence separators without replacing any official-raw byte."""

    sentences = list(
        _get_kiwi_sentence_splitter().split_into_sents(
            official_raw, return_sub_sents=False
        )
    )
    if len(sentences) < 2:
        return official_raw

    spans: list[tuple[int, int]] = []
    for sentence in sentences:
        try:
            start = int(sentence.start)
            end = int(sentence.end)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "Kiwi가 유효한 문장 start/end span을 반환하지 않았습니다"
            ) from exc
        if not 0 <= start < end <= len(official_raw):
            raise ValueError(
                "Kiwi 문장 span이 원문 범위를 벗어났습니다: "
                f"start={start}, end={end}, essay_length={len(official_raw)}"
            )

        # Kiwi 0.23.2 can emit two overlapping top-level spans around a typo,
        # for example ``안됀`` followed by ``됀다.``.  They describe one visible
        # character interval, so coalesce their interval union and put a
        # boundary only after the union.  Slicing the original string below
        # still preserves every input character exactly once.
        if spans and start < spans[-1][0]:
            raise ValueError(
                "Kiwi 문장 span 순서가 역전되었습니다: "
                f"start={start}, previous_start={spans[-1][0]}"
            )
        if spans and start < spans[-1][1]:
            previous_start, previous_end = spans[-1]
            spans[-1] = (min(previous_start, start), max(previous_end, end))
        else:
            spans.append((start, end))

    boundaries = [end for _, end in spans[:-1]]
    pieces: list[str] = []
    cursor = 0
    for boundary in boundaries:
        pieces.append(official_raw[cursor:boundary])
        pieces.append("\n")
        cursor = boundary
    pieces.append(official_raw[cursor:])
    return "".join(pieces)


def essay_for_input(
    row: dict[str, Any],
    config: RegressionConfig,
    *,
    essay_surface: str | None = None,
) -> str:
    """Return the essay surface that the scorer is meant to see.

    Canonical NIKL rows retain paragraph breaks reconstructed from the raw
    corpus.  ``flat`` is the legacy all-whitespace-collapse transform.  The
    official competition actually concatenates raw paragraph forms byte for
    byte, so ``official_raw`` retains its leading/trailing, repeated-space and tab
    cues. ``official_raw_kiwi_sentence_newline_v1`` retains those characters and
    inserts only sentence separators inferred from the same visible string.
    Keeping each transform in the collator makes the train/eval surface an
    explicit experiment setting without mutating labels or fingerprints.
    """

    text = essay_text(row)
    selected_surface = config.essay_surface if essay_surface is None else essay_surface
    if selected_surface == "canonical":
        return text
    if selected_surface == "flat":
        return " ".join(unicodedata.normalize("NFC", text).split())
    if selected_surface in {
        "official_raw",
        "official_gap_newline",
        "official_raw_kiwi_sentence_newline_v1",
    }:
        surfaces = row.get("essay_surfaces")
        if isinstance(surfaces, dict) and "official_raw" in surfaces:
            stored_raw = surfaces["official_raw"]
            if not isinstance(stored_raw, str) or not stored_raw.strip():
                raise ValueError(
                    "essay_surfaces.official_raw에는 비어 있지 않은 문자열이 필요합니다"
                )
            normalized_canonical = " ".join(unicodedata.normalize("NFC", text).split())
            normalized_raw = " ".join(unicodedata.normalize("NFC", stored_raw).split())
            if normalized_raw != normalized_canonical:
                raise ValueError(
                    "essay_surfaces.official_raw가 canonical essay와 일치하지 않습니다"
                )
            official_raw = stored_raw
        elif (
            "essay" in row
            and "essay_text" not in row
            and not any(
                key in row
                for key in ("schema_version", "source_dataset", "dataset_group")
            )
        ):
            # Official inference rows do not use the prepared-data schema.  Their
            # essay value is already the deployment string, including the rare
            # CR/tab and leading/trailing/repeated-space bytes.  Do not infer the
            # row type from whitespace: doing so would reject valid official raw
            # rows and accept a lossy one-paragraph prepared row by accident.
            official_raw = text
        else:
            raise ValueError(
                "official_raw surface에는 보존된 essay_surfaces.official_raw 또는 "
                "공식 raw inference row가 필요합니다. 데이터를 다시 빌드하세요."
            )
        if selected_surface == "official_raw":
            return official_raw
        if selected_surface == "official_raw_kiwi_sentence_newline_v1":
            return _official_raw_with_kiwi_sentence_newlines(official_raw)
        # In the current 12k source, every strict interior 2+ space run is a
        # true paragraph seam (8,489/8,489), but it covers only about 25.35%
        # of all seams.  This deployable
        # ablation exposes only that observed cue; it never consults labels or
        # hidden canonical boundaries.
        return re.sub(r"(?<=\S) {2,}(?=\S)", "\n\n", official_raw)
    raise ValueError(f"지원하지 않는 essay_surface입니다: {selected_surface}")


INPUT_TEMPLATE = """[task]
너는 한국어 논증적 글을 일관되게 직접 채점하는 평가자이다.
essay_text를 읽고 content, organization, expression 세 기준을 모두 평가하라.

[평가 기준 정의]
1. content
- 글의 주장과 핵심 내용이 문제에 적절하게 대응하는가
- 근거가 충분하고 구체적인가
- 주장과 근거 사이의 논리적 연결이 타당한가

2. organization
- 서론, 본론, 결론의 구조가 드러나는가
- 문단 간 연결이 자연스러운가
- 논리 전개 순서가 일관적인가

3. expression
- 문장이 자연스럽고 이해하기 쉬운가
- 어휘 사용이 적절한가
- 맞춤법, 띄어쓰기, 문법, 주술 호응에 문제가 없는가

[prompt_text]
{prompt}

[essay_text]
{essay}"""

# 문항 지문을 넣지 않는 입력. 2026-08-20 진단에서 나온 두 사실이 근거다.
#   1) 문항은 target을 거의 설명하지 못한다. 문항별 평균 예측기의 RMSE는 0.6470으로
#      전체 평균 예측기(0.6533)와 사실상 같다 — 전체 분산의 1% 수준이다.
#   2) 그런데 지문은 평균 333자로 baseline_v1 입력(평균 1,715자)의 **19.4%**를
#      차지하고, pooling="mean"은 그 token들의 hidden state를 essay token과 **똑같은
#      무게로** 평균한다. 문항이 9개뿐이라 같은 문항 글 1,300여 편이 이 성분을
#      글자 그대로 공유한다. 입력 길이는 20.4% 줄어든다(1,715 -> 1,366자).
# 즉 지문은 신호를 거의 주지 않으면서 pooled 표현을 희석한다. 이 template은 그
# 희석을 제거하고, 동시에 평가셋에 미학습 문항이 있더라도 입력이 변하지 않게 한다.
#
# 주의: content 준거 C1(문제 상황 제시)과 C2(주장)는 정의상 문항에 상대적이다.
# 그래서 이건 pooling="essay_mean"(지문을 attention에는 남기고 pooling에서만 빼기)의
# 대체가 아니라 **다른 가설**이다. 둘을 각각 재야 한다.
ESSAY_ONLY_INPUT_TEMPLATE = """[task]
너는 한국어 논증적 글을 일관되게 직접 채점하는 평가자이다.
essay_text를 읽고 content, organization, expression 세 기준을 모두 평가하라.

[평가 기준 정의]
1. content
- 글의 주장과 핵심 내용이 논제에 적절하게 대응하는가
- 근거가 충분하고 구체적인가
- 주장과 근거 사이의 논리적 연결이 타당한가

2. organization
- 서론, 본론, 결론의 구조가 드러나는가
- 문단 간 연결이 자연스러운가
- 논리 전개 순서가 일관적인가

3. expression
- 문장이 자연스럽고 이해하기 쉬운가
- 어휘 사용이 적절한가
- 맞춤법, 띄어쓰기, 문법, 주술 호응에 문제가 없는가

[essay_text]
{essay}"""

NEUTRAL_INPUT_TEMPLATE = """[task]
너는 주어진 쓰기 문항에 대한 한국어 글을 일관되게 직접 채점하는 평가자이다.
essay_text를 읽고 content, organization, expression 세 기준을 모두 평가하라.

[평가 기준 정의]
1. content: 문항에 적절하게 답하고, 핵심 내용과 근거가 충분하고 타당한가
2. organization: 글의 전체 구조와 정보 배열, 문장·문단 사이의 흐름이 자연스러운가
3. expression: 문장과 어휘가 자연스럽고 맞춤법·띄어쓰기·문법이 적절한가

[prompt_text]
{prompt}

[essay_text]
{essay}"""

# 별책1 「논증적 글쓰기 채점 매뉴얼」의 준거 설명과 1~5 경계만 압축했다.
# 예시 답안, validation 성능으로 고른 표현, 점수 분포 prior는 넣지 않는다.
FULL_RUBRIC_INPUT_TEMPLATE = """[task]
너는 주어진 쓰기 문항에 대한 한국어 논증적 글을 공식 준거로 직접 채점하는 평가자이다.

[채점 계약]
- C1~C5, O1~O2, E1~E2를 서로 독립적으로 판단하고 한 결함을 관련 없는 준거에 중복 감점하지 않는다.
- 개별 평가자는 각 준거에 1~5 정수를 준다. 공개 데이터의 대표 준거 target은 통상 두 평가자 평균이고 재평가가 확정된 행은 재평가자 점수이므로, 준거 예측은 1~5 연속값으로 유지하고 반올림하지 않는다.
- 최종 content=(C1+C2+C3+C4+C5)/5, organization=(O1+O2)/2, expression=(E1+E2)/2로 계산한다.
- 공식 입력에서는 원 줄바꿈이 보이지 않을 수 있다. 줄바꿈 부재 자체만으로 O1/O2를 감점하지 말고, 관측되는 공백과 의미 단위 및 담화 표지를 함께 판단한다.

[공식 세부 채점 준거]
C1 문제 상황 제시: 논제를 공론화할 배경·필요와 관련 정보를 적절하고 충분히 제시하는가? 5=논제와 매우 밀접하고 반드시 필요한 정보로 효과적; 4=대체로 밀접하고 대부분 필요한 정보; 3=관련은 있으나 정보성이 다소 낮음; 2=관련성이 낮은 정보 위주; 1=설명이 없거나 매우 빈약.
C2 주장: 주장이 논제에 부합하며 글 전체에서 일관되고 뚜렷한가? 5=부합하며 매우 일관·명확; 4=부합하며 대체로 일관·명확; 3=대체로 부합하나 일관성 또는 명료성이 다소 부족; 2=논제에서 다소 벗어나거나 일관성·명료성이 매우 부족; 1=논제에 부합하지 않거나 일관성·명료성이 극히 부족.
C3 이유·근거의 적절성: 주장과 이유·근거의 추론 연결이 질적으로 타당한가? 5=매우 적절해 설득력이 높음; 4=대체로 적절해 어느 정도 설득력 있음; 3=일부 부적절하지만 대체로 수용 가능; 2=부적절한 경우가 많아 설득력이 낮음; 1=대부분 부적절해 설득력이 매우 낮음.
C4 이유·근거의 충분성: 적절한 하위주장의 수와 각 뒷받침의 깊이가 충분한가? 5=적절한 하위주장 2개 이상을 각각 깊이 뒷받침; 4=2개 이상을 각각 적절히 뒷받침; 3=1개를 비교적 깊이 뒷받침; 2=1개를 제시했으나 뒷받침이 얕음; 1=적절한 하위주장이 없음.
C5 다른 입장 고려: 다른 입장을 다루고 비교·논박 등으로 깊게 전개하는가? 5=고려하며 매우 깊게 전개; 4=대체로 깊게 전개; 3=고려했으나 깊이가 다소 낮음; 2=완곡하거나 비단정적인 표현으로 존재 인식만 간접적으로 드러남; 1=한 입장만 단정적으로 언급.
O1 글 전체 조직: 형식·내용 문단의 서론-본론-결론 역할 구분과 배열이 유기적인가? 5=구분이 매우 적절하고 배열이 매우 유기적; 4=대체로 적절하고 유기적; 3=구분 또는 배열 일부가 부자연스럽지만 읽기에 큰 방해 없음; 2=체계와 완성도가 낮고 부자연스러운 배열이 많음; 1=내용 문단이 거의 구분되지 않아 체계가 없음.
O2 문단 내 조직: 각 문단 또는 의미 단위가 완결성, 통일성, 문장 간 일관성을 갖추는가? 5=모두 갖춤; 4=대부분 갖춤; 3=일부 갖춤; 2=전반적으로 부족해 완성도가 낮음; 1=대부분 갖추지 못함.
E1 문장과 어휘: 문장이 자연스럽고 명료·효과적이며 어휘가 문맥에 적절한가? 5=문장·어휘가 자연스럽고 논증에 효과적; 4=대체로 자연스럽고 적절하지만 논증 효과가 부족; 3=부자연스러운 문장이나 부적절한 어휘가 있으나 의미 파악에 큰 방해 없음; 2=의미 파악을 방해; 1=대부분 부자연스럽고 문맥에 맞지 않아 의미 파악을 크게 방해.
E2 어문 규범과 관습: 맞춤법·띄어쓰기·오탈자와 문어체·종결어미 일관성을 지키는가? 5=둘 다 매우 정확; 4=규범 위반이 가끔 있거나 관습 위반 1유형; 3=규범 위반 가끔+관습 1유형, 규범 정확+관습 2유형, 또는 규범 빈번+관습 정확; 2=규범 빈번+관습 1유형 또는 규범 가끔+관습 2유형; 1=규범 빈번+관습 2유형. 가끔과 빈번은 글 길이에 비례해 판단한다.

[prompt_text]
{prompt}

[essay_text]
{essay}"""

RUBRIC_CRITERION_TAG_BY_NAME = {
    "content_1": "C1",
    "content_2": "C2",
    "content_3": "C3",
    "content_4": "C4",
    "content_5": "C5",
    "organization_1": "O1",
    "organization_2": "O2",
    "expression_1": "E1",
    "expression_2": "E2",
}
RUBRIC_CRITERION_TAGS = tuple(
    RUBRIC_CRITERION_TAG_BY_NAME[criterion] for criterion in DETAIL_CRITERIA
)
RUBRIC_READOUT_HEADER = "\n\n[criterion_readout]\n"
RUBRIC_TOKENIZATION_POLICY = "base_special_plus_suffix_no_special_v1"


@dataclass(frozen=True)
class FormattedInputSegments:
    """Scorer-visible text plus strict character boundaries for RC readout."""

    text: str
    essay_span: tuple[int, int]
    scoring_end: int
    criterion_anchor_spans: tuple[tuple[int, int], ...] = ()


def format_input_segments(
    row: dict[str, Any],
    config: RegressionConfig,
    *,
    essay_surface: str | None = None,
) -> FormattedInputSegments:
    if config.input_format not in {
        "baseline_v1",
        "source_aware_v1",
        "rubric_conditioned_v1",
        "essay_only_v1",
    }:
        raise ValueError(f"지원하지 않는 input_format입니다: {config.input_format}")

    # 학습 중 뽑힌 지문 dropout 표시. `_TrainingSurfaceRow`만 이 속성을 갖고
    # validation/inference row에는 없으므로 평가 경로는 영향을 받지 않는다.
    if getattr(row, "drop_prompt", False) or config.input_format == "essay_only_v1":
        # 지문을 아예 읽지 않는다. row에 prompt가 없어도 동작해야 하므로
        # prompt_text(row)를 호출하지 않는다.
        essay = essay_for_input(row, config, essay_surface=essay_surface)
        base = ESSAY_ONLY_INPUT_TEMPLATE.format(essay=essay)
        start = len(base) - len(essay)
        if start < 0 or base[start:] != essay:
            raise ValueError("essay 원문이 regression 입력의 마지막 span과 다릅니다")
        return FormattedInputSegments(base, (start, start + len(essay)), len(base))

    source = str(row.get("source_dataset", row.get("_dataset_name", "competition")))
    if config.input_format == "rubric_conditioned_v1":
        template = (
            FULL_RUBRIC_INPUT_TEMPLATE
            if config.rubric_profile == "full_9criterion_1to5_v1"
            else INPUT_TEMPLATE
        )
    else:
        template = (
            NEUTRAL_INPUT_TEMPLATE
            if config.input_format == "source_aware_v1"
            and source not in {"competition", "nikl_grading", "nikl_competition"}
            else INPUT_TEMPLATE
        )

    essay = essay_for_input(row, config, essay_surface=essay_surface)
    base = template.format(prompt=prompt_text(row), essay=essay)
    essay_start = len(base) - len(essay)
    if config.input_format == "rubric_conditioned_v1" and not essay:
        raise ValueError("rubric_conditioned_v1은 빈 essay를 허용하지 않습니다")
    if essay_start < 0 or base[essay_start:] != essay:
        raise ValueError("essay 원문이 regression 입력의 마지막 span과 다릅니다")
    essay_span = (essay_start, essay_start + len(essay))
    if config.input_format != "rubric_conditioned_v1":
        return FormattedInputSegments(base, essay_span, len(base))

    suffix = RUBRIC_READOUT_HEADER
    anchor_spans: list[tuple[int, int]] = []
    for tag in RUBRIC_CRITERION_TAGS:
        anchor = f"[{tag} 판단]:"
        start = len(base) + len(suffix)
        suffix += anchor + "\n"
        anchor_spans.append((start, start + len(anchor)))
    return FormattedInputSegments(
        text=base + suffix,
        essay_span=essay_span,
        scoring_end=len(base),
        criterion_anchor_spans=tuple(anchor_spans),
    )


def format_input(
    row: dict[str, Any],
    config: RegressionConfig,
    *,
    essay_surface: str | None = None,
) -> str:
    return format_input_segments(row, config, essay_surface=essay_surface).text


PromptKey = tuple[str, str]


def question_key(row: dict[str, Any]) -> PromptKey:
    """문제 번호와 정규화한 원문을 함께 써서 추가 데이터의 Q1 충돌을 막는다."""

    number = str(row.get("prompt_num", "")).strip()
    normalized_prompt = " ".join(
        unicodedata.normalize("NFKC", prompt_text(row)).split()
    )
    return number, normalized_prompt


def build_prompt_registry(
    rows: Sequence[dict[str, Any]],
) -> tuple[PromptKey, ...]:
    """Train split에서만 deterministic 문제 registry를 만든다."""

    return tuple(sorted({question_key(row) for row in rows}))


def prompt_group_id(row: dict[str, Any]) -> int:
    """문항을 **지표 집계용** 정수로 바꾼다. registry와 무관하게 안정적이어야 한다.

    `prompt_index`는 학습되는 prompt head의 embedding 색인이라 registry 순서에
    묶여 있다. 여기 필요한 것은 그런 학습 대상이 아니라 "같은 문항끼리 묶는 열쇠"다.
    Qn 형식은 n을 그대로 쓰고(사람이 로그에서 바로 읽을 수 있다), 형식이 다르면
    문자열 해시를 쓴다. 어느 경우든 같은 문항은 항상 같은 값이 된다.
    """

    number = str(row.get("prompt_num", "")).strip()
    if number.startswith(("Q", "q")) and number[1:].isdigit():
        return int(number[1:])
    if number.isdigit():
        return int(number)
    if not number:
        return -1
    # 안정적인 해시. Python의 str.__hash__는 프로세스마다 달라서 못 쓴다.
    digest = hashlib.sha256(number.encode("utf-8")).hexdigest()[:8]
    return 1000 + int(digest, 16) % 100000


def parse_unseen_prompt_holdout(value: str) -> tuple[str, ...]:
    """`RegressionConfig.unseen_prompt_holdout` 문자열을 문항 목록으로 만든다."""

    text = (value or "").strip()
    if not text:
        return ()
    return tuple(item.strip() for item in text.split(",") if item.strip())


def exclude_unseen_prompt_rows(
    rows: Sequence[dict[str, Any]], holdout: Sequence[str]
) -> list[dict[str, Any]]:
    """학습 행에서 보류 문항을 제거한다. 빈 목록이면 입력을 그대로 돌려준다.

    validation은 건드리지 않는다. 보류 문항의 validation 에세이가 바로 "미학습
    문항" 평가이기 때문이다.
    """

    if not holdout:
        return list(rows)
    blocked = {str(item) for item in holdout}
    return [row for row in rows if str(row.get("prompt_num", "")) not in blocked]


def prompt_index(row: dict[str, Any], registry: Sequence[PromptKey]) -> int:
    return {key: index for index, key in enumerate(registry)}.get(question_key(row), -1)


def _as_token_ids(value: Any) -> list[int]:
    """Fast/slow tokenizer의 서로 다른 반환형을 작은 list[int]로 통일한다."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise TypeError(
            f"tokenizer input_ids가 list가 아닙니다: {type(value).__name__}"
        )
    return [int(token) for token in value]


def _as_offset_pairs(value: Any) -> list[tuple[int, int]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise TypeError("offset_mapping이 list가 아닙니다")
    pairs: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise TypeError("offset_mapping item은 (start, end)여야 합니다")
        pairs.append((int(item[0]), int(item[1])))
    return pairs


def _tokenize_rubric_conditioned(
    tokenizer: Any,
    segments: FormattedInputSegments,
    *,
    max_length: int,
) -> tuple[list[int], list[int], list[int]]:
    """Tokenize RC input without silent truncation and locate nine anchors.

    The shared representation pools only the base prompt/rubric/question/essay
    prefix.  The suffix remains in the same single causal forward so each
    criterion anchor can attend to that prefix.  Missing offsets, overflow, or
    any missing/reordered anchor is a hard error rather than a changed method.
    """

    # Tokenize the scorer-visible base and the readout suffix separately.  A
    # byte-level tokenizer can otherwise merge the essay's trailing spaces
    # with the suffix's first newline.  Keeping this explicit boundary makes
    # every base token id exactly the one used by the no-suffix control while
    # the concatenated suffix still participates in the same causal forward.
    base_text = segments.text[: segments.scoring_end]
    suffix_text = segments.text[segments.scoring_end :]
    try:
        base_encoded = tokenizer(
            base_text,
            add_special_tokens=True,
            truncation=False,
            return_offsets_mapping=True,
        )
        suffix_encoded = tokenizer(
            suffix_text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        base_ids = _as_token_ids(base_encoded["input_ids"])
        base_offsets = _as_offset_pairs(base_encoded["offset_mapping"])
        suffix_ids = _as_token_ids(suffix_encoded["input_ids"])
        suffix_offsets = _as_offset_pairs(suffix_encoded["offset_mapping"])
    except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
        raise ValueError(
            "rubric_conditioned_v1에는 문자 offset을 제공하는 fast tokenizer가 "
            "필요합니다"
        ) from exc
    if len(base_ids) != len(base_offsets) or len(suffix_ids) != len(suffix_offsets):
        raise ValueError("input_ids와 offset_mapping 길이가 다릅니다")
    if not base_ids or not suffix_ids:
        raise ValueError(
            "rubric base 또는 criterion readout suffix token이 비어 있습니다"
        )
    ids = base_ids + suffix_ids
    if len(ids) > max_length:
        raise ValueError(
            "rubric_conditioned_v1 입력이 max_length를 초과했습니다: "
            f"tokens={len(ids)}, max_length={max_length}; truncation은 허용하지 않습니다"
        )

    suffix_start = len(base_ids)
    shared_pooling_mask = [1] * suffix_start + [0] * len(suffix_ids)
    if not any(shared_pooling_mask):
        raise ValueError("shared scoring token mask가 비어 있습니다")

    anchor_positions: list[int] = []
    for anchor_start, anchor_end in segments.criterion_anchor_spans:
        relative_start = anchor_start - segments.scoring_end
        relative_end = anchor_end - segments.scoring_end
        overlapping = [
            suffix_start + index
            for index, (token_start, token_end) in enumerate(suffix_offsets)
            if token_end > relative_start
            and token_start < relative_end
            and token_end > token_start
        ]
        if not overlapping:
            raise ValueError(
                "criterion anchor token을 찾지 못했습니다: "
                f"span=({anchor_start}, {anchor_end})"
            )
        anchor_positions.append(max(overlapping))
    if len(anchor_positions) != len(RUBRIC_CRITERION_TAGS):
        raise ValueError("criterion anchor는 정확히 9개여야 합니다")
    if anchor_positions != sorted(set(anchor_positions)):
        raise ValueError("criterion anchor token 위치가 중복되거나 순서가 잘못됐습니다")
    if any(position < suffix_start for position in anchor_positions):
        raise ValueError("criterion anchor가 readout suffix 밖에 있습니다")
    return ids, shared_pooling_mask, anchor_positions


def _tokenize_with_essay_mask(
    tokenizer: Any,
    text: str,
    essay: str,
    *,
    max_length: int,
) -> tuple[list[int], list[int]]:
    """전체 입력은 그대로 인코딩하되 essay에 겹치는 token만 1로 표시한다.

    essay-only pooling이 rubric/prompt token을 평균에 섞지 않게 하는 작은 경계다.
    fast tokenizer에서는 문자 offset을 사용하고, offset을 지원하지 않는 tokenizer는
    essay 직전 prefix와 전체 입력의 공통 token prefix를 이용한다.
    """

    start = text.rfind(essay)
    if not essay or start < 0:
        raise ValueError("essay 원문을 regression 입력에서 찾지 못했습니다")
    end = start + len(essay)
    common_kwargs = {
        "add_special_tokens": True,
        "truncation": True,
        "max_length": max_length,
    }
    try:
        encoded = tokenizer(text, return_offsets_mapping=True, **common_kwargs)
        ids = _as_token_ids(encoded["input_ids"])
        offsets = encoded["offset_mapping"]
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        if (
            isinstance(offsets, list)
            and len(offsets) == 1
            and isinstance(offsets[0], list)
        ):
            offsets = offsets[0]
        mask = [
            int(int(offset_end) > start and int(offset_start) < end)
            for offset_start, offset_end in offsets
        ]
        if len(mask) == len(ids) and any(mask):
            return ids, mask
    except (KeyError, TypeError, ValueError, NotImplementedError):
        pass

    # Slow tokenizer fallback. 입력 template에서 essay가 맨 끝이므로 prefix와
    # 전체 tokenization이 갈라지는 첫 위치부터 essay로 간주하면 된다.
    ids = _as_token_ids(tokenizer(text, **common_kwargs)["input_ids"])
    prefix_ids = _as_token_ids(tokenizer(text[:start], **common_kwargs)["input_ids"])
    boundary = 0
    for left, right in zip(ids, prefix_ids):
        if left != right:
            break
        boundary += 1
    special_ids = {int(token) for token in getattr(tokenizer, "all_special_ids", [])}
    mask = [
        int(index >= boundary and token not in special_ids)
        for index, token in enumerate(ids)
    ]
    if not any(mask):
        raise ValueError(
            "essay token mask를 만들지 못했습니다. fast tokenizer 사용 여부와 "
            "max_length를 확인하세요"
        )
    return ids, mask


_SENTENCE_BOUNDARY_PATTERN = re.compile(
    r"(?:[.!?。！？]+[\"'”’)\]}>〉》」』]*|[\r\n\u2028\u2029]+)"
)
_STRICT_PARAGRAPH_GAP_PATTERN = re.compile(r"(?<=\S) {2,}(?=\S)")


def _sentence_end_offsets(essay: str) -> list[int]:
    """Return deterministic sentence ends available from either input surface.

    Official raw essays retain sentence punctuation even though paragraph
    whitespace is flattened. Canonical-only newlines are accepted for a
    train-time augmented view, but validation and inference never require them.
    """

    raw_ends = [match.end() for match in _SENTENCE_BOUNDARY_PATTERN.finditer(essay)]
    ends: list[int] = []
    for boundary_end in raw_ends:
        # ``문장.\n다음``에서 punctuation과 newline은 같은 경계다. 사이에
        # whitespace밖에 없으면 두 sentence ID를 만들지 않고 뒤쪽 끝으로 합친다.
        if ends and not essay[ends[-1] : boundary_end].strip():
            ends[-1] = boundary_end
        else:
            ends.append(boundary_end)
    if not ends or essay[ends[-1] :].strip():
        ends.append(len(essay))
    return ends


def _paragraph_boundary_candidate_offsets(official_raw: str) -> list[int]:
    """Return sentence/gap offsets visible in the one-paragraph test input.

    The final sentence end cannot separate two paragraphs, so it is excluded.
    Repeated ASCII spaces are added because the official concatenation retains
    this high-precision cue even though it removes canonical newlines.
    """

    offsets = set(_sentence_end_offsets(official_raw)[:-1])
    offsets.update(
        match.start() for match in _STRICT_PARAGRAPH_GAP_PATTERN.finditer(official_raw)
    )
    return sorted(offset for offset in offsets if 0 < offset < len(official_raw))


def _tokenize_with_paragraph_boundary_supervision(
    tokenizer: Any,
    segments: FormattedInputSegments,
    row: dict[str, Any],
    *,
    max_length: int,
) -> tuple[list[int], list[float], list[int], dict[str, int]]:
    """Align train-only paragraph labels to scorer-visible raw token positions.

    The score model still receives the exact ``official_raw`` text.  For every
    deployable sentence/gap candidate, the auxiliary decision is placed on the
    first following token.  A causal decoder can therefore see the preceding
    sentence and the beginning of the next one; no gold newline or marker is
    inserted into its input.
    """

    from .paragraph_boundary import gold_paragraph_boundaries

    essay_start, essay_end = segments.essay_span
    official_raw = segments.text[essay_start:essay_end]
    try:
        encoded = tokenizer(
            segments.text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        ids = _as_token_ids(encoded["input_ids"])
        offsets = _as_offset_pairs(encoded["offset_mapping"])
    except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
        raise ValueError(
            "paragraph boundary multi-task에는 문자 offset을 제공하는 fast "
            "tokenizer가 필요합니다"
        ) from exc
    if len(ids) != len(offsets):
        raise ValueError("input_ids와 paragraph offset_mapping 길이가 다릅니다")

    candidates = _paragraph_boundary_candidate_offsets(official_raw)
    gold = set(gold_paragraph_boundaries(row))
    labels = [0.0] * len(ids)
    mask = [0] * len(ids)
    aligned_candidates = 0
    aligned_positives = 0
    collisions = 0
    for candidate in candidates:
        next_character = candidate
        while (
            next_character < len(official_raw)
            and official_raw[next_character].isspace()
        ):
            next_character += 1
        if next_character >= len(official_raw):
            continue
        absolute = essay_start + next_character
        following = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > token_start
            and token_start <= absolute < token_end
            and token_start < essay_end
        ]
        if not following:
            # Byte-level tokenizers can leave a small offset gap. In that case
            # use the first later essay token rather than the preceding sentence.
            following = [
                index
                for index, (token_start, token_end) in enumerate(offsets)
                if token_end > token_start
                and token_start >= absolute
                and token_start < essay_end
            ]
        if not following:
            continue
        position = min(following)
        label = float(candidate in gold)
        if mask[position]:
            collisions += 1
            labels[position] = max(labels[position], label)
        else:
            mask[position] = 1
            labels[position] = label
        aligned_candidates += 1
        aligned_positives += int(label)

    return ids, labels, mask, {
        "gold_boundaries": len(gold),
        "candidate_offsets": len(candidates),
        "candidate_positives": len(gold.intersection(candidates)),
        "aligned_candidates": aligned_candidates,
        "aligned_positives": aligned_positives,
        "token_collisions": collisions,
        "unaligned_candidates": len(candidates) - aligned_candidates,
    }


def _paragraph_segment_ends(essay: str) -> list[int]:
    """Return paragraph ends that are visible in the official one-paragraph input.

    공식 JSONL은 문단을 구분자 없이 이어 붙이지만 원래 문단 사이의 이중 이상
    공백은 그대로 남는다. 이 cue는 heldout precision 1.0이고 essay의 약 41%에
    존재한다. cue가 없는 글은 한 문단으로 취급하므로 이 함수는 항상 문단이 하나
    이상이 되도록 essay 끝을 마지막 문단 끝으로 넣는다.
    """

    ends = [
        match.start()
        for match in _STRICT_PARAGRAPH_GAP_PATTERN.finditer(essay)
        if 0 < match.start() < len(essay)
    ]
    ends.append(len(essay))
    return ends


def _tokenize_with_segment_ids(
    tokenizer: Any,
    text: str,
    essay: str,
    *,
    max_length: int,
    segment_ends: list[int],
) -> tuple[list[int], list[int], list[int]]:
    """Map essay tokens to segment IDs without needing test-time paragraphs.

    ``segment_ends``는 essay 안의 문자 offset이며 마지막 원소는 essay 끝이다.
    Fast tokenizers use exact character offsets. The slow-tokenizer fallback
    keeps the existing essay mask and maps visible token ranks onto segment
    character spans. The fallback is approximate but deterministic and does
    not import canonical boundaries into a flat validation input.
    """

    start = text.rfind(essay)
    if not essay or start < 0:
        raise ValueError("essay 원문을 regression 입력에서 찾지 못했습니다")
    end = start + len(essay)
    common_kwargs = {
        "add_special_tokens": True,
        "truncation": True,
        "max_length": max_length,
    }
    try:
        encoded = tokenizer(text, return_offsets_mapping=True, **common_kwargs)
        ids = _as_token_ids(encoded["input_ids"])
        offsets = encoded["offset_mapping"]
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        if (
            isinstance(offsets, list)
            and len(offsets) == 1
            and isinstance(offsets[0], list)
        ):
            offsets = offsets[0]
        essay_mask = [
            int(int(offset_end) > start and int(offset_start) < end)
            for offset_start, offset_end in offsets
        ]
        if len(essay_mask) == len(ids) and any(essay_mask):
            segment_ids = []
            for is_essay, (offset_start, offset_end) in zip(
                essay_mask, offsets, strict=True
            ):
                if not is_essay:
                    segment_ids.append(-1)
                    continue
                overlap_start = max(int(offset_start), start) - start
                overlap_end = min(int(offset_end), end) - start
                midpoint = max(overlap_start, (overlap_start + overlap_end - 1) // 2)
                segment_ids.append(bisect_right(segment_ends, midpoint))
            return ids, essay_mask, segment_ids
    except (KeyError, TypeError, ValueError, NotImplementedError):
        pass

    ids, essay_mask = _tokenize_with_essay_mask(
        tokenizer, text, essay, max_length=max_length
    )
    essay_positions = [index for index, value in enumerate(essay_mask) if value]
    segment_ids = [-1] * len(ids)
    for rank, token_index in enumerate(essay_positions):
        midpoint = int((rank + 0.5) * len(essay) / len(essay_positions))
        segment_ids[token_index] = bisect_right(segment_ends, midpoint)
    return ids, essay_mask, segment_ids


def _tokenize_with_sentence_ids(
    tokenizer: Any,
    text: str,
    essay: str,
    *,
    max_length: int,
) -> tuple[list[int], list[int], list[int]]:
    """Segment essay tokens by deterministic sentence ends."""

    return _tokenize_with_segment_ids(
        tokenizer,
        text,
        essay,
        max_length=max_length,
        segment_ends=_sentence_end_offsets(essay),
    )


def _tokenize_with_paragraph_ids(
    tokenizer: Any,
    text: str,
    essay: str,
    *,
    max_length: int,
) -> tuple[list[int], list[int], list[int]]:
    """Segment essay tokens by the strict double-space paragraph cue."""

    return _tokenize_with_segment_ids(
        tokenizer,
        text,
        essay,
        max_length=max_length,
        segment_ends=_paragraph_segment_ends(essay),
    )


# Data loading, subset identity and leakage checks ---------------------------
def load_rows(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """대회 JSONL을 합치고 source tag를 붙인다."""

    rows: list[dict[str, Any]] = []
    for source_index, path in enumerate(paths):
        for row in read_jsonl(path):
            item = dict(row)
            item["_source_index"] = source_index
            item.setdefault("source_dataset", "competition")
            item["_dataset_name"] = "competition"
            rows.append(item)
    if not rows:
        raise ValueError("입력 데이터가 비어 있습니다")
    return rows


def stratified_training_subset(
    rows: Sequence[dict[str, Any]], limit: int, *, seed: int | str
) -> list[dict[str, Any]]:
    """Select a deterministic, approximately prompt-proportional train subset.

    ``--limit`` remains a quick smoke-test truncation.  Real data-size
    comparisons use this helper so row order in JSONL does not define the
    experiment and the same seed gives the same source pool.
    """

    if limit <= 0 or limit >= len(rows):
        return list(rows)

    groups: dict[PromptKey, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[question_key(row)].append(row)

    rng = random.Random(seed)
    ordered_groups = sorted(groups.items())
    for _, group_rows in ordered_groups:
        rng.shuffle(group_rows)

    # Largest-remainder allocation keeps each prompt close to its original
    # share while making the exact total equal to ``limit``.
    exact = [limit * len(group_rows) / len(rows) for _, group_rows in ordered_groups]
    quotas = [int(value) for value in exact]
    # AIHub처럼 답안 하나짜리 prompt가 많으면 remainder가 모두 같을 수 있다.
    # 그런 경우 prompt 문자열의 사전순 앞부분만 뽑히지 않도록 seed tie-break를 쓴다.
    tie_breakers = [rng.random() for _ in ordered_groups]
    remainder_order = sorted(
        range(len(quotas)),
        key=lambda index: (-(exact[index] - quotas[index]), tie_breakers[index]),
    )
    for index in remainder_order[: limit - sum(quotas)]:
        quotas[index] += 1

    selected = [
        row
        for (_, group_rows), quota in zip(ordered_groups, quotas, strict=True)
        for row in group_rows[:quota]
    ]
    rng.shuffle(selected)
    return selected


def split_validation_holdout(
    rows: Sequence[dict[str, Any]], size: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """학습 풀에서 추가 validation을 떼어내 ``(남은 학습, holdout)``으로 돌려준다.

    공식 validation 400편만으로는 arm 사이 차이의 짝지은 표준오차가 약 .005라 그보다
    작은 효과를 판정할 수 없다. holdout을 N편 붙이면 그 표준오차가 대략
    ``1/sqrt(1 + N/400)``배가 된다.

    두 가지를 지킨다. ``official_train`` 2,000편은 공식 validation과 같은 수집 절차를
    거친 행이므로 전부 학습에 남기고 ``origin_pool_extra``에서만 뽑는다. 그리고
    선택은 essay_hash 정렬만으로 결정해 seed와 무관하게 항상 같은 집합이 나오게 한다.
    같은 holdout을 쓰는 run끼리만 비교할 수 있기 때문이다.
    """

    if size <= 0:
        return list(rows), []

    pool = [row for row in rows if row.get("source_split") == "origin_pool_extra"]
    if len(pool) < size:
        raise ValueError(
            f"origin_pool_extra가 {len(pool)}편뿐이라 holdout {size}편을 뗄 수 없습니다"
        )

    # prompt 구성을 원 분포에 맞춰 뽑는다. 문항 난이도가 달라 한 문항만 몰리면
    # holdout이 학습 분포와도 공식 validation과도 어긋난다.
    groups: dict[PromptKey, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        groups[question_key(row)].append(row)
    ordered = sorted(groups.items())
    exact = [size * len(group) / len(pool) for _, group in ordered]
    quotas = [int(value) for value in exact]
    remainder_order = sorted(
        range(len(quotas)), key=lambda index: -(exact[index] - quotas[index])
    )
    for index in remainder_order[: size - sum(quotas)]:
        quotas[index] += 1

    chosen: set[int] = set()
    for (_, group), quota in zip(ordered, quotas, strict=True):
        for row in sorted(group, key=lambda item: str(item.get("essay_hash", "")))[:quota]:
            chosen.add(id(row))

    holdout = [row for row in rows if id(row) in chosen]
    remaining = [row for row in rows if id(row) not in chosen]
    return remaining, holdout


def rows_fingerprint(rows: Sequence[dict[str, Any]]) -> str:
    """Hash selected IDs, essays and labels for result/data identity."""

    records = []
    for row in rows:
        identities = tuple(sorted(row_identity_values(row)))
        records.append(
            (
                str(row.get("source_dataset", "competition")),
                identities,
                normalized_essay_hash(row),
                human_scores(row),
            )
        )
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: (item[0], item[1], item[2])):
        digest.update(
            json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def deployment_surface_fingerprints(
    rows: Sequence[dict[str, Any]], config: RegressionConfig
) -> dict[str, Any]:
    """Hash the exact essay bytes and exact formatted scorer inputs.

    ``rows_fingerprint`` deliberately normalizes whitespace for leakage checks.
    That is the wrong identity for a surface experiment, where one leading or
    repeated space can change tokenizer IDs.  These digests are length-prefixed
    UTF-8 and therefore distinguish every preserved space, tab, CR, newline and
    normalization form without putting full essays in run artifacts.
    """

    essay_digest = hashlib.sha256()
    input_digest = hashlib.sha256()

    def update(digest: Any, value: str) -> None:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)

    # Prepared validation rows and the official inference JSONL intentionally
    # use different schemas/provenance fields.  Hash only the scorer-visible
    # values plus the stable public essay ID so those two views compare equal.
    # The value tie-breakers also make duplicate IDs deterministic without
    # depending on input row order.
    records = [
        (essay_id(row), essay_for_input(row, config), format_input(row, config))
        for row in rows
    ]
    for identity, essay, formatted in sorted(records):
        for digest, value in (
            (essay_digest, essay),
            (input_digest, formatted),
        ):
            update(digest, identity)
            update(digest, value)

    return {
        "count": len(rows),
        "essay_surface": config.essay_surface,
        "input_format": config.input_format,
        "rubric_profile": config.rubric_profile,
        "criterion_readout": config.criterion_readout,
        "tokenization_policy": (
            RUBRIC_TOKENIZATION_POLICY
            if config.input_format == "rubric_conditioned_v1"
            else "single_text_v1"
        ),
        "essay_utf8_sha256": essay_digest.hexdigest(),
        "formatted_input_utf8_sha256": input_digest.hexdigest(),
    }


_IDENTITY_FIELDS = ("essay_id", "id", "document_id", "original_id")


def normalized_essay_hash(row: dict[str, Any]) -> str:
    """전처리기와 같은 NFC+공백 정규화로 essay leakage fingerprint를 만든다."""

    text = essay_text(row).replace("#@문장구분#", "")
    normalized = unicodedata.normalize("NFC", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def row_identity_values(row: dict[str, Any]) -> set[str]:
    """대회/외부 canonical이 사용하는 모든 문서 ID 표현을 반환한다."""

    return {
        str(row[field]).strip()
        for field in _IDENTITY_FIELDS
        if row.get(field) is not None and str(row[field]).strip()
    }


def validation_overlap_report(
    training_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    *,
    max_examples: int = 10,
) -> dict[str, Any]:
    """Train/validation의 ID 및 essay 본문 교집합을 fail-fast용으로 계산한다.

    source prefix가 붙은 canonical ``id``만 비교하면 NIKL의 원래 대회 ID를 놓칠 수
    있으므로 ``original_id``도 함께 본다. ID가 전혀 다른 복제본도 잡도록 정규화한
    essay 본문 SHA-256을 별도로 비교한다.
    """

    train_ids: set[str] = set()
    validation_ids: set[str] = set()
    for row in training_rows:
        train_ids.update(row_identity_values(row))
    for row in validation_rows:
        validation_ids.update(row_identity_values(row))
    train_hashes = {normalized_essay_hash(row) for row in training_rows}
    validation_hashes = {normalized_essay_hash(row) for row in validation_rows}
    shared_ids = sorted(train_ids & validation_ids)
    shared_hashes = sorted(train_hashes & validation_hashes)
    overlap_rows = sum(
        bool(row_identity_values(row) & validation_ids)
        or normalized_essay_hash(row) in validation_hashes
        for row in training_rows
    )
    return {
        "checked": True,
        "training_rows": len(training_rows),
        "validation_rows": len(validation_rows),
        "identifier_overlap_count": len(shared_ids),
        # Compatibility name used by older result aggregators. This counts
        # distinct ID strings, not rows; use overlap_training_row_count for rows.
        "id_overlap_count": len(shared_ids),
        "essay_text_overlap_count": len(shared_hashes),
        "overlap_training_row_count": overlap_rows,
        "has_overlap": bool(shared_ids or shared_hashes),
        "id_overlap_examples": shared_ids[:max_examples],
        "essay_text_hash_overlap_examples": shared_hashes[:max_examples],
        "normalization": "NFC+collapse_whitespace+SHA256",
    }


def parse_extended_dataset_names(value: str) -> tuple[str, ...]:
    """`a,b`와 `a b`를 모두 받아 중복 없는 alias 순서를 보존한다."""

    names: list[str] = []
    for item in value.replace(",", " ").split():
        if item not in names:
            names.append(item)
    return tuple(names)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset_profile(
    directory: str | Path,
    *,
    expected_dataset: str | None = None,
) -> dict[str, Any]:
    """Verify one processed profile and return a checkpoint-safe snapshot."""

    profile = Path(directory)
    manifest_path = profile / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"전처리 manifest가 없습니다: {manifest_path}; "
            "bash main_code/build_datasets.sh를 먼저 실행하세요"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"손상된 manifest입니다: {manifest_path}") from exc
    if manifest.get("schema_version") != 1:
        raise ValueError(f"현재 schema와 맞지 않는 manifest입니다: {manifest_path}")
    if expected_dataset is not None and manifest.get("dataset") != expected_dataset:
        raise ValueError(
            f"dataset profile 이름이 다릅니다: "
            f"{manifest.get('dataset')} != {expected_dataset}"
        )
    files: dict[str, Any] = {}
    for split, info in manifest.get("files", {}).items():
        if not isinstance(info, dict):
            raise ValueError(f"manifest files.{split}이 잘못됐습니다")
        path = profile / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"전처리 파일이 없습니다: {path}")
        size = path.stat().st_size
        sha256 = _file_sha256(path)
        if size != info.get("bytes") or sha256 != info.get("sha256"):
            raise ValueError(f"manifest와 다른 canonical 파일입니다: {path}")
        files[split] = {
            "path": str(path.resolve()),
            "rows": int(info.get("rows", 0)),
            "bytes": size,
            "sha256": sha256,
        }
    return {
        "schema_version": 1,
        "dataset": manifest.get("dataset"),
        "policy": manifest.get("policy"),
        "contains_validation_labels": bool(
            manifest.get("contains_validation_labels", False)
        ),
        "safe_for_model_selection": bool(
            manifest.get("safe_for_model_selection", True)
        ),
        "manifest_path": str(manifest_path.resolve()),
        "files": files,
    }


def validate_prepared_manifest(
    directory: str | Path,
    dataset_names: Sequence[str],
) -> dict[str, Any]:
    """Verify each selected external profile against its package-local manifest."""

    root = Path(directory)
    selected: dict[str, Any] = {}
    for name in dataset_names:
        if name not in EXTERNAL_DATASETS:
            raise ValueError(f"지원하지 않는 external dataset입니다: {name}")
        profile = root / EXTERNAL_DATASETS[name].name
        selected[name] = validate_dataset_profile(profile, expected_dataset=name)
    return {"schema_version": 1, "datasets": selected}


def load_prepared_extended_rows(
    directory: str | Path,
    dataset_names: Sequence[str],
    *,
    limit_per_dataset: int | None = None,
    sample_seed: int = 42,
    validate_manifest: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """전처리된 source별 JSONL을 기존 row schema로 읽는다.

    원본 AIHub/NIKL parsing은 학습 loop에 섞지 않는다. 각 파일은
    prepare_extended_data.py가 만든 canonical `<alias>.jsonl`이어야 한다.
    """

    root = Path(directory)
    if validate_manifest:
        validate_prepared_manifest(root, dataset_names)
    all_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for name in dataset_names:
        if name not in EXTERNAL_DATASETS:
            raise ValueError(f"지원하지 않는 external dataset입니다: {name}")
        path = root / EXTERNAL_DATASETS[name].name / "train.jsonl"
        if not path.is_file():
            raise FileNotFoundError(
                f"전처리 데이터가 없습니다: {path}; "
                "bash main_code/build_datasets.sh를 먼저 실행하세요"
            )
        rows = read_jsonl(path)
        if limit_per_dataset is not None:
            rows = stratified_training_subset(
                rows,
                limit_per_dataset,
                seed=f"{sample_seed}:{name}",
            )
        normalized: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["source_dataset"] = name
            item["_dataset_name"] = name
            scores = human_scores(item)
            if scores is None or any(
                not math.isfinite(score) or not 1.0 <= score <= 5.0
                for score in scores.values()
            ):
                raise ValueError(
                    f"{path}의 score는 finite 1~5 content/organization/expression이어야 합니다"
                )
            normalized.append(item)
        counts[name] = len(normalized)
        all_rows.extend(normalized)
    return all_rows, counts


def competition_steps_per_epoch(
    rows: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    gradient_accumulation: int,
    inbatch_sampling: str,
) -> int:
    """대회 데이터 한 pass에 필요한 optimizer step 수를 계산한다."""

    if inbatch_sampling == "same_question":
        group_sizes = Counter(question_key(row) for row in rows).values()
        physical_batches = sum(math.ceil(size / batch_size) for size in group_sizes)
        return math.ceil(physical_batches / gradient_accumulation)
    return math.ceil(len(rows) / (batch_size * gradient_accumulation))


# Dataset and train samplers --------------------------------------------------
class _TrainingSurfaceRow(dict[str, Any]):
    """A copied train row carrying the sampled essay view.

    ``drop_prompt``는 이 접근에서 지문을 빼기로 뽑혔다는 표시다. 같은 row를 다시
    읽으면 다시 뽑으므로 epoch마다 다른 view가 된다.
    """

    def __init__(
        self, row: dict[str, Any], *, essay_surface: str, drop_prompt: bool = False
    ):
        super().__init__(row)
        self.essay_surface = essay_surface
        self.drop_prompt = drop_prompt


def _row_with_rounded_labels(row: dict[str, Any]) -> dict[str, Any]:
    """train row의 trait label만 사사오입 정수로 바꾼 얕은 복사본을 만든다.

    2026-08-06 공지로 채점이 정수 기준이 됐으므로 train label도 정수로 맞추는 실험을
    할 수 있게 한다. 원본 row는 바꾸지 않고, validation/inference 경로는 이 함수를
    호출하지 않는다. ``human_score``가 있으면 그쪽이 우선 읽히므로 함께 갱신한다.
    """

    scores = human_scores(row)
    if scores is None:
        return row
    rounded = {
        trait: float(min(5.0, max(1.0, math.floor(scores[trait] + 0.5))))
        for trait in TRAITS
    }
    average = sum(rounded.values()) / len(TRAITS)
    item = dict(row)
    for key in ("score", "human_score"):
        if isinstance(row.get(key), dict):
            updated = dict(row[key])
            updated.update(rounded)
            updated["average"] = average
            item[key] = updated
    return item


class EssayRegressionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        config: RegressionConfig,
        *,
        split: str,
        require_labels: bool,
    ):
        self.rows = list(rows)
        self.config = config
        self.split = split
        self.require_labels = require_labels
        self.surface_view_counts: Counter[str] = Counter()
        if require_labels:
            for row in self.rows:
                if human_scores(row) is None:
                    raise ValueError(f"학습 점수가 없습니다: {essay_id(row)}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        probability = self.config.train_canonical_surface_probability
        if self.split != "train":
            return row
        if self.config.train_label_rounding == "integer":
            row = _row_with_rounded_labels(row)
        # 지문 dropout은 train split에서만, 접근마다 새로 뽑는다. p=0이면 아래 두
        # 분기가 예전과 완전히 같아 기존 run과 bit-exact 동일하다.
        dropout = float(getattr(self.config, "prompt_dropout_probability", 0.0))
        drop_prompt = dropout > 0 and random.random() < dropout
        if probability <= 0 and not drop_prompt:
            self.surface_view_counts[self.config.essay_surface] += 1
            return row

        # Repeated accesses can sample a different view while labels and IDs stay
        # unchanged.
        selected_surface = (
            "canonical" if random.random() < probability else self.config.essay_surface
        )
        self.surface_view_counts[selected_surface] += 1
        if drop_prompt:
            self.surface_view_counts["prompt_dropped"] += 1
        return _TrainingSurfaceRow(
            row,
            essay_surface=selected_surface,
            drop_prompt=drop_prompt,
        )


class SameQuestionSampler(Sampler[int]):
    """같은 문제의 답안이 한 physical batch에 오도록 index 순서를 만든다.

    pair를 미리 저장하지 않고 DataLoader가 묶는 순서만 바꾼다. 마지막 작은
    묶음은 같은 문제의 다른 답안을 조금 재사용해 채운다. 따라서 모든 batch는
    고정 크기이고, batch 안의 all-pairs를 그대로 비교할 수 있다.
    """

    def __init__(
        self,
        dataset: EssayRegressionDataset,
        *,
        batch_size: int,
        seed: int,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.groups: dict[PromptKey, list[int]] = defaultdict(list)
        for index, row in enumerate(dataset.rows):
            key = question_key(row)
            self.groups[key].append(index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return sum(
            ((len(indices) + self.batch_size - 1) // self.batch_size) * self.batch_size
            for indices in self.groups.values()
        )

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for indices in self.groups.values():
            shuffled = list(indices)
            rng.shuffle(shuffled)
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start : start + self.batch_size]
                # 보통 문제당 답안 수가 batch보다 훨씬 크다. 남는 칸에는 같은
                # 문제의 답안을 넣되 가능하면 현재 batch와 중복되지 않게 한다.
                while len(batch) < self.batch_size:
                    candidates = [index for index in shuffled if index not in batch]
                    batch.append(rng.choice(candidates or shuffled))
                batches.append(batch)
        rng.shuffle(batches)
        return iter(index for batch in batches for index in batch)


class MultiSourceStepSampler(Sampler[int]):
    """optimizer step 단위로 competition/확장 source를 배치한다.

    한 optimizer step에 속한 ``gradient_accumulation``개 physical batch는 모두
    같은 source에서 뽑는다. 따라서 alternating의 "1 step씩 교대" 의미가 GA>1에서도
    유지되고, same_question이면 각 physical batch 안의 prompt도 하나로 고정된다.
    """

    def __init__(
        self,
        dataset: EssayRegressionDataset,
        *,
        batch_size: int,
        gradient_accumulation: int,
        main_steps: int,
        final_competition_steps: int,
        schedule: str,
        competition_mix_ratio: float,
        competition_every_n_steps: int,
        extended_source_sampling: str,
        extended_source_weights: str,
        inbatch_sampling: str,
        seed: int,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.gradient_accumulation = int(gradient_accumulation)
        self.main_steps = int(main_steps)
        self.final_competition_steps = int(final_competition_steps)
        self.schedule = schedule
        self.competition_mix_ratio = float(competition_mix_ratio)
        self.competition_every_n_steps = int(competition_every_n_steps)
        self.extended_source_sampling = extended_source_sampling
        self.extended_source_weights = extended_source_weights
        self.inbatch_sampling = inbatch_sampling
        self.seed = int(seed)
        self.epoch = 0

        self.source_indices: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(dataset.rows):
            source = str(
                row.get("_dataset_name", row.get("source_dataset", "competition"))
            )
            self.source_indices[source].append(index)
        if not self.source_indices.get("competition"):
            raise ValueError("대회 train 데이터는 모든 schedule에 필수입니다")
        self.extended_sources = sorted(
            source for source in self.source_indices if source != "competition"
        )
        if schedule != "competition_only" and not self.extended_sources:
            raise ValueError("확장 데이터 schedule인데 사용할 external row가 없습니다")
        if schedule == "mixed" and (
            not 0 < self.competition_mix_ratio < 1 or self.main_steps < 2
        ):
            raise ValueError(
                "mixed schedule은 competition/external step이 모두 필요합니다"
            )
        if schedule == "alternating" and (
            self.competition_every_n_steps < 2
            or self.main_steps < self.competition_every_n_steps
        ):
            raise ValueError(
                "alternating main phase에는 external/competition step이 모두 필요합니다"
            )

    @property
    def total_steps(self) -> int:
        return self.main_steps + self.final_competition_steps

    def __len__(self) -> int:
        return self.total_steps * self.gradient_accumulation * self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _allocate(total: int, weights: dict[str, float]) -> dict[str, int]:
        """largest-remainder 방식으로 총 step 수를 정확히 나눈다."""

        if total <= 0:
            return {name: 0 for name in weights}
        denominator = sum(weights.values())
        if denominator <= 0:
            raise ValueError("source sampling weight 합은 양수여야 합니다")
        exact = {name: total * weight / denominator for name, weight in weights.items()}
        counts = {name: math.floor(value) for name, value in exact.items()}
        remainder = total - sum(counts.values())
        order = sorted(weights, key=lambda name: (-(exact[name] - counts[name]), name))
        for name in order[:remainder]:
            counts[name] += 1
        return counts

    def _extended_plan(self, steps: int, rng: random.Random) -> list[str]:
        if steps < len(self.extended_sources):
            raise ValueError(
                "선택한 모든 external dataset을 한 step 이상 쓰려면 external step 수가 "
                "dataset 수 이상이어야 합니다"
            )
        if self.extended_source_weights.strip():
            weights: dict[str, float] = {}
            for item in self.extended_source_weights.replace(",", " ").split():
                name, separator, raw_weight = item.partition("=")
                if not separator or name not in self.extended_sources:
                    raise ValueError(
                        "extended_source_weights는 선택한 alias=weight 목록이어야 합니다"
                    )
                weight = float(raw_weight)
                if weight <= 0:
                    raise ValueError("extended source weight는 양수여야 합니다")
                weights[name] = weight
            if set(weights) != set(self.extended_sources):
                raise ValueError(
                    "extended_source_weights에는 선택한 모든 external alias가 필요합니다"
                )
        else:
            weights = {
                source: (
                    float(len(self.source_indices[source]))
                    if self.extended_source_sampling == "proportional"
                    else 1.0
                )
                for source in self.extended_sources
            }
        counts = self._allocate(steps, weights)
        unused = sorted(name for name, count in counts.items() if count == 0)
        if unused:
            raise ValueError(
                f"source weight가 너무 작아 학습 step이 0인 dataset이 있습니다: {unused}"
            )
        plan = [source for source, count in counts.items() for _ in range(count)]
        rng.shuffle(plan)
        return plan

    def source_plan(self, *, epoch: int = 0) -> list[str]:
        """학습 전에도 run.json에 저장할 수 있는 deterministic source plan."""

        rng = random.Random(self.seed + int(epoch))
        if self.schedule == "competition_only":
            main = ["competition"] * self.main_steps
        elif self.schedule == "external_only":
            main = self._extended_plan(self.main_steps, rng)
        elif self.schedule == "mixed":
            competition_steps = round(self.main_steps * self.competition_mix_ratio)
            # ratio 반올림으로 어느 한쪽이 0 step이 되지 않게 최소 한 step씩 둔다.
            competition_steps = min(self.main_steps - 1, max(1, competition_steps))
            main = ["competition"] * competition_steps
            main += self._extended_plan(self.main_steps - competition_steps, rng)
            rng.shuffle(main)
        elif self.schedule == "alternating":
            extension_count = sum(
                (step + 1) % self.competition_every_n_steps != 0
                for step in range(self.main_steps)
            )
            extension_plan = iter(self._extended_plan(extension_count, rng))
            main = [
                (
                    "competition"
                    if (step + 1) % self.competition_every_n_steps == 0
                    else next(extension_plan)
                )
                for step in range(self.main_steps)
            ]
        elif self.schedule == "pretrain_then_competition":
            main = self._extended_plan(self.main_steps, rng)
        else:
            raise ValueError(f"지원하지 않는 dataset schedule: {self.schedule}")
        return main + ["competition"] * self.final_competition_steps

    def planned_summary(self) -> dict[str, Any]:
        steps = Counter(self.source_plan())
        return {
            "main_steps": self.main_steps,
            "final_competition_steps": self.final_competition_steps,
            "total_steps": self.total_steps,
            "source_steps": dict(sorted(steps.items())),
            "source_examples": {
                source: count * self.batch_size * self.gradient_accumulation
                for source, count in sorted(steps.items())
            },
        }

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        random_pools: dict[str, list[int]] = {}
        question_batches: dict[str, list[list[int]]] = {}

        def random_batch(source: str) -> list[int]:
            pool = random_pools.setdefault(source, [])
            result: list[int] = []
            while len(result) < self.batch_size:
                if not pool:
                    pool.extend(self.source_indices[source])
                    rng.shuffle(pool)
                needed = self.batch_size - len(result)
                result.extend(pool[:needed])
                del pool[:needed]
            return result

        def refill_question_batches(source: str) -> None:
            groups: dict[PromptKey, list[int]] = defaultdict(list)
            for index in self.source_indices[source]:
                groups[question_key(self.dataset.rows[index])].append(index)
            batches: list[list[int]] = []
            for indices in groups.values():
                shuffled = list(indices)
                rng.shuffle(shuffled)
                for start in range(0, len(shuffled), self.batch_size):
                    batch = shuffled[start : start + self.batch_size]
                    while len(batch) < self.batch_size:
                        # 작은 prompt group도 허용한다. 중복은 padding 부분에만 생긴다.
                        batch.append(rng.choice(shuffled))
                    batches.append(batch)
            rng.shuffle(batches)
            question_batches[source] = batches

        def next_batch(source: str) -> list[int]:
            if self.inbatch_sampling != "same_question":
                return random_batch(source)
            if not question_batches.get(source):
                refill_question_batches(source)
            return question_batches[source].pop()

        indices: list[int] = []
        for source in self.source_plan(epoch=self.epoch):
            for _ in range(self.gradient_accumulation):
                indices.extend(next_batch(source))
        return iter(indices)


# Batch collation -------------------------------------------------------------
def _collation_essay_surface(row: dict[str, Any], config: RegressionConfig) -> str:
    """Resolve one already-sampled train view; eval/inference use config surface."""

    selected = (
        row.essay_surface
        if isinstance(row, _TrainingSurfaceRow)
        else config.essay_surface
    )
    if selected not in ESSAY_SURFACES:
        raise ValueError(f"지원하지 않는 학습 essay surface입니다: {selected}")
    return str(selected)


def effective_max_length(tokenizer: Any, requested: int) -> int:
    """Clamp a requested length to a real tokenizer limit.

    Hugging Face uses enormous sentinel values for tokenizers without a known
    limit; those are deliberately ignored rather than clamping to the sentinel.
    """

    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
        return min(int(requested), tokenizer_limit)
    return int(requested)


class RegressionCollator:
    def __init__(
        self,
        tokenizer: Any,
        config: RegressionConfig,
        *,
        include_labels: bool,
        include_metadata: bool = True,
        max_length: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.config = config
        requested = config.max_length if max_length is None else max_length
        self.max_length = effective_max_length(tokenizer, requested)
        self.include_labels = include_labels
        self.include_metadata = include_metadata
        self.prompt_lookup = {
            key: index for index, key in enumerate(config.prompt_registry)
        }
        self.include_detail_labels = include_labels and (
            getattr(config, "detail_head_mode", "none") != "none"
            or float(getattr(config, "detail_rater_loss_weight", 0.0)) > 0
        )
        self.include_detail_raters = (
            self.include_detail_labels
            and float(getattr(config, "detail_rater_loss_weight", 0.0)) > 0
        )
        self.include_official_rater_set = (
            self.include_detail_labels
            and getattr(config, "detail_head_mode", "none") in RATER_SET_HEAD_MODES
            and float(getattr(config, "detail_rater_set_loss_weight", 0.0)) > 0
        )
        registry = tuple(getattr(config, "detail_rater_registry", ()))
        try:
            self.detail_rater_lookup = {
                (str(source), str(evaluator_id)): index
                for index, (source, evaluator_id) in enumerate(registry)
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "detail_rater_registry item은 (source_dataset, evaluator_id)여야 합니다"
            ) from exc
        if len(self.detail_rater_lookup) != len(registry):
            raise ValueError("detail_rater_registry에는 중복 평가자가 없어야 합니다")
        if self.include_detail_raters and not self.detail_rater_lookup:
            raise ValueError(
                "평가자 auxiliary에는 train에서 만든 detail_rater_registry가 필요합니다"
            )
        self.include_paragraph_boundaries = (
            include_labels and config.paragraph_boundary_loss_weight > 0
        )
        self._paragraph_boundary_row_audits: dict[str, dict[str, int]] = {}
        self._observed_input_lengths: list[int] = []

    def input_length_summary(self) -> dict[str, Any]:
        """Return compact observed token-length evidence for run artifacts."""

        values = sorted(self._observed_input_lengths)
        fail_closed = self.config.input_format == "rubric_conditioned_v1"
        if not values:
            return {
                "count": 0,
                "min": None,
                "p50": None,
                "p95": None,
                "p99": None,
                "max": None,
                "max_length": self.max_length,
                "overflow_count": 0 if fail_closed else None,
                "truncation_policy": (
                    "fail_closed" if fail_closed else "tokenizer_truncate"
                ),
                "tokenization_policy": (
                    RUBRIC_TOKENIZATION_POLICY if fail_closed else "single_text_v1"
                ),
            }

        def percentile(fraction: float) -> int:
            index = max(0, math.ceil(fraction * len(values)) - 1)
            return int(values[min(index, len(values) - 1)])

        return {
            "count": len(values),
            "min": int(values[0]),
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": int(values[-1]),
            "max_length": self.max_length,
            "overflow_count": 0 if fail_closed else None,
            "truncation_policy": (
                "fail_closed" if fail_closed else "tokenizer_truncate"
            ),
            "tokenization_policy": (
                RUBRIC_TOKENIZATION_POLICY if fail_closed else "single_text_v1"
            ),
        }

    def paragraph_boundary_supervision_summary(self) -> dict[str, int] | None:
        """Summarize unique rows aligned by the joint auxiliary collator."""

        if not self.include_paragraph_boundaries:
            return None
        totals: Counter[str] = Counter()
        for audit in self._paragraph_boundary_row_audits.values():
            totals.update(audit)
        return {
            "unique_rows": len(self._paragraph_boundary_row_audits),
            **{key: int(value) for key, value in sorted(totals.items())},
        }

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        essay_surfaces = [_collation_essay_surface(row, self.config) for row in rows]
        segments = [
            format_input_segments(row, self.config, essay_surface=essay_surface)
            for row, essay_surface in zip(rows, essay_surfaces, strict=True)
        ]
        texts = [item.text for item in segments]
        use_rubric_conditioning = self.config.input_format == "rubric_conditioned_v1"
        use_essay_mask = (
            self.config.pooling in {"essay_mean", "essay_attention"}
            or self.config.organization_pooling != "shared"
        )
        if use_rubric_conditioning:
            examples = []
            shared_pooling_masks: list[list[int]] = []
            anchor_positions_rows: list[list[int]] = []
            for item in segments:
                ids, shared_pooling_mask, anchor_positions = (
                    _tokenize_rubric_conditioned(
                        self.tokenizer,
                        item,
                        max_length=self.max_length,
                    )
                )
                examples.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
                shared_pooling_masks.append(shared_pooling_mask)
                anchor_positions_rows.append(anchor_positions)
                self._observed_input_lengths.append(len(ids))
            encoded = self.tokenizer.pad(examples, padding=True, return_tensors="pt")
            padded_length = int(encoded["input_ids"].shape[1])
            padding_side = getattr(self.tokenizer, "padding_side", "right")
            padded_shared_pooling_masks: list[list[int]] = []
            padded_anchor_positions: list[list[int]] = []
            for shared_pooling_mask, anchor_positions in zip(
                shared_pooling_masks, anchor_positions_rows, strict=True
            ):
                padding_length = padded_length - len(shared_pooling_mask)
                padding = [0] * padding_length
                padded_shared_pooling_masks.append(
                    padding + shared_pooling_mask
                    if padding_side == "left"
                    else shared_pooling_mask + padding
                )
                position_offset = padding_length if padding_side == "left" else 0
                padded_anchor_positions.append(
                    [position + position_offset for position in anchor_positions]
                )
        elif self.include_paragraph_boundaries:
            examples = []
            paragraph_boundary_labels: list[list[float]] = []
            paragraph_boundary_masks: list[list[int]] = []
            for row, item in zip(rows, segments, strict=True):
                has_train_boundary_labels = str(row.get("source_split") or "") in {
                    "official_train",
                    "origin_pool_extra",
                }
                if has_train_boundary_labels:
                    ids, boundary_labels, boundary_mask, audit = (
                        _tokenize_with_paragraph_boundary_supervision(
                            self.tokenizer,
                            item,
                            row,
                            max_length=self.max_length,
                        )
                    )
                else:
                    ids = _as_token_ids(
                        self.tokenizer(
                            item.text,
                            add_special_tokens=True,
                            truncation=True,
                            max_length=self.max_length,
                        )["input_ids"]
                    )
                    boundary_labels = [0.0] * len(ids)
                    boundary_mask = [0] * len(ids)
                    audit = None
                examples.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
                paragraph_boundary_labels.append(boundary_labels)
                paragraph_boundary_masks.append(boundary_mask)
                if audit is not None:
                    row_key = essay_id(row)
                    previous = self._paragraph_boundary_row_audits.setdefault(
                        row_key, audit
                    )
                    if previous != audit:
                        raise ValueError(
                            "같은 essay의 paragraph supervision이 달라졌습니다: "
                            f"{row_key}"
                        )
            encoded = self.tokenizer.pad(examples, padding=True, return_tensors="pt")
            padded_length = int(encoded["input_ids"].shape[1])
            padding_side = getattr(self.tokenizer, "padding_side", "right")
            padded_paragraph_boundary_labels: list[list[float]] = []
            padded_paragraph_boundary_masks: list[list[int]] = []
            for boundary_labels, boundary_mask in zip(
                paragraph_boundary_labels, paragraph_boundary_masks, strict=True
            ):
                padding_length = padded_length - len(boundary_labels)
                label_padding = [0.0] * padding_length
                mask_padding = [0] * padding_length
                padded_paragraph_boundary_labels.append(
                    label_padding + boundary_labels
                    if padding_side == "left"
                    else boundary_labels + label_padding
                )
                padded_paragraph_boundary_masks.append(
                    mask_padding + boundary_mask
                    if padding_side == "left"
                    else boundary_mask + mask_padding
                )
        elif use_essay_mask:
            examples: list[dict[str, list[int]]] = []
            essay_masks: list[list[int]] = []
            # sentence_transition과 paragraph_mean은 같은 segment-ID 채널을 쓰지만
            # 서로 배타적인 organization_pooling 값이므로 batch key만 다르게 낸다.
            segment_ids_rows: list[list[int]] = []
            use_sentence_ids = self.config.organization_pooling == "sentence_transition"
            use_paragraph_ids = self.config.organization_pooling == "paragraph_mean"
            for row, text, essay_surface in zip(
                rows, texts, essay_surfaces, strict=True
            ):
                input_essay = essay_for_input(
                    row, self.config, essay_surface=essay_surface
                )
                if use_sentence_ids:
                    ids, essay_mask, segment_ids = _tokenize_with_sentence_ids(
                        self.tokenizer,
                        text,
                        input_essay,
                        max_length=self.max_length,
                    )
                    segment_ids_rows.append(segment_ids)
                elif use_paragraph_ids:
                    ids, essay_mask, segment_ids = _tokenize_with_paragraph_ids(
                        self.tokenizer,
                        text,
                        input_essay,
                        max_length=self.max_length,
                    )
                    segment_ids_rows.append(segment_ids)
                else:
                    ids, essay_mask = _tokenize_with_essay_mask(
                        self.tokenizer,
                        text,
                        input_essay,
                        max_length=self.max_length,
                    )
                examples.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
                essay_masks.append(essay_mask)
            encoded = self.tokenizer.pad(examples, padding=True, return_tensors="pt")
            padded_length = int(encoded["input_ids"].shape[1])
            padding_side = getattr(self.tokenizer, "padding_side", "right")
            padded_essay_masks = []
            padded_segment_ids = []
            for row_index, essay_mask in enumerate(essay_masks):
                padding = [0] * (padded_length - len(essay_mask))
                padded_essay_masks.append(
                    padding + essay_mask
                    if padding_side == "left"
                    else essay_mask + padding
                )
                if use_sentence_ids or use_paragraph_ids:
                    segment_ids = segment_ids_rows[row_index]
                    segment_padding = [-1] * (padded_length - len(segment_ids))
                    padded_segment_ids.append(
                        segment_padding + segment_ids
                        if padding_side == "left"
                        else segment_ids + segment_padding
                    )
        else:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        if not use_rubric_conditioning:
            self._observed_input_lengths.extend(
                int(length) for length in encoded["attention_mask"].sum(dim=1).tolist()
            )
        batch: dict[str, Any] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        # BERT/RoBERTa 계열 tokenizer가 segment ids를 만들면 버리지 않는다.
        # decoder tokenizer에는 보통 이 key가 없어 기존 batch와 동일하다.
        # prompt_ids는 출력용 metadata가 아니라 문제별 head routing용 모델 입력이다.
        # Train에 없던 문제는 -1로 두고 모델에서 공용 head만 사용한다.
        if self.config.prompt_head_mode != "none":
            batch["prompt_ids"] = torch.tensor(
                [self.prompt_lookup.get(question_key(row), -1) for row in rows],
                dtype=torch.long,
            )
        if "token_type_ids" in encoded:
            batch["token_type_ids"] = encoded["token_type_ids"]
        if use_rubric_conditioning:
            batch["shared_pooling_mask"] = torch.tensor(
                padded_shared_pooling_masks, dtype=torch.long
            )
            batch["criterion_anchor_positions"] = torch.tensor(
                padded_anchor_positions, dtype=torch.long
            )
        if use_essay_mask and not use_rubric_conditioning:
            batch["essay_mask"] = torch.tensor(padded_essay_masks, dtype=torch.long)
            if self.config.organization_pooling == "sentence_transition":
                batch["sentence_ids"] = torch.tensor(
                    padded_segment_ids, dtype=torch.long
                )
            elif self.config.organization_pooling == "paragraph_mean":
                batch["paragraph_ids"] = torch.tensor(
                    padded_segment_ids, dtype=torch.long
                )
        if self.include_paragraph_boundaries:
            batch["paragraph_boundary_labels"] = torch.tensor(
                padded_paragraph_boundary_labels, dtype=torch.float32
            )
            batch["paragraph_boundary_mask"] = torch.tensor(
                padded_paragraph_boundary_masks, dtype=torch.bool
            )
        if self.include_metadata:
            batch["essay_ids"] = [essay_id(row) for row in rows]
            batch["rows"] = rows
        if self.include_labels:
            label_rows: list[list[float]] = []
            average_rows: list[float] = []
            for row in rows:
                scores = human_scores(row)
                if scores is None:
                    raise ValueError(f"점수가 없습니다: {essay_id(row)}")
                label_rows.append([scores[trait] for trait in TRAITS])
                # Prepared validation can opt in to the exact public raw
                # ``score.average`` without replacing the processed row's
                # training labels.  This private field is injected only by
                # train.py's fail-closed validation overlay.
                stored_average = row.get(
                    "_official_metric_average", official_average_score(row)
                )
                average_rows.append(
                    float(stored_average)
                    if stored_average is not None
                    else float(sum(scores[trait] for trait in TRAITS) / len(TRAITS))
                )
            batch["labels"] = torch.tensor(
                label_rows,
                # Trainer detaches this input tensor as label_ids before the
                # model casts labels to float32 for its loss.  Keep canonical
                # competition decimals (and any genuinely
                # continuous external labels) in float64 so checkpoint
                # Spearman uses the same tie groups as final inference.
                dtype=torch.float64,
            )
            # Trainer checkpoint selection must see the same unrounded stored
            # score.average as final inference.  The scorer accepts this as a
            # metric-only label and deliberately does not use it in the loss.
            batch["average_labels"] = torch.tensor(
                average_rows, dtype=torch.float64
            )
            # 문항별 지표(prompt macro / worst / unseen)를 checkpoint 선택에 쓰기
            # 위한 **지표 전용** 입력이다. loss는 이 값을 보지 않는다.
            # `label_names`에 등록되어 eval_prediction.label_ids로 전달된다.
            batch["prompt_group_ids"] = torch.tensor(
                [prompt_group_id(row) for row in rows], dtype=torch.int64
            )
        if self.include_detail_labels:
            criterion_rows = [_criterion_supervision(row) for row in rows]
            batch["criterion_scores"] = torch.tensor(
                [item[0] for item in criterion_rows], dtype=torch.float32
            )
            batch["criterion_mask"] = torch.tensor(
                [item[1] for item in criterion_rows], dtype=torch.bool
            )
            batch["criterion_distributions"] = torch.tensor(
                [item[2] for item in criterion_rows], dtype=torch.float32
            )
            batch["criterion_distribution_mask"] = torch.tensor(
                [item[3] for item in criterion_rows], dtype=torch.bool
            )

        if self.include_official_rater_set:
            official_rater_rows = [_official_rater_set_supervision(row) for row in rows]
            batch["official_rater_labels"] = torch.tensor(
                [item[0] for item in official_rater_rows], dtype=torch.long
            )
            batch["official_rater_mask"] = torch.tensor(
                [item[1] for item in official_rater_rows], dtype=torch.bool
            )

        if self.include_detail_raters:
            rater_rows = [
                _rater_supervision(row, self.detail_rater_lookup) for row in rows
            ]
            max_raters = max((len(item[0]) for item in rater_rows), default=0)
            rater_ids = torch.full(
                (len(rows), max_raters), DETAIL_PAD_RATER_ID, dtype=torch.long
            )
            rater_labels = torch.full(
                (len(rows), max_raters, len(DETAIL_CRITERIA)),
                DETAIL_MISSING_CLASS,
                dtype=torch.long,
            )
            rater_mask = torch.zeros(
                (len(rows), max_raters, len(DETAIL_CRITERIA)), dtype=torch.bool
            )
            for row_index, (ids, labels, masks) in enumerate(rater_rows):
                rater_count = len(ids)
                if not rater_count:
                    continue
                rater_ids[row_index, :rater_count] = torch.tensor(ids, dtype=torch.long)
                rater_labels[row_index, :rater_count] = torch.tensor(
                    labels, dtype=torch.long
                )
                rater_mask[row_index, :rater_count] = torch.tensor(
                    masks, dtype=torch.bool
                )
            batch["detail_rater_ids"] = rater_ids
            batch["detail_rater_labels"] = rater_labels
            batch["detail_rater_mask"] = rater_mask
        return batch
