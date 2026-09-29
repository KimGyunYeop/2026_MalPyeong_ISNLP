from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import torch
from torch import nn


SPLITS = ("train", "calibration", "heldout")
PROMPTS = tuple(f"Q{index}" for index in range(1, 10))
STRICT_TWO_SPACE = re.compile(r"(?<=\S) {2,}(?=\S)")
AUDITED_TRAIN_SHA256 = (
    "0583815b6cc22b1189c84851f103fd230266af0ed97716456da592b832f75736"
)
AUDITED_FULL_COUNTS = {
    "rows": 11600,
    "gold_seams": 32308,
    "candidates": 185435,
    "candidate_positives": 31383,
    "candidate_negatives": 154052,
}
SEAM_COUNT_BINS = 8
SEQUENCE_ATTENTION_HEADS = 4
SEQUENCE_TRANSFORMER_LAYERS = 2
POSITIONAL_ROLE_NAMES = ("first", "middle", "last")


@dataclass(frozen=True)
class BoundaryPoint:
    offset: int
    is_kiwi_end: bool
    is_strict_two_space: bool
    label: int


@dataclass(frozen=True)
class BoundaryExample:
    row_id: str
    document_id: str
    prompt_num: str
    year: str
    source_split: str
    official_raw: str
    gold_boundaries: tuple[int, ...]
    candidates: tuple[BoundaryPoint, ...]


@dataclass(frozen=True)
class CandidateRecord:
    example_index: int
    candidate_index: int
    point: BoundaryPoint


@dataclass
class ClassifierTrainingResult:
    model: "TinyBoundaryClassifier"
    history: list[dict[str, float | int]]
    best_epoch: int
    best_calibration_ap: float


@dataclass
class SequenceTrainingResult:
    model: "SequenceBoundaryClassifier"
    history: list[dict[str, float | int]]
    best_epoch: int
    best_calibration_count_decode_f1: float


def file_sha256(path: Any) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collapsed_whitespace_with_raw_map(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace while retaining each normalized character's raw end."""

    characters: list[str] = []
    raw_ends: list[int] = []
    index = 0
    while index < len(text):
        if text[index].isspace():
            end = index + 1
            while end < len(text) and text[end].isspace():
                end += 1
            if characters and end < len(text):
                characters.append(" ")
                raw_ends.append(end)
            index = end
            continue
        characters.append(text[index])
        raw_ends.append(index + 1)
        index += 1
    return "".join(characters), raw_ends


def gold_paragraph_boundaries(row: dict[str, Any]) -> tuple[int, ...]:
    """Map canonical paragraph seams back onto the byte-preserved visible string.

    The prepared canonical essay contains ``\n\n`` only from source
    ``paragraph[].form`` boundaries.  Matching whitespace-collapsed paragraph
    text against official_raw recovers the offset immediately after the last
    visible character of each left paragraph.  No validation row or score is
    involved.
    """

    canonical = row.get("essay")
    surfaces = row.get("essay_surfaces")
    official_raw = surfaces.get("official_raw") if isinstance(surfaces, dict) else None
    if not isinstance(canonical, str) or not isinstance(official_raw, str):
        raise ValueError(
            "paragraph label에는 canonical essay와 official_raw가 필요합니다"
        )

    normalized_raw, raw_ends = _collapsed_whitespace_with_raw_map(official_raw)
    if normalized_raw != " ".join(canonical.split()):
        raise ValueError(
            f"canonical/official_raw text가 일치하지 않습니다: {row.get('id')}"
        )

    paragraphs = canonical.split("\n\n")
    boundaries: list[int] = []
    search_start = 0
    for index, paragraph in enumerate(paragraphs):
        normalized_paragraph = " ".join(paragraph.split())
        if not normalized_paragraph:
            raise ValueError(f"빈 canonical paragraph가 있습니다: {row.get('id')}")
        found = normalized_raw.find(normalized_paragraph, search_start)
        if found < 0:
            raise ValueError(
                f"canonical paragraph를 official_raw에 정렬할 수 없습니다: "
                f"{row.get('id')} paragraph={index}"
            )
        paragraph_end = found + len(normalized_paragraph)
        if index < len(paragraphs) - 1:
            boundaries.append(raw_ends[paragraph_end - 1])
        search_start = paragraph_end

    recorded_count = row.get("metadata", {}).get("paragraph_count")
    if recorded_count not in (None, "") and int(recorded_count) != len(boundaries) + 1:
        raise ValueError(
            f"paragraph_count와 canonical seam 수가 다릅니다: {row.get('id')}"
        )
    return tuple(boundaries)


def coalesce_sentence_spans(
    official_raw: str, spans: Iterable[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    """Validate Kiwi spans and union its rare overlapping top-level spans."""

    merged: list[tuple[int, int]] = []
    for start, end in spans:
        start, end = int(start), int(end)
        if not 0 <= start < end <= len(official_raw):
            raise ValueError(
                "문장 span이 원문 범위를 벗어났습니다: "
                f"start={start}, end={end}, essay_length={len(official_raw)}"
            )
        if merged and start < merged[-1][0]:
            raise ValueError("문장 span 순서가 역전되었습니다")
        if merged and start < merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def kiwi_sentence_spans(official_raw: str) -> tuple[tuple[int, int], ...]:
    """Run the package-local Kiwi 0.23.2 splitter on one deployment string."""

    from .datasets import _get_kiwi_sentence_splitter

    raw_spans = (
        (int(sentence.start), int(sentence.end))
        for sentence in _get_kiwi_sentence_splitter().split_into_sents(
            official_raw, return_sub_sents=False
        )
    )
    return coalesce_sentence_spans(official_raw, raw_spans)


def boundary_candidates(
    official_raw: str,
    *,
    sentence_spans: Sequence[tuple[int, int]] | None = None,
) -> tuple[tuple[int, bool, bool], ...]:
    """Return online-visible candidate offsets and their two source flags."""

    spans = (
        kiwi_sentence_spans(official_raw)
        if sentence_spans is None
        else coalesce_sentence_spans(official_raw, sentence_spans)
    )
    flags: dict[int, list[bool]] = {}
    for _, end in spans[:-1]:
        flags.setdefault(end, [False, False])[0] = True
    for match in STRICT_TWO_SPACE.finditer(official_raw):
        flags.setdefault(match.start(), [False, False])[1] = True
    return tuple(
        (offset, source_flags[0], source_flags[1])
        for offset, source_flags in sorted(flags.items())
    )


def build_boundary_examples(
    rows: Sequence[dict[str, Any]],
    *,
    sentence_span_fn: Callable[[str], Sequence[tuple[int, int]]] = kiwi_sentence_spans,
) -> list[BoundaryExample]:
    examples: list[BoundaryExample] = []
    for row in rows:
        if row.get("source_dataset") != "nikl_competition":
            raise ValueError("boundary parser는 competition train row만 사용합니다")
        if row.get("dataset_group") != "competition":
            raise ValueError("boundary parser에 external dataset을 섞을 수 없습니다")
        surfaces = row.get("essay_surfaces")
        official_raw = (
            surfaces.get("official_raw") if isinstance(surfaces, dict) else None
        )
        if not isinstance(official_raw, str) or not official_raw:
            raise ValueError(f"official_raw가 없습니다: {row.get('id')}")
        gold = gold_paragraph_boundaries(row)
        gold_set = set(gold)
        candidates = tuple(
            BoundaryPoint(
                offset=offset,
                is_kiwi_end=is_kiwi,
                is_strict_two_space=is_two_space,
                label=int(offset in gold_set),
            )
            for offset, is_kiwi, is_two_space in boundary_candidates(
                official_raw, sentence_spans=sentence_span_fn(official_raw)
            )
        )
        examples.append(
            BoundaryExample(
                row_id=str(row["id"]),
                document_id=str(row.get("document_id") or row["id"]),
                prompt_num=str(row.get("prompt_num") or ""),
                year=str(row.get("metadata", {}).get("year") or ""),
                source_split=str(row.get("source_split") or ""),
                official_raw=official_raw,
                gold_boundaries=gold,
                candidates=candidates,
            )
        )
    return examples


def flatten_candidates(examples: Sequence[BoundaryExample]) -> list[CandidateRecord]:
    return [
        CandidateRecord(example_index, candidate_index, point)
        for example_index, example in enumerate(examples)
        for candidate_index, point in enumerate(example.candidates)
    ]


def candidate_audit(examples: Sequence[BoundaryExample]) -> dict[str, int | float]:
    gold = sum(len(example.gold_boundaries) for example in examples)
    candidate_count = sum(len(example.candidates) for example in examples)
    positives = sum(point.label for example in examples for point in example.candidates)
    return {
        "rows": len(examples),
        "gold_seams": gold,
        "candidates": candidate_count,
        "candidate_positives": positives,
        "candidate_negatives": candidate_count - positives,
        "rows_without_candidates": sum(not example.candidates for example in examples),
        "candidate_recall_ceiling": positives / gold if gold else 0.0,
    }


def validate_audited_full_counts(
    examples: Sequence[BoundaryExample], dataset_sha256: str
) -> None:
    if dataset_sha256 != AUDITED_TRAIN_SHA256:
        return
    audit = candidate_audit(examples)
    mismatches = {
        name: (audit[name], expected)
        for name, expected in AUDITED_FULL_COUNTS.items()
        if audit[name] != expected
    }
    if mismatches:
        raise ValueError(f"audited boundary count가 바뀌었습니다: {mismatches}")


def _stable_document_key(document_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{document_id}".encode()).hexdigest()


def normalized_essay_identity(row: dict[str, Any]) -> str:
    """Hash normalized official_raw, conservatively ignoring the prompt identity."""

    surfaces = row.get("essay_surfaces")
    official_raw = surfaces.get("official_raw") if isinstance(surfaces, dict) else None
    if not isinstance(official_raw, str):
        raise ValueError(f"essay identity에 official_raw가 필요합니다: {row.get('id')}")

    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFC", official_raw)).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deterministic_document_split(
    rows: Sequence[dict[str, Any]], seed: int
) -> tuple[dict[str, str], dict[str, Any]]:
    """Split connected document/input-identity groups without train/dev leakage."""

    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_by_document: dict[str, int] = {}
    first_by_identity: dict[str, int] = {}
    rows_by_identity: dict[str, list[int]] = defaultdict(list)
    identities: list[str] = []
    document_ids: list[str] = []
    for index, row in enumerate(rows):
        document_id = str(row.get("document_id") or row.get("id") or "")
        if not document_id:
            raise ValueError("split row에 document/id가 없습니다")
        identity = normalized_essay_identity(row)
        document_ids.append(document_id)
        identities.append(identity)
        rows_by_identity[identity].append(index)
        if document_id in first_by_document:
            union(index, first_by_document[document_id])
        else:
            first_by_document[document_id] = index
        if identity in first_by_identity:
            union(index, first_by_identity[identity])
        else:
            first_by_identity[identity] = index

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[find(index)].append(index)

    component_records: list[dict[str, Any]] = []
    mixed_stratum_components: list[list[str]] = []
    for indices in components.values():
        stratum_frequency: dict[tuple[str, str], int] = defaultdict(int)
        for index in indices:
            row = rows[index]
            stratum_frequency[
                (
                    str(row.get("metadata", {}).get("year") or ""),
                    str(row.get("prompt_num") or ""),
                )
            ] += 1
        stratum = sorted(
            stratum_frequency,
            key=lambda item: (-stratum_frequency[item], item),
        )[0]
        documents = sorted({document_ids[index] for index in indices})
        if len(stratum_frequency) > 1:
            mixed_stratum_components.append(documents)
        stable_identity = hashlib.sha256(
            "\0".join(
                documents + sorted({identities[index] for index in indices})
            ).encode()
        ).hexdigest()
        component_records.append(
            {
                "indices": indices,
                "documents": documents,
                "stratum": stratum,
                "stable_identity": stable_identity,
            }
        )

    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for component in component_records:
        strata[component["stratum"]].append(component)

    split_by_document: dict[str, str] = {}
    split_by_row: dict[int, str] = {}
    stratum_counts: dict[str, dict[str, int]] = {}
    for stratum, stratum_components in sorted(strata.items()):
        ordered = sorted(
            stratum_components,
            key=lambda component: hashlib.sha256(
                f"{seed}\0{component['stable_identity']}".encode()
            ).hexdigest(),
        )
        stratum_rows = sum(len(component["indices"]) for component in ordered)
        if len(ordered) < 3:
            raise ValueError(
                f"80/10/10 split에는 stratum당 identity group 3개가 필요합니다: {stratum}"
            )
        targets = {
            "heldout": max(1, round(stratum_rows * 0.10)),
            "calibration": max(1, round(stratum_rows * 0.10)),
        }
        assigned_rows = {split: 0 for split in SPLITS}
        for component in ordered:
            if assigned_rows["heldout"] < targets["heldout"]:
                split = "heldout"
            elif assigned_rows["calibration"] < targets["calibration"]:
                split = "calibration"
            else:
                split = "train"
            for index in component["indices"]:
                split_by_row[index] = split
            for document_id in component["documents"]:
                previous = split_by_document.setdefault(document_id, split)
                if previous != split:
                    raise RuntimeError(f"document split이 갈라졌습니다: {document_id}")
            assigned_rows[split] += len(component["indices"])
        stratum_counts["|".join(stratum)] = assigned_rows

    split_ids = {
        split: sorted(
            document_id
            for document_id, assigned in split_by_document.items()
            if assigned == split
        )
        for split in SPLITS
    }
    source_split_counts = {
        split: dict(
            sorted(
                {
                    source: sum(
                        split_by_row[index] == split
                        and str(rows[index].get("source_split") or "") == source
                        for index in range(len(rows))
                    )
                    for source in {str(row.get("source_split") or "") for row in rows}
                }.items()
            )
        )
        for split in SPLITS
    }
    identity_splits: dict[str, set[str]] = defaultdict(set)
    for index, identity in enumerate(identities):
        identity_splits[identity].add(split_by_row[index])
    duplicate_identity_groups = [
        identity for identity, indices in rows_by_identity.items() if len(indices) > 1
    ]
    cross_source_duplicate_groups = sum(
        len(
            {
                str(rows[index].get("source_split") or "")
                for index in rows_by_identity[identity]
            }
        )
        > 1
        for identity in duplicate_identity_groups
    )
    manifest = {
        "schema_version": 1,
        "strategy": (
            "document_or_normalized_official_raw_identity_group_"
            "prompt_year_stratified_80_10_10"
        ),
        "seed": seed,
        "group_fields": [
            "document_id",
            "normalized_NFC_whitespace_collapsed(official_raw)",
        ],
        "stratum_fields": ["metadata.year", "prompt_num"],
        "source_split_role": "audit_only",
        "counts": {split: len(split_ids[split]) for split in SPLITS},
        "row_counts": {
            split: sum(assigned == split for assigned in split_by_row.values())
            for split in SPLITS
        },
        "stratum_counts": stratum_counts,
        "source_split_counts": source_split_counts,
        "identity_group_count": len(first_by_identity),
        "connected_component_count": len(component_records),
        "duplicate_identity_group_count": len(duplicate_identity_groups),
        "cross_source_duplicate_identity_group_count": cross_source_duplicate_groups,
        "normalized_essay_identity_cross_split_count": sum(
            len(splits) > 1 for splits in identity_splits.values()
        ),
        "mixed_prompt_year_component_count": len(mixed_stratum_components),
        "mixed_prompt_year_component_documents": mixed_stratum_components,
        "document_ids": split_ids,
    }
    return split_by_document, manifest


def select_smoke_rows(
    rows: Sequence[dict[str, Any]],
    split_by_document: dict[str, str],
    rows_per_split: int,
    seed: int,
) -> list[dict[str, Any]]:
    if rows_per_split <= 0:
        return list(rows)
    selected_documents: set[str] = set()
    for split in SPLITS:
        candidates = {
            str(row.get("document_id") or row["id"])
            for row in rows
            if split_by_document[str(row.get("document_id") or row["id"])] == split
        }
        selected_documents.update(
            sorted(candidates, key=lambda item: _stable_document_key(item, seed))[
                :rows_per_split
            ]
        )
    return [
        row
        for row in rows
        if str(row.get("document_id") or row["id"]) in selected_documents
    ]


SURFACE_FEATURE_NAMES = (
    "is_kiwi_end",
    "is_strict_two_space",
    "relative_character_position",
    "relative_candidate_index",
    "log_essay_characters",
    "log_candidate_count",
    "previous_candidate_distance_fraction",
    "next_candidate_distance_fraction",
    "log_right_whitespace_run",
    "right_gap_has_tab",
    "right_gap_has_lf",
    "right_gap_has_cr",
    "previous_is_period",
    "previous_is_question",
    "previous_is_exclamation",
    "previous_is_closing_quote",
    "previous_terminal_before_quote",
    "next_is_digit",
    "next_is_quote",
    "next_is_enumeration_cue",
    "next_is_contrast_cue",
    "next_is_addition_cue",
    "next_is_conclusion_cue",
    "left_context_has_question",
    *(f"prompt_{prompt.lower()}" for prompt in PROMPTS),
)


def _right_whitespace(raw: str, offset: int) -> str:
    end = offset
    while end < len(raw) and raw[end].isspace():
        end += 1
    return raw[offset:end]


def _surface_features_for_point(
    example: BoundaryExample, candidate_index: int, point: BoundaryPoint
) -> list[float]:
    raw = example.official_raw
    offset = point.offset
    offsets = [candidate.offset for candidate in example.candidates]
    previous_offset = offsets[candidate_index - 1] if candidate_index else 0
    next_offset = (
        offsets[candidate_index + 1] if candidate_index + 1 < len(offsets) else len(raw)
    )
    left = raw[:offset].rstrip()
    right = raw[offset:].lstrip()
    previous = left[-1:] or ""
    next_character = right[:1]
    right_gap = _right_whitespace(raw, offset)
    next_prefix = right[:24].replace(" ", "")
    quote_characters = "'\"‘’“”「」『』()[]{}"
    terminal_characters = ".?!。？！"
    before_quote = left[-2:-1] if previous in quote_characters else ""
    denominator = max(1, len(raw))
    candidate_denominator = max(1, len(offsets) - 1)
    values = [
        float(point.is_kiwi_end),
        float(point.is_strict_two_space),
        offset / denominator,
        candidate_index / candidate_denominator,
        math.log1p(len(raw)),
        math.log1p(len(offsets)),
        (offset - previous_offset) / denominator,
        (next_offset - offset) / denominator,
        math.log1p(len(right_gap)),
        float("\t" in right_gap),
        float("\n" in right_gap),
        float("\r" in right_gap),
        float(previous in ".。"),
        float(previous in "?？"),
        float(previous in "!！"),
        float(previous in quote_characters),
        float(before_quote in terminal_characters),
        float(next_character.isdigit()),
        float(next_character in quote_characters),
        float(
            next_prefix.startswith(
                ("첫째", "둘째", "셋째", "넷째", "먼저", "다음으로", "마지막으로")
            )
        ),
        float(next_prefix.startswith(("그러나", "하지만", "반면", "그렇지만"))),
        float(next_prefix.startswith(("또한", "그리고", "한편", "더불어"))),
        float(
            next_prefix.startswith(
                ("따라서", "그러므로", "결론적으로", "이처럼", "결국", "요컨대")
            )
        ),
        float(any(character in left[-80:] for character in "?？")),
    ]
    values.extend(float(example.prompt_num == prompt) for prompt in PROMPTS)
    return values


def surface_feature_matrix(
    examples: Sequence[BoundaryExample], records: Sequence[CandidateRecord]
) -> torch.Tensor:
    matrix = [
        _surface_features_for_point(
            examples[record.example_index], record.candidate_index, record.point
        )
        for record in records
    ]
    if not matrix:
        raise ValueError("boundary candidate가 없습니다")
    result = torch.tensor(matrix, dtype=torch.float32)
    if result.shape[1] != len(SURFACE_FEATURE_NAMES):
        raise RuntimeError("surface feature 이름과 값의 수가 다릅니다")
    return result


SEQUENCE_SURFACE_FEATURE_NAMES = tuple(
    name for name in SURFACE_FEATURE_NAMES if not name.startswith("prompt_")
)


def sequence_surface_feature_matrix(
    examples: Sequence[BoundaryExample], records: Sequence[CandidateRecord]
) -> torch.Tensor:
    """Keep deployment-visible structure features, but remove prompt identity.

    The sequence detector should learn paragraph layout rather than memorize the
    nine training prompts.  D98/D99 continue to use ``surface_feature_matrix``
    unchanged.
    """

    all_features = surface_feature_matrix(examples, records)
    retained = [
        index
        for index, name in enumerate(SURFACE_FEATURE_NAMES)
        if not name.startswith("prompt_")
    ]
    result = all_features[:, retained]
    if result.shape[1] != len(SEQUENCE_SURFACE_FEATURE_NAMES):
        raise RuntimeError("sequence surface feature 이름과 값의 수가 다릅니다")
    return result


def records_for_split(
    examples: Sequence[BoundaryExample],
    records: Sequence[CandidateRecord],
    split_by_document: dict[str, str],
    split: str,
) -> list[int]:
    return [
        index
        for index, record in enumerate(records)
        if split_by_document[examples[record.example_index].document_id] == split
    ]


def examples_for_split(
    examples: Sequence[BoundaryExample],
    split_by_document: dict[str, str],
    split: str,
) -> set[int]:
    return {
        index
        for index, example in enumerate(examples)
        if split_by_document[example.document_id] == split
    }


def average_precision(
    probabilities: Sequence[float], labels: Sequence[int], denominator: int
) -> float:
    """Threshold-integrated precision with tied scores handled as one group."""

    if denominator <= 0 or not probabilities:
        return 0.0
    ordered = sorted(
        zip(probabilities, labels, strict=True), key=lambda item: item[0], reverse=True
    )
    area = 0.0
    true_positives = 0
    predicted = 0
    index = 0
    while index < len(ordered):
        score = ordered[index][0]
        group_positives = 0
        group_count = 0
        while index < len(ordered) and ordered[index][0] == score:
            group_positives += int(ordered[index][1])
            group_count += 1
            index += 1
        true_positives += group_positives
        predicted += group_count
        area += (group_positives / denominator) * (true_positives / predicted)
    return area


def choose_calibration_threshold(
    probabilities: Sequence[float], labels: Sequence[int], total_gold_seams: int
) -> tuple[float, float]:
    """Choose one global threshold by end-to-end seam F1 on calibration only."""

    if not probabilities or total_gold_seams <= 0:
        raise ValueError("threshold calibration에는 candidate와 gold seam이 필요합니다")
    ordered = sorted(
        zip(probabilities, labels, strict=True), key=lambda item: item[0], reverse=True
    )
    best_threshold = math.nextafter(ordered[0][0], math.inf)
    best_f1 = 0.0
    true_positives = 0
    predicted = 0
    index = 0
    while index < len(ordered):
        threshold = ordered[index][0]
        while index < len(ordered) and ordered[index][0] == threshold:
            true_positives += int(ordered[index][1])
            predicted += 1
            index += 1
        f1 = 2 * true_positives / (predicted + total_gold_seams)
        # Descending iteration deliberately retains the higher threshold on ties.
        if f1 > best_f1 + 1e-12:
            best_f1 = f1
            best_threshold = threshold
    return float(best_threshold), float(best_f1)


def _metrics_without_prompt_macro(
    examples: Sequence[BoundaryExample],
    records: Sequence[CandidateRecord],
    probabilities: Sequence[float],
    threshold: float,
    example_indices: Sequence[int] | None = None,
    predicted_offsets_by_example: dict[int, set[int]] | None = None,
) -> dict[str, float | int]:
    if len(records) != len(probabilities):
        raise ValueError("candidate와 probability 수가 다릅니다")
    if example_indices is None:
        selected_example_indices = sorted({record.example_index for record in records})
    else:
        selected_example_indices = sorted(set(example_indices))
    labels = [record.point.label for record in records]
    gold_count = sum(
        len(examples[index].gold_boundaries) for index in selected_example_indices
    )
    candidate_positive_count = sum(labels)
    predictions = (
        [
            record.point.offset
            in predicted_offsets_by_example.get(record.example_index, set())
            for record in records
        ]
        if predicted_offsets_by_example is not None
        else [probability >= threshold for probability in probabilities]
    )
    true_positive = sum(
        int(predicted and label)
        for predicted, label in zip(predictions, labels, strict=True)
    )
    predicted_count = sum(predictions)
    false_positive = predicted_count - true_positive
    false_negative = gold_count - true_positive
    precision = true_positive / predicted_count if predicted_count else 0.0
    recall = true_positive / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    predicted_by_example: dict[int, set[int]] = defaultdict(set)
    for record, predicted in zip(records, predictions, strict=True):
        if predicted:
            predicted_by_example[record.example_index].add(record.point.offset)
    count_errors: list[int] = []
    exact_sets = 0
    for example_index in selected_example_indices:
        gold = set(examples[example_index].gold_boundaries)
        predicted = predicted_by_example[example_index]
        count_errors.append(abs(len(predicted) - len(gold)))
        exact_sets += int(predicted == gold)

    candidate_ap = average_precision(probabilities, labels, candidate_positive_count)
    end_to_end_ap = average_precision(probabilities, labels, gold_count)
    essay_count = len(selected_example_indices)
    return {
        "essay_count": essay_count,
        "candidate_count": len(records),
        "gold_seam_count": gold_count,
        "candidate_positive_count": candidate_positive_count,
        "candidate_recall_ceiling": (
            candidate_positive_count / gold_count if gold_count else 0.0
        ),
        "candidate_average_precision": candidate_ap,
        "end_to_end_average_precision": end_to_end_ap,
        "predicted_boundary_count": predicted_count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "paragraph_count_mae": (
            sum(count_errors) / essay_count if essay_count else 0.0
        ),
        "paragraph_count_exact_accuracy": (
            sum(error == 0 for error in count_errors) / essay_count
            if essay_count
            else 0.0
        ),
        "exact_boundary_set_accuracy": exact_sets / essay_count if essay_count else 0.0,
    }


def boundary_metrics(
    examples: Sequence[BoundaryExample],
    records: Sequence[CandidateRecord],
    probabilities: Sequence[float],
    threshold: float,
    *,
    example_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    overall = _metrics_without_prompt_macro(
        examples, records, probabilities, threshold, example_indices
    )
    prompt_values: dict[str, dict[str, float | int]] = {}
    selected_examples = (
        sorted({record.example_index for record in records})
        if example_indices is None
        else sorted(set(example_indices))
    )
    for prompt in sorted({examples[index].prompt_num for index in selected_examples}):
        indices = [
            index
            for index, record in enumerate(records)
            if examples[record.example_index].prompt_num == prompt
        ]
        prompt_records = [records[index] for index in indices]
        prompt_probabilities = [probabilities[index] for index in indices]
        prompt_example_indices = [
            index for index in selected_examples if examples[index].prompt_num == prompt
        ]
        prompt_values[prompt] = _metrics_without_prompt_macro(
            examples,
            prompt_records,
            prompt_probabilities,
            threshold,
            prompt_example_indices,
        )

    macro_names = (
        "candidate_recall_ceiling",
        "candidate_average_precision",
        "end_to_end_average_precision",
        "precision",
        "recall",
        "f1",
        "paragraph_count_mae",
        "paragraph_count_exact_accuracy",
        "exact_boundary_set_accuracy",
    )
    prompt_macro = {
        name: (
            sum(float(values[name]) for values in prompt_values.values())
            / len(prompt_values)
            if prompt_values
            else 0.0
        )
        for name in macro_names
    }
    return {
        "threshold": threshold,
        "overall": overall,
        "prompt_macro": prompt_macro,
        "per_prompt": prompt_values,
    }


def anchor_count_top_k_predictions(
    examples: Sequence[BoundaryExample],
    records: Sequence[CandidateRecord],
    probabilities: Sequence[float],
    predicted_seam_counts: dict[int, int],
    *,
    example_indices: Sequence[int],
    force_strict_two_space_anchors: bool,
) -> dict[int, set[int]]:
    """Decode a document count, optionally retaining train-audited anchors."""

    if len(records) != len(probabilities):
        raise ValueError("candidate와 probability 수가 다릅니다")
    candidates_by_example: dict[int, list[tuple[CandidateRecord, float]]] = defaultdict(
        list
    )
    for record, probability in zip(records, probabilities, strict=True):
        candidates_by_example[record.example_index].append((record, probability))

    predictions: dict[int, set[int]] = {}
    for example_index in example_indices:
        candidates = candidates_by_example.get(example_index, [])
        anchors = (
            {
                record.point.offset
                for record, _ in candidates
                if record.point.is_strict_two_space
            }
            if force_strict_two_space_anchors
            else set()
        )
        requested = max(0, int(predicted_seam_counts.get(example_index, 0)))
        target_count = min(len(candidates), max(requested, len(anchors)))
        ranked = sorted(
            (
                (probability, record.point.offset)
                for record, probability in candidates
                if record.point.offset not in anchors
            ),
            key=lambda item: (-item[0], item[1]),
        )
        selected = set(anchors)
        selected.update(
            offset for _, offset in ranked[: max(0, target_count - len(selected))]
        )
        predictions[example_index] = selected
    return predictions


def anchor_count_top_k_metrics(
    examples: Sequence[BoundaryExample],
    records: Sequence[CandidateRecord],
    probabilities: Sequence[float],
    predicted_seam_counts: dict[int, int],
    *,
    example_indices: Sequence[int],
    force_strict_two_space_anchors: bool,
) -> dict[str, Any]:
    """Measure the ordinal-count top-k decoder on the normal gold denominator."""

    selected_examples = sorted(set(example_indices))
    predictions = anchor_count_top_k_predictions(
        examples,
        records,
        probabilities,
        predicted_seam_counts,
        example_indices=selected_examples,
        force_strict_two_space_anchors=force_strict_two_space_anchors,
    )
    overall = _metrics_without_prompt_macro(
        examples,
        records,
        probabilities,
        0.5,
        selected_examples,
        predictions,
    )
    prompt_values: dict[str, dict[str, float | int]] = {}
    for prompt in sorted({examples[index].prompt_num for index in selected_examples}):
        positions = [
            position
            for position, record in enumerate(records)
            if examples[record.example_index].prompt_num == prompt
        ]
        prompt_examples = [
            index for index in selected_examples if examples[index].prompt_num == prompt
        ]
        prompt_values[prompt] = _metrics_without_prompt_macro(
            examples,
            [records[position] for position in positions],
            [probabilities[position] for position in positions],
            0.5,
            prompt_examples,
            predictions,
        )

    macro_names = (
        "candidate_recall_ceiling",
        "candidate_average_precision",
        "end_to_end_average_precision",
        "precision",
        "recall",
        "f1",
        "paragraph_count_mae",
        "paragraph_count_exact_accuracy",
        "exact_boundary_set_accuracy",
    )
    prompt_macro = {
        name: (
            sum(float(values[name]) for values in prompt_values.values())
            / len(prompt_values)
            if prompt_values
            else 0.0
        )
        for name in macro_names
    }
    ordinal_count_errors = [
        abs(
            int(predicted_seam_counts.get(index, 0))
            - min(len(examples[index].gold_boundaries), SEAM_COUNT_BINS - 1)
        )
        for index in selected_examples
    ]
    return {
        "decoder": (
            "strict_two_space_anchor_then_boundary_probability_top_k"
            if force_strict_two_space_anchors
            else "ordinal_count_then_boundary_probability_top_k"
        ),
        "count_bins": [*range(SEAM_COUNT_BINS - 1), f"{SEAM_COUNT_BINS - 1}+"],
        "overall": overall,
        "prompt_macro": prompt_macro,
        "per_prompt": prompt_values,
        "ordinal_count_bin_mae": (
            sum(ordinal_count_errors) / len(ordinal_count_errors)
            if ordinal_count_errors
            else 0.0
        ),
        "ordinal_count_bin_accuracy": (
            sum(error == 0 for error in ordinal_count_errors)
            / len(ordinal_count_errors)
            if ordinal_count_errors
            else 0.0
        ),
    }


def training_strict_anchor_policy(
    records: Sequence[CandidateRecord], train_record_indices: Sequence[int]
) -> dict[str, int | float | bool | str | None]:
    """Choose anchor forcing from train labels alone.

    Calibration and heldout labels must not influence this deployment policy.
    An absent anchor type is not evidence of perfect precision, so it remains off.
    """

    anchors = [
        records[index]
        for index in train_record_indices
        if records[index].point.is_strict_two_space
    ]
    positives = sum(record.point.label for record in anchors)
    total = len(anchors)
    force = total > 0 and positives == total
    return {
        "selected_on": "train",
        "strict_two_space_candidate_count": total,
        "strict_two_space_positive_count": positives,
        "strict_two_space_negative_count": total - positives,
        "strict_two_space_precision": positives / total if total else None,
        "force_strict_two_space_anchors": force,
        "selection_rule": "force_only_when_train_count>0_and_precision==1",
    }


def deterministic_boundary_baselines(
    examples: Sequence[BoundaryExample],
    records: Sequence[CandidateRecord],
    *,
    example_indices: Sequence[int],
) -> dict[str, Any]:
    """Evaluate three label-free candidate rules on the identical denominator."""

    probability_sets = {
        "all_candidates": [1.0 for _ in records],
        "kiwi_only": [float(record.point.is_kiwi_end) for record in records],
        "strict_two_space_only": [
            float(record.point.is_strict_two_space) for record in records
        ],
    }
    return {
        name: boundary_metrics(
            examples,
            records,
            probabilities,
            0.5,
            example_indices=example_indices,
        )
        for name, probabilities in probability_sets.items()
    }


class TinyBoundaryClassifier(nn.Module):
    def __init__(
        self,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.register_buffer("feature_mean", feature_mean.float())
        self.register_buffer("feature_scale", feature_scale.float())
        input_size = int(feature_mean.numel())
        if hidden_size > 0:
            self.classifier = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )
        else:
            self.classifier = nn.Linear(input_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features.float() - self.feature_mean) / self.feature_scale
        return self.classifier(normalized).squeeze(-1)


def candidate_paragraph_role(
    example: BoundaryExample, point: BoundaryPoint
) -> int:
    """Return first/middle/last for a candidate's left sentence, or -100.

    The source data provides paragraph boundaries, not discourse-role labels.
    Consequently this is deliberately only a positional auxiliary label.  One-
    and two-paragraph essays are masked because first/middle/last is ambiguous.
    """

    if not point.is_kiwi_end:
        return -100
    previous_kiwi_end = max(
        (
            candidate.offset
            for candidate in example.candidates
            if candidate.is_kiwi_end and candidate.offset < point.offset
        ),
        default=0,
    )
    if any(
        previous_kiwi_end < seam < point.offset
        for seam in example.gold_boundaries
    ):
        # Kiwi occasionally merges text across a gold paragraph seam.  Such a
        # sentence has no unambiguous single paragraph role.
        return -100
    paragraph_count = len(example.gold_boundaries) + 1
    if paragraph_count < 3:
        return -100
    paragraph_index = bisect_left(example.gold_boundaries, point.offset)
    if paragraph_index == 0:
        return 0
    if paragraph_index == paragraph_count - 1:
        return 2
    return 1


def positional_role_supervision_audit(
    examples: Sequence[BoundaryExample], records: Sequence[CandidateRecord]
) -> dict[str, Any]:
    """Count usable weak positional-role labels without evaluating predictions."""

    labels = [
        candidate_paragraph_role(examples[record.example_index], record.point)
        for record in records
    ]
    class_counts = {
        name: sum(label == index for label in labels)
        for index, name in enumerate(POSITIONAL_ROLE_NAMES)
    }
    valid_count = sum(label >= 0 for label in labels)
    return {
        "candidate_count": len(records),
        "valid_count": valid_count,
        "masked_count": len(records) - valid_count,
        "class_counts": class_counts,
        "mask_contract": (
            "paragraph_count<3, non-Kiwi candidate, or Kiwi sentence crossing "
            "a gold paragraph seam"
        ),
        "label_semantics": "positional first/middle/last, not gold discourse roles",
    }


def positional_role_metrics(
    examples: Sequence[BoundaryExample],
    records: Sequence[CandidateRecord],
    predictions: Sequence[int | None],
) -> dict[str, Any]:
    """Evaluate the optional role head only on its predeclared valid labels."""

    if len(records) != len(predictions):
        raise ValueError("role prediction과 candidate 수가 다릅니다")
    gold_and_predictions = []
    for record, prediction in zip(records, predictions, strict=True):
        label = candidate_paragraph_role(
            examples[record.example_index], record.point
        )
        if label < 0:
            continue
        if prediction is None:
            raise ValueError("role head가 켜졌지만 role prediction이 없습니다")
        gold_and_predictions.append((label, int(prediction)))

    per_class: dict[str, dict[str, float | int]] = {}
    for class_index, name in enumerate(POSITIONAL_ROLE_NAMES):
        true_positive = sum(
            gold == class_index and prediction == class_index
            for gold, prediction in gold_and_predictions
        )
        false_positive = sum(
            gold != class_index and prediction == class_index
            for gold, prediction in gold_and_predictions
        )
        false_negative = sum(
            gold == class_index and prediction != class_index
            for gold, prediction in gold_and_predictions
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[name] = {
            "support": sum(gold == class_index for gold, _ in gold_and_predictions),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    valid_count = len(gold_and_predictions)
    return {
        **positional_role_supervision_audit(examples, records),
        "accuracy": (
            sum(gold == prediction for gold, prediction in gold_and_predictions)
            / valid_count
            if valid_count
            else 0.0
        ),
        "macro_f1": sum(values["f1"] for values in per_class.values())
        / len(POSITIONAL_ROLE_NAMES),
        "per_class": per_class,
        "checkpoint_selection_used": False,
    }


def ordinal_seam_count_targets(
    seam_counts: Sequence[int], *, device: torch.device | None = None
) -> torch.Tensor:
    """Encode 8 bins (0..6, 7+) as seven cumulative ordinal decisions."""

    clipped = torch.tensor(
        [min(max(0, int(count)), SEAM_COUNT_BINS - 1) for count in seam_counts],
        dtype=torch.long,
        device=device,
    )
    thresholds = torch.arange(SEAM_COUNT_BINS - 1, device=device)
    return (clipped.unsqueeze(1) > thresholds.unsqueeze(0)).float()


class SequenceBoundaryClassifier(nn.Module):
    """Model all candidate sentence ends jointly within one essay."""

    def __init__(
        self,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        hidden_size: int,
        dropout: float,
        *,
        predict_roles: bool,
    ) -> None:
        super().__init__()
        if hidden_size % SEQUENCE_ATTENTION_HEADS:
            raise ValueError(
                f"sequence hidden_size는 {SEQUENCE_ATTENTION_HEADS}의 배수여야 합니다"
            )
        self.register_buffer("feature_mean", feature_mean.float())
        self.register_buffer("feature_scale", feature_scale.float())
        self.input_projection = nn.Sequential(
            nn.Linear(int(feature_mean.numel()), hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=SEQUENCE_ATTENTION_HEADS,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sequence_encoder = nn.TransformerEncoder(
            layer,
            num_layers=SEQUENCE_TRANSFORMER_LAYERS,
            enable_nested_tensor=False,
        )
        self.boundary_head = nn.Linear(hidden_size, 1)
        self.ordinal_count_head = nn.Linear(hidden_size, SEAM_COUNT_BINS - 1)
        self.role_head = nn.Linear(hidden_size, 3) if predict_roles else None

    def forward(
        self, features: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("sequence batch에는 candidate가 없는 essay를 넣을 수 없습니다")
        normalized = (features.float() - self.feature_mean) / self.feature_scale
        encoded = self.sequence_encoder(
            self.input_projection(normalized),
            src_key_padding_mask=~valid_mask,
        )
        boundary_logits = self.boundary_head(encoded).squeeze(-1)
        denominator = valid_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (encoded * valid_mask.unsqueeze(-1)).sum(dim=1) / denominator
        count_logits = self.ordinal_count_head(pooled)
        role_logits = self.role_head(encoded) if self.role_head is not None else None
        return boundary_logits, count_logits, role_logits


def _record_indices_by_example(
    records: Sequence[CandidateRecord], record_indices: Sequence[int]
) -> dict[int, list[int]]:
    result: dict[int, list[int]] = defaultdict(list)
    for record_index in record_indices:
        result[records[record_index].example_index].append(record_index)
    for indices in result.values():
        indices.sort(key=lambda index: records[index].candidate_index)
    return result


def _padded_sequence_batch(
    features: torch.Tensor,
    records: Sequence[CandidateRecord],
    record_indices_by_example: dict[int, list[int]],
    example_indices: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_lists = [record_indices_by_example[index] for index in example_indices]
    if not candidate_lists or any(not indices for indices in candidate_lists):
        raise ValueError("sequence batch의 모든 essay에 candidate가 필요합니다")
    max_candidates = max(len(indices) for indices in candidate_lists)
    batch_features = torch.zeros(
        (len(candidate_lists), max_candidates, features.shape[1]),
        dtype=features.dtype,
    )
    valid_mask = torch.zeros(
        (len(candidate_lists), max_candidates), dtype=torch.bool
    )
    record_index_matrix = torch.full(
        (len(candidate_lists), max_candidates), -1, dtype=torch.long
    )
    for batch_index, indices in enumerate(candidate_lists):
        positions = torch.tensor(indices, dtype=torch.long)
        length = len(indices)
        batch_features[batch_index, :length] = features[positions]
        valid_mask[batch_index, :length] = True
        record_index_matrix[batch_index, :length] = positions
    return batch_features, valid_mask, record_index_matrix


def _feature_moments(
    features: torch.Tensor, indices: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = features[torch.tensor(indices, dtype=torch.long)].float()
    mean = selected.mean(dim=0)
    scale = selected.std(dim=0, unbiased=False).clamp_min(1e-5)
    return mean, scale


@torch.no_grad()
def predict_probabilities(
    model: TinyBoundaryClassifier,
    features: torch.Tensor,
    indices: Sequence[int],
    *,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    model.eval()
    result: list[float] = []
    for start in range(0, len(indices), batch_size):
        batch_indices = torch.tensor(
            indices[start : start + batch_size], dtype=torch.long
        )
        batch = features[batch_indices].to(device)
        result.extend(torch.sigmoid(model(batch)).cpu().tolist())
    return result


def train_tiny_classifier(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_indices: Sequence[int],
    calibration_indices: Sequence[int],
    *,
    hidden_size: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    seed: int,
    device: torch.device,
) -> ClassifierTrainingResult:
    if not train_indices or not calibration_indices:
        raise ValueError("train/calibration candidate가 모두 필요합니다")
    train_labels = labels[torch.tensor(train_indices, dtype=torch.long)]
    positives = int(train_labels.sum().item())
    negatives = len(train_indices) - positives
    if positives == 0 or negatives == 0:
        raise ValueError(
            "train split에 positive와 negative candidate가 모두 필요합니다"
        )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mean, scale = _feature_moments(features, train_indices)
    model = TinyBoundaryClassifier(mean, scale, hidden_size, dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, device=device)
    )
    generator = torch.Generator().manual_seed(seed)
    train_index_tensor = torch.tensor(train_indices, dtype=torch.long)
    calibration_labels = [int(labels[index].item()) for index in calibration_indices]
    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_ap = -1.0
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_indices), generator=generator)
        total_loss = 0.0
        seen = 0
        for start in range(0, len(train_indices), batch_size):
            positions = permutation[start : start + batch_size]
            batch_indices = train_index_tensor[positions]
            batch_features = features[batch_indices].to(device)
            batch_labels = labels[batch_indices].to(device).float()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * len(batch_indices)
            seen += len(batch_indices)

        calibration_probabilities = predict_probabilities(
            model,
            features,
            calibration_indices,
            batch_size=batch_size,
            device=device,
        )
        calibration_ap = average_precision(
            calibration_probabilities,
            calibration_labels,
            sum(calibration_labels),
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / seen,
                "calibration_candidate_ap": calibration_ap,
            }
        )
        if calibration_ap > best_ap + 1e-8:
            best_ap = calibration_ap
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("classifier best state가 만들어지지 않았습니다")
    model.load_state_dict(best_state)
    return ClassifierTrainingResult(model, history, best_epoch, best_ap)


@torch.no_grad()
def predict_sequence_outputs(
    model: SequenceBoundaryClassifier,
    features: torch.Tensor,
    examples: Sequence[BoundaryExample],
    records: Sequence[CandidateRecord],
    record_indices: Sequence[int],
    example_indices: Sequence[int],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[list[float], dict[int, int], list[int | None]]:
    """Return boundary, count and optional positional-role predictions."""

    model.eval()
    by_example = _record_indices_by_example(records, record_indices)
    nonempty_examples = [index for index in example_indices if by_example.get(index)]
    probabilities_by_record: dict[int, float] = {}
    roles_by_record: dict[int, int] = {}
    predicted_counts = {int(index): 0 for index in example_indices}
    for start in range(0, len(nonempty_examples), batch_size):
        batch_examples = nonempty_examples[start : start + batch_size]
        batch_features, valid_mask, record_matrix = _padded_sequence_batch(
            features, records, by_example, batch_examples
        )
        boundary_logits, count_logits, role_logits = model(
            batch_features.to(device), valid_mask.to(device)
        )
        boundary_probabilities = torch.sigmoid(boundary_logits).cpu()
        # Each logit estimates P(seam_count > k).  The rounded sum is the
        # ordinal expected count and is less brittle than seven independent
        # threshold decisions when adjacent logits are not perfectly monotone.
        count_decisions = torch.round(torch.sigmoid(count_logits).sum(dim=1)).cpu()
        role_predictions = (
            role_logits.argmax(dim=-1).cpu() if role_logits is not None else None
        )
        for batch_index, example_index in enumerate(batch_examples):
            valid_positions = valid_mask[batch_index]
            indices = record_matrix[batch_index, valid_positions].tolist()
            values = boundary_probabilities[batch_index, valid_positions].tolist()
            probabilities_by_record.update(zip(indices, values, strict=True))
            if role_predictions is not None:
                roles_by_record.update(
                    zip(
                        indices,
                        role_predictions[batch_index, valid_positions].tolist(),
                        strict=True,
                    )
                )
            predicted_counts[example_index] = int(count_decisions[batch_index].item())
    return (
        [probabilities_by_record[index] for index in record_indices],
        predicted_counts,
        [roles_by_record.get(index) for index in record_indices],
    )


def train_sequence_classifier(
    features: torch.Tensor,
    examples: Sequence[BoundaryExample],
    records: Sequence[CandidateRecord],
    train_record_indices: Sequence[int],
    calibration_record_indices: Sequence[int],
    train_example_indices: Sequence[int],
    calibration_example_indices: Sequence[int],
    *,
    hidden_size: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    count_loss_weight: float,
    role_loss_weight: float,
    force_strict_two_space_anchors: bool,
    seed: int,
    device: torch.device,
) -> SequenceTrainingResult:
    """Train the two-layer document sequence model on train-only gold seams."""

    train_by_example = _record_indices_by_example(records, train_record_indices)
    nonempty_train_examples = [
        index for index in train_example_indices if train_by_example.get(index)
    ]
    if not nonempty_train_examples or not calibration_record_indices:
        raise ValueError("sequence train/calibration candidate가 모두 필요합니다")

    train_labels = [records[index].point.label for index in train_record_indices]
    positives = sum(train_labels)
    negatives = len(train_labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("sequence train에 positive와 negative candidate가 필요합니다")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mean, scale = _feature_moments(features, train_record_indices)
    model = SequenceBoundaryClassifier(
        mean,
        scale,
        hidden_size,
        dropout,
        predict_roles=role_loss_weight > 0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    boundary_loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, device=device)
    )
    # Unweighted ordinal BCE preserves each cumulative probability's natural
    # prevalence; inverse-frequency weights would bias the decoded seam count up.
    count_loss_function = nn.BCEWithLogitsLoss()
    role_counts = torch.zeros(3, dtype=torch.float32)
    role_by_record: dict[int, int] = {}
    if role_loss_weight > 0:
        for record_index in train_record_indices:
            record = records[record_index]
            role = candidate_paragraph_role(
                examples[record.example_index], record.point
            )
            role_by_record[record_index] = role
            if role >= 0:
                role_counts[role] += 1
    role_weights = torch.where(
        role_counts > 0,
        role_counts.sum() / (3 * role_counts.clamp_min(1)),
        torch.ones_like(role_counts),
    ).to(device)
    role_loss_function = nn.CrossEntropyLoss(
        weight=role_weights, ignore_index=-100
    )
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(nonempty_train_examples), generator=generator)
        shuffled_examples = [
            nonempty_train_examples[position] for position in permutation.tolist()
        ]
        loss_sums = {"total": 0.0, "boundary": 0.0, "count": 0.0, "role": 0.0}
        seen_examples = 0
        for start in range(0, len(shuffled_examples), batch_size):
            batch_examples = shuffled_examples[start : start + batch_size]
            batch_features, valid_mask, record_matrix = _padded_sequence_batch(
                features, records, train_by_example, batch_examples
            )
            boundary_targets = torch.zeros_like(valid_mask, dtype=torch.float32)
            role_targets = torch.full_like(record_matrix, -100)
            for batch_index, example_index in enumerate(batch_examples):
                for candidate_index, record_index in enumerate(
                    train_by_example[example_index]
                ):
                    record = records[record_index]
                    boundary_targets[batch_index, candidate_index] = record.point.label
                    if role_loss_weight > 0:
                        role_targets[batch_index, candidate_index] = role_by_record[
                            record_index
                        ]
            count_targets = ordinal_seam_count_targets(
                [len(examples[index].gold_boundaries) for index in batch_examples],
                device=device,
            )
            valid_device = valid_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            boundary_logits, count_logits, role_logits = model(
                batch_features.to(device), valid_device
            )
            boundary_loss = boundary_loss_function(
                boundary_logits[valid_device],
                boundary_targets.to(device)[valid_device],
            )
            count_loss = count_loss_function(count_logits, count_targets)
            role_loss = boundary_loss.new_zeros(())
            if role_logits is not None:
                flat_roles = role_targets.to(device)[valid_device]
                if bool((flat_roles != -100).any()):
                    role_loss = role_loss_function(
                        role_logits[valid_device], flat_roles
                    )
            loss = (
                boundary_loss
                + count_loss_weight * count_loss
                + role_loss_weight * role_loss
            )
            loss.backward()
            optimizer.step()
            batch_example_count = len(batch_examples)
            loss_sums["total"] += float(loss.detach().item()) * batch_example_count
            loss_sums["boundary"] += (
                float(boundary_loss.detach().item()) * batch_example_count
            )
            loss_sums["count"] += (
                float(count_loss.detach().item()) * batch_example_count
            )
            loss_sums["role"] += float(role_loss.detach().item()) * batch_example_count
            seen_examples += batch_example_count

        (
            calibration_probabilities,
            calibration_counts,
            _,
        ) = predict_sequence_outputs(
            model,
            features,
            examples,
            records,
            calibration_record_indices,
            calibration_example_indices,
            batch_size=batch_size,
            device=device,
        )
        calibration_records = [
            records[index] for index in calibration_record_indices
        ]
        count_metrics = anchor_count_top_k_metrics(
            examples,
            calibration_records,
            calibration_probabilities,
            calibration_counts,
            example_indices=calibration_example_indices,
            force_strict_two_space_anchors=force_strict_two_space_anchors,
        )
        calibration_f1 = float(count_metrics["overall"]["f1"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sums["total"] / seen_examples,
                "train_boundary_loss": loss_sums["boundary"] / seen_examples,
                "train_ordinal_count_loss": loss_sums["count"] / seen_examples,
                "train_positional_role_loss": loss_sums["role"] / seen_examples,
                "calibration_anchor_count_top_k_f1": calibration_f1,
                "calibration_anchor_count_top_k_count_mae": count_metrics["overall"][
                    "paragraph_count_mae"
                ],
            }
        )
        if calibration_f1 > best_f1 + 1e-8:
            best_f1 = calibration_f1
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("sequence classifier best state가 만들어지지 않았습니다")
    model.load_state_dict(best_state)
    return SequenceTrainingResult(model, history, best_epoch, best_f1)


@torch.no_grad()
def encode_klue_contexts(
    examples: Sequence[BoundaryExample],
    records: Sequence[CandidateRecord],
    *,
    model_id: str,
    revision: str,
    max_length: int,
    local_characters: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Encode local left/right pairs once; only the learned tiny head is saved."""

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=True
    )
    encoder = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=True,
        add_pooling_layer=False,
    )
    encoder.requires_grad_(False).eval().to(device)
    use_bfloat16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if use_bfloat16:
        encoder.to(dtype=torch.bfloat16)
    hidden_size = int(encoder.config.hidden_size)
    embeddings = torch.empty((len(records), hidden_size), dtype=torch.float16)

    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        left_texts: list[str] = []
        right_texts: list[str] = []
        for record in batch_records:
            example = examples[record.example_index]
            offset = record.point.offset
            left_texts.append(
                example.official_raw[max(0, offset - local_characters) : offset]
            )
            right_texts.append(example.official_raw[offset : offset + local_characters])
        tokenized = tokenizer(
            left_texts,
            right_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        # transformers 5.x resolves KLUE's tokenizer as BertTokenizer and emits
        # pair segment id 1, but KLUE RoBERTa has type_vocab_size=1.
        tokenized.pop("token_type_ids", None)
        tokenized = {name: value.to(device) for name, value in tokenized.items()}
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if use_bfloat16
            else torch.autocast("cpu", enabled=False)
        )
        with context:
            output = encoder(**tokenized).last_hidden_state
            mask = tokenized["attention_mask"].unsqueeze(-1)
            pooled = (output.float() * mask).sum(dim=1) / mask.sum(dim=1)
        embeddings[start : start + len(batch_records)] = pooled.cpu().half()

    metadata = {
        "model_id": model_id,
        "revision": revision,
        "hidden_size": hidden_size,
        "pooling": "masked_mean_last_hidden_state",
        "max_length": max_length,
        "local_characters_each_side": local_characters,
        "token_type_ids": "dropped_for_type_vocab_size_1",
        "frozen": True,
    }
    return embeddings, metadata
