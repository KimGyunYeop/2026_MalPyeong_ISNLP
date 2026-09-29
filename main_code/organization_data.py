from __future__ import annotations

import hashlib
import random
from copy import deepcopy
from typing import Any, Sequence

from .paragraph_boundary import gold_paragraph_boundaries, kiwi_sentence_spans


EXTERNAL_ORGANIZATION_LABEL_POLICIES = (
    "official",
    "aihub24_highschool_structural_v1",
    "aihub24_structural_aihub26_persuasion_v1",
)
ORGANIZATION_AUGMENTATIONS = (
    "none",
    "paragraph_order_high_confidence_v1",
    "sentence_order_high_confidence_v1",
)


def apply_external_organization_policy(
    rows: Sequence[dict[str, Any]], policy: str
) -> list[dict[str, Any]]:
    """Apply one audited, source-local organization data policy.

    AIHub24's stored organization score mixes paragraph structure, coherence,
    and length with task-dependent weights.  The competition organization
    target instead averages two structural criteria.  This policy keeps only
    high-school rows where both AIHub24 structural criteria were active and
    gives them an equal-weight organization target.  The A24+A26 pack policy
    additionally keeps only A26 persuasion rows without changing their two
    expert ratings.  The later hard-rank loss uses only score ordering, not
    either source's absolute scale.
    """

    if policy == "official":
        return list(rows)
    if policy not in {
        "aihub24_highschool_structural_v1",
        "aihub24_structural_aihub26_persuasion_v1",
    }:
        raise ValueError(f"지원하지 않는 external organization policy: {policy}")

    selected: list[dict[str, Any]] = []
    for row in rows:
        source = row.get("_dataset_name")
        if source != "aihub24_essay":
            # The policy is source-local: in a multi-source essay pack it
            # rewrites AIHub24 and leaves the already audited NIKL rows intact.
            # The A24+A26 policy additionally keeps only A26 persuasion rows.
            # Its two expert organization ratings remain unchanged here.
            if (
                policy == "aihub24_structural_aihub26_persuasion_v1"
                and source == "aihub26_essay"
                and str((row.get("metadata") or {}).get("purpose", "")).strip()
                != "설득"
            ):
                continue
            selected.append(row)
            continue
        metadata = row.get("metadata") or {}
        if not str(metadata.get("grade", "")).startswith("고등_"):
            continue
        organization = row["score_details"]["traits"]["organization"]
        criteria = organization["criteria"]
        essay_structure = criteria["org_essay"]
        paragraph_structure = criteria["org_paragraph"]
        coherence = criteria["org_coherence"]
        if not (
            float(essay_structure["weight"]) > 0
            and float(paragraph_structure["weight"]) > 0
            and float(coherence["weight"]) > 0
        ):
            continue

        item = deepcopy(row)
        score = dict(item["score"])
        score["organization"] = (
            float(essay_structure["official_score"])
            + float(paragraph_structure["official_score"])
        ) / 2.0
        score["average"] = (
            float(score["content"])
            + float(score["organization"])
            + float(score["expression"])
        ) / 3.0
        item["score"] = score
        item["organization_label_policy"] = policy
        selected.append(item)
    return selected


def _selected_organization_ratings(row: dict[str, Any]) -> list[float]:
    details = row.get("score_details") or {}
    if details.get("label_policy") != "mean_of_two_raters":
        return []
    organization = (details.get("traits") or {}).get("organization") or {}
    criteria = organization.get("criteria") or {}
    ratings: list[float] = []
    for criterion_name in ("organization_1", "organization_2"):
        criterion = criteria.get(criterion_name) or {}
        rater_scores = criterion.get("rater_scores") or {}
        selected = criterion.get("selected_for_target") or {}
        ratings.extend(
            float(score)
            for evaluator, score in rater_scores.items()
            if selected.get(evaluator) and score is not None
        )
    return ratings


def _stable_seed(row: dict[str, Any], kind: str, seed: int) -> int:
    identity = str(row.get("essay_hash") or row.get("id") or row.get("document_id"))
    digest = hashlib.sha256(f"{identity}|{kind}|seed{seed}|v1".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _sattolo_order(size: int, *, seed: int) -> list[int]:
    order = list(range(size))
    rng = random.Random(seed)
    for index in range(size - 1, 0, -1):
        other = rng.randrange(index)
        order[index], order[other] = order[other], order[index]
    return order


def _official_raw_paragraph_spans(row: dict[str, Any]) -> list[str]:
    surfaces = row.get("essay_surfaces") or {}
    official_raw = surfaces.get("official_raw")
    if not isinstance(official_raw, str):
        raise ValueError(f"official_raw가 없는 competition row: {row.get('id')}")
    boundaries = gold_paragraph_boundaries(row)
    offsets = (0, *boundaries, len(official_raw))
    spans = [
        official_raw[offsets[index] : offsets[index + 1]]
        for index in range(len(offsets) - 1)
    ]
    if "".join(spans) != official_raw:
        raise ValueError(f"paragraph span 복원 실패: {row.get('id')}")
    return spans


def _organization_pair_row(
    row: dict[str, Any],
    *,
    augmentation_name: str,
    kind: str,
    role: str,
    essay: str,
    organization_score: float,
) -> dict[str, Any]:
    parent_id = str(row.get("id", row.get("document_id")))
    pair_id = f"{parent_id}:{kind}"
    return {
        "schema_version": 1,
        "dataset_group": "augmentation",
        # This is target-derived writing, so source-aware prompting and score
        # conventions should remain identical to the competition rows.  The
        # augmentation provenance stays explicit in `_dataset_name`/metadata.
        "source_dataset": "nikl_competition",
        "source_split": "competition_train_derived",
        "id": f"{augmentation_name}:{pair_id}:{role}",
        "document_id": f"{augmentation_name}:{pair_id}:{role}",
        # The sampler groups by prompt_num + visible prompt.  Appending the
        # parent identity makes every physical B2 batch one exact pair.
        "prompt_num": f"{augmentation_name}:{pair_id}",
        "prompt": row["prompt"],
        "essay": essay,
        "score": {
            "content": 3.0,
            "organization": organization_score,
            "expression": 3.0,
            "average": (6.0 + organization_score) / 3.0,
        },
        "metadata": {
            "parent_id": parent_id,
            "corruption": kind,
            "role": role,
        },
        "_dataset_name": augmentation_name,
    }


def build_paragraph_order_rows(
    competition_rows: Sequence[dict[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    """Create exact original>moderate/severe paragraph-order pairs online.

    Gold paragraph boundaries select and construct training corruptions only.
    Each generated essay is a concatenation of exact official-raw spans, with
    no marker or character added; the caller then applies the same configured
    input surface as every other prefix arm.  Validation is never passed here.
    """

    pairs: list[dict[str, Any]] = []
    for row in competition_rows:
        paragraph_count = int((row.get("metadata") or {}).get("paragraph_count", 0))
        ratings = _selected_organization_ratings(row)
        if paragraph_count < 4 or len(ratings) != 4 or min(ratings) < 4.0:
            continue

        spans = _official_raw_paragraph_spans(row)
        if len(spans) != paragraph_count:
            raise ValueError(
                f"paragraph_count와 span 수가 다릅니다: {row.get('id')} "
                f"{paragraph_count}!={len(spans)}"
            )
        original = "".join(spans)

        moderate_order = list(range(paragraph_count))
        moderate_seed = _stable_seed(row, "moderate", seed)
        internal_index = 1 + moderate_seed % (paragraph_count - 3)
        moderate_order[internal_index], moderate_order[internal_index + 1] = (
            moderate_order[internal_index + 1],
            moderate_order[internal_index],
        )
        moderate = "".join(spans[index] for index in moderate_order)

        severe_order = _sattolo_order(
            paragraph_count,
            seed=_stable_seed(row, "severe", seed),
        )
        severe = "".join(spans[index] for index in severe_order)
        if len(set(severe_order)) != paragraph_count or any(
            index == value for index, value in enumerate(severe_order)
        ):
            raise ValueError(f"severe order가 derangement가 아닙니다: {row.get('id')}")
        if len({original, moderate, severe}) != 3:
            raise ValueError(
                f"paragraph corruption이 서로 다르지 않습니다: {row.get('id')}"
            )

        for kind, corrupted, low_score in (
            ("moderate", moderate, 2.0),
            ("severe", severe, 1.0),
        ):
            pairs.append(
                _organization_pair_row(
                    row,
                    augmentation_name="paragraph_order_high_confidence_v1",
                    kind=kind,
                    role="original",
                    essay=original,
                    organization_score=3.0,
                )
            )
            pairs.append(
                _organization_pair_row(
                    row,
                    augmentation_name="paragraph_order_high_confidence_v1",
                    kind=kind,
                    role="corrupted",
                    essay=corrupted,
                    organization_score=low_score,
                )
            )
    return pairs


def _official_raw_sentence_spans(row: dict[str, Any]) -> list[str]:
    surfaces = row.get("essay_surfaces") or {}
    official_raw = surfaces.get("official_raw")
    if not isinstance(official_raw, str):
        raise ValueError(f"official_raw가 없는 competition row: {row.get('id')}")

    sentences = kiwi_sentence_spans(official_raw)
    # Kiwi spans can leave whitespace between sentences.  End offsets turn
    # them into a complete, non-overlapping partition of the original string;
    # no character is generated or dropped when the pieces are reordered.
    offsets = (0, *(end for _, end in sentences[:-1]), len(official_raw))
    spans = [
        official_raw[offsets[index] : offsets[index + 1]]
        for index in range(len(offsets) - 1)
    ]
    if "".join(spans) != official_raw:
        raise ValueError(f"sentence span 복원 실패: {row.get('id')}")
    return spans


def build_sentence_order_rows(
    competition_rows: Sequence[dict[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    """Create target-domain sentence-order preference pairs online.

    The high-confidence filter follows the same human O-rating contract as the
    paragraph augmentation.  Moderate corruption swaps one adjacent internal
    sentence; severe corruption deranges every sentence.  Only the direction
    ``original > corrupted`` is supervised, so invented absolute scores never
    enter the final regression loss.
    """

    pairs: list[dict[str, Any]] = []
    for row in competition_rows:
        ratings = _selected_organization_ratings(row)
        if len(ratings) != 4 or min(ratings) < 4.0:
            continue

        spans = _official_raw_sentence_spans(row)
        sentence_count = len(spans)
        if sentence_count < 8:
            continue
        original = "".join(spans)

        moderate_order = list(range(sentence_count))
        moderate_seed = _stable_seed(row, "sentence-moderate", seed)
        internal_index = 1 + moderate_seed % (sentence_count - 3)
        moderate_order[internal_index], moderate_order[internal_index + 1] = (
            moderate_order[internal_index + 1],
            moderate_order[internal_index],
        )
        moderate = "".join(spans[index] for index in moderate_order)

        severe_order = _sattolo_order(
            sentence_count,
            seed=_stable_seed(row, "sentence-severe", seed),
        )
        severe = "".join(spans[index] for index in severe_order)

        for kind, corrupted, low_score in (
            ("moderate", moderate, 2.0),
            ("severe", severe, 1.0),
        ):
            # Repeated adjacent sentences can make a particular permutation
            # text-identical.  Such a pair has no learnable ordering signal.
            if corrupted == original:
                continue
            pairs.append(
                _organization_pair_row(
                    row,
                    augmentation_name="sentence_order_high_confidence_v1",
                    kind=kind,
                    role="original",
                    essay=original,
                    organization_score=3.0,
                )
            )
            pairs.append(
                _organization_pair_row(
                    row,
                    augmentation_name="sentence_order_high_confidence_v1",
                    kind=kind,
                    role="corrupted",
                    essay=corrupted,
                    organization_score=low_score,
                )
            )
    return pairs
