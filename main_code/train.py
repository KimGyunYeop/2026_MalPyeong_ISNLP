from __future__ import annotations

import argparse
import hashlib
import json
import math
from functools import partial
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from transformers import TrainerCallback
except ImportError:
    # 설정/metric 단위 테스트는 transformers 없이도 import할 수 있게 둔다.
    class TrainerCallback:  # type: ignore[no-redef]
        pass


from .config import (
    LEGACY_BASELINE_CONFIG_PATH,
    RATER_SET_HEAD_MODES,
    PRIMARY_DATA_PROFILES,
    TRAITS,
    RegressionConfig,
    add_config_arguments,
    config_updates_from_namespace,
    load_config,
    resolve_model,
    save_config,
)
from .datasets import (
    EssayRegressionDataset,
    MultiSourceStepSampler,
    RegressionCollator,
    SameQuestionSampler,
    build_detail_rater_registry,
    build_prompt_registry,
    competition_steps_per_epoch,
    deployment_surface_fingerprints,
    detail_supervision_summary,
    essay_id,
    exclude_unseen_prompt_rows,
    filter_origin_extra_by_rater_agreement,
    human_scores,
    load_prepared_extended_rows,
    load_rows,
    normalized_essay_hash,
    official_average_score,
    parse_extended_dataset_names,
    parse_unseen_prompt_holdout,
    question_key,
    rows_fingerprint,
    split_primary_rows_by_source,
    split_validation_holdout,
    stratified_training_subset,
    validate_dataset_profile,
    validate_prepared_manifest,
    validation_overlap_report,
)
from .models import (
    LoadedRegressionModel,
    RegressionScorer,
    build_model,
    quantization_objective_active,
    save_checkpoint,
)
from .organization_data import (
    apply_external_organization_policy,
    build_paragraph_order_rows,
    build_sentence_order_rows,
)
from .distribution_shift import shifted_rmse
from .utils import (
    average_matched_integer_scores,
    regression_metrics,
    score_metric_matrix,
    set_seed,
    write_json,
    write_jsonl,
)


# CLI and experiment identity ------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hugging Face Trainer three-head regression training"
    )
    config_source = parser.add_mutually_exclusive_group()
    config_source.add_argument("--config")
    config_source.add_argument(
        "--baseline",
        action="store_true",
        help="2026-08-10 이전 frozen legacy baseline config 사용",
    )
    parser.add_argument(
        "--model", default=None, help="registry alias 또는 등록된 Hugging Face model ID"
    )
    parser.add_argument(
        "--train-file",
        action="append",
        help="생략하면 --primary-data-profile의 train.jsonl을 사용",
    )
    parser.add_argument(
        "--validation-file",
        help="생략하면 --primary-data-profile의 validation.jsonl을 사용",
    )
    parser.add_argument(
        "--validation-average-overlay-file",
        help=(
            "checkpoint metric에 쓸 저장 score.average의 원본 JSONL. "
            "ID·본문·trait label이 validation과 1:1 일치할 때만 적용"
        ),
    )
    parser.add_argument(
        "--training-average-overlay-file",
        help=(
            "quantized mean-first loss에 쓸 저장 score.average 원본 JSONL. "
            "train의 strict subset이어야 하며 ID·본문·trait label을 검증"
        ),
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="모델 선택 metric 없는 최종 refit에서만 validation을 끄기",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, help="작은 smoke test용")
    # 아래 metadata는 학습 방법에 영향을 주지 않는다. 네 suite shell과 단일
    # aggregator가 경로를 추측하지 않고 실험을 식별하기 위한 공통 계약이다.
    parser.add_argument("--experiment-suite", default="manual")
    parser.add_argument("--experiment-base", default="default")
    parser.add_argument("--experiment-case", default="manual")
    parser.add_argument("--experiment-description", default="")
    parser.add_argument("--experiment-config-id", default="manual")
    parser.add_argument("--experiment-base-args", default="")
    parser.add_argument("--experiment-overrides", default="")
    parser.add_argument("--experiment-inference-args", default="")
    add_config_arguments(parser)
    return parser.parse_args()


def experiment_record(
    args: argparse.Namespace,
    model_argument: str,
    config: RegressionConfig | None = None,
) -> dict[str, Any]:
    """Build the path-independent identity shared by runners and aggregation."""

    # Runner가 넘긴 JSON 배열을 다시 문자열로 감싸 두지 않는다. experiment.json을
    # 사람이 열었을 때도 공통 설정과 case별 차이를 바로 비교할 수 있게 한다.
    base_args = json.loads(args.experiment_base_args or "[]")
    override_args = json.loads(args.experiment_overrides or "[]")
    inference_args = json.loads(args.experiment_inference_args or "[]")
    record = {
        "schema_version": 1,
        "suite": args.experiment_suite,
        "base_name": args.experiment_base,
        "case_name": args.experiment_case,
        "description": args.experiment_description,
        "config_id": args.experiment_config_id,
        "model_argument": model_argument,
        "base_args": base_args,
        "override_args": override_args,
        "inference_args": inference_args,
        "command": [sys.executable, "-m", "main_code.train", *sys.argv[1:]],
    }
    if config is not None:
        record.update(
            model_slug=config.model_slug,
            model_id=config.model_id,
            model_revision=config.model_revision,
            backbone_type=config.backbone_type,
        )
    return record


def competition_selection_description(config: RegressionConfig) -> str:
    agreement_filter = config.origin_extra_max_rater_disagreement >= 0
    subset = config.competition_train_limit > 0
    if not agreement_filter and not subset:
        return "full"
    if not agreement_filter:
        return "prompt-proportional deterministic subset"
    if not subset:
        return (
            "official rows retained; origin_pool_extra filtered by "
            "9-criterion primary-rater absolute-difference sum"
        )
    return "agreement filter, then prompt-proportional deterministic subset"


def training_data_signature(
    config: RegressionConfig,
    competition_rows: list[dict[str, Any]],
    extended_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fingerprint the exact rows used, independent of their absolute paths."""

    validation_metric_average_records = sorted(
        (essay_id(row), float(row["_official_metric_average"]))
        for row in validation_rows
        if "_official_metric_average" in row
    )
    training_metric_average_records = sorted(
        (essay_id(row), float(row["_official_metric_average"]))
        for row in competition_rows
        if "_official_metric_average" in row
    )

    def overlay_digest(records: list[tuple[str, float]]) -> str | None:
        if not records:
            return None
        return hashlib.sha256(
            (
                "".join(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                    for record in records
                )
            ).encode("utf-8")
        ).hexdigest()

    payload = {
        "primary_data_profile": config.primary_data_profile,
        "dataset_schedule": config.dataset_schedule,
        "extended_datasets": config.extended_datasets,
        "external_purpose_filter": config.external_purpose_filter,
        "external_rankable_trait": config.external_rankable_trait,
        "external_organization_label_policy": (
            config.external_organization_label_policy
        ),
        "organization_augmentation": config.organization_augmentation,
        "essay_surface": config.essay_surface,
        "origin_extra_max_rater_disagreement": (
            None
            if config.origin_extra_max_rater_disagreement < 0
            else config.origin_extra_max_rater_disagreement
        ),
        "train_surface_augmentation": {
            "method": (
                "canonical_or_configured_surface_per_access"
                if config.train_canonical_surface_probability > 0
                else "none"
            ),
            "canonical_probability": config.train_canonical_surface_probability,
            "configured_surface": config.essay_surface,
            "seed": config.seed,
        },
        "competition_rows": len(competition_rows),
        "competition_source_split_counts": dict(
            sorted(
                Counter(
                    str(row.get("source_split", "")) for row in competition_rows
                ).items()
            )
        ),
        "extended_rows": len(extended_rows),
        "validation_rows": len(validation_rows),
        "competition_sha256": rows_fingerprint(competition_rows),
        "extended_sha256": rows_fingerprint(extended_rows),
        "validation_sha256": rows_fingerprint(validation_rows),
        "validation_metric_average_overlay": (
            {
                "rows": len(validation_metric_average_records),
                "sha256": overlay_digest(validation_metric_average_records),
            }
            if validation_metric_average_records
            else None
        ),
        "training_metric_average_overlay": (
            {
                "rows": len(training_metric_average_records),
                "sha256": overlay_digest(training_metric_average_records),
            }
            if training_metric_average_records
            else None
        ),
        "deployment_surface_fingerprints": {
            "competition": deployment_surface_fingerprints(
                competition_rows, config
            ),
            "external": deployment_surface_fingerprints(extended_rows, config),
            "validation": deployment_surface_fingerprints(validation_rows, config),
        },
    }
    semantic = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return {
        **payload,
        "sha256": hashlib.sha256(semantic.encode("utf-8")).hexdigest(),
        "selection": {
            "competition": competition_selection_description(config),
            "external": "; ".join(
                [
                    *(
                        [f"metadata.purpose={config.external_purpose_filter!r}"]
                        if config.external_purpose_filter
                        else []
                    ),
                    *(
                        [
                            "same-source/question groups with >=2 distinct visible "
                            "essay hashes and >=2 distinct "
                            f"{config.external_rankable_trait} labels"
                        ]
                        if config.external_rankable_trait
                        else []
                    ),
                    *(
                        ["prompt-proportional deterministic subset per source"]
                        if config.external_train_limit
                        else []
                    ),
                    *(
                        [
                            "external organization labels="
                            f"{config.external_organization_label_policy}"
                        ]
                        if config.external_organization_label_policy != "official"
                        else []
                    ),
                    *(
                        [f"online augmentation={config.organization_augmentation}"]
                        if config.organization_augmentation != "none"
                        else []
                    ),
                ]
            )
            or "full",
        },
        "seed": config.seed,
    }


@dataclass
class PreparedData:
    """Rows and provenance needed by the rest of the training pipeline."""

    competition_rows: list[dict[str, Any]]
    extended_rows: list[dict[str, Any]]
    validation_rows: list[dict[str, Any]]
    train_files: list[str]
    validation_file: str | None
    primary_manifest: dict[str, Any]
    external_manifest: dict[str, Any] | None
    extended_names: tuple[str, ...]
    extended_source_counts: dict[str, int]
    overlap_report: dict[str, Any]
    overlap_allowed: bool
    primary_manifest_used_for_training: bool

    @property
    def train_rows(self) -> list[dict[str, Any]]:
        return self.competition_rows + self.extended_rows


def overlay_validation_metric_averages(
    validation_rows: list[dict[str, Any]],
    overlay_file: str | Path,
    *,
    allow_source_superset: bool = False,
) -> list[dict[str, Any]]:
    """Attach exact stored averages after strict row/label identity checks.

    Processed validation intentionally carries richer supervision and can
    recompute ``score.average`` from the three traits.  Official evaluation,
    however, reads the two-decimal value stored in the public raw JSONL.  The
    overlay affects metric labels only; C/O/E training labels and visible
    scorer input stay byte-identical.
    """

    source_rows = load_rows([overlay_file])
    source_by_id: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        identity = essay_id(row)
        if identity in source_by_id:
            raise ValueError(
                f"validation average overlay에 중복 essay_id가 있습니다: {identity}"
            )
        source_by_id[identity] = row

    validation_ids = [essay_id(row) for row in validation_rows]
    if len(validation_ids) != len(set(validation_ids)):
        raise ValueError("validation rows에 중복 essay_id가 있어 average overlay 불가")
    missing = sorted(set(validation_ids) - set(source_by_id))
    extra = sorted(set(source_by_id) - set(validation_ids))
    if missing or (extra and not allow_source_superset):
        raise ValueError(
            "validation average overlay ID 집합 불일치: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    overlaid: list[dict[str, Any]] = []
    for row in validation_rows:
        identity = essay_id(row)
        source = source_by_id[identity]
        if question_key(row) != question_key(source):
            raise ValueError(f"validation average overlay prompt 불일치: {identity}")
        if normalized_essay_hash(row) != normalized_essay_hash(source):
            raise ValueError(f"validation average overlay essay 불일치: {identity}")
        labels = human_scores(row)
        source_labels = human_scores(source)
        if labels is None or source_labels is None or any(
            not math.isclose(
                float(labels[trait]),
                float(source_labels[trait]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for trait in TRAITS
        ):
            raise ValueError(f"validation average overlay trait label 불일치: {identity}")
        average = official_average_score(source)
        if average is None or not math.isfinite(float(average)):
            raise ValueError(f"validation average overlay score.average 누락: {identity}")
        copied = dict(row)
        copied["_official_metric_average"] = float(average)
        overlaid.append(copied)
    return overlaid


def overlay_training_metric_averages(
    training_rows: list[dict[str, Any]], overlay_file: str | Path
) -> list[dict[str, Any]]:
    """Overlay stored averages for a strict, identity-checked train subset.

    The public raw train contains 2,000 official rows while the final prepared
    pool contains 11,600 rows.  Missing overlay IDs are therefore expected only
    on the prepared side; every source ID must match exactly one prepared row.
    """

    source_rows = load_rows([overlay_file])
    source_by_id: dict[str, dict[str, Any]] = {}
    for source in source_rows:
        identity = essay_id(source)
        if identity in source_by_id:
            raise ValueError(
                f"training average overlay에 중복 essay_id가 있습니다: {identity}"
            )
        source_by_id[identity] = source
    training_by_id = {essay_id(row): row for row in training_rows}
    if len(training_by_id) != len(training_rows):
        raise ValueError("training rows에 중복 essay_id가 있어 average overlay 불가")
    unknown = sorted(set(source_by_id) - set(training_by_id))
    if unknown:
        raise ValueError(
            f"training average overlay에 train 밖 ID가 있습니다: {unknown[:5]}"
        )

    overlaid: list[dict[str, Any]] = []
    for row in training_rows:
        identity = essay_id(row)
        source = source_by_id.get(identity)
        if source is None:
            overlaid.append(row)
            continue
        if question_key(row) != question_key(source):
            raise ValueError(f"training average overlay prompt 불일치: {identity}")
        if normalized_essay_hash(row) != normalized_essay_hash(source):
            raise ValueError(f"training average overlay essay 불일치: {identity}")
        labels = human_scores(row)
        source_labels = human_scores(source)
        if labels is None or source_labels is None or any(
            not math.isclose(
                float(labels[trait]),
                float(source_labels[trait]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for trait in TRAITS
        ):
            raise ValueError(f"training average overlay trait label 불일치: {identity}")
        average = official_average_score(source)
        if average is None or not math.isfinite(float(average)):
            raise ValueError(f"training average overlay score.average 누락: {identity}")
        copied = dict(row)
        copied["_official_metric_average"] = float(average)
        overlaid.append(copied)
    if not source_by_id:
        raise ValueError("training average overlay가 비어 있습니다")
    return overlaid


# Data preparation and schedule ----------------------------------------------
def filter_external_rows_by_purpose(
    rows: list[dict[str, Any]], purpose: str
) -> list[dict[str, Any]]:
    """Keep one explicit writing purpose from prepared AIHub metadata."""

    selected = purpose.strip()
    if not selected:
        return rows
    return [
        row
        for row in rows
        if isinstance(row.get("metadata"), dict)
        and str(row["metadata"].get("purpose", "")).strip() == selected
    ]


def filter_external_rankable_groups(
    rows: list[dict[str, Any]], trait: str
) -> list[dict[str, Any]]:
    """Keep external question groups that contain a real pairwise preference.

    The source is part of the grouping key so aliases with the same prompt text
    never create an artificial cross-corpus pair.  Hashes are recomputed from
    the visible essay field rather than trusting provider identifiers.
    """

    selected_trait = trait.strip()
    if not selected_trait:
        return rows
    if selected_trait not in {"content", "organization", "expression"}:
        raise ValueError(f"지원하지 않는 rankable trait입니다: {selected_trait}")

    groups: dict[tuple[str, tuple[str, str]], list[dict[str, Any]]] = {}
    for row in rows:
        source = str(
            row.get("_dataset_name", row.get("source_dataset", ""))
        ).strip()
        groups.setdefault((source, question_key(row)), []).append(row)

    usable_keys = set()
    for key, group in groups.items():
        essay_hashes = {normalized_essay_hash(row) for row in group}
        labels = {
            scores[selected_trait]
            for row in group
            if (scores := human_scores(row)) is not None
        }
        if len(essay_hashes) >= 2 and len(labels) >= 2:
            usable_keys.add(key)

    return [
        row
        for row in rows
        if (
            str(row.get("_dataset_name", row.get("source_dataset", ""))).strip(),
            question_key(row),
        )
        in usable_keys
    ]


def _target_quantile_value(sorted_target: list[float], fraction: float) -> float:
    """정렬된 대회 label 분포에서 분위수 ``fraction``의 값을 선형 보간으로 읽는다."""

    if not sorted_target:
        raise ValueError("대회 label 분포가 비어 있습니다")
    if len(sorted_target) == 1:
        return sorted_target[0]
    position = fraction * (len(sorted_target) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_target) - 1)
    weight = position - lower
    return sorted_target[lower] * (1.0 - weight) + sorted_target[upper] * weight


def align_external_scores_to_competition(
    external_rows: list[dict[str, Any]],
    competition_rows: list[dict[str, Any]],
    policy: str,
) -> list[dict[str, Any]]:
    """외부 corpus의 trait 점수를 대회 train label 분포로 사상한다.

    외부 자료는 척도와 분포가 대회와 다르다(AIHub 에세이는 고득점 편향이 크다).
    절대 MSE는 그 차이를 그대로 학습하고 hard RankNet은 절대값을 전부 버린다. 이
    변환은 그 중간으로, source 안에서 trait별 순위를 그대로 두고 그 순위의 분위수에
    해당하는 대회 label 값으로 점수를 바꾼다. 동점은 같은 분위수로 보내 원래의
    tie를 보존한다. 원본 row를 변형하지 않고 새 list를 반환하며, validation row는
    이 함수에 전달하지 않는다.
    """

    if policy == "none":
        return external_rows
    if policy != "quantile_to_competition_v1":
        raise ValueError(f"지원하지 않는 external_score_alignment입니다: {policy}")

    target_by_trait: dict[str, list[float]] = {}
    for trait in TRAITS:
        values = sorted(
            float(scores[trait])
            for row in competition_rows
            if (scores := human_scores(row)) is not None
        )
        if not values:
            raise ValueError(f"대회 train label에서 {trait} 분포를 만들 수 없습니다")
        target_by_trait[trait] = values

    original = [dict(row.get("score") or {}) for row in external_rows]
    aligned = [dict(scores) for scores in original]
    indices_by_source: dict[str, list[int]] = {}
    for index, row in enumerate(external_rows):
        source = str(row.get("_dataset_name", row.get("source_dataset", ""))).strip()
        indices_by_source.setdefault(source, []).append(index)

    for source_indices in indices_by_source.values():
        for trait in TRAITS:
            ranked = sorted(
                (index for index in source_indices if trait in original[index]),
                key=lambda index: float(original[index][trait]),
            )
            if not ranked:
                continue
            denominator = max(1, len(ranked) - 1)
            position = 0
            while position < len(ranked):
                last = position
                while (
                    last + 1 < len(ranked)
                    and float(original[ranked[last + 1]][trait])
                    == float(original[ranked[position]][trait])
                ):
                    last += 1
                fraction = ((position + last) / 2.0) / denominator
                value = _target_quantile_value(target_by_trait[trait], fraction)
                for index in ranked[position : last + 1]:
                    aligned[index][trait] = value
                position = last + 1

    aligned_rows = []
    for row, scores in zip(external_rows, aligned, strict=True):
        item = dict(row)
        if all(trait in scores for trait in TRAITS):
            # official_average_score가 읽는 average도 사상된 값과 맞춘다.
            scores["average"] = sum(scores[trait] for trait in TRAITS) / len(TRAITS)
        item["score"] = scores
        aligned_rows.append(item)
    return aligned_rows


def prepare_training_data(
    args: argparse.Namespace, config: RegressionConfig
) -> PreparedData:
    """Load, sample and leakage-check all selected training sources."""

    profile_directory = (
        Path(config.dataset_root)
        / PRIMARY_DATA_PROFILES[config.primary_data_profile].name
    )
    train_files = args.train_file or [str(profile_directory / "train.jsonl")]
    if args.no_validation and args.validation_file:
        raise ValueError("--no-validation과 --validation-file은 같이 쓸 수 없습니다")
    validation_file = (
        None
        if args.no_validation
        else args.validation_file or str(profile_directory / "validation.jsonl")
    )
    primary_manifest = validate_dataset_profile(
        profile_directory,
        expected_dataset={
            "full": "competition",
            "official": "official_only",
            "validation_leaked": "competition_leaked",
        }[config.primary_data_profile],
    )

    competition_rows = filter_origin_extra_by_rater_agreement(
        load_rows(train_files), config.origin_extra_max_rater_disagreement
    )
    # 미학습 문항 보류. **어떤 파생 row보다 먼저** 뗀다. 증강·source 분할 뒤에 떼면
    # 같은 문항에서 만든 파생 row가 학습에 남아 "본 적 없는 문항"이 아니게 된다.
    # validation은 건드리지 않는다 — 그쪽의 같은 문항 에세이가 바로 평가 대상이다.
    unseen_prompts = parse_unseen_prompt_holdout(config.unseen_prompt_holdout)
    if unseen_prompts:
        before = len(competition_rows)
        competition_rows = exclude_unseen_prompt_rows(competition_rows, unseen_prompts)
        if len(competition_rows) == before:
            raise ValueError(
                f"unseen_prompt_holdout={unseen_prompts}에 해당하는 학습 행이 없습니다. "
                "prompt_num 표기를 확인하세요"
            )
    # holdout은 어떤 파생 row(증강, source 분할)보다 먼저 떼어낸다. 나중에 떼면 같은
    # essay에서 만든 증강 row가 학습에 남아 누수가 된다.
    competition_rows, holdout_rows = split_validation_holdout(
        competition_rows, config.validation_holdout_size
    )
    competition_rows = stratified_training_subset(
        competition_rows, config.competition_train_limit, seed=config.seed
    )
    if config.split_primary_sources:
        competition_rows = split_primary_rows_by_source(competition_rows)
    if args.limit is not None:
        competition_rows = competition_rows[: args.limit]

    extended_names = parse_extended_dataset_names(config.extended_datasets)
    extended_manifest = None
    extended_rows: list[dict[str, Any]] = []
    extended_source_counts: dict[str, int] = {}
    if extended_names:
        extended_manifest = validate_prepared_manifest(
            config.dataset_root, extended_names
        )
        extended_rows, extended_source_counts = load_prepared_extended_rows(
            config.dataset_root,
            extended_names,
            # purpose를 먼저 고른 뒤 limit을 적용해야 선택한 장르 안에서 재현 가능한
            # subset이 된다. filter가 없을 때는 기존 loader 경로를 그대로 쓴다.
            limit_per_dataset=(
                None
                if (
                    config.external_purpose_filter
                    or config.external_rankable_trait
                    or config.external_organization_label_policy != "official"
                )
                else (config.external_train_limit or None)
            ),
            sample_seed=config.seed,
            validate_manifest=False,
        )
        if (
            config.external_purpose_filter
            or config.external_rankable_trait
            or config.external_organization_label_policy != "official"
        ):
            by_source = {}
            for name in extended_names:
                source_rows = [
                    row for row in extended_rows if row.get("_dataset_name") == name
                ]
                source_rows = filter_external_rows_by_purpose(
                    source_rows, config.external_purpose_filter
                )
                source_rows = apply_external_organization_policy(
                    source_rows, config.external_organization_label_policy
                )
                source_rows = filter_external_rankable_groups(
                    source_rows, config.external_rankable_trait
                )
                if not source_rows:
                    filters = []
                    if config.external_purpose_filter:
                        filters.append(
                            f"metadata.purpose={config.external_purpose_filter!r}"
                        )
                    if config.external_rankable_trait:
                        filters.append(
                            f"rankable_trait={config.external_rankable_trait!r}"
                        )
                    if config.external_organization_label_policy != "official":
                        filters.append(
                            "organization_label_policy="
                            f"{config.external_organization_label_policy!r}"
                        )
                    raise ValueError(f"{name}에 {'; '.join(filters)} 행이 없습니다")
                if config.external_train_limit:
                    source_rows = stratified_training_subset(
                        source_rows,
                        config.external_train_limit,
                        seed=f"{config.seed}:{name}:"
                        f"purpose={config.external_purpose_filter}",
                    )
                by_source[name] = source_rows
            extended_rows = [row for name in extended_names for row in by_source[name]]
            extended_source_counts = {
                name: len(by_source[name]) for name in extended_names
            }

    # 순서 교란 augmentation은 대회 label을 그대로 물려받으므로 사상 대상이 아니다.
    # 따라서 실제 외부 corpus row만 있는 이 지점에서 한 번만 적용한다.
    extended_rows = align_external_scores_to_competition(
        extended_rows, competition_rows, config.external_score_alignment
    )

    augmentation_builders = {
        "paragraph_order_high_confidence_v1": build_paragraph_order_rows,
        "sentence_order_high_confidence_v1": build_sentence_order_rows,
    }
    if config.organization_augmentation in augmentation_builders:
        augmentation_rows = augmentation_builders[config.organization_augmentation](
            competition_rows, seed=config.seed
        )
        if args.limit is not None:
            augmentation_rows = augmentation_rows[: args.limit]
        augmentation_name = config.organization_augmentation
        extended_rows.extend(augmentation_rows)
        extended_source_counts[augmentation_name] = len(augmentation_rows)
        extended_names = (*extended_names, augmentation_name)

    if args.limit is not None and extended_names:
        by_source = {
            name: [
                row for row in extended_rows if row.get("_dataset_name") == name
            ][: args.limit]
            for name in extended_names
        }
        extended_rows = [row for name in extended_names for row in by_source[name]]
        extended_source_counts = {
            name: len(by_source[name]) for name in extended_names
        }

    if args.training_average_overlay_file:
        competition_rows = overlay_training_metric_averages(
            competition_rows, args.training_average_overlay_file
        )

    validation_rows = load_rows([validation_file]) if validation_file else []
    if args.limit is not None:
        validation_rows = validation_rows[: args.limit]
    # 공식 400편 뒤에 붙인다. 아래 overlap 검사가 두 집합을 함께 보므로 holdout이
    # 학습에서 제대로 빠졌는지도 같은 검사에서 확인된다.
    validation_rows = validation_rows + holdout_rows
    if args.validation_average_overlay_file:
        if not validation_rows:
            raise ValueError(
                "--validation-average-overlay-file은 validation이 있을 때만 사용합니다"
            )
        validation_rows = overlay_validation_metric_averages(
            validation_rows,
            args.validation_average_overlay_file,
            allow_source_superset=args.limit is not None,
        )

    train_rows = competition_rows + extended_rows
    if validation_rows:
        overlap_report = validation_overlap_report(train_rows, validation_rows)
        # validation_leaked는 일반 overlap 탈출구와 진단 전용 동의를 모두 요구한다.
        # 둘 중 하나만 실수로 켠 일반 실험이 누수 metric을 만들지 못하게 한다.
        overlap_allowed = config.allow_validation_overlap and (
            config.primary_data_profile != "validation_leaked"
            or config.allow_validation_leaked
        )
        if overlap_report["has_overlap"] and not overlap_allowed:
            raise ValueError(
                "train/validation leakage를 발견했습니다: "
                f"중복 train row {overlap_report['overlap_training_row_count']}개, "
                f"identifier {overlap_report['identifier_overlap_count']}개, "
                f"정규화 essay 본문 {overlap_report['essay_text_overlap_count']}개"
            )
    else:
        overlap_allowed = config.allow_validation_overlap
        overlap_report = {
            "checked": False,
            "training_rows": len(train_rows),
            "validation_rows": 0,
            "identifier_overlap_count": 0,
            "id_overlap_count": 0,
            "essay_text_overlap_count": 0,
            "overlap_training_row_count": 0,
            "has_overlap": False,
            "id_overlap_examples": [],
            "essay_text_hash_overlap_examples": [],
            "normalization": "NFC+collapse_whitespace+SHA256",
        }

    return PreparedData(
        competition_rows=competition_rows,
        extended_rows=extended_rows,
        validation_rows=validation_rows,
        train_files=[str(path) for path in train_files],
        validation_file=validation_file,
        primary_manifest=primary_manifest,
        external_manifest=extended_manifest,
        extended_names=extended_names,
        extended_source_counts=extended_source_counts,
        overlap_report=overlap_report,
        overlap_allowed=overlap_allowed,
        primary_manifest_used_for_training=args.train_file is None,
    )


def resolve_step_schedule(
    config: RegressionConfig, competition_rows: list[dict[str, Any]]
) -> tuple[RegressionConfig, int, int]:
    """Resolve automatic step settings once and save them in the config."""

    schedule_competition_rows = competition_rows
    if config.split_primary_sources:
        schedule_competition_rows = [
            row
            for row in competition_rows
            if row.get("_dataset_name") == "competition"
        ]
        if not schedule_competition_rows:
            raise ValueError("split_primary_sources에서 official_train row가 없습니다")
    steps_per_epoch = competition_steps_per_epoch(
        schedule_competition_rows,
        batch_size=config.batch_size,
        gradient_accumulation=config.gradient_accumulation,
        inbatch_sampling=config.inbatch_sampling,
    )
    if config.max_train_steps > 0:
        # head_warmup_steps is a real stage boundary only for two-stage runs.
        # Previously ``0 or automatic`` also rewrote LoRA-only artifacts to a
        # non-zero value (for example p28 recorded 1,104), even though those
        # runs trained the head and LoRA jointly from step zero.  Keep the
        # resolved config faithful to the stage that actually ran.
        resolved_head_warmup_steps = 0
        if config.training_mode == "two_stage":
            resolved_head_warmup_steps = (
                config.head_warmup_steps or steps_per_epoch * config.head_epochs
            )
        config = config.with_updates(
            head_warmup_steps=resolved_head_warmup_steps,
            eval_steps=(config.eval_steps or steps_per_epoch),
        )
    final_steps = (
        steps_per_epoch * config.final_competition_epochs
        if config.max_train_steps > 0
        else 0
    )
    if (
        config.max_train_steps > 0
        and config.training_mode == "two_stage"
        and config.head_warmup_steps >= config.max_train_steps + final_steps
    ):
        raise ValueError(
            "two_stage의 head_warmup_steps는 전체 optimizer step보다 작아야 합니다"
        )
    return config, steps_per_epoch, final_steps


def stage_plan_record(
    config: RegressionConfig, final_competition_steps: int
) -> list[dict[str, Any]]:
    """Serialize the human-readable stage plan for ``run.json``."""

    if config.max_train_steps == 0:
        return [
            {"stage": stage, "epochs": epochs} for stage, epochs in config.stage_plan()
        ]
    total_steps = config.max_train_steps + final_competition_steps
    if config.training_mode == "two_stage":
        return [
            {"stage": "head", "steps": min(config.head_warmup_steps, total_steps)},
            {
                "stage": "joint",
                "steps": max(0, total_steps - config.head_warmup_steps),
            },
        ]
    stage = "head" if config.training_mode == "head_only" else "joint"
    return [{"stage": stage, "steps": total_steps}]


# Resolved config and metrics -------------------------------------------------
def resolved_config(args: argparse.Namespace) -> RegressionConfig:
    config_argument = getattr(args, "config", None)
    use_legacy_baseline = bool(getattr(args, "baseline", False))
    if config_argument and use_legacy_baseline:
        raise ValueError("--config와 --baseline은 함께 사용할 수 없습니다")
    config_path = LEGACY_BASELINE_CONFIG_PATH if use_legacy_baseline else config_argument
    config = load_config(config_path) if config_path else RegressionConfig()
    updates = config_updates_from_namespace(args)
    model_argument = args.model if args.model is not None else config.model_id
    spec = resolve_model(model_argument)
    if args.model is not None:
        updates["model_id"] = spec.model_id
        updates["model_slug"] = spec.slug
        updates["model_source_run"] = spec.source_run
        if "model_revision" not in updates:
            updates["model_revision"] = spec.revision
        if "trust_remote_code" not in updates:
            updates["trust_remote_code"] = spec.trust_remote_code
        # registry가 아는 encoder/decoder 종류도 resolved config에 함께 저장한다.
        # 명시적인 --backbone-type이 있으면 사용자의 값을 우선한다.
        if "backbone_type" not in updates:
            updates["backbone_type"] = spec.backbone_type
    elif config_path is None:
        # The dataclass default also becomes a fully resolved/reproducible
        # config when --model is omitted.
        updates.setdefault("model_slug", spec.slug)
        updates.setdefault("model_source_run", spec.source_run)
        updates.setdefault("model_revision", spec.revision)
        updates.setdefault("trust_remote_code", spec.trust_remote_code)
        updates.setdefault("backbone_type", spec.backbone_type)
    return config.with_updates(**updates)


def _tie_aware_spearman(prediction: np.ndarray, truth: np.ndarray) -> float | None:
    """저장소 전체가 쓰는 average-rank Spearman과 같은 계산."""

    from .utils import _average_ranks

    if len(set(truth.tolist())) < 2 or len(set(prediction.tolist())) < 2:
        return None
    value = float(
        np.corrcoef(_average_ranks(truth), _average_ranks(prediction))[0, 1]
    )
    return None if math.isnan(value) else value


def prompt_group_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    groups: np.ndarray,
    holdout: tuple[str, ...] = (),
) -> dict[str, float]:
    """문항별로 쪼갠 지표. 문항 구성에 불변인 저울을 만든다.

    왜 필요한가: 공개 validation 400편의 문항 구성은 우리 학습 편수에 비례한다
    (Q1 25편 … Q5 51편). 그런데 2025 채점 데이터 4,000편은 Q11 1,512 / Q12 1,331편으로
    구성이 전혀 다르다. essay micro 평균으로 checkpoint를 고르면 우리 validation의
    구성에만 맞춘 모델을 고르게 된다.

    `unseen_prompt_*`는 `RegressionConfig.unseen_prompt_holdout`에 적힌 문항만 모은
    부분집합이다. 그 문항은 학습에서 완전히 빠져 있으므로 **미학습 문항 성능**이다.
    보류가 없으면 이 키들은 만들지 않는다(있는 척하면 선택이 조용히 틀어진다).
    """

    from .datasets import prompt_group_id

    result: dict[str, float] = {}
    rmses: list[float] = []
    spearmans: list[float] = []
    for key in sorted(set(np.asarray(groups).ravel().tolist())):
        mask = groups == key
        if int(mask.sum()) < 5:
            continue
        rmses.append(float(np.sqrt(np.mean((prediction[mask] - truth[mask]) ** 2))))
        if len(set(prediction[mask].tolist())) >= 2:
            value = _tie_aware_spearman(prediction[mask], truth[mask])
            if value is not None:
                spearmans.append(value)
    if rmses:
        result["prompt_macro_rmse"] = float(np.mean(rmses))
        result["worst_prompt_rmse"] = float(np.max(rmses))
    if spearmans:
        result["prompt_macro_spearman"] = float(np.mean(spearmans))

    if holdout:
        wanted = {prompt_group_id({"prompt_num": item}) for item in holdout}
        mask = np.isin(groups, list(wanted))
        if int(mask.sum()) >= 5:
            result["unseen_prompt_rmse"] = float(
                np.sqrt(np.mean((prediction[mask] - truth[mask]) ** 2))
            )
            if len(set(prediction[mask].tolist())) >= 2:
                value = _tie_aware_spearman(prediction[mask], truth[mask])
                if value is not None:
                    result["unseen_prompt_spearman"] = value
    return result


def compute_trainer_metrics(
    eval_prediction: Any, *, unseen_prompt_holdout: tuple[str, ...] = ()
) -> dict[str, float | int]:
    """Convert the shared nested metrics into Trainer-friendly scalar keys."""

    predictions = eval_prediction.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    # Keep metric computation in float64.  Labels can contain evaluator means
    # whose distinct JSON values collapse to the same float32 value; that
    # changes tie-aware Spearman and can select a different best checkpoint.
    # Model outputs arrive as float32, but widening them here also keeps the
    # Trainer metric contract identical to final inference.
    label_payload = eval_prediction.label_ids
    average_labels: np.ndarray | None = None
    prompt_groups: np.ndarray | None = None
    if isinstance(label_payload, (tuple, list)):
        if len(label_payload) not in {2, 3}:
            raise ValueError(
                "Trainer label_ids는 labels, average_labels, (선택) prompt_group_ids "
                "여야 합니다"
            )
        labels = np.asarray(label_payload[0], dtype=np.float64)
        average_labels = np.asarray(label_payload[1], dtype=np.float64).reshape(-1)
        if len(label_payload) == 3:
            prompt_groups = np.asarray(label_payload[2]).reshape(-1)
    else:
        # Historical/unit-test callers can still provide only the trait labels.
        labels = np.asarray(label_payload, dtype=np.float64)
    scores = np.asarray(predictions, dtype=np.float64)
    nested = regression_metrics(labels, scores, average_labels)
    flat: dict[str, float | int] = {"count": nested["count"]}
    for trait, values in nested["traits"].items():
        flat[f"{trait}_rmse"] = values["rmse"]
        if values["spearman"] is not None:
            flat[f"{trait}_spearman"] = values["spearman"]
    flat["overall_rmse"] = nested["overall"]["rmse"]
    if nested["overall"]["spearman"] is not None:
        flat["overall_spearman"] = nested["overall"]["spearman"]
    # 위 overall은 trait별 지표의 산술평균(정의 A)이라 리더보드와 다르다. 공식 지표와
    # 2026-08-06 공지의 정수 반올림 지표를 학습 로그에도 함께 남겨, checkpoint 선택과
    # 곡선 판독을 실제 채점 기준으로 할 수 있게 한다. 학습 중에는 `score.average`를
    # collator가 저장된 score.average를 별도 label로 주므로 final inference와 같은 gold를
    # 쓴다. 과거/unit-test caller가 그 label을 생략한 경우에만 trait 평균으로 대체한다.
    official = nested["official"]
    raw = official["raw_continuous"]
    flat["official_rmse"] = raw["rmse"]
    if raw["spearman"] is not None:
        flat["official_spearman"] = raw["spearman"]
    matched = official["average_matched"]
    flat["official_matched_rmse"] = matched["rmse"]
    if matched["spearman"] is not None:
        flat["official_matched_spearman"] = matched["spearman"]
    matched_scores = average_matched_integer_scores(scores)
    # 리더보드 라벨 분포로 옮겨 잰 RMSE. 근거는 main_code/distribution_shift.py에 있다.
    # 요약: 로컬 400편은 정답 SD 0.653인데 채점은 SD 0.77쯤인 집합에서 이루어진다.
    # 기존 official_matched_rmse는 그대로 두고 저울을 하나 더 놓는다.
    flat["official_matched_rmse_shifted"] = shifted_rmse(
        matched_scores.mean(axis=1),
        average_labels if average_labels is not None else labels.mean(axis=1),
    )
    matched_trait_rmses = [
        float(np.sqrt(np.mean((matched_scores[:, index] - labels[:, index]) ** 2)))
        for index in range(labels.shape[1])
    ]
    flat["submitted_trait_macro_rmse"] = float(np.mean(matched_trait_rmses))
    if prompt_groups is not None and prompt_groups.size == scores.shape[0]:
        flat.update(
            prompt_group_metrics(
                matched_scores.mean(axis=1),
                average_labels if average_labels is not None else labels.mean(axis=1),
                prompt_groups,
                unseen_prompt_holdout,
            )
        )
    # 평가식 문의 결과가 어느 축으로 확정돼도 재학습 로그에서 같은 checkpoint를
    # 재선택할 수 있도록 3 surface × 3 aggregation × 2 metric을 모두 남긴다.
    # 기존 alias 위 값은 그대로 보존한다.
    for surface, metrics in score_metric_matrix(
        labels, scores, average_labels
    ).items():
        for metric_name, value in metrics.items():
            if value is not None:
                flat[f"{surface}_{metric_name}"] = float(value)
    return flat


# Trainer construction and callbacks ----------------------------------------
ParameterUpdateSnapshot = dict[str, dict[str, torch.Tensor]]


def capture_parameter_update_snapshot(
    model: RegressionScorer,
) -> ParameterUpdateSnapshot:
    """Copy only score-side and LoRA parameters to CPU before training.

    A full backbone snapshot would be prohibitively large and is unnecessary:
    the optimizer can update only ``scoring_parameters()`` and parameters whose
    names contain ``lora_``.  Keeping the original dtype also makes the later
    exact ``changed`` check independent of metric rounding.
    """

    scoring_ids = {id(parameter) for parameter in model.scoring_parameters()}
    snapshot: ParameterUpdateSnapshot = {"score_side": {}, "lora": {}}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if id(parameter) in scoring_ids:
                group = "score_side"
            elif name.startswith("backbone.") and "lora_" in name:
                group = "lora"
            else:
                continue
            snapshot[group][name] = parameter.detach().cpu().clone()
    return snapshot


def _parameter_group_update_summary(
    initial: dict[str, torch.Tensor],
    final: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Measure one small parameter group's movement without concatenating it."""

    if initial.keys() != final.keys():
        missing = sorted(initial.keys() - final.keys())
        unexpected = sorted(final.keys() - initial.keys())
        raise RuntimeError(
            "학습 전후 parameter group 구성이 달라졌습니다: "
            f"missing={missing}, unexpected={unexpected}"
        )

    tensor_count = len(initial)
    parameter_count = sum(tensor.numel() for tensor in initial.values())
    if tensor_count == 0:
        return {
            "parameter_count": 0,
            "tensor_count": 0,
            "initial_l2": None,
            "final_l2": None,
            "delta_l2": None,
            "max_abs_delta": None,
            "changed": False,
        }

    initial_squared = 0.0
    final_squared = 0.0
    delta_squared = 0.0
    max_abs_delta = 0.0
    changed = False
    for name, initial_tensor in initial.items():
        final_tensor = final[name]
        if initial_tensor.shape != final_tensor.shape:
            raise RuntimeError(
                f"학습 전후 parameter shape이 달라졌습니다: {name}: "
                f"{tuple(initial_tensor.shape)} != {tuple(final_tensor.shape)}"
            )
        initial_float = initial_tensor.to(dtype=torch.float64)
        final_float = final_tensor.to(dtype=torch.float64)
        delta = final_float - initial_float
        initial_squared += float(initial_float.square().sum().item())
        final_squared += float(final_float.square().sum().item())
        delta_squared += float(delta.square().sum().item())
        if delta.numel():
            max_abs_delta = max(max_abs_delta, float(delta.abs().max().item()))
        changed = changed or not torch.equal(initial_tensor, final_tensor)

    return {
        "parameter_count": int(parameter_count),
        "tensor_count": int(tensor_count),
        "initial_l2": math.sqrt(initial_squared),
        "final_l2": math.sqrt(final_squared),
        "delta_l2": math.sqrt(delta_squared),
        "max_abs_delta": max_abs_delta,
        "changed": changed,
    }


def summarize_parameter_updates(
    model: RegressionScorer,
    initial: ParameterUpdateSnapshot,
) -> dict[str, Any]:
    """Return JSON-safe evidence that the score side and LoRA were updated."""

    final = capture_parameter_update_snapshot(model)
    initial_score_side = initial.get("score_side", {})
    final_score_side = final["score_side"]

    def score_side_subset(prefixes: tuple[str, ...]) -> tuple[dict, dict]:
        initial_subset = {
            name: tensor
            for name, tensor in initial_score_side.items()
            if name.startswith(prefixes)
        }
        final_subset = {name: final_score_side[name] for name in initial_subset}
        return initial_subset, final_subset

    direct_initial, direct_final = score_side_subset(("heads.",))
    detail_initial, detail_final = score_side_subset(("detail_heads.",))
    rater_initial, rater_final = score_side_subset(("detail_evaluator_severity",))
    anchor_initial, anchor_final = score_side_subset(("criterion_anchor_gates",))
    boundary_initial, boundary_final = score_side_subset(
        ("paragraph_boundary_head.",)
    )
    return {
        "snapshot_device": "cpu",
        "score_side": _parameter_group_update_summary(
            initial_score_side, final_score_side
        ),
        # Detail aggregate leaves keep the legacy direct heads for checkpoint
        # compatibility.  Separate summaries make an intentionally dormant
        # direct branch distinguishable from a broken detail update.
        "direct_heads": _parameter_group_update_summary(direct_initial, direct_final),
        "detail_heads": _parameter_group_update_summary(detail_initial, detail_final),
        "detail_rater_severity": _parameter_group_update_summary(
            rater_initial, rater_final
        ),
        "criterion_anchor_gates": _parameter_group_update_summary(
            anchor_initial, anchor_final
        ),
        "paragraph_boundary_head": _parameter_group_update_summary(
            boundary_initial, boundary_final
        ),
        "lora": _parameter_group_update_summary(initial.get("lora", {}), final["lora"]),
    }


def optimizer_for_training(loaded: LoadedRegressionModel) -> torch.optim.Optimizer:
    """한 optimizer에 head와 LoRA group을 미리 등록한다.

    two-stage의 head 구간에는 LoRA가 frozen이라 gradient가 없고, callback이
    unfreeze한 뒤부터 같은 optimizer가 LoRA group을 업데이트한다.
    """

    model = loaded.scorer
    groups = [
        {
            # head뿐 아니라 선택적으로 생기는 attention-pool query/projection과
            # scalar layer-mix weight도 같은 score-side group에서 학습한다.
            "params": list(model.scoring_parameters()),
            "lr": loaded.config.head_learning_rate,
        }
    ]
    adapters = [
        (name, parameter)
        for name, parameter in model.backbone.named_parameters()
        if "lora_" in name
    ]
    if adapters:
        ratio = float(loaded.config.lora_plus_lr_ratio)
        if ratio == 1.0:
            # 기본값은 parameter 구성과 optimizer group 수까지 기존 동작과 같다.
            groups.append(
                {
                    "params": [parameter for _, parameter in adapters],
                    "lr": loaded.config.lora_learning_rate,
                }
            )
        else:
            lora_a = [
                parameter for name, parameter in adapters if "lora_A." in name
            ]
            lora_b = [
                parameter for name, parameter in adapters if "lora_B." in name
            ]
            if len(lora_a) + len(lora_b) != len(adapters) or not lora_a or not lora_b:
                raise ValueError("LoRA+에는 lora_A와 lora_B parameter가 모두 필요합니다")
            groups.extend(
                [
                    {
                        "params": lora_a,
                        "lr": loaded.config.lora_learning_rate,
                        "lora_lr_scale": 1.0,
                    },
                    {
                        "params": lora_b,
                        "lr": loaded.config.lora_learning_rate * ratio,
                        "lora_lr_scale": ratio,
                    },
                ]
            )
    elif loaded.config.training_mode != "head_only":
        raise ValueError("LoRA 학습 모드인데 optimizer에 넣을 adapter가 없습니다")
    return torch.optim.AdamW(groups, weight_decay=loaded.config.weight_decay)


def audit_rubric_conditioned_inputs(
    tokenizer: Any,
    config: RegressionConfig,
    *,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Fail before optimization if any RC essay or readout anchor would truncate."""

    if config.input_format != "rubric_conditioned_v1":
        return None
    split_summaries: dict[str, Any] = {}
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        audit_collator = RegressionCollator(
            tokenizer,
            config,
            include_labels=False,
            include_metadata=False,
        )
        for start in range(0, len(rows), 128):
            audit_collator(rows[start : start + 128])
        summary = audit_collator.input_length_summary()
        summary["essay_span_preserved_count"] = len(rows)
        summary["standalone_base_token_prefix_count"] = len(rows)
        summary["complete_anchor_count"] = len(rows)
        split_summaries[split] = summary
    return {
        "input_format": config.input_format,
        "rubric_profile": config.rubric_profile,
        "criterion_readout": config.criterion_readout,
        "expected_anchor_count_per_row": 9,
        "splits": split_summaries,
    }


class StageSwitchCallback(TrainerCallback):
    """epoch 또는 optimizer-step 경계에서 head-only를 head+LoRA로 전환한다."""

    def __init__(self, model: RegressionScorer, config: RegressionConfig):
        self.model = model
        self.config = config
        self.active_stage: str | None = model.current_stage

    def switch_stage(self, stage: str) -> None:
        """Freeze/unfreeze the large backbone only when the stage changes."""

        if stage == self.active_stage:
            return
        self.model.set_training_stage(stage)
        self.active_stage = stage

    def stage_for_epoch(self, epoch_index: int) -> str:
        if self.config.training_mode == "head_only":
            return "head"
        if self.config.training_mode == "lora_only":
            return "joint"
        return "head" if epoch_index < self.config.head_epochs else "joint"

    def on_epoch_begin(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> None:
        if self.config.max_train_steps > 0:
            return
        stage = self.stage_for_epoch(int(state.epoch or 0))
        self.switch_stage(stage)

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self.config.max_train_steps <= 0:
            return
        if self.config.training_mode == "head_only":
            stage = "head"
        elif self.config.training_mode == "lora_only":
            stage = "joint"
        else:
            stage = (
                "head"
                if int(state.global_step) < self.config.head_warmup_steps
                else "joint"
            )
        self.switch_stage(stage)


class QuantizationScheduleCallback(TrainerCallback):
    """Expose Trainer optimizer steps to the opt-in quantization schedule."""

    def __init__(self, model: RegressionScorer):
        self.model = model

    def on_train_begin(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> None:
        self.model.set_quantization_global_step(int(state.global_step))

    def on_step_begin(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> None:
        self.model.set_quantization_global_step(int(state.global_step))


def lora_schedule_multiplier(
    scheduler_type: str,
    current_step: int,
    *,
    total_steps: int,
    warmup_steps: int,
) -> float:
    """Return the joint-local LoRA LR multiplier for one optimizer step.

    ``current_step`` is zero-based within the joint stage.  Linear/cosine
    follow the same warmup-then-decay shape used by Transformers; constant
    keeps the post-warmup LR fixed.  Clamping also makes the helper safe for
    the final callback event at ``current_step == total_steps``.
    """

    if scheduler_type not in {"constant", "linear", "cosine"}:
        raise ValueError(f"unsupported scheduler_type={scheduler_type!r}")
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps]")

    step = min(max(int(current_step), 0), int(total_steps))
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    if scheduler_type == "constant":
        return 1.0

    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    if scheduler_type == "linear":
        return 1.0 - progress
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class LoraJointSchedulerCallback(TrainerCallback):
    """Give the LoRA optimizer group a scheduler local to the joint stage.

    Trainer still owns the original global scheduler.  It therefore keeps
    controlling the score/head parameter group exactly as before.  This
    callback overrides only the LoRA group's LR after each global scheduler
    update and again immediately before the next optimizer step.
    """

    def __init__(
        self,
        model: RegressionScorer,
        optimizer: torch.optim.Optimizer,
        config: RegressionConfig,
    ):
        if config.lora_scheduler_scope != "joint":
            raise ValueError("LoraJointSchedulerCallback requires joint scope")
        adapter_ids = {
            id(parameter)
            for name, parameter in model.backbone.named_parameters()
            if "lora_" in name
        }
        matching_groups = [
            index
            for index, group in enumerate(optimizer.param_groups)
            if any(id(parameter) in adapter_ids for parameter in group["params"])
        ]
        if not adapter_ids or not matching_groups:
            raise ValueError("LoRA optimizer group을 찾을 수 없습니다")

        self.optimizer = optimizer
        self.config = config
        self.group_indexes = matching_groups
        self.group_lr_scales = [
            float(optimizer.param_groups[index].get("lora_lr_scale", 1.0))
            for index in matching_groups
        ]
        self.base_learning_rate = float(config.lora_learning_rate)
        self.joint_start_step: int | None = (
            int(config.head_warmup_steps) if config.max_train_steps > 0 else None
        )
        self.joint_total_steps: int | None = None
        self.warmup_steps: int | None = None
        self.last_learning_rate = 0.0

    def _maybe_start_epoch_joint(self, state: Any) -> None:
        if self.config.max_train_steps > 0 or self.joint_start_step is not None:
            return
        epoch_index = int(state.epoch or 0)
        if epoch_index >= self.config.head_epochs:
            self.joint_start_step = int(state.global_step)

    def _resolve_plan(self, state: Any) -> None:
        if self.joint_start_step is None:
            return
        if self.joint_total_steps is not None:
            return
        remaining = int(state.max_steps) - self.joint_start_step
        if remaining < 1:
            raise ValueError("joint stage에는 optimizer step이 하나 이상 필요합니다")
        self.joint_total_steps = remaining
        self.warmup_steps = min(
            remaining, int(math.ceil(remaining * self.config.warmup_ratio))
        )

    def learning_rate_at(self, global_step: int, state: Any) -> float:
        self._resolve_plan(state)
        if (
            self.joint_start_step is None
            or self.joint_total_steps is None
            or self.warmup_steps is None
            or int(global_step) < self.joint_start_step
        ):
            return 0.0
        multiplier = lora_schedule_multiplier(
            self.config.lr_scheduler_type,
            int(global_step) - self.joint_start_step,
            total_steps=self.joint_total_steps,
            warmup_steps=self.warmup_steps,
        )
        return self.base_learning_rate * multiplier

    def _apply(self, state: Any) -> None:
        learning_rate = self.learning_rate_at(int(state.global_step), state)
        for group_index, scale in zip(self.group_indexes, self.group_lr_scales):
            self.optimizer.param_groups[group_index]["lr"] = learning_rate * scale
        self.last_learning_rate = learning_rate

    def on_train_begin(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> None:
        self._apply(state)

    def on_epoch_begin(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> None:
        self._maybe_start_epoch_joint(state)
        self._apply(state)

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._maybe_start_epoch_joint(state)
        self._apply(state)

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        # Trainer's global scheduler has just touched every param group.  Put
        # the LoRA group back on its joint-local schedule for the next step.
        self._apply(state)

    def summary(self) -> dict[str, Any]:
        return {
            "scope": self.config.lora_scheduler_scope,
            "scheduler_type": self.config.lr_scheduler_type,
            "base_learning_rate": self.base_learning_rate,
            "group_lr_scales": self.group_lr_scales,
            "joint_start_step": self.joint_start_step,
            "joint_total_steps": self.joint_total_steps,
            "warmup_steps": self.warmup_steps,
            "last_learning_rate": self.last_learning_rate,
        }


class BestCheckpointCallback(TrainerCallback):
    """RMSE/Spearman best와 기존 선택-metric checkpoint를 함께 보존한다.

    ``best_checkpoint/``는 이전 runner와 artifact reader가 기대하는 경로이므로
    ``config.best_checkpoint_metric``이 좋아질 때 그대로 갱신한다. 두 지표의
    독립적인 best는 그와 별도로 sibling directory에 additive하게 저장한다.
    """

    CORE_METRICS = {
        "rmse": ("eval_overall_rmse", False),
        "spearman": ("eval_overall_spearman", True),
        # 공식 지표(essay별 세 trait 평균 하나)와 정수 반올림 지표. rmse/spearman은
        # trait별 지표의 평균이라 리더보드와 정의가 다르므로 새 실험은 아래를 쓴다.
        "official_rmse": ("eval_official_rmse", False),
        "official_spearman": ("eval_official_spearman", True),
        "official_matched_rmse": ("eval_official_matched_rmse", False),
        # 문항 구성에 불변인 지표들. legacy_core retention에서 이 목록의 best가 모두
        # sibling으로 저장되므로, run 하나에서 여러 저울의 승자를 동시에 얻는다.
        "prompt_macro_rmse": ("eval_prompt_macro_rmse", False),
        "worst_prompt_rmse": ("eval_worst_prompt_rmse", False),
        "prompt_macro_spearman": ("eval_prompt_macro_spearman", True),
        # unseen_prompt_holdout이 없는 run에서는 이 키가 생성되지 않아 선택이 그냥
        # 갱신되지 않는다(있는 척하지 않는다).
        "unseen_prompt_rmse": ("eval_unseen_prompt_rmse", False),
        "unseen_prompt_spearman": ("eval_unseen_prompt_spearman", True),
        # 평가셋 분포로 옮겨 잰 같은 지표. 꼬리 표본을 무겁게 본다.
        "official_matched_rmse_shifted": (
            "eval_official_matched_rmse_shifted",
            False,
        ),
        "submitted_trait_macro_rmse": (
            "eval_submitted_trait_macro_rmse",
            False,
        ),
        "official_matched_spearman": (
            "eval_official_matched_spearman",
            True,
        ),
    }
    TRAIT_METRICS = {
        f"{trait}_{metric}": (f"eval_{trait}_{metric}", metric == "spearman")
        for trait in ("content", "organization", "expression")
        for metric in ("rmse", "spearman")
    }
    SURFACE_METRICS = {
        f"{surface}_{aggregation}_{metric}": (
            f"eval_{surface}_{aggregation}_{metric}",
            metric == "spearman",
        )
        for surface in ("raw_continuous", "independent_half_up", "average_matched")
        for aggregation in ("trait_macro", "mean_first", "pooled")
        for metric in ("rmse", "spearman")
    }

    @classmethod
    def metric_contract(cls, metric: str) -> tuple[str, bool]:
        if metric in cls.CORE_METRICS:
            return cls.CORE_METRICS[metric]
        if metric in cls.TRAIT_METRICS:
            return cls.TRAIT_METRICS[metric]
        if metric in cls.SURFACE_METRICS:
            return cls.SURFACE_METRICS[metric]
        raise ValueError(f"checkpoint metric 구현이 없습니다: {metric}")

    @staticmethod
    def resolved_secondary_metric(config: RegressionConfig) -> str | None:
        value = getattr(config, "secondary_checkpoint_metric", "auto")
        if value == "none":
            return None
        if value != "auto":
            return value
        return {
            "rmse": "spearman",
            "spearman": "rmse",
            "official_rmse": "official_matched_rmse",
            "official_matched_rmse": "official_rmse",
            "official_matched_rmse_shifted": "official_matched_spearman",
            "submitted_trait_macro_rmse": "official_matched_spearman",
            "official_matched_spearman": "submitted_trait_macro_rmse",
            "average_matched_trait_macro_rmse": (
                "average_matched_mean_first_spearman"
            ),
            "average_matched_mean_first_spearman": (
                "average_matched_trait_macro_rmse"
            ),
        }.get(config.best_checkpoint_metric)

    def __init__(self, loaded: LoadedRegressionModel, directory: Path):
        self.loaded = loaded
        self.directory = directory
        self.selected_metric = loaded.config.best_checkpoint_metric
        self.retention = getattr(loaded.config, "checkpoint_retention", "legacy_core")
        if self.retention == "legacy_core":
            # Bit-for-bit historical artifact policy.  Current mixed runs may
            # start new Python processes while this code is being extended.
            self.metrics = dict(self.CORE_METRICS)
            if self.selected_metric not in self.metrics:
                self.metrics[self.selected_metric] = self.metric_contract(
                    self.selected_metric
                )
        else:
            self.metrics = {
                self.selected_metric: self.metric_contract(self.selected_metric)
            }
            secondary = self.resolved_secondary_metric(loaded.config)
            if secondary is not None and secondary != self.selected_metric:
                self.metrics[secondary] = self.metric_contract(secondary)
        self.metric_name, self.greater_is_better = self.metrics[self.selected_metric]
        self.metric_directories = {
            metric: directory.parent / f"best_checkpoint_{metric}"
            for metric in self.metrics
        }
        self.selections: dict[str, dict[str, Any]] = {
            metric: {
                "checkpoint": None,
                "metric": metric_name,
                "metric_value": None,
                "epoch": None,
                "global_step": None,
            }
            for metric, (metric_name, _) in self.metrics.items()
        }

    @property
    def best_metric(self) -> float | None:
        return self.selections[self.selected_metric]["metric_value"]

    @property
    def best_epoch(self) -> float | None:
        return self.selections[self.selected_metric]["epoch"]

    @property
    def best_global_step(self) -> int | None:
        return self.selections[self.selected_metric]["global_step"]

    def selection_summary(self) -> dict[str, dict[str, Any]]:
        """Return a JSON-safe copy for ``run.json`` and trainer summaries."""

        return {
            metric: dict(selection) for metric, selection in self.selections.items()
        }

    def _save_selection(
        self,
        *,
        directory: Path,
        metric_name: str,
        metric_value: float,
        epoch: float | None,
        global_step: int,
    ) -> None:
        save_checkpoint(self.loaded, directory)
        write_json(
            directory / "selection.json",
            {
                "metric": metric_name,
                "metric_value": metric_value,
                "epoch": epoch,
                "global_step": global_step,
            },
        )

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        epoch = float(state.epoch) if isinstance(state.epoch, (int, float)) else None
        global_step = int(state.global_step)
        for metric_key, (metric_name, greater_is_better) in self.metrics.items():
            value = (metrics or {}).get(metric_name)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                continue
            metric_value = float(value)
            previous = self.selections[metric_key]["metric_value"]
            if previous is not None:
                improved = (
                    metric_value > previous
                    if greater_is_better
                    else metric_value < previous
                )
                if not improved:
                    continue

            selection = self.selections[metric_key]
            selected_directory = self.metric_directories[metric_key]
            if (
                self.retention == "selected_pair"
                and metric_key == self.selected_metric
            ):
                selected_directory = self.directory
            selection.update(
                checkpoint=str(selected_directory),
                metric_value=metric_value,
                epoch=epoch,
                global_step=global_step,
            )

            # 선택 지표는 먼저 기존 경로를 갱신한다. 두 번째 additive 저장이
            # 실패하더라도 기존 best_checkpoint 계약은 온전히 남는다.
            if metric_key == self.selected_metric:
                self._save_selection(
                    directory=self.directory,
                    metric_name=metric_name,
                    metric_value=metric_value,
                    epoch=epoch,
                    global_step=global_step,
                )
            if not (
                self.retention == "selected_pair"
                and metric_key == self.selected_metric
            ):
                self._save_selection(
                    directory=self.metric_directories[metric_key],
                    metric_name=metric_name,
                    metric_value=metric_value,
                    epoch=epoch,
                    global_step=global_step,
                )


def training_arguments(
    config: RegressionConfig,
    *,
    output_dir: Path,
    has_validation: bool,
    total_steps: int | None = None,
) -> Any:
    try:
        from transformers import TrainingArguments
    except ImportError as exc:
        raise RuntimeError("학습에는 transformers Trainer가 필요합니다") from exc

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    step_mode = config.max_train_steps > 0
    if step_mode and (total_steps is None or total_steps < 1):
        raise ValueError("step mode에는 total_steps가 필요합니다")
    evaluate = has_validation and config.eval_every_epoch
    return TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=has_validation and config.eval_every_epoch,
        num_train_epochs=(
            1.0
            if step_mode
            else float(sum(epochs for _, epochs in config.stage_plan()))
        ),
        max_steps=(int(total_steps) if step_mode else -1),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.head_learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        bf16=use_bf16,
        eval_strategy=(
            "steps" if step_mode and evaluate else "epoch" if evaluate else "no"
        ),
        eval_steps=(config.eval_steps if step_mode and evaluate else None),
        logging_strategy=("steps" if step_mode else "epoch"),
        logging_steps=(config.eval_steps if step_mode else 500),
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels", "average_labels", "prompt_group_ids"],
        seed=config.seed,
        data_seed=config.seed,
    )


def run_training(
    loaded: LoadedRegressionModel,
    *,
    train_dataset: EssayRegressionDataset,
    eval_dataset: EssayRegressionDataset | None,
    collator: RegressionCollator,
    output_dir: Path,
    best_checkpoint_dir: Path,
    final_competition_steps: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from transformers import Trainer
    except ImportError as exc:
        raise RuntimeError("학습에는 transformers Trainer가 필요합니다") from exc

    step_sampler = None
    if loaded.config.max_train_steps > 0:
        step_sampler = MultiSourceStepSampler(
            train_dataset,
            batch_size=loaded.config.batch_size,
            gradient_accumulation=loaded.config.gradient_accumulation,
            main_steps=loaded.config.max_train_steps,
            final_competition_steps=final_competition_steps,
            schedule=loaded.config.dataset_schedule,
            competition_mix_ratio=loaded.config.competition_mix_ratio,
            competition_every_n_steps=loaded.config.competition_every_n_steps,
            extended_source_sampling=loaded.config.extended_source_sampling,
            extended_source_weights=loaded.config.extended_source_weights,
            inbatch_sampling=loaded.config.inbatch_sampling,
            seed=loaded.config.seed,
        )

    class RegressionTrainer(Trainer):
        """기본 Trainer에서 train index sampler만 선택적으로 바꾼다."""

        def _get_train_sampler(self, train_dataset: Any | None = None):
            dataset = train_dataset if train_dataset is not None else self.train_dataset
            if step_sampler is not None:
                return step_sampler
            if loaded.config.inbatch_sampling == "same_question":
                return SameQuestionSampler(
                    dataset,
                    batch_size=loaded.config.batch_size,
                    seed=loaded.config.seed,
                )
            return super()._get_train_sampler(train_dataset)

    callback = StageSwitchCallback(loaded.scorer, loaded.config)
    optimizer = optimizer_for_training(loaded)
    arguments = training_arguments(
        loaded.config,
        output_dir=output_dir,
        has_validation=eval_dataset is not None,
        total_steps=(step_sampler.total_steps if step_sampler is not None else None),
    )
    callbacks: list[TrainerCallback] = [callback]
    if quantization_objective_active(loaded.config):
        callbacks.append(QuantizationScheduleCallback(loaded.scorer))
    lora_scheduler_callback = None
    if loaded.config.lora_scheduler_scope == "joint":
        lora_scheduler_callback = LoraJointSchedulerCallback(
            loaded.scorer, optimizer, loaded.config
        )
        # StageSwitchCallback runs first at the shared boundary so the adapter
        # is trainable before its joint-local LR is applied.
        callbacks.append(lora_scheduler_callback)
    best_callback = None
    if eval_dataset is not None and loaded.config.eval_every_epoch:
        best_callback = BestCheckpointCallback(loaded, best_checkpoint_dir)
        callbacks.append(best_callback)
    trainer = RegressionTrainer(
        model=loaded.scorer,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        # 미학습 문항 목록은 config에만 있고 Trainer는 config를 모른다. 전역 변수나
        # 환경변수로 넘기면 run 사이에 새는 상태가 되므로 부분 적용으로 묶는다.
        compute_metrics=(
            partial(
                compute_trainer_metrics,
                unseen_prompt_holdout=parse_unseen_prompt_holdout(
                    loaded.config.unseen_prompt_holdout
                ),
            )
            if eval_dataset is not None
            else None
        ),
        optimizers=(optimizer, None),
        callbacks=callbacks,
    )
    initial_parameter_snapshot = capture_parameter_update_snapshot(loaded.scorer)
    train_result = trainer.train()
    # eval_steps의 배수가 아닌 schedule도 마지막 모델을 한 번은 비교한다.
    # Trainer.evaluate()는 on_evaluate callback도 호출하므로 best checkpoint 갱신과
    # log_history 기록이 기존 주기 평가와 같은 경로를 탄다.
    final_evaluation_performed = False
    if (
        step_sampler is not None
        and eval_dataset is not None
        and loaded.config.eval_every_epoch
        and int(trainer.state.global_step) % loaded.config.eval_steps != 0
    ):
        trainer.evaluate()
        final_evaluation_performed = True
    trainer.save_state()
    history = [dict(record) for record in trainer.state.log_history]
    parameter_updates = summarize_parameter_updates(
        loaded.scorer, initial_parameter_snapshot
    )
    summary = {
        "epochs": (
            None if step_sampler is not None else int(arguments.num_train_epochs)
        ),
        "planned_steps": (
            step_sampler.total_steps if step_sampler is not None else None
        ),
        "actual_steps": int(trainer.state.global_step),
        "final_evaluation_performed": final_evaluation_performed,
        "source_schedule": (
            step_sampler.planned_summary() if step_sampler is not None else None
        ),
        "trainer_output_dir": str(output_dir),
        "train_metrics": dict(train_result.metrics),
        "parameter_updates": parameter_updates,
        "lr_scheduler_type": loaded.config.lr_scheduler_type,
        "warmup_ratio": loaded.config.warmup_ratio,
        "lora_scheduler_scope": loaded.config.lora_scheduler_scope,
        "lora_scheduler": (
            lora_scheduler_callback.summary()
            if lora_scheduler_callback is not None
            else None
        ),
        "best_checkpoint": (
            str(best_checkpoint_dir)
            if best_callback is not None and best_callback.best_metric is not None
            else None
        ),
        "best_checkpoint_metric": (
            best_callback.best_metric if best_callback is not None else None
        ),
        "best_checkpoint_metric_name": (
            best_callback.metric_name if best_callback is not None else None
        ),
        "best_checkpoint_epoch": (
            best_callback.best_epoch if best_callback is not None else None
        ),
        "best_checkpoint_global_step": (
            best_callback.best_global_step if best_callback is not None else None
        ),
        "best_checkpoints": (
            best_callback.selection_summary() if best_callback is not None else None
        ),
    }
    return history, summary


def stage_for_log(
    config: RegressionConfig, epoch: Any, step: Any = None
) -> tuple[str, float | None]:
    """Trainer log에 현재 stage와 stage 내부 epoch/step을 붙인다."""

    if config.max_train_steps > 0:
        value = float(step) if isinstance(step, (int, float)) else None
        if config.training_mode == "head_only":
            return "head", value
        if config.training_mode == "lora_only":
            return "joint", value
        if value is None or value <= config.head_warmup_steps:
            return "head", value
        return "joint", value - config.head_warmup_steps

    value = float(epoch) if isinstance(epoch, (int, float)) else None
    if config.training_mode == "head_only":
        return "head", value
    if config.training_mode == "lora_only":
        return "joint", value
    if value is None or value <= config.head_epochs:
        return "head", value
    return "joint", value - config.head_epochs


# End-to-end entrypoint -------------------------------------------------------
def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_argument = (
        args.model
        or args.config
        or ("legacy_baseline" if args.baseline else "config_default")
    )
    # Config 해석이나 데이터 검증이 실패해도 runner/aggregator가 이 시도를 찾을 수
    # 있도록 raw experiment identity를 먼저 남긴 뒤, resolve 후 model 정보를 보강한다.
    experiment = experiment_record(args, model_argument)
    write_json(output / "experiment.json", experiment)

    config = resolved_config(args)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit은 1 이상이어야 합니다")
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # batch/gradient-checkpointing 속도 실험을 결과만 보고도 비교할 수 있게
        # 이 process의 model load+train peak를 한 번의 run에 기록한다.
        torch.cuda.reset_peak_memory_stats(device)
    log_path = output / "train_log.jsonl"
    experiment = experiment_record(args, model_argument, config)
    write_json(output / "experiment.json", experiment)

    data = prepare_training_data(args, config)
    config, competition_epoch_steps, final_competition_steps = resolve_step_schedule(
        config, data.competition_rows
    )
    # AIHub의 수십~수천 문제마다 별도 head를 만들지 않는다. registry는 대회
    # train만으로 고정한다. 같은 원문을 가진 허용 데이터는 기존 head로 가고,
    # 외부 문제는 prompt_id=-1이라 공용 head만 사용한다.
    prompt_registry = (
        build_prompt_registry(data.competition_rows)
        if config.prompt_head_mode != "none"
        else ()
    )
    # 평가자 ID는 validation/test에서 만들지 않는다. H5의 registry는 이 run이
    # 실제로 학습에 쓰는 row만으로 고정하고 checkpoint config에 함께 저장한다.
    detail_rater_registry = (
        build_detail_rater_registry(data.train_rows)
        if config.detail_rater_loss_weight > 0
        else ()
    )
    config = config.with_updates(
        prompt_registry=prompt_registry,
        detail_rater_registry=detail_rater_registry,
    )

    train_dataset = EssayRegressionDataset(
        data.train_rows, config, split="train", require_labels=True
    )
    eval_dataset = (
        EssayRegressionDataset(
            data.validation_rows, config, split="validation", require_labels=True
        )
        if data.validation_rows
        else None
    )
    data_signature = training_data_signature(
        config, data.competition_rows, data.extended_rows, data.validation_rows
    )
    detail_summary = (
        detail_supervision_summary(
            data.train_rows,
            config.detail_rater_registry,
            include_official_rater_set=(
                config.detail_head_mode in RATER_SET_HEAD_MODES
            ),
        )
        if config.detail_head_mode != "none"
        else None
    )
    save_config(config, output / "resolved_config.json")
    prompt_lookup = set(config.prompt_registry)
    validation_unknown_prompts = (
        sum(question_key(row) not in prompt_lookup for row in data.validation_rows)
        if config.prompt_head_mode != "none"
        else 0
    )

    loaded = build_model(config, device=device)
    input_token_audit = audit_rubric_conditioned_inputs(
        loaded.tokenizer,
        config,
        train_rows=data.train_rows,
        validation_rows=data.validation_rows,
    )
    if input_token_audit is not None:
        write_json(output / "input_token_audit.json", input_token_audit)
    collator = RegressionCollator(
        loaded.tokenizer,
        config,
        include_labels=True,
        include_metadata=False,
    )

    history, trainer_summary = run_training(
        loaded,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        collator=collator,
        output_dir=output / "trainer",
        best_checkpoint_dir=output / "best_checkpoint",
        final_competition_steps=final_competition_steps,
    )
    annotated_history = []
    for record in history:
        stage, stage_progress = stage_for_log(
            config, record.get("epoch"), record.get("step")
        )
        annotated = {"stage": stage, **record}
        if stage_progress is not None:
            annotated["stage_step" if config.max_train_steps > 0 else "stage_epoch"] = (
                stage_progress
            )
        annotated_history.append(annotated)
    write_jsonl(log_path, annotated_history)

    checkpoint = save_checkpoint(loaded, output / "checkpoint")
    best_checkpoint = trainer_summary.get("best_checkpoint")
    best_checkpoints = trainer_summary.get("best_checkpoints")
    peak_reserved_bytes = (
        int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
    )
    peak_allocated_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    write_json(
        output / "run.json",
        {
            "experiment": experiment,
            "checkpoint": str(checkpoint),
            "best_checkpoint": best_checkpoint,
            "best_checkpoints": best_checkpoints,
            "checkpoint_policy": (
                f"final_and_best_eval_{config.best_checkpoint_metric}"
                if best_checkpoint
                else "final_step" if config.max_train_steps > 0 else "final_epoch"
            ),
            "checkpoint_epoch": (
                None
                if config.max_train_steps > 0
                else sum(epochs for _, epochs in config.stage_plan())
            ),
            "checkpoint_step": (
                trainer_summary.get("actual_steps")
                if config.max_train_steps > 0
                else None
            ),
            "train_files": [str(Path(path).resolve()) for path in data.train_files],
            "validation_file": (
                str(Path(data.validation_file).resolve())
                if data.validation_file
                else None
            ),
            "primary_data_profile": config.primary_data_profile,
            "primary_data_manifest": data.primary_manifest,
            "primary_manifest_used_for_training": (
                data.primary_manifest_used_for_training
            ),
            "data_signature": data_signature,
            "input_token_audit": input_token_audit,
            "observed_runtime_input_lengths": collator.input_length_summary(),
            "paragraph_boundary_supervision": (
                collator.paragraph_boundary_supervision_summary()
            ),
            "criterion_anchor_gate_values": (
                loaded.scorer.criterion_anchor_gates.detach().float().cpu().tolist()
                if loaded.scorer.criterion_anchor_gates is not None
                else None
            ),
            "training_rows": len(data.train_rows),
            "competition_training_rows": len(data.competition_rows),
            "extended_training_rows": len(data.extended_rows),
            "training_source_counts": {
                "competition": len(data.competition_rows),
                **data.extended_source_counts,
            },
            "training_surface_views": {
                "deployment_surface": config.essay_surface,
                "canonical_probability": (
                    config.train_canonical_surface_probability
                ),
                "observed_counts": dict(train_dataset.surface_view_counts),
                "observed_total": sum(train_dataset.surface_view_counts.values()),
            },
            "detail_supervision": detail_summary,
            "extended_datasets": list(data.extended_names),
            "prepared_data_manifest": data.external_manifest,
            "validation_rows": len(data.validation_rows),
            "validation_overlap_guard": {
                **data.overlap_report,
                "allowed_by_config": data.overlap_allowed,
            },
            "prompt_count": len(config.prompt_registry),
            "validation_unknown_prompts": validation_unknown_prompts,
            "competition_steps_per_epoch": competition_epoch_steps,
            "stage_plan": stage_plan_record(config, final_competition_steps),
            "trainer": trainer_summary,
            "peak_gpu_memory_reserved_bytes": peak_reserved_bytes,
            "peak_gpu_memory_allocated_bytes": peak_allocated_bytes,
        },
    )
    print(f"checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
