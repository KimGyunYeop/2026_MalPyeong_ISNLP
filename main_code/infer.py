from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import (
    DETAIL_CRITERIA,
    DETAIL_CRITERIA_BY_TRAIT,
    ESSAY_SURFACES,
    PRIMARY_DATA_PROFILES,
    TRAITS,
)
from .datasets import (
    EssayRegressionDataset,
    RegressionCollator,
    deployment_surface_fingerprints,
    essay_id,
    human_scores,
    normalized_essay_hash,
    question_key,
    read_jsonl,
    official_average_score,
)
from .models import LoadedRegressionModel, load_checkpoint
from .postprocess import ScorePostprocessor
from .utils import (
    average_matched_integer_scores,
    prediction_record,
    regression_metrics,
    set_seed,
    write_json,
    write_jsonl,
)


# CLI and metric helpers ------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regression checkpoint inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--input",
        help="생략하면 checkpoint의 primary profile validation.jsonl을 사용",
    )
    parser.add_argument(
        "--detail-label-input",
        help=(
            "모델 입력은 --input 그대로 유지하고 9개 criterion 정답만 이 JSONL에서 "
            "essay_id로 결합. ID·문항·공백 정규화 본문·최종 점수가 모두 같지 않으면 실패"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int)
    parser.add_argument(
        "--essay-surface",
        choices=ESSAY_SURFACES,
        help="checkpoint 설정을 바꾸지 않고 입력 essay 공백 표면만 비교",
    )
    parser.add_argument(
        "--load-in-4bit", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def _mean_available(values: list[float | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    return float(np.mean(available)) if available else None


def prompt_macro_metrics(prompt_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """문제별 metric을 동일 가중 평균해 큰 문제의 표본 수 편향을 줄인다."""

    macro_traits: dict[str, dict[str, float | None]] = {}
    for trait in TRAITS:
        macro_traits[trait] = {
            metric: _mean_available(
                [row["metrics"]["traits"][trait][metric] for row in prompt_rows]
            )
            for metric in ("rmse", "spearman")
        }
    return {
        "prompt_count": len(prompt_rows),
        "traits": macro_traits,
        "overall": {
            metric: _mean_available(
                [row["metrics"]["overall"][metric] for row in prompt_rows]
            )
            for metric in ("rmse", "spearman")
        },
    }


@dataclass
class ScoringOutput:
    """한 번의 score-only 순회에서 이후 저장/평가에 필요한 산출물."""

    records: list[dict[str, Any]]
    score_records: list[dict[str, Any]]
    prediction_arrays: list[np.ndarray]
    detail_prediction_arrays: list[np.ndarray]
    detail_probability_arrays: list[np.ndarray]
    unknown_prompt_count: int


def score_surface_payload(scores: dict[str, float]) -> dict[str, Any]:
    """한 예측의 raw/실제 제출 표면을 명시적인 공통 schema로 만든다."""

    raw_scores = {trait: float(scores[trait]) for trait in TRAITS}
    raw_values = np.asarray(
        [[raw_scores[trait] for trait in TRAITS]], dtype=np.float64
    )
    matched_values = average_matched_integer_scores(raw_values)[0]
    matched_scores = {
        trait: int(matched_values[index])
        for index, trait in enumerate(TRAITS)
    }
    return {
        "raw_continuous": {
            "scores": raw_scores,
            "average": float(raw_values.mean()),
        },
        "average_matched": {
            "scores": matched_scores,
            "average": float(matched_values.mean()),
        },
    }


def attach_score_surfaces(scoring: ScoringOutput) -> None:
    """두 prediction JSONL 행에 같은 canonical score surface를 덧붙인다."""

    for prediction, score_record in zip(
        scoring.records, scoring.score_records, strict=True
    ):
        surfaces = score_surface_payload(score_record["scores"])
        submission_rule = str(score_record.get("score_postprocess") or "unknown")
        matches_canonical = submission_rule == "average_matched"
        # 두 파일이 독립적으로 후처리되어도 서로 영향을 주지 않게 중첩 dict를 복사한다.
        prediction["score_surfaces"] = {
            name: {"scores": dict(block["scores"]), "average": block["average"]}
            for name, block in surfaces.items()
        }
        # 과거 진단 config가 none/per_trait_round를 명시한 경우 submission.json은 canonical
        # average_matched와 다를 수 있다. 이를 숨기지 않고 두 JSONL 모두에 명시한다.
        prediction["score_postprocess"] = submission_rule
        prediction["submission_matches_average_matched"] = matches_canonical
        score_record["score_surfaces"] = surfaces
        score_record["submission_matches_average_matched"] = matches_canonical


# Score-only inference --------------------------------------------------------
def score_batches(
    loaded: LoadedRegressionModel,
    loader: DataLoader,
    device: torch.device,
    prompt_lookup: dict[tuple[str, str], int],
    prompt_routing_active: bool,
) -> ScoringOutput:
    """모델 scoring loop. rationale과 label 처리는 여기서 섞지 않는다."""

    records: list[dict[str, Any]] = []
    # D 파이프라인의 1단계 산출물. 이후 rationale generator는 이 float
    # scores를 고정 조건으로 받고 원문 입력과 essay_id로 결합하면 된다.
    score_records: list[dict[str, Any]] = []
    prediction_arrays: list[np.ndarray] = []
    detail_prediction_arrays: list[np.ndarray] = []
    detail_probability_arrays: list[np.ndarray] = []
    unknown_prompt_count = 0
    detail_active = loaded.config.detail_head_mode != "none"

    loaded.scorer.eval()
    with torch.no_grad():
        for batch in loader:
            model_inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            if "essay_mask" in batch:
                model_inputs["essay_mask"] = batch["essay_mask"].to(device)
            if "sentence_ids" in batch:
                model_inputs["sentence_ids"] = batch["sentence_ids"].to(device)
            if "paragraph_ids" in batch:
                model_inputs["paragraph_ids"] = batch["paragraph_ids"].to(device)
            if "shared_pooling_mask" in batch:
                model_inputs["shared_pooling_mask"] = batch[
                    "shared_pooling_mask"
                ].to(device)
            if "criterion_anchor_positions" in batch:
                model_inputs["criterion_anchor_positions"] = batch[
                    "criterion_anchor_positions"
                ].to(device)
            if "token_type_ids" in batch:
                model_inputs["token_type_ids"] = batch["token_type_ids"].to(device)
            if "prompt_ids" in batch:
                model_inputs["prompt_ids"] = batch["prompt_ids"].to(device)
                unknown_prompt_count += int((batch["prompt_ids"] < 0).sum().item())

            result = loaded.scorer(
                **model_inputs,
                return_probabilities=(loaded.config.score_head == "distribution"),
                return_detail_predictions=detail_active,
            )
            scores = result["scores"].float().cpu().numpy()  # type: ignore[union-attr]
            probabilities_tensor = result.get("probabilities")
            probabilities = (
                probabilities_tensor.float().cpu().numpy()
                if probabilities_tensor is not None
                else None
            )
            detail_scores_tensor = result.get("detail_scores")
            detail_scores = (
                detail_scores_tensor.float().cpu().numpy()
                if detail_scores_tensor is not None
                else None
            )
            detail_probabilities_tensor = result.get("detail_probabilities")
            detail_probabilities = (
                detail_probabilities_tensor.float().cpu().numpy()
                if detail_probabilities_tensor is not None
                else None
            )
            prediction_arrays.append(scores)
            if detail_scores is not None:
                detail_prediction_arrays.append(detail_scores)
            if detail_probabilities is not None:
                detail_probability_arrays.append(detail_probabilities)

            for index, (item_id, row_scores) in enumerate(
                zip(batch["essay_ids"], scores, strict=True)
            ):
                source_row = batch["rows"][index]
                key = question_key(source_row)
                routed_prompt_id = (
                    prompt_lookup.get(key, -1) if prompt_routing_active else None
                )
                prompt_known = (
                    routed_prompt_id >= 0 if routed_prompt_id is not None else None
                )

                record = prediction_record(item_id, row_scores)
                record.update(
                    prompt_num=key[0],
                    prompt_id=routed_prompt_id,
                    prompt_known=prompt_known,
                )
                records.append(record)

                score_record: dict[str, Any] = {
                    "essay_id": item_id,
                    "prompt_num": key[0],
                    "prompt_id": routed_prompt_id,
                    "prompt_known": prompt_known,
                    "scores": {
                        trait: float(row_scores[trait_index])
                        for trait_index, trait in enumerate(TRAITS)
                    },
                    "pipeline_stage": "scoring",
                }
                if probabilities is not None:
                    score_record["score_distribution"] = {
                        trait: [
                            float(value) for value in probabilities[index, trait_index]
                        ]
                        for trait_index, trait in enumerate(TRAITS)
                    }
                if detail_scores is not None:
                    score_record["criterion_scores"] = {
                        criterion: float(detail_scores[index, criterion_index])
                        for criterion_index, criterion in enumerate(DETAIL_CRITERIA)
                    }
                if detail_probabilities is not None:
                    if loaded.config.detail_head_mode == "rater_set":
                        score_record["rater_set_distributions"] = {
                            f"predicted_rater_{rater_slot + 1}": {
                                criterion: [
                                    float(value)
                                    for value in detail_probabilities[
                                        index, rater_slot, criterion_index
                                    ]
                                ]
                                for criterion_index, criterion in enumerate(
                                    DETAIL_CRITERIA
                                )
                            }
                            for rater_slot in range(2)
                        }
                        score_record["rater_set_class_values"] = [1, 2, 3, 4, 5]
                    else:
                        score_record["criterion_distributions"] = {
                            criterion: [
                                float(value)
                                for value in detail_probabilities[
                                    index, criterion_index
                                ]
                            ]
                            for criterion_index, criterion in enumerate(DETAIL_CRITERIA)
                        }
                        score_record["criterion_distribution_values"] = (
                            [1.0 + 0.5 * value for value in range(9)]
                            if loaded.config.detail_head_mode == "halfstep_categorical"
                            else [1, 2, 3, 4, 5]
                        )
                score_records.append(score_record)

    return ScoringOutput(
        records=records,
        score_records=score_records,
        prediction_arrays=prediction_arrays,
        detail_prediction_arrays=detail_prediction_arrays,
        detail_probability_arrays=detail_probability_arrays,
        unknown_prompt_count=unknown_prompt_count,
    )


def attach_human_labels(
    rows: list[dict[str, Any]], scoring: ScoringOutput
) -> tuple[list[dict[str, float] | None], int]:
    """정답이 있는 입력만 prediction 행에 붙인다. submission은 건드리지 않는다."""

    labelled = [human_scores(row) for row in rows]
    label_count = 0
    for source_row, record, score_record, scores in zip(
        rows, scoring.records, scoring.score_records, labelled, strict=True
    ):
        detail_labels = criterion_labels(source_row)
        if detail_labels and "criterion_scores" in score_record:
            score_record["criterion_labels"] = detail_labels
        if scores is None:
            continue
        labels = {trait: float(scores[trait]) for trait in TRAITS}
        record["labels"] = labels
        score_record["labels"] = dict(labels)
        label_count += 1
    return labelled, label_count


def criterion_labels(row: dict[str, Any]) -> dict[str, float]:
    """Read available official 9-criterion labels without filling missing data."""

    details = row.get("score_details")
    traits = details.get("traits") if isinstance(details, dict) else None
    if not isinstance(traits, dict):
        return {}

    labels: dict[str, float] = {}
    for trait, trait_criteria in zip(TRAITS, DETAIL_CRITERIA_BY_TRAIT, strict=True):
        trait_details = traits.get(trait)
        criteria = (
            trait_details.get("criteria") if isinstance(trait_details, dict) else None
        )
        if not isinstance(criteria, dict):
            continue
        for criterion in trait_criteria:
            record = criteria.get(criterion)
            value = record.get("official_score") if isinstance(record, dict) else None
            if isinstance(value, bool):
                continue
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(score) and 1.0 <= score <= 5.0:
                labels[criterion] = score
    return labels


def align_detail_label_rows(
    input_rows: list[dict[str, Any]],
    detail_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach detail labels without changing the rows used for model input.

    The competition JSONL has the official final labels but omits the nine
    criterion records.  A processed copy may provide those records.  Joining
    only by an ID would be unsafe, so every model-visible identity field and
    the final labels are checked before a copied ``score_details`` object is
    attached to the primary row.
    """

    detail_by_id: dict[str, dict[str, Any]] = {}
    for row in detail_rows:
        item_id = essay_id(row)
        if item_id in detail_by_id:
            raise ValueError(
                f"detail label input에 중복 essay_id가 있습니다: {item_id}"
            )
        detail_by_id[item_id] = row

    input_ids = [essay_id(row) for row in input_rows]
    if len(set(input_ids)) != len(input_ids):
        raise ValueError("inference input에 중복 essay_id가 있습니다")
    missing = sorted(set(input_ids) - set(detail_by_id))
    extra = sorted(set(detail_by_id) - set(input_ids))
    if missing or extra:
        raise ValueError(
            "detail label input의 essay_id 집합이 inference input과 다릅니다: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    aligned: list[dict[str, Any]] = []
    for input_row in input_rows:
        item_id = essay_id(input_row)
        detail_row = detail_by_id[item_id]
        if question_key(input_row) != question_key(detail_row):
            raise ValueError(f"detail label input의 문항이 다릅니다: {item_id}")
        if normalized_essay_hash(input_row) != normalized_essay_hash(detail_row):
            raise ValueError(f"detail label input의 본문이 다릅니다: {item_id}")

        input_scores = human_scores(input_row)
        detail_scores = human_scores(detail_row)
        if input_scores is None or detail_scores is None:
            raise ValueError(
                f"detail label 결합에는 양쪽 최종 점수가 필요합니다: {item_id}"
            )
        if any(
            not math.isclose(
                input_scores[trait], detail_scores[trait], rel_tol=0.0, abs_tol=1e-9
            )
            for trait in TRAITS
        ):
            raise ValueError(f"detail label input의 최종 점수가 다릅니다: {item_id}")

        details = detail_row.get("score_details")
        if not isinstance(details, dict):
            raise ValueError(
                f"detail label input에 score_details가 없습니다: {item_id}"
            )
        merged = dict(input_row)
        merged["score_details"] = details
        aligned.append(merged)
    return aligned


def _one_dimensional_metrics(
    labels: np.ndarray, predictions: np.ndarray
) -> dict[str, float | int | None]:
    """Reuse the competition's tie-aware metric implementation for one criterion."""

    repeated_labels = np.repeat(labels[:, None], len(TRAITS), axis=1)
    repeated_predictions = np.repeat(predictions[:, None], len(TRAITS), axis=1)
    metric = regression_metrics(repeated_labels, repeated_predictions)["traits"][
        TRAITS[0]
    ]
    return {
        "count": int(labels.shape[0]),
        "rmse": metric["rmse"],
        "spearman": metric["spearman"],
    }


def evaluate_detail_predictions(
    rows: list[dict[str, Any]],
    scoring: ScoringOutput,
    detail_head_mode: str | None = None,
) -> dict[str, Any] | None:
    """Evaluate nine criterion heads and their deterministic 5/2/2 aggregate."""

    if not scoring.detail_prediction_arrays:
        return None
    predictions = np.concatenate(scoring.detail_prediction_arrays).astype(
        np.float64, copy=False
    )
    if predictions.shape != (len(rows), len(DETAIL_CRITERIA)):
        raise ValueError("detail prediction은 입력 row와 같은 [N,9]여야 합니다")

    row_labels = [criterion_labels(row) for row in rows]
    criterion_metrics: dict[str, dict[str, float | int | None]] = {}
    for criterion_index, criterion in enumerate(DETAIL_CRITERIA):
        valid_indices = [
            index for index, labels in enumerate(row_labels) if criterion in labels
        ]
        if not valid_indices:
            continue
        labels = np.asarray(
            [row_labels[index][criterion] for index in valid_indices],
            dtype=np.float64,
        )
        criterion_predictions = predictions[
            np.asarray(valid_indices, dtype=np.int64), criterion_index
        ]
        criterion_metrics[criterion] = _one_dimensional_metrics(
            labels, criterion_predictions
        )

    if not criterion_metrics:
        return None

    trait_macro: dict[str, dict[str, float | int | None]] = {}
    for trait, trait_criteria in zip(TRAITS, DETAIL_CRITERIA_BY_TRAIT, strict=True):
        available = [
            criterion_metrics[name]
            for name in trait_criteria
            if name in criterion_metrics
        ]
        spearman_values = [
            float(metric["spearman"])
            for metric in available
            if metric["spearman"] is not None
        ]
        trait_macro[trait] = {
            "criterion_count": len(available),
            "rmse": (
                float(np.mean([float(metric["rmse"]) for metric in available]))
                if available
                else None
            ),
            "spearman": (float(np.mean(spearman_values)) if spearman_values else None),
        }

    available_trait_rmse = [
        float(metric["rmse"])
        for metric in trait_macro.values()
        if metric["rmse"] is not None
    ]
    available_trait_spearman = [
        float(metric["spearman"])
        for metric in trait_macro.values()
        if metric["spearman"] is not None
    ]

    complete_indices = [
        index
        for index, labels in enumerate(row_labels)
        if all(criterion in labels for criterion in DETAIL_CRITERIA)
        and human_scores(rows[index]) is not None
    ]
    aggregate_metrics: dict[str, Any] | None = None
    decode_metrics: dict[str, Any] | None = None
    if complete_indices:
        selected_indices = np.asarray(complete_indices, dtype=np.int64)

        def aggregate(selected_detail: np.ndarray) -> np.ndarray:
            aggregate_columns: list[np.ndarray] = []
            start = 0
            for trait_criteria in DETAIL_CRITERIA_BY_TRAIT:
                stop = start + len(trait_criteria)
                aggregate_columns.append(selected_detail[:, start:stop].mean(axis=1))
                start = stop
            return np.stack(aggregate_columns, axis=1)

        aggregate_predictions = aggregate(predictions[selected_indices])
        aggregate_labels = np.asarray(
            [
                [human_scores(rows[index])[trait] for trait in TRAITS]  # type: ignore[index]
                for index in complete_indices
            ],
            dtype=np.float64,
        )
        aggregate_metrics = regression_metrics(aggregate_labels, aggregate_predictions)

        native_steps = np.asarray([0.1, 0.25, 0.25], dtype=np.float64)
        snapped_predictions = (
            np.rint(aggregate_predictions / native_steps) * native_steps
        ).clip(1.0, 5.0)
        decode_metrics = {
            "soft_expectation": aggregate_metrics,
            "soft_expectation_native_grid_snap": regression_metrics(
                aggregate_labels,
                snapped_predictions,
            ),
        }

        if scoring.detail_probability_arrays and detail_head_mode in {
            "categorical",
            "halfstep_categorical",
            "rater_set",
        }:
            probabilities = np.concatenate(scoring.detail_probability_arrays).astype(
                np.float64, copy=False
            )
            if detail_head_mode == "categorical":
                class_values = np.arange(1.0, 6.0, dtype=np.float64)
                hard_detail = class_values[probabilities.argmax(axis=-1)]
            elif detail_head_mode == "halfstep_categorical":
                class_values = np.arange(1.0, 5.01, 0.5, dtype=np.float64)
                hard_detail = class_values[probabilities.argmax(axis=-1)]
            else:
                class_values = np.arange(1.0, 6.0, dtype=np.float64)
                hard_per_rater = class_values[probabilities.argmax(axis=-1)]
                hard_detail = hard_per_rater.mean(axis=1)
            hard_aggregate = aggregate(hard_detail[selected_indices])
            decode_metrics["hard_argmax"] = regression_metrics(
                aggregate_labels,
                hard_aggregate,
            )

    return {
        "row_count": len(rows),
        "criteria": criterion_metrics,
        "criterion_trait_macro": trait_macro,
        "criterion_overall_macro": {
            "rmse": (
                float(np.mean(available_trait_rmse)) if available_trait_rmse else None
            ),
            "spearman": (
                float(np.mean(available_trait_spearman))
                if available_trait_spearman
                else None
            ),
        },
        "criterion_aggregate_vs_final_labels": aggregate_metrics,
        "aggregate_decode_metrics": decode_metrics,
    }


def evaluate_predictions(
    rows: list[dict[str, Any]],
    labelled: list[dict[str, float] | None],
    prediction_arrays: list[np.ndarray],
    prompt_lookup: dict[tuple[str, str], int],
    prompt_routing_active: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """전체 metric과 문제별/macro metric을 같은 예측 배열에서 계산한다."""

    # JSON label decimals are the evaluation contract.  Casting them to
    # float32 can collapse distinct values into artificial Spearman ties, so
    # inference metrics use the same float64 values that a submission
    # evaluator sees after reading score_predictions.jsonl.  Model outputs
    # originate as float32, but widening preserves their serialized values.
    labels = np.asarray(
        [[scores[trait] for trait in TRAITS] for scores in labelled],  # type: ignore[index]
        dtype=np.float64,
    )
    predictions = np.concatenate(prediction_arrays).astype(np.float64, copy=False)
    # 공식 지표의 human_avg는 데이터셋의 score.average다. 라벨이 있는 행 전부에서
    # 읽히면 그 값을 쓰고, 한 행이라도 없으면 세 trait 평균으로 조용히 되돌린다.
    labelled_rows = [row for row, scores in zip(rows, labelled) if scores is not None]
    official_averages = [official_average_score(row) for row in labelled_rows]
    average_labels = (
        np.asarray(official_averages, dtype=np.float64)
        if official_averages and all(value is not None for value in official_averages)
        else None
    )
    metrics = regression_metrics(labels, predictions, average_labels)

    # validation_holdout_size를 켜면 평가 집합이 공식 400편 + 학습 풀 holdout이 된다.
    # 위 metrics는 둘을 합친 값이라 이전 라운드와 직접 비교할 수 없으므로 source_split
    # 별로도 남긴다. holdout이 없으면 official_validation 하나만 생기고 값은 위와 같다.
    split_indices: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        split_indices.setdefault(str(row.get("source_split") or "unknown"), []).append(index)
    if len(split_indices) > 1:
        metrics["by_source_split"] = {
            name: regression_metrics(
                labels[np.asarray(indices, dtype=np.int64)],
                predictions[np.asarray(indices, dtype=np.int64)],
                None
                if average_labels is None
                else average_labels[np.asarray(indices, dtype=np.int64)],
            )
            for name, indices in sorted(split_indices.items())
        }

    grouped_indices: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        grouped_indices.setdefault(question_key(row), []).append(index)

    by_prompt: list[dict[str, Any]] = []
    for key in sorted(grouped_indices):
        indices = np.asarray(grouped_indices[key], dtype=np.int64)
        by_prompt.append(
            {
                "prompt_num": key[0],
                "prompt_text": key[1],
                "prompt_id": (
                    prompt_lookup.get(key, -1) if prompt_routing_active else None
                ),
                "known_to_train": (
                    key in prompt_lookup if prompt_routing_active else None
                ),
                "metrics": regression_metrics(
                    labels[indices],
                    predictions[indices],
                    None if average_labels is None else average_labels[indices],
                ),
            }
        )

    prompt_metrics = {
        "macro": prompt_macro_metrics(by_prompt),
        "prompts": by_prompt,
    }
    metrics["prompt_macro"] = prompt_metrics["macro"]
    metrics["prompts"] = by_prompt
    return metrics, prompt_metrics


# End-to-end entrypoint -------------------------------------------------------
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_checkpoint(
        args.checkpoint, device=device, load_in_4bit=args.load_in_4bit
    )
    if args.essay_surface is not None:
        loaded.config = loaded.config.with_updates(essay_surface=args.essay_surface)
    set_seed(loaded.config.seed)
    input_path = (
        Path(args.input)
        if args.input
        else (
            Path(loaded.config.dataset_root)
            / PRIMARY_DATA_PROFILES[loaded.config.primary_data_profile].name
            / "validation.jsonl"
        )
    )
    rows = read_jsonl(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    detail_label_path = (
        Path(args.detail_label_input) if args.detail_label_input else None
    )
    detail_label_rows = rows
    if detail_label_path is not None:
        source_detail_rows = read_jsonl(detail_label_path)
        if args.limit is not None:
            requested_ids = {essay_id(row) for row in rows}
            source_detail_rows = [
                row for row in source_detail_rows if essay_id(row) in requested_ids
            ]
        detail_label_rows = align_detail_label_rows(rows, source_detail_rows)
    max_length = args.max_length or loaded.config.max_length
    collator = RegressionCollator(
        loaded.tokenizer,
        loaded.config,
        include_labels=False,
        max_length=max_length,
    )
    loader = DataLoader(
        EssayRegressionDataset(
            rows, loaded.config, split="inference", require_labels=False
        ),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # 1) 점수 예측
    prompt_lookup = {
        key: index for index, key in enumerate(loaded.config.prompt_registry)
    }
    prompt_routing_active = loaded.config.prompt_head_mode != "none"
    scoring = score_batches(
        loaded, loader, device, prompt_lookup, prompt_routing_active
    )

    # 2) 검증 label 결합. submission JSON에는 label이 들어가지 않는다.
    labelled, label_count = attach_human_labels(rows, scoring)
    if detail_label_path is not None:
        for detail_row, score_record in zip(
            detail_label_rows, scoring.score_records, strict=True
        ):
            labels = criterion_labels(detail_row)
            if labels and "criterion_scores" in score_record:
                score_record["criterion_labels"] = labels

    # 3) 예측 파일과 실행 manifest 저장
    checkpoint_path = Path(args.checkpoint).resolve()
    if checkpoint_path.name == "best_checkpoint":
        checkpoint_type = "best_checkpoint"
    elif checkpoint_path.name in {"best_checkpoint_rmse", "best_checkpoint_spearman"}:
        checkpoint_type = checkpoint_path.name
    elif checkpoint_path.name == "checkpoint":
        checkpoint_type = "final_checkpoint"
    else:
        checkpoint_type = "custom_checkpoint"

    output = Path(args.output_dir)
    # 원점수는 그대로 두고 제출 점수를 **덧붙인다**. 축 판정은 연속값으로, 보고와 제출은
    # 정수로 하므로 둘 다 남아야 한다. 서빙 엔진이 부르는 것과 같은 class다.
    postprocessor = ScorePostprocessor(loaded.config.score_postprocess)
    postprocessor.apply_rows(scoring.score_records)
    attach_score_surfaces(scoring)
    write_jsonl(output / "predictions.jsonl", scoring.records)
    write_jsonl(output / "score_predictions.jsonl", scoring.score_records)
    detail_metrics = evaluate_detail_predictions(
        detail_label_rows,
        scoring,
        detail_head_mode=loaded.config.detail_head_mode,
    )
    if detail_metrics is not None:
        write_json(output / "detail_metrics.json", detail_metrics)
    # submission.json은 **그대로 제출되는 형식**이라 반드시 후처리된 점수여야 한다.
    # 이전에는 실수 원점수를 써서 서빙 출력(.4249)과 .023 어긋나 있었다.
    submitted_by_id = {
        row["essay_id"]: row["submitted_scores"] for row in scoring.score_records
    }
    write_json(
        output / "submission.json",
        [
            {
                "essay_id": row["essay_id"],
                "judge": {
                    trait: {
                        **row["judge"][trait],
                        "score": submitted_by_id[row["essay_id"]][trait],
                    }
                    for trait in row["judge"]
                },
            }
            for row in scoring.records
        ],
    )
    runtime_load_in_4bit = (
        loaded.config.use_qlora if args.load_in_4bit is None else args.load_in_4bit
    )
    input_surface_fingerprints = deployment_surface_fingerprints(rows, loaded.config)
    write_json(
        output / "inference_manifest.json",
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_type": checkpoint_type,
            "label_count": label_count,
            "input": str(input_path.resolve()),
            "detail_label_input": (
                str(detail_label_path.resolve())
                if detail_label_path is not None
                else None
            ),
            "detail_label_alignment_count": (
                len(detail_label_rows) if detail_label_path is not None else 0
            ),
            "model_id": loaded.config.model_id,
            "model_slug": loaded.config.model_slug,
            "model_revision": loaded.config.model_revision,
            "model_source_run": loaded.config.model_source_run,
            "pipeline_stage": "score_only",
            "rationale_generated": False,
            "score_prediction_schema_version": 2,
            "score_surfaces": ["raw_continuous", "average_matched"],
            "score_postprocess": loaded.config.score_postprocess,
            "submission_matches_average_matched": (
                loaded.config.score_postprocess == "average_matched"
            ),
            "input_format": loaded.config.input_format,
            "rubric_profile": loaded.config.rubric_profile,
            "criterion_readout": loaded.config.criterion_readout,
            "criterion_anchor_gate_values": (
                loaded.scorer.criterion_anchor_gates.detach()
                .float()
                .cpu()
                .tolist()
                if loaded.scorer.criterion_anchor_gates is not None
                else None
            ),
            "essay_surface": loaded.config.essay_surface,
            "input_surface_fingerprints": input_surface_fingerprints,
            "input_token_audit": collator.input_length_summary(),
            "backbone_type": loaded.config.backbone_type,
            "score_head": loaded.config.score_head,
            "score_values": [1, 2, 3, 4, 5],
            "distribution_label_smoothing": (
                loaded.config.distribution_label_smoothing
            ),
            "detail_head_mode": loaded.config.detail_head_mode,
            "detail_final_source": loaded.config.detail_final_source,
            "detail_criteria": list(DETAIL_CRITERIA),
            "detail_loss_weights": {
                "expected": loaded.config.detail_expected_loss_weight,
                "distribution": loaded.config.detail_distribution_loss_weight,
                "halfstep": loaded.config.detail_halfstep_loss_weight,
                "rater_set": loaded.config.detail_rater_set_loss_weight,
                "hierarchy": loaded.config.detail_hierarchy_loss_weight,
                "rater": loaded.config.detail_rater_loss_weight,
            },
            "detail_rater_registry_count": len(loaded.config.detail_rater_registry),
            "detail_predictions_saved": bool(scoring.detail_prediction_arrays),
            "detail_probabilities_saved": bool(scoring.detail_probability_arrays),
            "detail_metrics_saved": detail_metrics is not None,
            "load_in_4bit": runtime_load_in_4bit,
            "batch_size": args.batch_size,
            "requested_max_length": max_length,
            "max_length": collator.max_length,
            "pooling": loaded.config.pooling,
            "normalize_features": loaded.config.normalize_features,
            "layer_aggregation": loaded.config.layer_aggregation,
            "last_n_layers": loaded.config.last_n_layers,
            "attention_pool_dim": loaded.config.attention_pool_dim,
            "head_type": loaded.config.head_type,
            "prompt_head_mode": loaded.config.prompt_head_mode,
            "prompt_head_traits": loaded.config.prompt_head_traits,
            "prompt_head_weight": loaded.config.prompt_head_weight,
            "prompt_count": len(loaded.config.prompt_registry),
            "unknown_prompt_count": scoring.unknown_prompt_count,
            "organization_pooling": loaded.config.organization_pooling,
            "trait_loss_weights": {
                "content": loaded.config.content_loss_weight,
                "organization": loaded.config.organization_loss_weight,
                "expression": loaded.config.expression_loss_weight,
            },
            "ranking_trait_weights": {
                "content": loaded.config.ranking_content_weight,
                "organization": loaded.config.ranking_organization_weight,
                "expression": loaded.config.ranking_expression_weight,
            },
            "mixed_head_weight": loaded.config.mixed_head_weight,
            "head_hidden_size": loaded.config.head_hidden_size,
            "count": len(scoring.records),
        },
    )

    # 4) 모든 행에 label이 있을 때만 전체/문제별 metric 저장
    if scoring.records and label_count == len(scoring.records):
        metrics, prompt_metrics = evaluate_predictions(
            rows,
            labelled,
            scoring.prediction_arrays,
            prompt_lookup,
            prompt_routing_active,
        )
        write_json(output / "metrics.json", metrics)
        write_json(output / "prompt_metrics.json", prompt_metrics)
        print(metrics)
    print(
        f"scores={output / 'score_predictions.jsonl'} "
        f"predictions={output / 'predictions.jsonl'} count={len(scoring.records)}"
    )


if __name__ == "__main__":
    main()
