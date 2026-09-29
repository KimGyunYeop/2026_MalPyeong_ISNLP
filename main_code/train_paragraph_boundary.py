from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from .paragraph_boundary import (
    SEAM_COUNT_BINS,
    SEQUENCE_ATTENTION_HEADS,
    SEQUENCE_SURFACE_FEATURE_NAMES,
    SEQUENCE_TRANSFORMER_LAYERS,
    SPLITS,
    SURFACE_FEATURE_NAMES,
    anchor_count_top_k_metrics,
    anchor_count_top_k_predictions,
    boundary_metrics,
    build_boundary_examples,
    candidate_audit,
    candidate_paragraph_role,
    choose_calibration_threshold,
    deterministic_document_split,
    deterministic_boundary_baselines,
    encode_klue_contexts,
    examples_for_split,
    file_sha256,
    flatten_candidates,
    positional_role_metrics,
    positional_role_supervision_audit,
    predict_probabilities,
    predict_sequence_outputs,
    records_for_split,
    select_smoke_rows,
    sequence_surface_feature_matrix,
    surface_feature_matrix,
    train_sequence_classifier,
    train_tiny_classifier,
    training_strict_anchor_policy,
    validate_audited_full_counts,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
DEFAULT_TRAIN = PACKAGE_ROOT / "datasets" / "processed_dataset" / "train.jsonl"
DEFAULT_KLUE_REVISION = "02f94ba5e3fcb7e2a58a390b8639b0fac974a8da"


@dataclass(frozen=True)
class BoundaryRunConfig:
    detector: str
    train_file: str
    output_dir: str
    seed: int
    device: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden_size: int
    dropout: float
    patience: int
    smoke_rows_per_split: int
    klue_model_id: str
    klue_revision: str
    context_max_length: int
    context_characters: int
    embedding_batch_size: int
    sequence_batch_size: int
    count_loss_weight: float
    role_loss_weight: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "competition train 문단 seam만으로 online paragraph-boundary "
            "검출기의 독립 offline gate를 학습"
        )
    )
    parser.add_argument(
        "--detector",
        choices=("surface", "klue_frozen", "klue_sequence"),
        required=True,
    )
    parser.add_argument("--train-file", default=str(DEFAULT_TRAIN))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--smoke-rows-per-split",
        type=int,
        default=0,
        help="0이면 전체, 양수면 full split을 만든 뒤 각 split에서 이 수만 사용",
    )
    parser.add_argument("--klue-model-id", default="klue/roberta-base")
    parser.add_argument("--klue-revision", default=DEFAULT_KLUE_REVISION)
    parser.add_argument("--context-max-length", type=int, default=256)
    parser.add_argument("--context-characters", type=int, default=256)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--sequence-batch-size", type=int, default=64)
    parser.add_argument("--count-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--role-loss-weight",
        type=float,
        default=0.0,
        help=(
            "0이면 positional role head 없음. 양수면 paragraph_count>=3에서만 "
            "first/middle/last 약지도 auxiliary를 사용"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="split과 candidate 구성을 검증하고 artifact/training 없이 종료",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSON object가 아닙니다")
            rows.append(value)
    if not rows:
        raise ValueError(f"train JSONL이 비었습니다: {path}")
    return rows


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda이지만 CUDA를 사용할 수 없습니다")
    return torch.device(name)


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for value in values:
            file.write(json.dumps(value, ensure_ascii=False) + "\n")


def selected_config(args: argparse.Namespace) -> BoundaryRunConfig:
    return BoundaryRunConfig(
        detector=args.detector,
        train_file=str(Path(args.train_file).resolve()),
        output_dir=str(Path(args.output_dir).resolve()),
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        patience=args.patience,
        smoke_rows_per_split=args.smoke_rows_per_split,
        klue_model_id=args.klue_model_id,
        klue_revision=args.klue_revision,
        context_max_length=args.context_max_length,
        context_characters=args.context_characters,
        embedding_batch_size=args.embedding_batch_size,
        sequence_batch_size=args.sequence_batch_size,
        count_loss_weight=args.count_loss_weight,
        role_loss_weight=args.role_loss_weight,
    )


def validate_config(config: BoundaryRunConfig) -> None:
    positive_values = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "hidden_size": config.hidden_size,
        "patience": config.patience,
        "context_max_length": config.context_max_length,
        "context_characters": config.context_characters,
        "embedding_batch_size": config.embedding_batch_size,
        "sequence_batch_size": config.sequence_batch_size,
    }
    invalid = {name: value for name, value in positive_values.items() if value <= 0}
    if invalid:
        raise ValueError(f"양수여야 하는 설정이 잘못되었습니다: {invalid}")
    if config.smoke_rows_per_split < 0:
        raise ValueError("smoke_rows_per_split은 음수일 수 없습니다")
    if not 0 <= config.dropout < 1:
        raise ValueError("dropout은 [0, 1)이어야 합니다")
    if config.weight_decay < 0:
        raise ValueError("weight_decay는 음수일 수 없습니다")
    if config.count_loss_weight < 0 or config.role_loss_weight < 0:
        raise ValueError("sequence auxiliary loss weight는 음수일 수 없습니다")
    if config.detector == "klue_sequence" and config.count_loss_weight <= 0:
        raise ValueError("klue_sequence는 count_loss_weight > 0이어야 합니다")
    if config.detector != "klue_sequence" and config.role_loss_weight != 0:
        raise ValueError("role_loss_weight는 klue_sequence에서만 사용할 수 있습니다")
    if config.detector == "klue_sequence" and (
        config.hidden_size % SEQUENCE_ATTENTION_HEADS
    ):
        raise ValueError(
            f"klue_sequence hidden_size는 {SEQUENCE_ATTENTION_HEADS}의 배수여야 합니다"
        )


def main() -> None:
    args = parse_args()
    config = selected_config(args)
    validate_config(config)
    started = time.time()
    train_path = Path(config.train_file)
    dataset_sha256 = file_sha256(train_path)
    all_rows = load_jsonl(train_path)
    split_by_document, split_manifest = deterministic_document_split(
        all_rows, config.seed
    )
    used_rows = select_smoke_rows(
        all_rows,
        split_by_document,
        config.smoke_rows_per_split,
        config.seed,
    )
    examples = build_boundary_examples(used_rows)
    if config.smoke_rows_per_split == 0:
        validate_audited_full_counts(examples, dataset_sha256)
    audit = candidate_audit(examples)
    used_document_ids = {
        split: sorted(
            example.document_id
            for example in examples
            if split_by_document[example.document_id] == split
        )
        for split in SPLITS
    }
    plan = {
        "detector": config.detector,
        "dataset_sha256": dataset_sha256,
        "full_row_count": len(all_rows),
        "used_candidate_audit": audit,
        "full_split_counts": split_manifest["counts"],
        "used_split_counts": {split: len(used_document_ids[split]) for split in SPLITS},
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("[DRY RUN] artifact를 쓰거나 classifier를 학습하지 않았습니다.")
        return

    device = resolve_device(config.device)
    if (
        config.detector in {"klue_frozen", "klue_sequence"}
        and device.type == "cpu"
        and config.smoke_rows_per_split == 0
    ):
        raise RuntimeError("전체 KLUE boundary leaf는 CUDA에서 실행하세요")

    records = flatten_candidates(examples)
    labels = torch.tensor(
        [record.point.label for record in records], dtype=torch.float32
    )
    split_record_indices = {
        split: records_for_split(examples, records, split_by_document, split)
        for split in SPLITS
    }
    split_example_indices = {
        split: sorted(examples_for_split(examples, split_by_document, split))
        for split in SPLITS
    }
    for split in SPLITS:
        if not split_record_indices[split] or not split_example_indices[split]:
            raise ValueError(f"{split} split이 비었습니다")

    is_sequence = config.detector == "klue_sequence"
    if is_sequence and audit["rows_without_candidates"]:
        raise ValueError(
            "klue_sequence에는 candidate가 없는 essay가 없어야 합니다: "
            f"{audit['rows_without_candidates']}"
        )
    strict_anchor_policy = (
        training_strict_anchor_policy(records, split_record_indices["train"])
        if is_sequence
        else None
    )
    force_strict_anchors = bool(
        strict_anchor_policy["force_strict_two_space_anchors"]
        if strict_anchor_policy is not None
        else False
    )
    surface_features = (
        sequence_surface_feature_matrix(examples, records)
        if is_sequence
        else surface_feature_matrix(examples, records)
    )
    context_metadata: dict[str, Any] | None = None
    if config.detector in {"klue_frozen", "klue_sequence"}:
        contexts, context_metadata = encode_klue_contexts(
            examples,
            records,
            model_id=config.klue_model_id,
            revision=config.klue_revision,
            max_length=config.context_max_length,
            local_characters=config.context_characters,
            batch_size=config.embedding_batch_size,
            device=device,
        )
        features = torch.cat((contexts, surface_features.half()), dim=1)
        feature_schema = {
            "context_dimensions": int(contexts.shape[1]),
            "surface_features": list(
                SEQUENCE_SURFACE_FEATURE_NAMES
                if is_sequence
                else SURFACE_FEATURE_NAMES
            ),
            "concatenation_order": ["klue_masked_mean", "surface_features"],
        }
        if is_sequence:
            feature_schema["prompt_identity_features"] = False
    else:
        features = surface_features
        feature_schema = {
            "context_dimensions": 0,
            "surface_features": list(SURFACE_FEATURE_NAMES),
            "concatenation_order": ["surface_features"],
        }

    if is_sequence:
        training = train_sequence_classifier(
            features,
            examples,
            records,
            split_record_indices["train"],
            split_record_indices["calibration"],
            split_example_indices["train"],
            split_example_indices["calibration"],
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            epochs=config.epochs,
            batch_size=config.sequence_batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            patience=config.patience,
            count_loss_weight=config.count_loss_weight,
            role_loss_weight=config.role_loss_weight,
            force_strict_two_space_anchors=force_strict_anchors,
            seed=config.seed,
            device=device,
        )
        (
            calibration_probabilities,
            calibration_predicted_counts,
            calibration_role_predictions,
        ) = (
            predict_sequence_outputs(
                training.model,
                features,
                examples,
                records,
                split_record_indices["calibration"],
                split_example_indices["calibration"],
                batch_size=config.sequence_batch_size,
                device=device,
            )
        )
    else:
        training = train_tiny_classifier(
            features,
            labels,
            split_record_indices["train"],
            split_record_indices["calibration"],
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            patience=config.patience,
            seed=config.seed,
            device=device,
        )
        calibration_probabilities = predict_probabilities(
            training.model,
            features,
            split_record_indices["calibration"],
            batch_size=config.batch_size,
            device=device,
        )
        calibration_predicted_counts = {}
        calibration_role_predictions = []
    calibration_labels = [
        int(labels[index].item()) for index in split_record_indices["calibration"]
    ]
    calibration_gold = sum(
        len(examples[index].gold_boundaries)
        for index in split_example_indices["calibration"]
    )
    threshold, calibration_best_f1 = choose_calibration_threshold(
        calibration_probabilities, calibration_labels, calibration_gold
    )
    calibration_metrics = boundary_metrics(
        examples,
        [records[index] for index in split_record_indices["calibration"]],
        calibration_probabilities,
        threshold,
        example_indices=split_example_indices["calibration"],
    )
    if is_sequence:
        (
            heldout_probabilities,
            heldout_predicted_counts,
            heldout_role_predictions,
        ) = predict_sequence_outputs(
            training.model,
            features,
            examples,
            records,
            split_record_indices["heldout"],
            split_example_indices["heldout"],
            batch_size=config.sequence_batch_size,
            device=device,
        )
    else:
        heldout_probabilities = predict_probabilities(
            training.model,
            features,
            split_record_indices["heldout"],
            batch_size=config.batch_size,
            device=device,
        )
        heldout_predicted_counts = {}
        heldout_role_predictions = []
    heldout_records = [records[index] for index in split_record_indices["heldout"]]
    heldout_metrics = boundary_metrics(
        examples,
        heldout_records,
        heldout_probabilities,
        threshold,
        example_indices=split_example_indices["heldout"],
    )
    heldout_baselines = deterministic_boundary_baselines(
        examples,
        heldout_records,
        example_indices=split_example_indices["heldout"],
    )
    calibration_count_metrics = None
    heldout_count_metrics = None
    role_supervision = None
    if is_sequence:
        train_records = [records[index] for index in split_record_indices["train"]]
        calibration_records = [
            records[index] for index in split_record_indices["calibration"]
        ]
        calibration_count_metrics = anchor_count_top_k_metrics(
            examples,
            calibration_records,
            calibration_probabilities,
            calibration_predicted_counts,
            example_indices=split_example_indices["calibration"],
            force_strict_two_space_anchors=force_strict_anchors,
        )
        heldout_count_metrics = anchor_count_top_k_metrics(
            examples,
            heldout_records,
            heldout_probabilities,
            heldout_predicted_counts,
            example_indices=split_example_indices["heldout"],
            force_strict_two_space_anchors=force_strict_anchors,
        )
        role_supervision = {
            "train": positional_role_supervision_audit(examples, train_records),
            "calibration": positional_role_supervision_audit(
                examples, calibration_records
            ),
            "heldout": positional_role_supervision_audit(examples, heldout_records),
            "prediction_metrics_used_for_checkpoint_selection": False,
        }
        if config.role_loss_weight > 0:
            role_supervision["calibration_prediction_metrics"] = (
                positional_role_metrics(
                    examples,
                    calibration_records,
                    calibration_role_predictions,
                )
            )
            role_supervision["heldout_prediction_metrics"] = positional_role_metrics(
                examples,
                heldout_records,
                heldout_role_predictions,
            )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = asdict(config)
    resolved.update(
        dataset_sha256=dataset_sha256,
        split_strategy=split_manifest["strategy"],
        threshold_source="calibration_end_to_end_seam_f1",
        offset_unit="python_unicode_codepoint_index",
    )
    primary_decoder = (
        str(heldout_count_metrics["decoder"])
        if is_sequence and heldout_count_metrics is not None
        else "calibration_selected_global_threshold"
    )
    if is_sequence:
        resolved.update(
            primary_decoder=primary_decoder,
            strict_two_space_anchor_policy=strict_anchor_policy,
            positional_role_mask_contract=(
                "paragraph_count<3, non-Kiwi candidate, or Kiwi sentence "
                "crossing a gold paragraph seam"
            ),
            global_threshold_decoder_role="secondary_diagnostic",
        )
    split_manifest.update(
        dataset_sha256=dataset_sha256,
        used_document_ids=used_document_ids,
        smoke_rows_per_split=config.smoke_rows_per_split,
    )
    metrics = {
        "schema_version": 1,
        "candidate_audit": audit,
        "threshold": threshold,
        "threshold_selected_on": "calibration",
        "calibration_best_f1_during_threshold_search": calibration_best_f1,
        "calibration": calibration_metrics,
        "heldout": heldout_metrics,
        "heldout_deterministic_baselines": heldout_baselines,
    }
    if is_sequence:
        metrics.update(
            checkpoint_selected_by="calibration_primary_count_decode_f1",
            primary_decoder=primary_decoder,
            primary_calibration=calibration_count_metrics,
            primary_heldout=heldout_count_metrics,
            strict_two_space_anchor_policy=strict_anchor_policy,
            positional_role_supervision=role_supervision,
            global_threshold_decoder_role="secondary_diagnostic",
            calibration_anchor_count_top_k=calibration_count_metrics,
            heldout_anchor_count_top_k=heldout_count_metrics,
        )
    threshold_artifact = {
        "threshold": threshold,
        "selected_on": "calibration",
        "objective": "end_to_end_seam_f1_including_unproposable_gold_seams",
        "heldout_used": False,
    }
    if is_sequence:
        threshold_artifact.update(
            decoder_role="secondary_diagnostic",
            primary_decoder=primary_decoder,
        )
    model_artifact = {
        "schema_version": 2 if is_sequence else 1,
        "detector": config.detector,
        "classifier_hidden_size": config.hidden_size,
        "classifier_dropout": config.dropout,
        "classifier_state_dict": {
            name: value.detach().cpu()
            for name, value in training.model.state_dict().items()
        },
        "feature_schema": feature_schema,
        "offset_unit": "python_unicode_codepoint_index",
        "context_encoder": context_metadata,
        "threshold": threshold,
    }
    if is_sequence:
        model_artifact.update(
            threshold_role="secondary_diagnostic_global_threshold",
            primary_decoder={
                "name": primary_decoder,
                "ordinal_count_decode": "round(sum(sigmoid(P(count>k))))",
                "strict_two_space_anchor_policy": strict_anchor_policy,
            },
            global_threshold_decoder={
                "role": "secondary_diagnostic",
                "threshold": threshold,
                "selected_on": "calibration",
            },
            sequence_encoder={
                "layers": SEQUENCE_TRANSFORMER_LAYERS,
                "attention_heads": SEQUENCE_ATTENTION_HEADS,
                "feedforward_multiplier": 4,
                "activation": "gelu",
                "norm_first": True,
                "predict_roles": config.role_loss_weight > 0,
                "ordinal_seam_count_bins": [
                    *range(SEAM_COUNT_BINS - 1),
                    f"{SEAM_COUNT_BINS - 1}+",
                ],
                "count_decode": primary_decoder,
                "ordinal_count_decode": "round(sum(sigmoid(P(count>k))))",
                "role_auxiliary": (
                    "masked_positional_first_middle_last; paragraph_count<3 "
                    "masked because rhetorical roles are ambiguous"
                    if config.role_loss_weight > 0
                    else "none"
                ),
                "train_role_supervision_audit": (
                    role_supervision["train"]
                    if role_supervision is not None
                    else None
                ),
            }
        )
    run = {
        "schema_version": 1,
        "detector": config.detector,
        "runtime_seconds": time.time() - started,
        "device": str(device),
        "best_epoch": training.best_epoch,
        "training_rows": len(split_example_indices["train"]),
        "calibration_rows": len(split_example_indices["calibration"]),
        "heldout_rows": len(split_example_indices["heldout"]),
        "used_rows": len(used_rows),
        "training_candidates": len(split_record_indices["train"]),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "kiwipiepy": package_version("kiwipiepy"),
        },
    }
    if is_sequence:
        run.update(
            best_calibration_anchor_count_top_k_f1=(
                training.best_calibration_count_decode_f1
            ),
            checkpoint_calibration_candidate_ap=calibration_metrics["overall"][
                "candidate_average_precision"
            ],
            primary_decoder=primary_decoder,
            strict_two_space_anchor_policy=strict_anchor_policy,
        )
    else:
        run["best_calibration_candidate_ap"] = training.best_calibration_ap

    write_json(output_dir / "resolved_config.json", resolved)
    write_json(output_dir / "split_manifest.json", split_manifest)
    write_json(output_dir / "threshold.json", threshold_artifact)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "run.json", run)
    write_jsonl(output_dir / "train_log.jsonl", training.history)
    torch.save(model_artifact, output_dir / "model.pt")

    heldout_rows: list[dict[str, Any]] = []
    threshold_predictions: dict[int, set[int]] = {}
    for record, probability in zip(heldout_records, heldout_probabilities, strict=True):
        if probability >= threshold:
            threshold_predictions.setdefault(record.example_index, set()).add(
                record.point.offset
            )
    count_predictions = (
        anchor_count_top_k_predictions(
            examples,
            heldout_records,
            heldout_probabilities,
            heldout_predicted_counts,
            example_indices=split_example_indices["heldout"],
            force_strict_two_space_anchors=force_strict_anchors,
        )
        if is_sequence
        else threshold_predictions
    )
    for position, (record, probability) in enumerate(
        zip(heldout_records, heldout_probabilities, strict=True)
    ):
        example = examples[record.example_index]
        heldout_rows.append(
            {
                "id": example.row_id,
                "document_id": example.document_id,
                "prompt_num": example.prompt_num,
                "candidate_offset": record.point.offset,
                "is_kiwi_end": record.point.is_kiwi_end,
                "is_strict_two_space": record.point.is_strict_two_space,
                "label": record.point.label,
                "probability": probability,
                "prediction": int(probability >= threshold),
                **(
                    {
                        "primary_prediction": int(
                            record.point.offset
                            in count_predictions.get(record.example_index, set())
                        ),
                        "anchor_count_top_k_prediction": int(
                            record.point.offset
                            in count_predictions.get(record.example_index, set())
                        ),
                        "predicted_seam_count_bin": heldout_predicted_counts.get(
                            record.example_index, 0
                        ),
                        "positional_role_label": candidate_paragraph_role(
                            example, record.point
                        ),
                        "positional_role_prediction": (
                            heldout_role_predictions[position]
                            if config.role_loss_weight > 0
                            else None
                        ),
                    }
                    if is_sequence
                    else {}
                ),
            }
        )
    write_jsonl(output_dir / "heldout_predictions.jsonl", heldout_rows)
    essay_summaries: list[dict[str, Any]] = []
    for example_index in split_example_indices["heldout"]:
        example = examples[example_index]
        gold = set(example.gold_boundaries)
        candidates = {point.offset for point in example.candidates}
        predicted = count_predictions.get(example_index, set())
        essay_summaries.append(
            {
                "id": example.row_id,
                "document_id": example.document_id,
                "prompt_num": example.prompt_num,
                "gold_boundary_offsets": sorted(gold),
                "candidate_offsets": sorted(candidates),
                "predicted_boundary_offsets": sorted(predicted),
                "missed_gold_offsets": sorted(gold - predicted),
                "unproposable_gold_offsets": sorted(gold - candidates),
                "false_positive_offsets": sorted(predicted - gold),
                "gold_paragraph_count": len(gold) + 1,
                "predicted_paragraph_count": len(predicted) + 1,
                **(
                    {
                        "global_threshold_predicted_boundary_offsets": sorted(
                            threshold_predictions.get(example_index, set())
                        ),
                        "ordinal_predicted_seam_count_bin": (
                            heldout_predicted_counts.get(example_index, 0)
                        ),
                    }
                    if is_sequence
                    else {}
                ),
            }
        )
    write_jsonl(output_dir / "heldout_essay_summary.jsonl", essay_summaries)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "threshold": threshold,
                "heldout": (
                    heldout_count_metrics["overall"]
                    if is_sequence and heldout_count_metrics is not None
                    else heldout_metrics["overall"]
                ),
                **(
                    {
                        "primary_decoder": primary_decoder,
                        "heldout_global_threshold": heldout_metrics["overall"],
                        "heldout_primary_count_top_k": heldout_count_metrics[
                            "overall"
                        ],
                    }
                    if is_sequence and heldout_count_metrics is not None
                    else {}
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    # 이 진단 학습은 이미 받은 기본 Hugging Face cache만 사용한다.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    main()
