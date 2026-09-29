from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from main_code.paragraph_boundary import (
    SEAM_COUNT_BINS,
    SEQUENCE_SURFACE_FEATURE_NAMES,
    BoundaryExample,
    BoundaryPoint,
    CandidateRecord,
    SequenceBoundaryClassifier,
    anchor_count_top_k_metrics,
    anchor_count_top_k_predictions,
    boundary_metrics,
    build_boundary_examples,
    choose_calibration_threshold,
    coalesce_sentence_spans,
    deterministic_boundary_baselines,
    deterministic_document_split,
    gold_paragraph_boundaries,
    normalized_essay_identity,
    ordinal_seam_count_targets,
    positional_role_metrics,
    positional_role_supervision_audit,
    predict_probabilities,
    candidate_paragraph_role,
    sequence_surface_feature_matrix,
    train_sequence_classifier,
    train_tiny_classifier,
    training_strict_anchor_policy,
)
from main_code.train_paragraph_boundary import BoundaryRunConfig, validate_config
import pathlib

# 대회 데이터가 없는 환경에서는 실데이터 테스트만 건너뛴다.
_DATA_ROOT = pathlib.Path(__file__).resolve().parents[2] / "main_code/datasets/processed_dataset"
_needs_data = pytest.mark.skipif(
    not (_DATA_ROOT / "train.jsonl").is_file(),
    reason="대회 데이터(main_code/datasets/processed_dataset)가 없는 환경",
)



def competition_row(index: int, *, paragraphs: tuple[str, ...]) -> dict:
    official_raw = "  ".join(paragraphs)
    return {
        "schema_version": 1,
        "id": f"row-{index}",
        "document_id": f"document-{index}",
        "prompt_num": "Q1",
        "prompt": "한 문단의 글을 쓰시오.",
        "essay": "\n\n".join(paragraphs),
        "essay_surfaces": {"official_raw": official_raw},
        "source_dataset": "nikl_competition",
        "dataset_group": "competition",
        "source_split": "official_train",
        "metadata": {"year": 2023, "paragraph_count": len(paragraphs)},
    }


def test_gold_alignment_and_candidate_label_preserve_raw_offsets() -> None:
    row = competition_row(0, paragraphs=("첫 문단.", "둘째 문단."))
    row["essay_surfaces"]["official_raw"] = " 첫 문단.  둘째 문단. "
    left_end = row["essay_surfaces"]["official_raw"].index(".") + 1
    final_end = row["essay_surfaces"]["official_raw"].rindex(".") + 1

    assert gold_paragraph_boundaries(row) == (left_end,)
    examples = build_boundary_examples(
        [row], sentence_span_fn=lambda _: ((1, left_end), (left_end + 2, final_end))
    )
    assert [(point.offset, point.label) for point in examples[0].candidates] == [
        (left_end, 1)
    ]


def test_overlapping_kiwi_spans_are_coalesced_before_candidates() -> None:
    text = "가나다라마바사아자차"
    assert coalesce_sentence_spans(text, ((0, 4), (3, 7), (7, 10))) == (
        (0, 7),
        (7, 10),
    )
    with pytest.raises(ValueError, match="순서"):
        coalesce_sentence_spans(text, ((4, 7), (2, 3)))


def test_document_split_is_stable_disjoint_and_exact_80_10_10() -> None:
    rows = [
        competition_row(index, paragraphs=(f"서로 다른 문장 {index}.",))
        for index in range(100)
    ]
    first, manifest = deterministic_document_split(rows, seed=42)
    second, _ = deterministic_document_split(list(reversed(rows)), seed=42)

    assert first == second
    assert manifest["counts"] == {"train": 80, "calibration": 10, "heldout": 10}
    document_sets = [
        set(manifest["document_ids"][split]) for split in manifest["counts"]
    ]
    assert not (document_sets[0] & document_sets[1])
    assert not (document_sets[0] & document_sets[2])
    assert not (document_sets[1] & document_sets[2])


def test_identical_visible_inputs_are_one_connected_split_group() -> None:
    rows = [
        competition_row(index, paragraphs=(f"서로 다른 문장 {index}.",))
        for index in range(100)
    ]
    rows[-1]["essay"] = rows[0]["essay"]
    rows[-1]["essay_surfaces"] = dict(rows[0]["essay_surfaces"])
    rows[-1]["prompt"] = "같은 글에 대한 다른 문항이다."
    rows[-1]["source_split"] = "origin_pool_extra"

    split_by_document, manifest = deterministic_document_split(rows, seed=42)

    assert normalized_essay_identity(rows[0]) == normalized_essay_identity(rows[-1])
    assert (
        split_by_document[rows[0]["document_id"]]
        == split_by_document[rows[-1]["document_id"]]
    )
    assert manifest["normalized_essay_identity_cross_split_count"] == 0
    assert manifest["duplicate_identity_group_count"] == 1
    assert manifest["cross_source_duplicate_identity_group_count"] == 1
    assert manifest["source_split_role"] == "audit_only"


@_needs_data
def test_real_train_has_no_normalized_input_identity_across_splits() -> None:
    train_path = (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "processed_dataset"
        / "train.jsonl"
    )
    rows = [json.loads(line) for line in train_path.open() if line.strip()]
    split_by_document, manifest = deterministic_document_split(rows, seed=42)
    splits_by_identity: dict[str, set[str]] = {}
    for row in rows:
        identity = normalized_essay_identity(row)
        splits_by_identity.setdefault(identity, set()).add(
            split_by_document[row["document_id"]]
        )

    assert all(len(splits) == 1 for splits in splits_by_identity.values())
    assert (
        split_by_document["GWRW2300023150.1"] == split_by_document["GWRW2300045280.1"]
    )
    assert manifest["normalized_essay_identity_cross_split_count"] == 0
    assert manifest["duplicate_identity_group_count"] == 1
    assert manifest["cross_source_duplicate_identity_group_count"] == 1


def test_metrics_keep_unproposable_seams_and_zero_candidate_essays() -> None:
    examples = [
        BoundaryExample(
            "a",
            "a",
            "Q1",
            "2023",
            "official_train",
            "abcdef",
            (2, 5),
            (
                BoundaryPoint(2, True, False, 1),
                BoundaryPoint(3, True, False, 0),
            ),
        ),
        BoundaryExample(
            "b",
            "b",
            "Q2",
            "2023",
            "official_train",
            "abcd",
            (2,),
            (),
        ),
    ]
    records = [
        CandidateRecord(0, 0, examples[0].candidates[0]),
        CandidateRecord(0, 1, examples[0].candidates[1]),
    ]
    probabilities = [0.9, 0.1]
    threshold, best_f1 = choose_calibration_threshold(probabilities, [1, 0], 3)
    metrics = boundary_metrics(
        examples,
        records,
        probabilities,
        threshold,
        example_indices=[0, 1],
    )

    assert threshold == pytest.approx(0.9)
    assert best_f1 == pytest.approx(0.5)
    assert metrics["overall"]["gold_seam_count"] == 3
    assert metrics["overall"]["candidate_recall_ceiling"] == pytest.approx(1 / 3)
    assert metrics["overall"]["recall"] == pytest.approx(1 / 3)
    assert metrics["overall"]["paragraph_count_mae"] == pytest.approx(1.0)
    assert set(metrics["per_prompt"]) == {"Q1", "Q2"}
    assert metrics["per_prompt"]["Q2"]["candidate_count"] == 0
    baselines = deterministic_boundary_baselines(
        examples, records, example_indices=[0, 1]
    )
    assert baselines["all_candidates"]["overall"]["predicted_boundary_count"] == 2
    assert baselines["kiwi_only"]["overall"]["predicted_boundary_count"] == 2
    assert (
        baselines["strict_two_space_only"]["overall"]["predicted_boundary_count"] == 0
    )


def test_tiny_torch_classifier_trains_without_sklearn() -> None:
    generator = torch.Generator().manual_seed(7)
    first = torch.randn(160, generator=generator)
    features = torch.stack((first, torch.randn(160, generator=generator)), dim=1)
    labels = (first > 0).float()
    result = train_tiny_classifier(
        features,
        labels,
        list(range(120)),
        list(range(120, 160)),
        hidden_size=8,
        dropout=0.0,
        epochs=12,
        batch_size=32,
        learning_rate=0.05,
        weight_decay=0.0,
        patience=5,
        seed=42,
        device=torch.device("cpu"),
    )
    probabilities = predict_probabilities(
        result.model,
        features,
        list(range(120, 160)),
        batch_size=40,
        device=torch.device("cpu"),
    )
    accuracy = (
        sum(
            (probability >= 0.5) == bool(label)
            for probability, label in zip(probabilities, labels[120:], strict=True)
        )
        / 40
    )
    assert accuracy >= 0.9


def test_sequence_labels_are_ordinal_and_role_auxiliary_is_masked() -> None:
    points = (
        BoundaryPoint(5, True, False, 0),
        BoundaryPoint(10, True, False, 1),
        BoundaryPoint(15, True, False, 0),
        BoundaryPoint(20, True, False, 1),
        BoundaryPoint(25, True, False, 0),
    )
    example = BoundaryExample(
        "essay",
        "essay",
        "Q1",
        "2023",
        "official_train",
        "가" * 30,
        (10, 20),
        points,
    )
    assert candidate_paragraph_role(example, points[0]) == 0
    assert candidate_paragraph_role(example, points[2]) == 1
    assert candidate_paragraph_role(example, points[4]) == 2
    assert candidate_paragraph_role(
        example, BoundaryPoint(12, False, True, 0)
    ) == -100
    crossing = BoundaryExample(
        "crossing",
        "crossing",
        "Q1",
        "2023",
        "official_train",
        "가" * 30,
        (10, 20),
        (
            BoundaryPoint(5, True, False, 0),
            BoundaryPoint(15, True, False, 0),
            BoundaryPoint(25, True, False, 0),
        ),
    )
    assert candidate_paragraph_role(crossing, crossing.candidates[1]) == -100
    two_paragraph = BoundaryExample(
        "short",
        "short",
        "Q1",
        "2023",
        "official_train",
        "가" * 20,
        (10,),
        (),
    )
    assert (
        candidate_paragraph_role(two_paragraph, BoundaryPoint(5, True, False, 0))
        == -100
    )

    targets = ordinal_seam_count_targets([0, 3, 20])
    assert targets.shape == (3, SEAM_COUNT_BINS - 1)
    assert targets[0].tolist() == [0.0] * 7
    assert targets[1].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert targets[2].tolist() == [1.0] * 7

    records = [
        CandidateRecord(0, index, point) for index, point in enumerate(points)
    ]
    audit = positional_role_supervision_audit([example], records)
    role_metrics = positional_role_metrics(
        [example], records, [0, 0, 1, 1, 2]
    )
    assert audit["class_counts"] == {"first": 2, "middle": 2, "last": 1}
    assert audit["masked_count"] == 0
    assert role_metrics["accuracy"] == pytest.approx(1.0)
    assert role_metrics["macro_f1"] == pytest.approx(1.0)
    assert role_metrics["checkpoint_selection_used"] is False


def test_sequence_features_drop_prompt_one_hot_and_forward_document_jointly() -> None:
    example = BoundaryExample(
        "essay",
        "essay",
        "Q9",
        "2023",
        "official_train",
        "첫 문장. 둘째 문장. 셋째 문장.",
        (6,),
        (
            BoundaryPoint(6, True, False, 1),
            BoundaryPoint(13, True, False, 0),
        ),
    )
    records = [
        CandidateRecord(0, index, point)
        for index, point in enumerate(example.candidates)
    ]
    features = sequence_surface_feature_matrix([example], records)
    assert features.shape == (2, len(SEQUENCE_SURFACE_FEATURE_NAMES))
    assert not any(name.startswith("prompt_") for name in SEQUENCE_SURFACE_FEATURE_NAMES)

    model = SequenceBoundaryClassifier(
        torch.zeros(features.shape[1]),
        torch.ones(features.shape[1]),
        hidden_size=8,
        dropout=0.0,
        predict_roles=True,
    )
    boundary, count, role = model(
        features.unsqueeze(0), torch.tensor([[True, True]])
    )
    assert boundary.shape == (1, 2)
    assert count.shape == (1, SEAM_COUNT_BINS - 1)
    assert role is not None and role.shape == (1, 2, 3)


def test_count_top_k_decoder_keeps_strict_two_space_anchor() -> None:
    example = BoundaryExample(
        "essay",
        "essay",
        "Q1",
        "2023",
        "official_train",
        "abcdefgh",
        (2,),
        (
            BoundaryPoint(2, False, True, 1),
            BoundaryPoint(5, True, False, 0),
        ),
    )
    records = [
        CandidateRecord(0, index, point)
        for index, point in enumerate(example.candidates)
    ]
    predictions = anchor_count_top_k_predictions(
        [example],
        records,
        [0.01, 0.99],
        {0: 1},
        example_indices=[0],
        force_strict_two_space_anchors=True,
    )
    metrics = anchor_count_top_k_metrics(
        [example],
        records,
        [0.01, 0.99],
        {0: 1},
        example_indices=[0],
        force_strict_two_space_anchors=True,
    )
    without_forcing = anchor_count_top_k_predictions(
        [example],
        records,
        [0.01, 0.99],
        {0: 1},
        example_indices=[0],
        force_strict_two_space_anchors=False,
    )
    assert predictions == {0: {2}}
    assert without_forcing == {0: {5}}
    assert metrics["overall"]["f1"] == pytest.approx(1.0)
    assert metrics["ordinal_count_bin_accuracy"] == pytest.approx(1.0)


def test_strict_anchor_policy_uses_train_records_only() -> None:
    points = (
        BoundaryPoint(2, False, True, 1),
        BoundaryPoint(5, False, True, 0),
        BoundaryPoint(8, True, False, 0),
    )
    records = [CandidateRecord(0, index, point) for index, point in enumerate(points)]

    perfect_train = training_strict_anchor_policy(records, [0, 2])
    imperfect_train = training_strict_anchor_policy(records, [0, 1])
    absent_train = training_strict_anchor_policy(records, [2])

    assert perfect_train["selected_on"] == "train"
    assert perfect_train["strict_two_space_candidate_count"] == 1
    assert perfect_train["strict_two_space_precision"] == pytest.approx(1.0)
    assert perfect_train["force_strict_two_space_anchors"] is True
    assert imperfect_train["strict_two_space_negative_count"] == 1
    assert imperfect_train["force_strict_two_space_anchors"] is False
    assert absent_train["strict_two_space_precision"] is None
    assert absent_train["force_strict_two_space_anchors"] is False


def test_sequence_training_selects_epoch_by_count_decode_f1_on_cpu() -> None:
    examples: list[BoundaryExample] = []
    records: list[CandidateRecord] = []
    for example_index in range(6):
        points = (
            BoundaryPoint(2, True, False, 1),
            BoundaryPoint(5, True, False, 0),
            BoundaryPoint(8, True, False, 0),
        )
        examples.append(
            BoundaryExample(
                str(example_index),
                str(example_index),
                "Q1",
                "2023",
                "official_train",
                "abcdefghij",
                (2,),
                points,
            )
        )
        records.extend(
            CandidateRecord(example_index, index, point)
            for index, point in enumerate(points)
        )
    features = torch.tensor(
        [[float(record.point.label), record.candidate_index / 2] for record in records]
    )
    result = train_sequence_classifier(
        features,
        examples,
        records,
        list(range(12)),
        list(range(12, 18)),
        list(range(4)),
        [4, 5],
        hidden_size=8,
        dropout=0.0,
        epochs=3,
        batch_size=2,
        learning_rate=0.05,
        weight_decay=0.0,
        patience=3,
        count_loss_weight=0.5,
        role_loss_weight=0.0,
        force_strict_two_space_anchors=False,
        seed=42,
        device=torch.device("cpu"),
    )
    recorded_f1 = [
        row["calibration_anchor_count_top_k_f1"] for row in result.history
    ]
    assert result.best_epoch == recorded_f1.index(max(recorded_f1)) + 1
    assert result.best_calibration_count_decode_f1 == pytest.approx(max(recorded_f1))


def test_sequence_only_loss_options_are_validated_explicitly() -> None:
    config = BoundaryRunConfig(
        detector="klue_sequence",
        train_file="train.jsonl",
        output_dir="result",
        seed=42,
        device="cpu",
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=0.0,
        hidden_size=8,
        dropout=0.0,
        patience=1,
        smoke_rows_per_split=1,
        klue_model_id="klue/roberta-base",
        klue_revision="revision",
        context_max_length=32,
        context_characters=32,
        embedding_batch_size=2,
        sequence_batch_size=2,
        count_loss_weight=0.5,
        role_loss_weight=0.0,
    )
    validate_config(config)
    with pytest.raises(ValueError, match="count_loss_weight > 0"):
        validate_config(replace(config, count_loss_weight=0.0))
    with pytest.raises(ValueError, match="klue_sequence에서만"):
        validate_config(replace(config, detector="surface", role_loss_weight=0.2))


def test_cli_dry_run_builds_candidates_without_writing_artifacts() -> None:
    repository = Path(__file__).resolve().parents[2]
    training_python = repository / ".venv-train" / "bin" / "python"
    if not training_python.is_file():
        pytest.skip("repository training environment가 없습니다")
    with tempfile.TemporaryDirectory() as temporary:
        output_dir = Path(temporary) / "boundary"
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(repository)
        result = subprocess.run(
            [
                str(training_python),
                "-m",
                "main_code.train_paragraph_boundary",
                "--detector",
                "surface",
                "--output-dir",
                str(output_dir),
                "--smoke-rows-per-split",
                "1",
                "--dry-run",
            ],
            cwd=repository,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "[DRY RUN]" in result.stdout
        assert '"used_split_counts"' in result.stdout
        assert not output_dir.exists()
