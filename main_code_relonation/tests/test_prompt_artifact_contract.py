from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from main_code_relonation.ab_experiment import (
    align_training_rows,
    build_evaluation_rows,
    make_submitted_score_file,
)
from main_code_relonation.artifacts import read_jsonl, write_jsonl
from main_code_relonation.config import (
    RationaleConfig,
    bind_adapter_prompt,
    load_config,
)
from main_code_relonation.prompts import (
    BASELINE_PROMPT_PATH,
    RATIONALE_PROMPT_V1_PATH,
    build_messages,
    load_prompt_template,
    prompt_template_sha256,
)
from main_code_relonation.train import (
    validate_training_prompt_rows,
    validate_training_score_rows,
)
from main_code_relonation.infer import parse_args as parse_infer_args


BASELINE_RENDER_SHA256 = (
    "9d6324347b778a768762f7145ed08afd363fe0f5585cd5f912b2625a19fce3cd"
)

# prompt 선언 규약을 검사하는 최소 recipe. 이전 A.X recipe 파일 대신 테스트 안에서 만든다.
_RECIPE_BASE = {
    "model_id": "skt/A.X-4.0-Light",
    "trust_remote_code": True,
    "torch_dtype": "bfloat16",
    "max_length": 8192,
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 42,
    "score_mode": "fixed",
    "chat_template_kwargs": {},
    "lora_dropout": 0.05,
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def _legacy_recipe(tmp_path: Path) -> RationaleConfig:
    """prompt를 선언하지 않는 과거 형식 recipe."""

    path = tmp_path / "legacy_recipe.json"
    path.write_text(
        json.dumps(
            {**_RECIPE_BASE, "load_in_4bit": True, "lora_rank": 16, "lora_alpha": 32}
        ),
        encoding="utf-8",
    )
    return load_config(path)


def _v1_recipe(tmp_path: Path) -> RationaleConfig:
    """rationale_prompt_v1.txt를 명시하는 recipe."""

    path = tmp_path / "v1_recipe.json"
    path.write_text(
        json.dumps(
            {
                **_RECIPE_BASE,
                "load_in_4bit": False,
                "lora_rank": 32,
                "lora_alpha": 64,
                "rationale_prompt_file": str(RATIONALE_PROMPT_V1_PATH),
            }
        ),
        encoding="utf-8",
    )
    return load_config(path)


def test_baseline_file_is_byte_equivalent_to_legacy_render() -> None:
    content = build_messages(
        "논제",
        "본문",
        scores={"content": 3.4, "organization": 3.75, "expression": 4.0},
        prompt_template=load_prompt_template(BASELINE_PROMPT_PATH),
    )[0]["content"]
    assert len(content) == 2860
    assert prompt_template_sha256(content) == BASELINE_RENDER_SHA256


def test_missing_config_is_baseline_and_next_recipe_is_v1(tmp_path: Path) -> None:
    legacy = _legacy_recipe(tmp_path)
    current = _v1_recipe(tmp_path)
    assert legacy.rationale_prompt_source == "baseline_fallback"
    assert legacy.rationale_prompt_text == load_prompt_template(BASELINE_PROMPT_PATH)
    assert current.rationale_prompt_id == "rationale_prompt_v1"
    assert current.rationale_prompt_text == load_prompt_template(
        RATIONALE_PROMPT_V1_PATH
    )
    assert current.rationale_prompt_sha256 != legacy.rationale_prompt_sha256
    assert legacy.load_in_4bit is True
    assert current.load_in_4bit is False
    assert (legacy.lora_rank, legacy.lora_alpha) == (16, 32)
    assert (current.lora_rank, current.lora_alpha) == (32, 64)


def test_partial_or_contradictory_prompt_declaration_fails(tmp_path: Path) -> None:
    partial = tmp_path / "partial.json"
    partial.write_text(
        json.dumps({"model_id": "stub", "rationale_prompt_id": "rationale_prompt_v1"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="메타데이터만 일부"):
        load_config(partial)

    baseline = load_prompt_template(BASELINE_PROMPT_PATH)
    contradictory = tmp_path / "contradictory.json"
    contradictory.write_text(
        json.dumps(
            {
                "model_id": "stub",
                "rationale_prompt_text": baseline,
                "rationale_prompt_id": "rationale_prompt_v1",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ID가 원문과 다릅니다"):
        load_config(contradictory)


def test_infer_explicit_bf16_override_is_available_for_docker_parity() -> None:
    with patch(
        "sys.argv",
        [
            "infer",
            "--recipe",
            "recipe.json",
            "--input",
            "input.jsonl",
            "--output-dir",
            "out",
            "--no-load-in-4bit",
        ],
    ):
        assert parse_infer_args().load_in_4bit is False


def test_adapter_sidecar_is_authoritative_and_mismatch_fails(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    template = load_prompt_template(RATIONALE_PROMPT_V1_PATH)
    digest = prompt_template_sha256(template)
    (adapter / "rationale_prompt.txt").write_text(template, encoding="utf-8")
    (adapter / "rationale_runtime_config.json").write_text(
        json.dumps(
            {
                "rationale_prompt_id": "rationale_prompt_v1",
                "rationale_prompt_text": template,
                "rationale_prompt_sha256": digest,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    legacy_config = RationaleConfig(model_id="stub")
    bound = bind_adapter_prompt(legacy_config, adapter, prompt_was_explicit=False)
    assert bound.rationale_prompt_sha256 == digest
    with pytest.raises(ValueError, match="명시한 rationale prompt"):
        bind_adapter_prompt(legacy_config, adapter, prompt_was_explicit=True)


def test_training_rows_cannot_mix_prompt_versions(tmp_path: Path) -> None:
    baseline = RationaleConfig(model_id="stub")
    v1 = _v1_recipe(tmp_path)
    legacy_row = {"essay_id": "old", "pseudo_meta": {}}
    validate_training_prompt_rows([legacy_row], baseline)
    with pytest.raises(ValueError, match="rationale prompt"):
        validate_training_prompt_rows([legacy_row], v1)
    v1_row = {
        "essay_id": "new",
        "pseudo_meta": {"rationale_prompt_sha256": v1.rationale_prompt_sha256},
    }
    validate_training_prompt_rows([v1_row], v1)


def _accepted(identifier: str, teacher: str, prompt_hash: str) -> dict:
    fixed = {trait: 3 for trait in ("content", "organization", "expression")}
    return {
        "essay_id": identifier,
        "prompt_text": "논제",
        "essay_text": "본문",
        "source_split": "train",
        "conditioning_scores": dict(fixed),
        "judge": {
            trait: {"score": fixed[trait], "rationale": f"{trait} 근거"}
            for trait in ("content", "organization", "expression")
        },
        "pseudo_meta": {
            "status": "accepted",
            "teacher_model": teacher,
            "rationale_prompt_sha256": prompt_hash,
            "score_source": "human_average_matched",
            "score_source_values": dict(fixed),
            "conditioning_score_postprocess": "average_matched",
            "canonical_fixed_scores": dict(fixed),
            "parse_ok": True,
            "teacher_score_copy_exact": True,
            "input_source_split": "official_train",
            "qc": {"pass": True},
            "proxy_judge": {"filter": {"accepted": True}},
        },
    }


def test_ab_alignment_and_docker_score_conversion(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    qwen = tmp_path / "qwen.jsonl"
    gemma = tmp_path / "gemma.jsonl"
    prompt_hash = "a" * 64
    write_jsonl(source, [{"id": "b"}, {"id": "a"}, {"id": "c"}])
    write_jsonl(
        qwen, [_accepted("a", "qwen", prompt_hash), _accepted("b", "qwen", prompt_hash)]
    )
    write_jsonl(
        gemma,
        [_accepted("b", "gemma", prompt_hash), _accepted("a", "gemma", prompt_hash)],
    )
    manifest = align_training_rows(
        source_path=source,
        qwen_path=qwen,
        gemma_path=gemma,
        qwen_output=tmp_path / "qwen_aligned.jsonl",
        gemma_output=tmp_path / "gemma_aligned.jsonl",
        manifest_path=tmp_path / "alignment.json",
    )
    assert manifest["intersection_count"] == 2
    assert manifest["score_source"] == "human_average_matched"
    assert manifest["conditioning_score_postprocess"] == "average_matched"
    assert [row["essay_id"] for row in read_jsonl(tmp_path / "qwen_aligned.jsonl")] == [
        "b",
        "a",
    ]

    v1 = _v1_recipe(tmp_path)
    score_contract = validate_training_score_rows(
        read_jsonl(tmp_path / "qwen_aligned.jsonl"), v1
    )
    assert score_contract["score_source"] == "human_average_matched"
    assert score_contract["conditioning_score_postprocess"] == "average_matched"
    assert score_contract["row_count"] == 2
    assert len(score_contract["conditioning_scores_sha256"]) == 64

    predictions = tmp_path / "scores.jsonl"
    write_jsonl(
        predictions,
        [
            {
                "essay_id": "a",
                "scores": {"content": 3.2, "organization": 3.2, "expression": 3.2},
                "submitted_scores": {"content": 4, "organization": 3, "expression": 3},
            }
        ],
    )
    make_submitted_score_file(
        source_path=predictions, output_path=tmp_path / "fixed.jsonl"
    )
    assert read_jsonl(tmp_path / "fixed.jsonl")[0]["scores"] == {
        "content": 4,
        "organization": 3,
        "expression": 3,
    }


def test_ab_alignment_rejects_rows_without_proxy_acceptance(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    qwen = tmp_path / "qwen.jsonl"
    gemma = tmp_path / "gemma.jsonl"
    prompt_hash = "a" * 64
    write_jsonl(source, [{"id": "a"}])
    qwen_row = _accepted("a", "qwen", prompt_hash)
    qwen_row["pseudo_meta"]["proxy_judge"]["filter"]["accepted"] = False
    write_jsonl(qwen, [qwen_row])
    write_jsonl(gemma, [_accepted("a", "gemma", prompt_hash)])
    with pytest.raises(ValueError, match="accepted train row"):
        align_training_rows(
            source_path=source,
            qwen_path=qwen,
            gemma_path=gemma,
            qwen_output=tmp_path / "qwen_aligned.jsonl",
            gemma_output=tmp_path / "gemma_aligned.jsonl",
            manifest_path=tmp_path / "alignment.json",
        )


def test_ab_alignment_rejects_different_fixed_scores(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    qwen = tmp_path / "qwen.jsonl"
    gemma = tmp_path / "gemma.jsonl"
    write_jsonl(source, [{"id": "a"}])
    qwen_row = _accepted("a", "qwen", "a" * 64)
    gemma_row = _accepted("a", "gemma", "a" * 64)
    gemma_row["conditioning_scores"]["content"] = 4
    gemma_row["pseudo_meta"]["canonical_fixed_scores"]["content"] = 4
    write_jsonl(qwen, [qwen_row])
    write_jsonl(gemma, [gemma_row])
    with pytest.raises(ValueError, match="score 값 불일치"):
        align_training_rows(
            source_path=source,
            qwen_path=qwen,
            gemma_path=gemma,
            qwen_output=tmp_path / "qwen_aligned.jsonl",
            gemma_output=tmp_path / "gemma_aligned.jsonl",
            manifest_path=tmp_path / "alignment.json",
        )


def test_v1_training_rejects_wrong_average_matched_value(tmp_path: Path) -> None:
    row = _accepted("a", "qwen", "a" * 64)
    row["pseudo_meta"]["score_source_values"] = {
        "content": 3.49,
        "organization": 3.49,
        "expression": 3.49,
    }
    row["conditioning_scores"] = {
        "content": 3,
        "organization": 3,
        "expression": 3,
    }
    row["pseudo_meta"]["canonical_fixed_scores"] = dict(row["conditioning_scores"])
    v1 = _v1_recipe(tmp_path)
    with pytest.raises(ValueError, match="재계산값"):
        validate_training_score_rows([row], v1)


def test_build_evaluation_rows_keeps_validation_out_of_training(tmp_path: Path) -> None:
    validation = tmp_path / "validation.jsonl"
    inference = tmp_path / "predictions.jsonl"
    write_jsonl(
        validation,
        [
            {
                "id": "v1",
                "prompt": "논제",
                "essay_surfaces": {"official_raw": "검증 본문"},
            }
        ],
    )
    write_jsonl(inference, [_accepted("v1", "student", "b" * 64)])
    build_evaluation_rows(
        validation_path=validation,
        inference_path=inference,
        output_path=tmp_path / "eval.jsonl",
        arm="qwen_teacher_student",
    )
    row = read_jsonl(tmp_path / "eval.jsonl")[0]
    assert row["source_split"] == "evaluation"
    assert row["pseudo_meta"]["input_source_split"] == "official_validation"
