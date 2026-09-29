from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from main_code_relonation import TRAIN_SOURCE_SPLITS
from main_code_relonation.api import (
    _openai_url,
    completion_parts,
    local_base_url,
    response_audit,
)
from main_code_relonation.artifacts import read_jsonl, sha256_text, write_jsonl
from main_code_relonation.data import RationaleSFTDataset
from main_code_relonation.teacher_generate import (
    REPAIR_PROTOCOL_ID,
    REPAIR_SAMPLING_ATTEMPTS,
    REPAIR_TEXT,
    _fixed_scores,
    _request_hash,
    generate_teacher_rows,
    select_training_rows,
)
from main_code_relonation.teacher_judge import (
    JUDGE_DIMENSIONS,
    evaluate_teacher_rows,
    parse_proxy_judge,
    passes_filter,
    select_pseudo_rows,
)
from main_code_relonation.train import accepted_rows


def api_response(content: str) -> dict[str, Any]:
    return {
        "id": "local-response",
        "model": "local-model",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": content, "reasoning": "내부 추론"},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        "_client_latency_seconds": 0.25,
        "_request_sha256": "api-request-hash",
    }


def teacher_json(scores: tuple[float, float, float] = (3, 4, 5)) -> str:
    return json.dumps(
        {
            trait: {"score": score, "rationale": f"{trait}의 본문 기반 판단 근거"}
            for trait, score in zip(
                ("content", "organization", "expression"), scores, strict=True
            )
        },
        ensure_ascii=False,
    )


def proxy_json(score: int = 4, *, low_groundedness: bool = False) -> str:
    value: dict[str, Any] = {}
    for trait in ("content", "organization", "expression"):
        value[trait] = {}
        for dimension in JUDGE_DIMENSIONS:
            cell_score = (
                2
                if low_groundedness
                and trait == "organization"
                and dimension == "groundedness"
                else score
            )
            value[trait][dimension] = {
                "evidence": f"{trait}.{dimension} 판정 근거",
                "score": cell_score,
            }
    return json.dumps(value, ensure_ascii=False)


def source_row(
    row_id: str = "essay-1",
    *,
    split: str = "official_train",
) -> dict[str, Any]:
    return {
        "id": row_id,
        "prompt": "로봇세 도입에 관한 의견을 쓰시오.",
        "essay_surfaces": {
            "official_raw": "나는 로봇세에 반대한다. 투자가 위축될 수 있기 때문이다."
        },
        "score": {"content": 3.5, "organization": 4.0, "expression": 4.25},
        "source_split": split,
    }


def test_api_rejects_remote_and_extracts_audit() -> None:
    assert local_base_url("http://127.0.0.1:8000/") == "http://127.0.0.1:8000"
    assert _openai_url("http://localhost:8000", "models").endswith("/v1/models")
    assert _openai_url("http://localhost:8000/v1", "models").endswith("/v1/models")
    with pytest.raises(ValueError, match="localhost"):
        local_base_url("https://api.openai.com")
    response = api_response("{}")
    assert completion_parts(response) == ("{}", "내부 추론")
    assert response_audit(response)["usage"]["total_tokens"] == 30


def test_training_selection_filters_and_blocks_eval_rows() -> None:
    rows = [source_row("a"), source_row("b", split="origin_pool_extra")]
    selected = select_training_rows(rows, source_splits={"official_train"}, limit=1)
    assert [row["id"] for row in selected] == ["a"]
    selected = select_training_rows(
        rows, source_splits=set(TRAIN_SOURCE_SPLITS), limit=None
    )
    assert [row["id"] for row in selected] == ["a", "b"]
    assert select_training_rows(rows, source_splits=None, limit=0) == []
    with pytest.raises(ValueError, match="validation/test"):
        select_training_rows(
            [source_row("bad", split="official_validation")],
            source_splits=None,
            limit=None,
        )
    missing_split = source_row("missing")
    missing_split.pop("source_split")
    with pytest.raises(ValueError, match="source_split"):
        select_training_rows(
            [missing_split], source_splits={"official_train"}, limit=None
        )


def test_teacher_generation_fixed_score_copy_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = tmp_path / "train.jsonl"
    output = tmp_path / "pseudo.jsonl"
    write_jsonl(train, [source_row(), source_row("essay-2", split="origin_pool_extra")])
    calls: list[list[dict[str, str]]] = []

    def fake_chat(
        _base_url: str,
        _model: str,
        messages: list[dict[str, str]],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        calls.append(messages)
        return api_response(
            "<think>완료</think> 앞말\n" + teacher_json((3.5, 4.0, 4.25))
        )

    monkeypatch.setattr(
        "main_code_relonation.teacher_generate.chat_completion", fake_chat
    )
    kwargs = {
        "train_path": train,
        "output_path": output,
        "base_url": "http://localhost:8000",
        "model": "teacher-local",
        "model_revision": "revision-a",
        "chat_template": "template-a",
        "score_source": "human",
        "source_splits": {"official_train"},
        "limit": None,
        "retries": 2,
    }
    first = generate_teacher_rows(**kwargs)
    second = generate_teacher_rows(**kwargs)
    assert first["generated_this_run"] == 1
    assert second["resumed_skips"] == 1
    assert len(calls) == 1

    rows = read_jsonl(output)
    assert len(rows) == 1
    row = rows[0]
    assert row["source_split"] == "train"
    assert row["conditioning_scores"] == {
        "content": 3.5,
        "organization": 4.0,
        "expression": 4.25,
    }
    assert row["judge"]["content"]["score"] == 3.5
    assert row["judge"]["expression"]["score"] == 4.25
    meta = row["pseudo_meta"]
    assert meta["teacher_generated_scores"]["content"] == 3.5
    assert meta["teacher_score_copy_exact"] is True
    assert meta["scores_were_canonicalized"] is False
    assert meta["model_revision"] == "revision-a"
    assert meta["chat_template"] == "template-a"
    assert meta["parse_ok"] is True and meta["qc"]["pass"] is True
    assert meta["response"]["usage"]["total_tokens"] == 30
    assert "앞말" in meta["raw_output"]

    audit_text = output.with_suffix(".jsonl.audit.jsonl").read_text(encoding="utf-8")
    assert row["essay_text"] not in audit_text


def test_teacher_generation_repairs_only_after_three_strict_base_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = tmp_path / "train.jsonl"
    output = tmp_path / "pseudo.jsonl"
    write_jsonl(train, [source_row()])
    scores = (3.5, 4.0, 4.25)
    bad_qc = json.dumps(
        {
            trait: {"score": score, "rationale": "문단 구분이 없다"}
            for trait, score in zip(
                ("content", "organization", "expression"), scores, strict=True
            )
        },
        ensure_ascii=False,
    )
    outputs = [bad_qc, bad_qc, bad_qc, "JSON 아님", teacher_json(scores)]
    calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    def fake_chat(
        _base_url: str,
        _model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls.append((messages, kwargs))
        return api_response(outputs[len(calls) - 1])

    monkeypatch.setattr(
        "main_code_relonation.teacher_generate.chat_completion", fake_chat
    )
    kwargs = {
        "train_path": train,
        "output_path": output,
        "base_url": "http://127.0.0.1:8000",
        "model": "teacher-local",
        "model_revision": "revision-a",
        "chat_template": "template-a",
        "score_source": "human",
        "source_splits": {"official_train"},
    }
    first = generate_teacher_rows(**kwargs)
    second = generate_teacher_rows(**kwargs)

    assert first["generated_this_run"] == 1
    assert second["resumed_skips"] == 1
    assert len(calls) == 5
    assert all(
        [message["role"] for message in messages] == ["user"]
        for messages, _request in calls[:3]
    )
    assert [message["role"] for message in calls[3][0]] == [
        "user",
        "assistant",
        "user",
    ]
    assert calls[3][0][1]["content"] == bad_qc
    repair_instruction = calls[3][0][2]["content"]
    assert (
        "qc_duplicate_rationales, qc_organization_surface_claim" in repair_instruction
    )
    assert "인용부호" in repair_instruction
    assert "담화 표지를 근거로 들지 말고" in repair_instruction
    assert "공백 포함 180자 이내" in repair_instruction
    assert calls[4][0][1]["content"] == "JSON 아님"
    assert "통과하지 못했다: parse_error" in calls[4][0][2]["content"]
    observed_sampling = [
        (request["temperature"], request["top_p"], request["seed"])
        for _messages, request in calls
    ]
    assert observed_sampling == [
        (0.0, 1.0, 42),
        (0.7, 0.95, 43),
        (1.0, 0.95, 44),
        (0.0, 1.0, 45),
        (0.35, 0.90, 46),
    ]

    row = read_jsonl(output)[0]
    meta = row["pseudo_meta"]
    assert meta["generation_protocol_id"] == REPAIR_PROTOCOL_ID
    assert meta["accepted_phase"] == "repair"
    assert meta["accepted_attempt_index"] == 1
    assert [attempt["phase"] for attempt in meta["attempts_this_run"]] == [
        "base",
        "base",
        "base",
        "repair",
        "repair",
    ]
    assert meta["repair_trigger"]["failure_codes"] == [
        "qc_duplicate_rationales",
        "qc_organization_surface_claim",
    ]
    assert meta["request_sha256"] == meta["base_request_sha256"]
    assert meta["base_request_sha256"] == _request_hash(
        "teacher-local", "revision-a", "template-a", calls[0][0], 512
    )
    manifest = json.loads(
        output.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["generation_protocol"]["repair_sampling"] == list(
        REPAIR_SAMPLING_ATTEMPTS
    )
    assert manifest["generation_protocol"]["repair_text_sha256"] == (
        sha256_text(REPAIR_TEXT)
    )
    assert manifest["rationale_prompt_sha256"] == (
        "9b343bf059c923e51f672bc7b4a6994d8092ae59f4a5fd9a881a66f170cb1c78"
    )


def test_teacher_generation_never_relaxes_qc_during_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = tmp_path / "train.jsonl"
    output = tmp_path / "pseudo.jsonl"
    write_jsonl(train, [source_row()])
    calls = 0
    long_duplicate = "가" * 181
    invalid = json.dumps(
        {
            trait: {"score": score, "rationale": long_duplicate}
            for trait, score in zip(
                ("content", "organization", "expression"),
                (3.5, 4.0, 4.25),
                strict=True,
            )
        },
        ensure_ascii=False,
    )

    def fake_chat(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return api_response(invalid)

    monkeypatch.setattr(
        "main_code_relonation.teacher_generate.chat_completion", fake_chat
    )
    kwargs = {
        "train_path": train,
        "output_path": output,
        "base_url": "http://127.0.0.1:8000",
        "model": "teacher-local",
        "score_source": "human",
        "source_splits": {"official_train"},
    }
    first = generate_teacher_rows(**kwargs)
    assert calls == 5
    assert first["failed_this_run"] == 1
    assert not output.exists()
    audit_path = output.with_suffix(".jsonl.audit.jsonl")
    first_audit = read_jsonl(audit_path)[-1]
    assert first_audit["failure_codes"] == [
        "qc_duplicate_rationales",
        "qc_over_180_characters",
    ]
    assert [attempt["phase"] for attempt in first_audit["attempts_this_run"]] == [
        "base",
        "base",
        "base",
        "repair",
        "repair",
    ]
    assert first_audit["candidate_json_parse_ok"] is True


def test_human_average_matched_uses_submission_integer_surface() -> None:
    row = source_row()
    converted = _fixed_scores(
        row,
        score_source="human_average_matched",
        predicted_scores={},
    )
    assert converted == {
        "content": 4.0,
        "organization": 4.0,
        "expression": 4.0,
    }
    assert all(float(value).is_integer() for value in converted.values())
    # Legacy 재현 경로는 기존 소수 인간 점수를 그대로 보존한다.
    assert _fixed_scores(
        row,
        score_source="human",
        predicted_scores={},
    ) == {"content": 3.5, "organization": 4.0, "expression": 4.25}


def test_human_average_matched_residual_tie_matches_submission_order() -> None:
    row = source_row()
    row["score"] = {"content": 3.49, "organization": 3.49, "expression": 3.49}
    assert _fixed_scores(
        row,
        score_source="human_average_matched",
        predicted_scores={},
    ) == {"content": 4.0, "organization": 3.0, "expression": 3.0}


def test_teacher_generation_uses_score_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = tmp_path / "train.jsonl"
    predictions = tmp_path / "scores.jsonl"
    output = tmp_path / "pseudo.jsonl"
    row = source_row()
    row.pop("score")
    write_jsonl(train, [row])
    write_jsonl(
        predictions,
        [
            {
                "essay_id": "essay-1",
                "scores": {"content": 2.75, "organization": 3.25, "expression": 4.5},
            }
        ],
    )
    monkeypatch.setattr(
        "main_code_relonation.teacher_generate.chat_completion",
        lambda *_args, **_kwargs: api_response(teacher_json((2.75, 3.25, 4.5))),
    )
    generate_teacher_rows(
        train_path=train,
        output_path=output,
        base_url="http://127.0.0.1:8000",
        model="teacher-local",
        score_source="score_predictions",
        score_predictions_path=predictions,
        source_splits={"official_train"},
    )
    pseudo = read_jsonl(output)[0]
    assert pseudo["conditioning_scores"]["content"] == 2.75
    assert pseudo["judge"]["organization"]["score"] == 3.25


def test_teacher_generation_rejects_changed_fixed_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = tmp_path / "train.jsonl"
    output = tmp_path / "pseudo.jsonl"
    write_jsonl(train, [source_row()])
    monkeypatch.setattr(
        "main_code_relonation.teacher_generate.chat_completion",
        lambda *_args, **_kwargs: api_response(teacher_json()),
    )
    manifest = generate_teacher_rows(
        train_path=train,
        output_path=output,
        base_url="http://127.0.0.1:8000",
        model="teacher-local",
        score_source="human",
        source_splits={"official_train"},
        retries=0,
    )
    assert manifest["failed_this_run"] == 1
    assert not output.exists()
    audit = read_jsonl(output.with_suffix(".jsonl.audit.jsonl"))[0]
    assert "고정 점수를 그대로 복사" in audit["errors"][0]


def test_sft_guards_check_teacher_input_provenance() -> None:
    bad = {
        "essay_id": "leaked-validation",
        "prompt_text": "논제",
        "essay_text": "에세이",
        "source_split": "train",
        "judge": json.loads(teacher_json()),
        "pseudo_meta": {
            "status": "passed",
            "parse_ok": True,
            "input_source_split": "official_validation",
        },
    }
    with pytest.raises(ValueError, match="validation/test"):
        RationaleSFTDataset([bad])
    with pytest.raises(ValueError, match="accepted pseudo"):
        accepted_rows([bad])

    no_provenance = dict(bad)
    no_provenance["essay_id"] = "missing-provenance"
    no_provenance["pseudo_meta"] = {"status": "passed", "parse_ok": True}
    with pytest.raises(ValueError, match="accepted pseudo"):
        accepted_rows([no_provenance])

    valid = dict(bad)
    valid["essay_id"] = "safe-train"
    valid["pseudo_meta"] = {
        "status": "passed",
        "parse_ok": True,
        "input_source_split": "official_train",
        "teacher_score_copy_exact": True,
        "qc": {"pass": True},
    }
    assert accepted_rows([valid]) == [valid]

    extra = dict(valid)
    extra["essay_id"] = "safe-origin-extra"
    extra["pseudo_meta"] = {
        **valid["pseudo_meta"],
        "input_source_split": "origin_pool_extra",
    }
    assert accepted_rows([extra]) == [extra]


def test_proxy_parser_and_filter() -> None:
    parsed = parse_proxy_judge("분석 완료\n" + proxy_json())
    accepted, reasons = passes_filter(
        parsed,
        minimum_mean=3.5,
        minimum_scores={name: 3 for name in JUDGE_DIMENSIONS},
    )
    assert accepted and not reasons
    low = parse_proxy_judge(proxy_json(low_groundedness=True))
    accepted, reasons = passes_filter(
        low,
        minimum_mean=3.0,
        minimum_scores={name: 3 for name in JUDGE_DIMENSIONS},
    )
    assert not accepted
    assert any("groundedness" in reason for reason in reasons)


def test_proxy_judge_append_resume_filter_and_proxy_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pseudo_path = tmp_path / "pseudo.jsonl"
    output = tmp_path / "proxy.jsonl"
    accepted_output = tmp_path / "accepted.jsonl"
    pseudo = {
        "essay_id": "essay-1",
        "prompt_text": "로봇세 도입에 관한 의견을 쓰시오.",
        "essay_text": "로봇세는 투자 위축을 일으킬 수 있으므로 반대한다.",
        "source_split": "train",
        "conditioning_scores": {
            "content": 3.5,
            "organization": 4.0,
            "expression": 4.25,
        },
        "judge": json.loads(teacher_json((3.5, 4.0, 4.25))),
        "pseudo_meta": {
            "parse_ok": True,
            "input_source_split": "official_train",
            "request_sha256": "teacher-request",
        },
    }
    write_jsonl(pseudo_path, [pseudo])
    calls = 0

    def fake_chat(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return api_response("<think>검토</think>\n" + proxy_json())

    monkeypatch.setattr("main_code_relonation.teacher_judge.chat_completion", fake_chat)
    kwargs = {
        "pseudo_path": pseudo_path,
        "output_path": output,
        "accepted_output_path": accepted_output,
        "base_url": "http://127.0.0.1:9000",
        "model": "proxy-local",
        "source_splits": {"official_train"},
        "minimum_mean": 3.5,
        "minimum_scores": {name: 3 for name in JUDGE_DIMENSIONS},
    }
    first = evaluate_teacher_rows(**kwargs)
    second = evaluate_teacher_rows(**kwargs)
    assert first["evaluated_this_run"] == 1
    assert second["resumed_skips"] == 1
    assert calls == 1

    evaluation = read_jsonl(output)[0]
    assert evaluation["filter"]["accepted"] is True
    assert evaluation["exact_official_judge"] is False
    assert "official LLM Judge" in evaluation["interpretation"]
    assert evaluation["result"]["content"]["specificity"]["score"] == 4
    accepted = read_jsonl(accepted_output)
    assert len(accepted) == 1
    assert accepted[0]["pseudo_meta"]["proxy_judge"]["exact_official_judge"] is False
    assert pseudo["essay_text"] not in output.read_text(encoding="utf-8")

    stricter = evaluate_teacher_rows(
        **{
            **kwargs,
            "minimum_mean": 5.0,
            "minimum_scores": {name: 5 for name in JUDGE_DIMENSIONS},
        }
    )
    assert stricter["resumed_skips"] == 1
    assert stricter["accepted_latest"] == 0
    assert read_jsonl(accepted_output) == []
    assert calls == 1


def test_proxy_selection_rejects_validation_pseudo() -> None:
    pseudo = {
        "essay_id": "bad",
        "source_split": "train",
        "pseudo_meta": {"input_source_split": "official_validation"},
    }
    with pytest.raises(ValueError, match="train pseudo"):
        select_pseudo_rows([pseudo], source_splits=None, limit=None)
