from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping

from main_code.postprocess import ScorePostprocessor

from . import TRAIN_SOURCE_SPLITS
from .api import (
    NONTHINKING_TEMPLATE_KWARGS,
    chat_completion,
    completion_parts,
    local_base_url,
    resolve_model_id,
    response_audit,
    sampling_for_attempt,
)
from .artifacts import append_jsonl, read_jsonl, sha256_text, write_json
from .prompts import (
    DEFAULT_TRAINING_PROMPT_PATH,
    DEFAULT_SKELETON_HINT,
    build_messages,
    load_prompt_template,
    prompt_template_id,
    prompt_template_sha256,
)
from .schema import (
    canonicalize_with_fixed_scores,
    conditioning_scores,
    essay_id,
    human_scores,
    official_raw_essay,
    parse_generated_judge,
    prompt_text,
    rationale_audit,
    score_prediction_map,
)


DEFAULT_SCORE_SOURCE = "human_average_matched"
SCORE_SOURCES = ("human_average_matched", "human", "score_predictions")
_AVERAGE_MATCHED = ScorePostprocessor("average_matched")

# Base 생성은 기존 prompt/QC와 기존 3회 sampling을 그대로 쓴다. 세 번 모두 실패한
# 행에만 직전 assistant 출력을 보여 주고, 동일한 엄격 QC로 완전한 JSON을 최대 두 번
# 고치게 한다. 두 teacher arm에 공통으로 적용되는 고정 protocol이다.
# rationale 길이 상한. v1 prompt는 "두 문장 이내 180자"를 요구하므로 기본값이 180이다.
# v2처럼 영역당 200~350자를 요구하는 prompt 세대에서는 이 값을 함께 올려야 한다.
# 올리지 않으면 base 3회가 전부 qc_over_180_characters로 실패하고, repair 단계가 출력을
# 강제로 180자 아래로 줄여 **prompt가 요구한 길이와 다른 데이터가 만들어진다.**
RATIONALE_CHAR_LIMIT = int(os.environ.get("TEACHER_RATIONALE_CHAR_LIMIT", "180"))
# v1 repair는 인용부호를 아예 금지한다. v2는 원문 직접 인용을 핵심 요구로 삼으므로
# 그 문구를 그대로 쓰면 repair가 prompt 요구를 정면으로 뒤집는다. 인용을 요구하는
# 세대에서는 "정확한 부분 문자열로 고쳐라"로 바꾼다.
RATIONALE_REQUIRE_QUOTES = os.environ.get("TEACHER_REQUIRE_QUOTES", "0") == "1"
REPAIR_PROTOCOL_ID = (
    "strict_full_json_repair_v2_quoted"
    if RATIONALE_REQUIRE_QUOTES
    else "strict_full_json_repair_v1"
)
REPAIR_TEXT = (
    "[재작성 요청]\n"
    "방금 응답은 자동 검증을 통과하지 못했다: <<FAILURE_CODES>>. "
    "처음 user 메시지의 동일한 essay_text와 고정 score를 사용해 JSON 객체 전체를 "
    "새로 작성하라.\n"
    "- score 세 값은 출력 스켈레톤의 숫자를 문자 그대로 복사한다.\n"
    "- domain_match, score_rationale_consistency, specificity, groundedness와 영역별 "
    "evidence atom을 모두 유지한다.\n"
    f"- 각 rationale은 공백 포함 {RATIONALE_CHAR_LIMIT}자 이내다.\n"
    + (
        "- 인용은 essay_text에 그대로 존재하는 연속된 부분 문자열이어야 한다. 고쳐 쓴 "
        "인용, 의역, 줄임표로 이어 붙인 인용은 금지한다. 정확히 옮길 수 없으면 그 인용을 "
        "빼고 서술하라. JSON 문법상 키와 문자열 경계의 큰따옴표는 반드시 유지한다.\n"
        if RATIONALE_REQUIRE_QUOTES
        else "- rationale 문자열 내용에는 인용부호(작은따옴표, 큰따옴표, ‘ ’ “ ”)를 사용하지 "
        "마라. JSON 문법상 키와 문자열 경계의 큰따옴표는 반드시 유지한다.\n"
    ) +
    "- 이번 재작성에서는 담화 표지를 근거로 들지 말고, essay_text에서 확인되는 실제 "
    "내용 단위의 기능적 순서를 서술하라. essay에 없는 표현·오탈자·사례를 추가하지 "
    "말며, 교정형을 원문 표현처럼 제시하지 마라.\n"
    "- 앞 응답의 설명이나 수정 목록 없이 content, organization, expression만 포함한 "
    "JSON 객체 하나를 출력한다."
)
REPAIR_SAMPLING_ATTEMPTS: tuple[dict[str, Any], ...] = (
    {"temperature": 0.0, "top_p": 1.0, "seed": 45},
    {"temperature": 0.35, "top_p": 0.90, "seed": 46},
)


def _source_split(row: Mapping[str, Any]) -> str:
    value = row.get("source_split", row.get("split"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"source_split이 명시되지 않은 row는 teacher data로 사용할 수 없습니다: "
            f"{essay_id(row)}"
        )
    return value


def select_training_rows(
    rows: Iterable[dict[str, Any]],
    *,
    source_splits: set[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    if limit is not None and limit < 0:
        raise ValueError("limit은 0 이상이어야 합니다")
    if limit == 0:
        return []
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        item_id = essay_id(row)
        split = _source_split(row)
        lowered = split.lower()
        if "validation" in lowered or "test" in lowered:
            raise ValueError(
                f"validation/test row는 teacher data로 쓸 수 없습니다: {item_id}"
            )
        if source_splits and split not in source_splits:
            continue
        if item_id in seen:
            raise ValueError(f"중복 train essay ID: {item_id}")
        seen.add(item_id)
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _fixed_scores(
    row: Mapping[str, Any],
    *,
    score_source: str,
    predicted_scores: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    # 프롬프트에 넣는 값과 복사 검증에 쓰는 값이 같아야 한다. `conditioning_scores`가 그
    # 하나의 값을 만든다(왜 필요한지는 schema.conditioning_scores 참고).
    if score_source in {"human", "human_average_matched"}:
        source = conditioning_scores(human_scores(row))
        if score_source == "human_average_matched":
            # Docker가 근거 생성 전에 확정하는 것과 같은 정수 표면이다. 인간 점수를
            # 출발점으로 써 teacher A/B를 특정 score checkpoint에 종속시키지 않는다.
            return conditioning_scores(_AVERAGE_MATCHED.apply_row(source))
        return source
    if score_source == "score_predictions":
        try:
            return conditioning_scores(predicted_scores[essay_id(row)])
        except KeyError as exc:
            raise ValueError(f"score prediction이 없습니다: {essay_id(row)}") from exc
    raise ValueError(f"score_source는 {SCORE_SOURCES} 중 하나여야 합니다")


def _score_source_values(
    row: Mapping[str, Any],
    *,
    score_source: str,
    predicted_scores: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """후처리 전 조건 점수 원천을 provenance용으로 반환한다."""

    if score_source in {"human", "human_average_matched"}:
        return conditioning_scores(human_scores(row))
    if score_source == "score_predictions":
        try:
            return conditioning_scores(predicted_scores[essay_id(row)])
        except KeyError as exc:
            raise ValueError(f"score prediction이 없습니다: {essay_id(row)}") from exc
    raise ValueError(f"score_source는 {SCORE_SOURCES} 중 하나여야 합니다")


def _conditioning_score_postprocess(score_source: str) -> str:
    return "average_matched" if score_source == "human_average_matched" else "none"


def _request_hash(
    model: str,
    model_revision: str,
    chat_template: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    sampling: Mapping[str, Any] | None = None,
) -> str:
    """Hash the exact attempt; no sampling argument means the base resume key."""

    resolved_sampling = sampling or sampling_for_attempt(0)
    payload = {
        "model": model,
        "model_revision": model_revision,
        "chat_template": chat_template,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": resolved_sampling["temperature"],
        "top_p": resolved_sampling["top_p"],
        "seed": resolved_sampling["seed"],
        "chat_template_kwargs": NONTHINKING_TEMPLATE_KWARGS,
    }
    return sha256_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _repair_text(failure_codes: Iterable[str]) -> str:
    codes = sorted(set(failure_codes))
    if not codes:
        raise ValueError("repair에는 하나 이상의 failure code가 필요합니다")
    return REPAIR_TEXT.replace("<<FAILURE_CODES>>", ", ".join(codes))


def _protocol_contract(base_attempts: int) -> dict[str, Any]:
    base_sampling = [sampling_for_attempt(index) for index in range(base_attempts)]
    contract: dict[str, Any] = {
        "id": REPAIR_PROTOCOL_ID,
        "base_attempts": base_attempts,
        "base_sampling": base_sampling,
        "repair_attempts": len(REPAIR_SAMPLING_ATTEMPTS),
        "repair_sampling": list(REPAIR_SAMPLING_ATTEMPTS),
        "repair_text_sha256": sha256_text(REPAIR_TEXT),
        "repair_message_roles": ["user", "assistant", "user"],
        "qc_relaxed_for_repair": False,
    }
    contract["sha256"] = sha256_text(
        json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return contract


def _qc_failure_codes(qc: Mapping[str, Any]) -> list[str]:
    checks = (
        (not qc.get("all_nonempty"), "qc_empty_rationale"),
        (not qc.get("all_distinct"), "qc_duplicate_rationales"),
        (bool(qc.get("template_placeholder_echo")), "qc_template_placeholder_echo"),
        (not qc.get("within_180_characters"), "qc_over_180_characters"),
        (bool(qc.get("organization_surface_claim")), "qc_organization_surface_claim"),
        (not qc.get("all_quoted_spans_grounded"), "qc_ungrounded_quote"),
    )
    return sorted(code for failed, code in checks if failed)


def _check_candidate(
    raw: str,
    *,
    scores: Mapping[str, float],
    essay: str,
) -> tuple[Any, Any, Any, list[str], str | None]:
    """Apply the same strict parse, fixed-score, and QC contract to every phase."""

    try:
        parsed = parse_generated_judge(raw)
    except Exception as exc:
        return None, None, None, ["parse_error"], f"{type(exc).__name__}: {exc}"

    generated_scores = {
        trait: float(parsed[trait]["score"])
        for trait in ("content", "organization", "expression")
    }
    if generated_scores != scores:
        message = (
            "teacher가 고정 점수를 그대로 복사하지 않았습니다: "
            f"expected={scores} generated={generated_scores}"
        )
        return parsed, None, None, ["score_copy_mismatch"], f"ValueError: {message}"

    try:
        canonical = canonicalize_with_fixed_scores(parsed, scores)
    except Exception as exc:
        return (
            parsed,
            None,
            None,
            ["parse_schema_error"],
            f"{type(exc).__name__}: {exc}",
        )

    qc = rationale_audit(canonical, essay)
    qc["within_180_characters"] = all(
        length <= RATIONALE_CHAR_LIMIT
        for length in qc["rationale_characters"].values()
    )
    qc["rationale_char_limit"] = RATIONALE_CHAR_LIMIT
    failure_codes = _qc_failure_codes(qc)
    qc["pass"] = not failure_codes
    error = (
        "ValueError: rationale QC 실패: " + ", ".join(failure_codes)
        if failure_codes
        else None
    )
    return parsed, canonical, qc, failure_codes, error


def _existing_successes(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    for row in read_jsonl(path):
        meta = row.get("pseudo_meta")
        qc = meta.get("qc") if isinstance(meta, Mapping) else None
        if (
            not isinstance(meta, Mapping)
            or meta.get("parse_ok") is not True
            or meta.get("teacher_score_copy_exact") is not True
            or not isinstance(qc, Mapping)
            or qc.get("pass") is not True
        ):
            continue
        request_hash = meta.get("request_sha256")
        if isinstance(request_hash, str):
            done.add((essay_id(row), request_hash))
    return done


def _log(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=False, allow_nan=False), file=sys.stderr)


def generate_teacher_rows(
    *,
    train_path: str | Path,
    output_path: str | Path,
    base_url: str,
    model: str | None = None,
    model_revision: str = "local_server_checkpoint",
    chat_template: str = "local_server_default_unverified",
    score_source: str = DEFAULT_SCORE_SOURCE,
    score_predictions_path: str | Path | None = None,
    source_splits: set[str] | None = None,
    limit: int | None = None,
    max_tokens: int = 512,
    timeout: float = 900.0,
    retries: int = 2,
    concurrency: int = 8,
    rationale_prompt_file: str | Path = DEFAULT_TRAINING_PROMPT_PATH,
    skeleton_hint: str = DEFAULT_SKELETON_HINT,
) -> dict[str, Any]:
    """Generate self-contained train pseudo rows and a separate failure audit.

    `concurrency`는 서버에 동시에 띄우는 요청 수다. teacher는 vLLM continuous batching을
    쓰므로 1건씩 보내면 GPU가 거의 논다(2,000편에 5~8시간). 요청끼리 완전히 독립이고
    temperature=0/seed=42라 건별 결정성은 유지된다. 다만 배치 구성에 따른 미세한 부동소수점
    차이는 남으므로 manifest에 값을 기록해 숨은 변인이 되지 않게 한다.
    """

    endpoint = local_base_url(base_url)
    if retries < 0:
        raise ValueError("retries는 0 이상이어야 합니다")
    if concurrency < 1:
        raise ValueError("concurrency는 1 이상이어야 합니다")
    if max_tokens < 1:
        raise ValueError("max_tokens는 양수여야 합니다")
    if score_source == "score_predictions" and score_predictions_path is None:
        raise ValueError("score_predictions score source에는 파일이 필요합니다")

    source_rows = read_jsonl(train_path)
    rows = select_training_rows(source_rows, source_splits=source_splits, limit=limit)
    predictions = (
        score_prediction_map(read_jsonl(score_predictions_path))
        if score_predictions_path is not None
        else {}
    )
    teacher_model = model or resolve_model_id(endpoint)
    prompt_source = Path(rationale_prompt_file).expanduser().resolve()
    rationale_prompt = load_prompt_template(prompt_source)
    rationale_prompt_hash = prompt_template_sha256(rationale_prompt)
    rationale_prompt_name = prompt_template_id(rationale_prompt)
    target = Path(output_path)
    audit_path = target.with_suffix(target.suffix + ".audit.jsonl")
    done = _existing_successes(target)
    done_ids = {item_id for item_id, _request in done}
    protocol = _protocol_contract(retries + 1)

    generated = 0
    skipped = 0
    failed = 0

    # 계획 단계는 순차로 돈다. resume 판정과 정합성 검사는 전부 여기서 끝내고, 아래에서는
    # 네트워크가 필요한 일만 병렬로 돌린다.
    pending: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        item_id = essay_id(row)
        prompt = prompt_text(row)
        essay = official_raw_essay(row)
        source_scores = _score_source_values(
            row, score_source=score_source, predicted_scores=predictions
        )
        scores = _fixed_scores(
            row, score_source=score_source, predicted_scores=predictions
        )
        messages = build_messages(
            prompt,
            essay,
            scores=scores,
            prompt_template=rationale_prompt,
            skeleton_hint=skeleton_hint,
        )
        request_hash = _request_hash(
            teacher_model, model_revision, chat_template, messages, max_tokens
        )
        if (item_id, request_hash) in done:
            skipped += 1
            continue
        if item_id in done_ids:
            raise ValueError(
                f"기존 output에 다른 request의 성공 row가 있습니다: {item_id}. "
                "새 output 경로를 사용하십시오"
            )
        pending.append(
            (
                row,
                {
                    "item_id": item_id,
                    "prompt": prompt,
                    "essay": essay,
                    "scores": scores,
                    "score_source_values": source_scores,
                    "conditioning_score_postprocess": (
                        _conditioning_score_postprocess(score_source)
                    ),
                    "messages": messages,
                    "request_hash": request_hash,
                },
            )
        )

    def attempt_one(task: Mapping[str, Any]) -> dict[str, Any]:
        """teacher 1건을 생성한다. 공유 상태를 읽지도 쓰지도 않아 스레드에서 안전하다."""
        scores = task["scores"]
        raw = ""
        reasoning: str | None = None
        errors: list[str] = []
        api_audit: dict[str, Any] = {}
        parsed: dict[str, dict[str, Any]] | None = None
        canonical: dict[str, dict[str, Any]] | None = None
        qc: dict[str, Any] | None = None
        sampling = sampling_for_attempt(0)
        failure_codes: list[str] = []
        attempts_this_run: list[dict[str, Any]] = []
        accepted_phase: str | None = None
        accepted_attempt_index: int | None = None
        repair_trigger: dict[str, Any] | None = None
        base_attempts = retries + 1
        for sequence_index in range(base_attempts + len(REPAIR_SAMPLING_ATTEMPTS)):
            if sequence_index < base_attempts:
                phase, attempt_index = "base", sequence_index
                messages = task["messages"]
                sampling = sampling_for_attempt(attempt_index)
            else:
                if not raw:
                    break
                phase, attempt_index = "repair", sequence_index - base_attempts
                if repair_trigger is None:
                    repair_trigger = {
                        "failure_codes": failure_codes,
                        "raw_output_sha256": sha256_text(raw),
                    }
                messages = [
                    *task["messages"],
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": _repair_text(failure_codes)},
                ]
                sampling = REPAIR_SAMPLING_ATTEMPTS[attempt_index]

            record: dict[str, Any] = {
                "phase": phase,
                "attempt_index": attempt_index,
                "sampling": dict(sampling),
                "request_sha256": _request_hash(
                    teacher_model,
                    model_revision,
                    chat_template,
                    messages,
                    max_tokens,
                    sampling,
                ),
            }
            if phase == "repair":
                record["trigger_failure_codes"] = failure_codes
            attempts_this_run.append(record)
            try:
                response = chat_completion(
                    endpoint,
                    teacher_model,
                    messages,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    temperature=sampling["temperature"],
                    top_p=sampling["top_p"],
                    seed=sampling["seed"],
                )
                raw, reasoning = completion_parts(response)
                api_audit = response_audit(response)
                parsed, canonical, qc, failure_codes, candidate_error = (
                    _check_candidate(raw, scores=scores, essay=task["essay"])
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                errors.append(error)
                record.update(
                    accepted=False,
                    failure_codes=["request_error"],
                    error=error,
                )
                if not raw:
                    failure_codes = ["request_error"]
                continue

            record.update(
                accepted=not failure_codes,
                failure_codes=failure_codes,
                api_request_sha256=api_audit.get("request_sha256"),
                raw_output_sha256=sha256_text(raw),
            )
            if candidate_error is not None:
                errors.append(candidate_error)
                record["error"] = candidate_error
            if not failure_codes:
                accepted_phase = phase
                accepted_attempt_index = attempt_index
                break

        return {
            "raw": raw,
            "reasoning": reasoning,
            "errors": errors,
            "api_audit": api_audit,
            "parsed": parsed,
            "canonical": canonical,
            "qc": qc,
            "sampling": sampling,
            "failure_codes": failure_codes,
            "accepted": accepted_phase is not None,
            "accepted_phase": accepted_phase,
            "accepted_attempt_index": accepted_attempt_index,
            "attempts_this_run": attempts_this_run,
            "repair_trigger": repair_trigger,
        }

    # 파일 쓰기는 전부 이 루프(주 스레드)에서만 한다. `map`은 제출 순서대로 결과를 돌려주므로
    # 출력 파일 순서는 순차 실행과 동일하게 유지된다.
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        outcomes = pool.map(attempt_one, (task for _row, task in pending))
        for index, ((row, task), outcome) in enumerate(zip(pending, outcomes), start=1):
            item_id = task["item_id"]
            prompt = task["prompt"]
            essay = task["essay"]
            scores = task["scores"]
            request_hash = task["request_hash"]
            raw = outcome["raw"]
            reasoning = outcome["reasoning"]
            errors = outcome["errors"]
            api_audit = outcome["api_audit"]
            parsed = outcome["parsed"]
            canonical = outcome["canonical"]
            qc = outcome["qc"]

            attempt_count = len(outcome["attempts_this_run"])
            retry_count = max(0, attempt_count - 1)
            common_audit = {
                "essay_id": item_id,
                "stage": "teacher_generate",
                "teacher_model": teacher_model,
                "model_revision": model_revision,
                "chat_template": chat_template,
                # template_sha256 is retained as a compatibility alias.  Unlike
                # the legacy partial hash it is now the complete exact template.
                "template_sha256": rationale_prompt_hash,
                "rationale_prompt_id": rationale_prompt_name,
                "rationale_prompt_sha256": rationale_prompt_hash,
                "prompt_text_sha256": sha256_text(prompt),
                "essay_sha256": sha256_text(essay),
                # request_sha256 is the unchanged base greedy resume key.
                "request_sha256": request_hash,
                "base_request_sha256": request_hash,
                "generation_protocol_id": protocol["id"],
                "generation_protocol_sha256": protocol["sha256"],
                "accepted_phase": outcome["accepted_phase"],
                "accepted_attempt_index": outcome["accepted_attempt_index"],
                "attempts_this_run": outcome["attempts_this_run"],
                "repair_trigger": outcome["repair_trigger"],
                "failure_codes": sorted(set(outcome["failure_codes"])),
                # 실제로 채택됐거나 마지막으로 실행된 시도의 sampling이다.
                "sampling": outcome["sampling"],
                "temperature": outcome["sampling"]["temperature"],
                "seed": outcome["sampling"]["seed"],
                "chat_template_kwargs": NONTHINKING_TEMPLATE_KWARGS,
                "score_source": score_source,
                "score_source_values": task["score_source_values"],
                "conditioning_score_postprocess": task[
                    "conditioning_score_postprocess"
                ],
                "input_source_split": _source_split(row),
                "attempt_count": attempt_count,
                "retry_count": retry_count,
                "errors": errors,
                "response": api_audit,
                "raw_output_sha256": sha256_text(raw),
                "raw_output_characters": len(raw),
                "reasoning_sha256": sha256_text(reasoning or ""),
                "reasoning_characters": len(reasoning or ""),
            }
            if index % 50 == 0 or index == len(pending):
                _log(
                    {
                        "stage": "teacher_generate",
                        "status": "progress",
                        "done": index,
                        "pending": len(pending),
                        "generated": generated,
                        "failed": failed,
                    }
                )
            if outcome["accepted"] is not True:
                failed += 1
                append_jsonl(
                    audit_path,
                    {
                        **common_audit,
                        "parse_ok": False,
                        "candidate_json_parse_ok": parsed is not None,
                        "raw_output": raw,
                    },
                )
                _log(
                    {
                        "stage": "teacher_generate",
                        "essay_id": item_id,
                        "status": "failed",
                        "attempt_count": common_audit["attempt_count"],
                        "error": errors[-1] if errors else "unknown",
                    }
                )
                continue

            if canonical is None or parsed is None or not isinstance(qc, Mapping):
                raise RuntimeError("accepted teacher outcome에 canonical/QC가 없습니다")
            generated_scores = {
                trait: float(parsed[trait]["score"])
                for trait in ("content", "organization", "expression")
            }
            score_copy_exact = generated_scores == scores
            pseudo_meta = {
                **common_audit,
                "status": "passed",
                "parse_ok": True,
                "qc": qc,
                "teacher_generated_scores": generated_scores,
                "canonical_fixed_scores": scores,
                "teacher_score_copy_exact": score_copy_exact,
                "scores_were_canonicalized": not score_copy_exact,
                "raw_output": raw,
            }
            output_row = {
                "essay_id": item_id,
                "prompt_text": prompt,
                "essay_text": essay,
                "source_split": "train",
                "conditioning_scores": scores,
                "judge": canonical,
                "pseudo_meta": pseudo_meta,
            }
            append_jsonl(target, output_row)
            append_jsonl(
                audit_path,
                {
                    **common_audit,
                    "parse_ok": True,
                    "qc_pass": True,
                    "teacher_score_copy_exact": score_copy_exact,
                },
            )
            generated += 1
            _log(
                {
                    "stage": "teacher_generate",
                    "essay_id": item_id,
                    "status": "ok",
                    "retry_count": retry_count,
                    "latency_seconds": api_audit.get("latency_seconds"),
                    "usage": api_audit.get("usage"),
                    "teacher_score_copy_exact": score_copy_exact,
                }
            )

    manifest = {
        "schema_version": 1,
        "stage": "teacher_generate",
        "train_path": str(Path(train_path).resolve()),
        "output_path": str(target.resolve()),
        "audit_path": str(audit_path.resolve()),
        "teacher_model": teacher_model,
        "model_revision": model_revision,
        "chat_template": chat_template,
        "template_sha256": rationale_prompt_hash,
        "rationale_prompt_id": rationale_prompt_name,
        "rationale_prompt_file": str(prompt_source),
        "rationale_prompt_text": rationale_prompt,
        "rationale_prompt_sha256": rationale_prompt_hash,
        # 스켈레톤 자리표시자의 길이 지시. 생성 직전에 붙어 본문 규칙보다 강하게
        # 작동하므로 숨은 변인이 되지 않도록 기록한다.
        "rationale_skeleton_hint": skeleton_hint,
        "rationale_char_limit": RATIONALE_CHAR_LIMIT,
        "score_source": score_source,
        "conditioning_score_postprocess": _conditioning_score_postprocess(score_source),
        "score_predictions_path": (
            str(Path(score_predictions_path).resolve())
            if score_predictions_path is not None
            else None
        ),
        "source_split_filter": sorted(source_splits) if source_splits else None,
        "limit": limit,
        "sampling_by_attempt": [
            sampling_for_attempt(index) for index in range(retries + 1)
        ],
        "generation_protocol": protocol,
        "concurrency": concurrency,
        "selected": len(rows),
        "generated_this_run": generated,
        "resumed_skips": skipped,
        "failed_this_run": failed,
    }
    write_json(target.with_suffix(target.suffix + ".manifest.json"), manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="local teacher로 fixed-score rationale pseudo train 생성"
    )
    parser.add_argument("--train", "--input", dest="train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--base-url", "--api-base", dest="base_url", default="http://127.0.0.1:8000"
    )
    parser.add_argument("--model")
    parser.add_argument("--model-revision", default="local_server_checkpoint")
    parser.add_argument("--chat-template", default="local_server_default_unverified")
    parser.add_argument(
        "--score-source", choices=SCORE_SOURCES, default=DEFAULT_SCORE_SOURCE
    )
    parser.add_argument("--score-predictions")
    parser.add_argument("--source-split", action="append", dest="source_splits")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=512)
    # 빈 문자열이면 역사적 기본 hint를 그대로 쓴다(v1/v3 재현 경로 불변).
    parser.add_argument("--skeleton-hint", default=DEFAULT_SKELETON_HINT)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--rationale-prompt-file",
        default=str(DEFAULT_TRAINING_PROMPT_PATH),
        help="fixed-score rationale prompt template (next-training default: v1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = generate_teacher_rows(
        train_path=args.train,
        output_path=args.output,
        base_url=args.base_url,
        model=args.model,
        model_revision=args.model_revision,
        chat_template=args.chat_template,
        score_source=args.score_source,
        score_predictions_path=args.score_predictions,
        source_splits=(
            set(args.source_splits) if args.source_splits else set(TRAIN_SOURCE_SPLITS)
        ),
        limit=args.limit,
        max_tokens=args.max_tokens,
        skeleton_hint=args.skeleton_hint or DEFAULT_SKELETON_HINT,
        timeout=args.timeout,
        retries=args.retries,
        concurrency=args.concurrency,
        rationale_prompt_file=args.rationale_prompt_file,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
