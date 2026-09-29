#!/usr/bin/env python3
"""Self-contained reproduction of the Malpyeong Docker HTTP evaluation contract.

Only Python's standard library is used.  The evaluator discovers the served model via
``/v1/models``, sends the announced single-user Chat Completions request, executes the
organizer-published first-balanced-JSON parser, retries parse failures twice, and
computes both the raw HTTP-score surface and the official per-trait half-up surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import socket
import statistics
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence

try:
    from .official_parser_20260715 import (
        _extract_first_json as extract_first_json,
        _parse_model_output as parse_model_output,
    )
except ImportError:  # Direct execution: python3 evaluate_http.py
    from official_parser_20260715 import (
        _extract_first_json as extract_first_json,
        _parse_model_output as parse_model_output,
    )


TRAITS = ("content", "organization", "expression")
STOP_SEQUENCES = ("Q:", "User:")
# 서버가 근거 생성에 실패해 template으로 대체한 응답의 서명.
# `main_code_submission/degrade.py`의 `TEMPLATE_RATIONALE_PATTERN`을 그대로 복사한 것이다
# (이 스크립트는 표준 라이브러리만 쓰는 self-contained 파일이라 import할 수 없다).
# `main_code_submission/tests/test_no_score_loss.py`가 두 리터럴의 동일성을 검사한다.
#
# 평가 서버에서는 template 대체가 그 에세이를 0점으로 잃는 것보다 낫다. 그러나 **로컬에서
# 하나라도 나오면 고칠 수 있는 버그**이므로 배포 gate를 떨어뜨린다.
TEMPLATE_RATIONALE_PATTERN = r"^(내용|조직|표현) 영역은 [1-5]점 수준으로, .+\.$"

# 2026-08-25. greedy 디코딩이 반복 루프에 빠져 JSON을 못 닫는 편이 400편 중 3편 있고,
# 그 편들은 engine이 파싱된 trait만 살리고 나머지를 template로 채운다. 점수는 온전하다
# (컨테이너 400편이 로컬 앙상블과 합 기준 400/400 일치, 공식 지표 소수 16자리까지 동일).
# repetition_penalty 1.05로 400편을 전수 재생성해 봤으나 루프가 **없어지지 않고 옮겨갔고**
# (실패편 3 -> 2, 겹침 0), 인용 정확도 93.39% -> 92.66%, 글 전체 grounded 191 -> 169편으로
# 근거 품질이 오히려 나빠졌다. 그래서 3편을 받아들이는 쪽을 택한다.
# 기본값은 0이라 이 상수를 명시적으로 넘기지 않는 한 기존과 bit-exact 같은 판정을 한다.
_TEMPLATE_RATIONALE_BUDGET = int(
    os.environ.get("HTTP_TEMPLATE_RATIONALE_BUDGET", "0")
)
REQUEST_CONFIG = {
    "max_tokens": 512,
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 42,
    "stop": list(STOP_SEQUENCES),
}
PROMPT_TRANSCRIPTION_SHA256 = (
    "d202a7f24aa1c8a8930885f41ef12faf8495163cdec5f9c209118322dc84e925"
)

# 과제기술서 9-1의 사람이 읽을 수 있는 내용을 옮긴 지시문. 공지대로 이 지시문과
# prompt/essay가 한 user role에 들어간다. PDF 표는 런타임 문자열 원본을 제공하지 않으므로
# whitespace/출력 예시까지 운영 서버와 byte-exact라고 주장하지 않는다. 서버는 마지막 두
# section marker 뒤의 payload만 사용하므로 검증 대상인 prompt/essay 입력 표면은 보존된다.
OFFICIAL_INSTRUCTION = """[역할]
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

[점수 기준]
5점: 매우 우수함. 결함이 거의 없고, essay_text에서 확인되는 구체적 강점이 뚜렷함.
4점: 우수함. 경미한 약점은 있으나 기준을 전반적으로 잘 충족함.
3점: 보통. 장점과 약점이 함께 있으며 기준을 부분적으로 충족함.
2점: 미흡함. 주요 결함이 있어 기준 충족이 제한적임.
1점: 매우 미흡함. 기준을 거의 충족하지 못하거나 심각한 결함이 있음.

[평가 원칙]
- 1~5점 전 구간을 적극적으로 사용하라.
- 각 기준은 서로 독립적으로 판단하라.
- essay_text에서 확인 가능한 내용만 근거로 삼아라.
- 전반적 인상만으로 높은 점수를 주지 말고 구체적 근거를 확인하라.
- 근거 설명은 기준별로 분리해 작성하라.

[출력 규칙]
- JSON 객체 하나만 출력하라. 코드블록 마크다운 사용 금지.
- 모든 점수는 1~5 정수.
- criterion_rationale의 각 값은 한국어로 작성하라.

[출력 형식]
{"content":{"score":3,"rationale":"content 판단 근거"},"organization":{"score":3,"rationale":"organization 판단 근거"},"expression":{"score":3,"rationale":"expression 판단 근거"}}"""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_official_user_content(prompt_text: str, essay_text: str) -> str:
    """Build the organizer-announced single-user message without normalizing bytes."""

    return (
        f"{OFFICIAL_INSTRUCTION}\n\n"
        f"[prompt_text]\n{prompt_text}\n\n"
        f"[essay_text]\n{essay_text}"
    )


def finite_score(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean은 점수가 아닙니다")
    number = float(value)
    if not math.isfinite(number) or not 1.0 <= number <= 5.0:
        raise ValueError(f"점수는 1~5의 유한한 수여야 합니다: {value!r}")
    return number


def half_up_score(value: Any) -> int:
    """Positive-range 사사오입: floor(x + 0.5), not Python banker's round."""

    return int(math.floor(finite_score(value) + 0.5))


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        shared_rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = shared_rank
        start = end
    return ranks


def rmse(predictions: Sequence[float], gold: Sequence[float]) -> float:
    if not predictions or len(predictions) != len(gold):
        raise ValueError("RMSE 입력 길이가 비었거나 다릅니다")
    return math.sqrt(
        sum((prediction - target) ** 2 for prediction, target in zip(predictions, gold))
        / len(gold)
    )


def spearman(predictions: Sequence[float], gold: Sequence[float]) -> float | None:
    if not predictions or len(predictions) != len(gold):
        raise ValueError("Spearman 입력 길이가 비었거나 다릅니다")
    pred_rank, gold_rank = average_ranks(predictions), average_ranks(gold)
    pred_mean, gold_mean = statistics.mean(pred_rank), statistics.mean(gold_rank)
    numerator = sum(
        (x - pred_mean) * (y - gold_mean) for x, y in zip(pred_rank, gold_rank)
    )
    denominator = math.sqrt(
        sum((x - pred_mean) ** 2 for x in pred_rank)
        * sum((y - gold_mean) ** 2 for y in gold_rank)
    )
    if denominator == 0:
        return None
    value = numerator / denominator
    return None if math.isnan(value) else value


def metric_pair(predictions: Sequence[float], gold: Sequence[float]) -> dict[str, Any]:
    return {"rmse": rmse(predictions, gold), "spearman": spearman(predictions, gold)}


def load_validation(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # str.splitlines() also splits U+2028/U+2029 inside JSON strings.  Iterate on
    # physical LF-delimited records instead so essay bytes remain valid JSONL.
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"{path}:{line_number}: JSON 파싱 실패: {exc}"
                ) from exc
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise SystemExit(f"입력 데이터가 비었습니다: {path}")
    ids: list[str] = []
    for index, row in enumerate(rows, start=1):
        essay_id = row.get("id", row.get("essay_id"))
        prompt = row.get("prompt", row.get("prompt_text"))
        essay = row.get("essay", row.get("essay_text"))
        gold = row.get("score", row.get("human_score"))
        if not isinstance(essay_id, str) or not essay_id:
            raise SystemExit(f"입력 {index}: id/essay_id가 없습니다")
        if not isinstance(prompt, str) or not isinstance(essay, str):
            raise SystemExit(f"{essay_id}: prompt/essay 문자열이 없습니다")
        if not isinstance(gold, dict) or any(name not in gold for name in TRAITS):
            raise SystemExit(f"{essay_id}: 공식 gold C/O/E가 없습니다")
        for name in TRAITS:
            finite_score(gold[name])
        if "average" not in gold:
            raise SystemExit(
                f"{essay_id}: score.average가 없습니다. trait 재평균으로 대체하지 않습니다"
            )
        average = float(gold["average"])
        if not math.isfinite(average):
            raise SystemExit(f"{essay_id}: score.average가 유한하지 않습니다")
        ids.append(essay_id)
    duplicates = sorted({essay_id for essay_id in ids if ids.count(essay_id) > 1})
    if duplicates:
        raise SystemExit(f"중복 essay id가 있습니다: {duplicates[:5]}")
    return rows


def wait_for_service(
    base_url: str,
    timeout_seconds: float,
    *,
    poll_interval_seconds: float = 5.0,
) -> dict[str, Any]:
    root = base_url.rstrip("/")
    started = time.monotonic()
    deadline = started + timeout_seconds
    attempts = 0
    last_error = "요청 전"
    while time.monotonic() < deadline:
        attempts += 1
        try:
            with urllib.request.urlopen(f"{root}/health", timeout=15) as response:
                body = response.read().decode("utf-8", errors="replace")
                if response.status == 200:
                    break
                last_error = f"HTTP {response.status}: {body[:200]}"
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            health_status = None
            try:
                payload = json.loads(body)
                if isinstance(payload, dict):
                    health_status = payload.get("status")
            except json.JSONDecodeError:
                pass
            last_error = f"HTTP {exc.code}: {body[:1000]}"
            # 우리 server의 명시적 loading 503만 계속 기다린다. failed/500뿐 아니라
            # 상태를 식별할 수 없는 응답도 startup contract 위반으로 즉시 노출한다.
            if exc.code != 503 or health_status != "loading":
                raise RuntimeError(f"/health terminal failure: {last_error}") from exc
        except Exception as exc:  # noqa: BLE001 - not-ready errors are expected
            last_error = repr(exc)
        time.sleep(min(poll_interval_seconds, max(0.0, deadline - time.monotonic())))
    else:
        raise RuntimeError(
            f"/health가 {timeout_seconds:g}초 안에 200이 되지 않았습니다: {last_error}"
        )

    with urllib.request.urlopen(f"{root}/v1/models", timeout=30) as response:
        models_text = response.read().decode("utf-8", errors="strict")
        models = json.loads(models_text)
    try:
        model_id = models["data"][0]["id"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"/v1/models에 data[0].id가 없습니다: {models}") from exc
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError(f"/v1/models id가 비었습니다: {model_id!r}")
    return {
        "health_attempts": attempts,
        "readiness_seconds": round(time.monotonic() - started, 3),
        "model_id": model_id,
        "models_response": models,
    }


def request_completion(
    base_url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    status: int | None = None
    response_headers: dict[str, str] = {}
    body_text = ""
    error: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = response.status
            response_headers = dict(response.headers.items())
            body_text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        response_headers = dict(exc.headers.items()) if exc.headers else {}
        body_text = exc.read().decode("utf-8", errors="replace")
        error = repr(exc)
    except Exception as exc:  # noqa: BLE001 - preserve transport failure for audit
        error = repr(exc)
    return {
        "status": status,
        "headers": response_headers,
        "body": body_text,
        "error": error,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def assistant_content(response_text: str) -> str:
    body = json.loads(response_text)
    content = body["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise TypeError("choices[0].message.content가 문자열이 아닙니다")
    return content


def evaluate_one(
    *,
    index: int,
    row: dict[str, Any],
    base_url: str,
    model_id: str,
    request_timeout: float,
    request_config: dict[str, Any],
    raw_handle: Any,
) -> dict[str, Any]:
    essay_id = str(row.get("id", row.get("essay_id")))
    prompt = row.get("prompt", row.get("prompt_text"))
    essay = row.get("essay", row.get("essay_text"))
    content = build_official_user_content(prompt, essay)
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        **request_config,
    }
    attempts: list[str] = []
    parsed: dict[str, Any] | None = None
    final_content = ""
    final_error: str | None = None
    for attempt in range(1, 4):  # 최초 1회 + 공식 규정의 재시도 2회
        outcome = request_completion(base_url, payload, request_timeout)
        attempt_content = ""
        attempt_record = {
            "index": index,
            "essay_id": essay_id,
            "attempt": attempt,
            "requested_at": now_iso(),
            "request": payload,
            "request_sha256": hashlib.sha256(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "http_status": outcome["status"],
            "response_headers": outcome["headers"],
            "response_body": outcome["body"],
            "transport_error": outcome["error"],
            "elapsed_seconds": outcome["elapsed_seconds"],
        }
        try:
            if outcome["status"] != 200:
                raise RuntimeError(f"HTTP status={outcome['status']}")
            attempt_content = assistant_content(outcome["body"])
            attempt_record["assistant_content"] = attempt_content
            parsed = parse_model_output(attempt_content)
            if parsed is None:
                raise ValueError("공식 first-balanced-JSON parser 실패")
            final_error = None
        except Exception as exc:  # noqa: BLE001 - retry and retain exact failure
            final_error = repr(exc)
            attempt_record["parse_error"] = final_error
            parsed = None
        raw_handle.write(json.dumps(attempt_record, ensure_ascii=False) + "\n")
        raw_handle.flush()
        final_content = attempt_content
        attempts.append(attempt_content if attempt_content else outcome["body"])
        if parsed is not None:
            break

    record: dict[str, Any] = {
        "index": index,
        "essay_id": essay_id,
        "attempts": len(attempts),
        "retry_outputs_identical": len(set(attempts)) <= 1,
        "parse_ok": parsed is not None,
        "score_processing_ok": False,
        "raw_response_content": final_content,
        "parsed_output": parsed,
        "raw_scores": None,
        "official_scores": {name: 0 for name in TRAITS},
        "rationales": None,
        "gold": row.get("score", row.get("human_score")),
        "final_error": final_error,
        "stop_sequence_hits": [
            token for token in STOP_SEQUENCES if token in final_content
        ],
        "output_chars": len(final_content),
        "json_object_only": False,
        "top_level_keys_exact": False,
    }
    if parsed is None:
        # 과제기술서: 두 번 재시도 뒤 파싱 실패는 0점. 공식 metric은 이 행도 포함한다.
        return record

    try:
        raw_scores = {name: finite_score(parsed[name]["score"]) for name in TRAITS}
        official_scores = {
            name: half_up_score(parsed[name]["score"]) for name in TRAITS
        }
    except (KeyError, TypeError, ValueError) as exc:
        record["final_error"] = f"score processing: {exc!r}"
        return record

    record.update(
        {
            "score_processing_ok": True,
            "raw_scores": raw_scores,
            "official_scores": official_scores,
            "response_scores_are_integers": all(
                isinstance(parsed[name]["score"], (int, float))
                and not isinstance(parsed[name]["score"], bool)
                and value.is_integer()
                for name, value in raw_scores.items()
            ),
            "official_rounding_changed": any(
                raw_scores[name] != official_scores[name] for name in TRAITS
            ),
            "rationales": {
                name: (
                    parsed[name].get("rationale", "")
                    if isinstance(parsed[name].get("rationale", ""), str)
                    else ""
                )
                for name in TRAITS
            },
            "json_object_only": final_content.strip()
            == extract_first_json(final_content),
            "top_level_keys_exact": set(parsed) == set(TRAITS),
        }
    )
    return record


def surface_metrics(
    records: Sequence[dict[str, Any]],
    score_key: str,
    *,
    valid_only: bool,
) -> dict[str, Any]:
    selected = [
        record
        for record in records
        if not valid_only or isinstance(record.get(score_key), dict)
    ]
    if not selected:
        unavailable = {"rmse": None, "spearman": None}
        return {
            "n": 0,
            "excluded": len(records),
            "average": dict(unavailable),
            "traits": {trait: dict(unavailable) for trait in TRAITS},
        }
    trait_metrics: dict[str, Any] = {}
    for trait in TRAITS:
        predictions = [float(record[score_key][trait]) for record in selected]
        gold = [float(record["gold"][trait]) for record in selected]
        trait_metrics[trait] = metric_pair(predictions, gold)
    pred_average = [
        statistics.mean(float(record[score_key][trait]) for trait in TRAITS)
        for record in selected
    ]
    # 반드시 dataset의 명시적 score.average를 쓴다. C/O/E를 재평균하지 않는다.
    gold_average = [float(record["gold"]["average"]) for record in selected]
    return {
        "n": len(selected),
        "excluded": len(records) - len(selected),
        "average": metric_pair(pred_average, gold_average),
        "traits": trait_metrics,
    }


def calculate_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    matched_surface = surface_metrics(records, "official_scores", valid_only=False)
    raw_surface = surface_metrics(records, "raw_scores", valid_only=True)
    official_after_half_up = {
        "n": matched_surface["n"],
        "excluded": matched_surface["excluded"],
        **matched_surface["average"],
        "rule": (
            "HTTP JSON의 C/O/E 각각에 floor(score + 0.5)를 적용한 뒤 평균; "
            "파싱/점수 실패 행은 0점"
        ),
    }
    return {
        "schema_version": 1,
        "count": len(records),
        "gold": {
            "trait_scores": "score.content/organization/expression 실수 유지",
            "average_source": "dataset score.average 실수 유지 (trait 재평균 아님)",
        },
        "official": {
            "gold_source": "score.average",
            "official_after_per_trait_half_up": official_after_half_up,
            "raw_continuous": {
                "n": raw_surface["n"],
                "excluded": raw_surface["excluded"],
                **raw_surface["average"],
                "rule": (
                    "HTTP JSON에 실제로 기록된 C/O/E를 evaluator 반올림 전에 평균; "
                    "파싱/점수 실패 행은 제외"
                ),
            },
        },
        "traits": {
            trait: {
                "gold_source": f"score.{trait}",
                "official_after_per_trait_half_up": matched_surface["traits"][trait],
                "raw_continuous": raw_surface["traits"][trait],
            }
            for trait in TRAITS
        },
    }


def run_evaluation(
    *,
    base_url: str,
    input_path: Path,
    output_dir: Path,
    limit: int | None,
    expected_count: int,
    readiness_timeout: float,
    request_timeout: float,
    request_max_tokens: int,
    expected_model: str | None,
    image_ref: str | None,
    container_name: str | None,
    docker_network: str | None,
    gpu_spec: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started_at = now_iso()
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = load_validation(input_path, limit)
    request_contract = {**REQUEST_CONFIG, "max_tokens": request_max_tokens}
    actual_expected = (
        min(expected_count, limit) if limit is not None else expected_count
    )
    service = wait_for_service(base_url, readiness_timeout)
    model_id = service["model_id"]
    if expected_model and model_id != expected_model:
        raise RuntimeError(
            f"/v1/models id 불일치: 실제={model_id!r}, EXPECTED_MODEL={expected_model!r}"
        )

    print(
        f"service ready: model={model_id!r}, readiness={service['readiness_seconds']}s, "
        f"rows={len(rows)}"
    )
    records: list[dict[str, Any]] = []
    raw_path = output_dir / "raw_http.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw_handle:
        for index, row in enumerate(rows, start=1):
            record = evaluate_one(
                index=index,
                row=row,
                base_url=base_url,
                model_id=model_id,
                request_timeout=request_timeout,
                request_config=request_contract,
                raw_handle=raw_handle,
            )
            records.append(record)
            if index == 1 or index % 25 == 0 or index == len(rows):
                elapsed = time.monotonic() - started
                print(
                    f"  {index:>3}/{len(rows)}  parse={sum(r['parse_ok'] for r in records)} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    records_path = output_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics = calculate_metrics(records)
    parse_ok = sum(bool(record["parse_ok"]) for record in records)
    processing_ok = sum(bool(record["score_processing_ok"]) for record in records)
    integer_responses = sum(
        bool(record.get("response_scores_are_integers")) for record in records
    )
    rationale_ok = sum(
        bool(record.get("rationales"))
        and all(record["rationales"][name].strip() for name in TRAITS)
        for record in records
    )
    stop_hits = sum(bool(record["stop_sequence_hits"]) for record in records)
    json_only = sum(bool(record["json_object_only"]) for record in records)
    exact_keys = sum(bool(record["top_level_keys_exact"]) for record in records)
    known_fallback = sum(
        bool(record.get("rationales"))
        and any(
            marker in record["rationales"][name]
            for name in TRAITS
            for marker in ("degraded fallback", "기본 문구로 대체")
        )
        for record in records
    )
    template_rationale_hits = sum(
        bool(record.get("rationales"))
        and any(
            re.match(TEMPLATE_RATIONALE_PATTERN, record["rationales"][name].strip())
            for name in TRAITS
        )
        for record in records
    )
    gates = {
        f"input_count_is_{actual_expected}": len(records) == actual_expected,
        "all_outputs_parse": parse_ok == len(records),
        "all_scores_process": processing_ok == len(records),
        "all_response_scores_are_1_to_5_integers": integer_responses == len(records),
        "all_outputs_are_exactly_one_json_object": json_only == len(records),
        "all_top_level_keys_are_exactly_the_three_traits": exact_keys == len(records),
        "all_three_rationales_are_nonempty": rationale_ok == len(records),
        "no_stop_sequence_in_returned_content": stop_hits == 0,
        "no_known_degraded_or_empty_fallback": known_fallback == 0,
        "no_template_rationale_substitution": (
            template_rationale_hits <= _TEMPLATE_RATIONALE_BUDGET
        ),
    }
    summary = {
        "schema_version": 1,
        "started_at": started_at,
        "completed_at": now_iso(),
        "base_url": base_url,
        "served_model": model_id,
        "count": len(records),
        "expected_count": actual_expected,
        "limit": limit,
        "parse_ok": parse_ok,
        "parse_rate": parse_ok / len(records),
        "score_processing_ok": processing_ok,
        "zero_scored": len(records) - processing_ok,
        "integer_response_count": integer_responses,
        "json_object_only_count": json_only,
        "exact_top_level_key_count": exact_keys,
        "responses_changed_by_official_rounding": sum(
            bool(record.get("official_rounding_changed")) for record in records
        ),
        "nonempty_rationale_count": rationale_ok,
        "stop_sequence_hit_count": stop_hits,
        "known_fallback_count": known_fallback,
        "template_rationale_count": template_rationale_hits,
        "total_http_attempts": sum(int(record["attempts"]) for record in records),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "seconds_per_essay": round(
            (time.monotonic() - started) / max(len(records), 1), 3
        ),
        "service": service,
        "request_contract": request_contract,
        "reproduction_scope": {
            "single_user_message": True,
            "published_parser_executed_directly": True,
            "parse_failure_attempts_total": 3,
            "official_per_trait_half_up": True,
            "explicit_continuous_gold_average": True,
            "container_on_isolated_bridge_network": bool(docker_network),
            "host_port_publish_added_for_local_evaluator": True,
            "host_publish_checks_non_loopback_container_bind": True,
            "orchestrator_gpu_flag_exact": gpu_spec == "all",
            "official_half_up_equals_returned_scores": integer_responses
            == len(records),
            "image_internal_raw_scores_observable_over_http": False,
            "not_reproduced": [
                "hidden test 400 essays",
                "organizer evaluator container and its private timeout/concurrency",
                "LLM Judge qualitative score",
                "L40S 48GB hardware unless this host is itself a single L40S",
                "byte-exact instruction prefix (runtime bytes were not published)",
            ],
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "metrics_file": "metrics.json",
        "records_file": "records.jsonl",
        "raw_http_file": "raw_http.jsonl",
    }
    metadata = {
        "schema_version": 1,
        "created_at": now_iso(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "command": sys.argv,
        "image_ref": image_ref,
        "container_name": container_name,
        "docker_network": docker_network,
        "gpu_spec": gpu_spec,
        "input": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "input_size_bytes": input_path.stat().st_size,
        "input_rows_used": len(rows),
        "official_instruction_sha256": hashlib.sha256(
            OFFICIAL_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
        "official_instruction_provenance": (
            "human-readable transcription of task specification appendix 9-1; "
            "runtime prompt bytes were not published"
        ),
        "served_model": model_id,
        "base_url": base_url,
        "request_contract": request_contract,
    }
    for filename, payload in (
        ("metrics.json", metrics),
        ("summary.json", summary),
        ("metadata.json", metadata),
    ):
        (output_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary, metrics


def image_ref_from_tar(path: Path) -> str:
    """Return the sole RepoTag from an archive produced by ``docker save``."""

    try:
        with tarfile.open(path, "r:*") as archive:
            member = archive.getmember("manifest.json")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("manifest.json을 읽을 수 없습니다")
            manifest = json.load(stream)
    except (tarfile.TarError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"docker save archive manifest를 읽지 못했습니다: {exc}"
        ) from exc
    tags = sorted(
        {str(tag) for item in manifest for tag in (item.get("RepoTags") or []) if tag}
    )
    if len(tags) != 1:
        raise SystemExit(
            f"IMAGE_TAR에는 정확히 한 RepoTag가 있어야 합니다. 발견={tags}. "
            "한 태그만 docker save 하세요."
        )
    return tags[0]


class _FakeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    retry_counts: dict[str, int] = {}
    retry_lock = threading.Lock()
    health_responses: list[tuple[int, dict[str, Any]]] = []
    health_attempts = 0
    health_lock = threading.Lock()

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/health":
            with self.health_lock:
                if self.health_responses:
                    index = min(
                        self.health_attempts,
                        len(self.health_responses) - 1,
                    )
                    status, payload = self.health_responses[index]
                    type(self).health_attempts += 1
                else:
                    status, payload = 200, {"status": "ok"}
            self._json(payload, status)
        elif self.path == "/v1/models":
            self._json(
                {
                    "object": "list",
                    "data": [{"id": "fake-model", "object": "model"}],
                }
            )
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length).decode("utf-8"))
        assert self.path == "/v1/chat/completions"
        assert request["model"] == "fake-model"
        assert request["messages"][0]["role"] == "user"
        assert len(request["messages"]) == 1
        assert {
            key: request[key]
            for key in ("max_tokens", "temperature", "top_p", "seed", "stop")
        } == REQUEST_CONFIG
        content = request["messages"][0]["content"]
        assert content.startswith(OFFICIAL_INSTRUCTION + "\n\n[prompt_text]\n")
        assert "\n\n[essay_text]\n" in content
        if "retry-two" in content:
            with self.retry_lock:
                attempt = self.retry_counts.get(content, 0) + 1
                self.retry_counts[content] = attempt
            if attempt <= 2:
                self._json(
                    {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "첫 두 응답은 JSON이 아님",
                                }
                            }
                        ]
                    }
                )
                return
        scores = (4, 3, 2) if "essay-A" in content else (2, 3, 4)
        output = {
            name: {"score": score, "rationale": f"{name} 근거"}
            for name, score in zip(TRAITS, scores)
        }
        self._json(
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(output, ensure_ascii=False),
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )


def self_test() -> None:
    assert (
        hashlib.sha256(OFFICIAL_INSTRUCTION.encode("utf-8")).hexdigest()
        == PROMPT_TRANSCRIPTION_SHA256
    )
    prompt_probe = "  prompt-leading\t\u2028끝  "
    essay_probe = "\t essay-leading\n둘째 줄  "
    user_probe = build_official_user_content(prompt_probe, essay_probe)
    assert user_probe.endswith(
        f"[prompt_text]\n{prompt_probe}\n\n[essay_text]\n{essay_probe}"
    )
    assert half_up_score(1.49) == 1
    assert half_up_score(2.50) == 3
    assert half_up_score(4.50) == 5
    parsed = parse_model_output(
        '앞말 {"content":{"score":1},"organization":{"score":2},'
        '"expression":{"score":3}} 뒷말'
    )
    assert parsed is not None and parsed["organization"]["score"] == 2
    # 공지 parser는 rationale 유무/점수 타입을 검사하지 않고, 첫 JSON만 시도하며,
    # 문자열 안의 중괄호도 depth에 포함한다. 이 다소 엄격한 동작까지 그대로 고정한다.
    assert (
        parse_model_output(
            '```json\n{"content":{"score":1},"organization":{"score":2},'
            '"expression":{"score":3}}\n```'
        )
        is not None
    )
    assert (
        parse_model_output(
            '{"content":{"score":1,"rationale":"{"},'
            '"organization":{"score":2},"expression":{"score":3}}'
        )
        is None
    )
    assert (
        parse_model_output(
            '{} {"content":{"score":1},"organization":{"score":2},'
            '"expression":{"score":3}}'
        )
        is None
    )
    failed_metrics = calculate_metrics(
        [
            {
                "official_scores": {name: 0 for name in TRAITS},
                "raw_scores": None,
                "gold": {
                    "content": 3.0,
                    "organization": 3.0,
                    "expression": 3.0,
                    "average": 3.0,
                },
            }
        ]
    )
    assert (
        failed_metrics["official"]["official_after_per_trait_half_up"]["rmse"]
        == 3
    )
    assert "average_matched" not in failed_metrics["official"]
    assert failed_metrics["official"]["raw_continuous"]["rmse"] is None
    explicit_average_metrics = calculate_metrics(
        [
            {
                "official_scores": {name: 3 for name in TRAITS},
                "raw_scores": {name: 3.0 for name in TRAITS},
                "gold": {
                    "content": 3.0,
                    "organization": 3.0,
                    "expression": 3.0,
                    "average": 4.0,
                },
            }
        ]
    )
    assert (
        explicit_average_metrics["official"]["official_after_per_trait_half_up"][
            "rmse"
        ]
        == 1
    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        _FakeHandler.health_responses = [
            (503, {"status": "loading"}),
            (503, {"status": "loading"}),
            (200, {"status": "ok"}),
        ]
        _FakeHandler.health_attempts = 0
        readiness = wait_for_service(
            base_url,
            timeout_seconds=2,
            poll_interval_seconds=0.01,
        )
        assert readiness["health_attempts"] == 3

        for terminal_code in (500, 503):
            _FakeHandler.health_responses = [
                (
                    terminal_code,
                    {
                        "status": "failed",
                        "error": {
                            "type": "RuntimeError",
                            "message": "sentinel-load-failure",
                        },
                    },
                )
            ]
            _FakeHandler.health_attempts = 0
            started = time.monotonic()
            try:
                wait_for_service(
                    base_url,
                    timeout_seconds=2,
                    poll_interval_seconds=0.01,
                )
            except RuntimeError as exc:
                assert "terminal failure" in str(exc)
                assert "sentinel-load-failure" in str(exc)
            else:
                raise AssertionError("terminal health failure가 성공으로 처리됐습니다")
            assert time.monotonic() - started < 1.0

        _FakeHandler.health_responses = []
        _FakeHandler.health_attempts = 0
        with tempfile.TemporaryDirectory(prefix="malpyeong-docker-check-") as temp:
            root = Path(temp)
            input_path = root / "validation.jsonl"
            rows = [
                {
                    "id": "A",
                    "prompt": "prompt-A",
                    "essay": "essay-A",
                    "score": {
                        "content": 4.0,
                        "organization": 3.0,
                        "expression": 2.0,
                        "average": 3.0,
                    },
                },
                {
                    "id": "B",
                    "prompt": "prompt-B",
                    "essay": "essay-B",
                    "score": {
                        "content": 2.0,
                        "organization": 3.0,
                        "expression": 4.0,
                        "average": 3.0,
                    },
                },
            ]
            input_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            summary, metrics = run_evaluation(
                base_url=base_url,
                input_path=input_path,
                output_dir=root / "results",
                limit=None,
                expected_count=2,
                readiness_timeout=5,
                request_timeout=5,
                request_max_tokens=512,
                expected_model="fake-model",
                image_ref="fake:image",
                container_name="fake-container",
                docker_network="fake-eval-net",
                gpu_spec="all",
            )
            assert summary["all_gates_passed"]
            assert (
                metrics["official"]["official_after_per_trait_half_up"]["rmse"]
                == 0
            )
            assert metrics["official"]["raw_continuous"]["rmse"] == 0
            assert sum(1 for _ in (root / "results" / "records.jsonl").open()) == 2
            assert sum(1 for _ in (root / "results" / "raw_http.jsonl").open()) == 2

            retry_raw = root / "retry_raw.jsonl"
            _FakeHandler.retry_counts.clear()
            with retry_raw.open("w", encoding="utf-8") as raw_handle:
                retried = evaluate_one(
                    index=1,
                    row={
                        "id": "retry",
                        "prompt": "retry-prompt",
                        "essay": "retry-two",
                        "score": {
                            "content": 2.0,
                            "organization": 3.0,
                            "expression": 4.0,
                            "average": 3.0,
                        },
                    },
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    model_id="fake-model",
                    request_timeout=5,
                    request_config=REQUEST_CONFIG,
                    raw_handle=raw_handle,
                )
            assert retried["parse_ok"] and retried["attempts"] == 3
            assert sum(1 for _ in retry_raw.open()) == 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    print(
        "SELF-TEST PASS: published parser quirks, exact request fields, "
        "two retries, half-up, explicit gold average, fake HTTP, artifacts"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--image-ref-from-tar",
        metavar="PATH",
        help="docker save tar의 유일한 RepoTag를 출력하고 종료",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--input")
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-count", type=int, default=400)
    parser.add_argument("--readiness-timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--request-max-tokens", type=int, default=512)
    parser.add_argument("--expected-model")
    parser.add_argument("--image-ref")
    parser.add_argument("--container-name")
    parser.add_argument("--docker-network")
    parser.add_argument("--gpu-spec")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        self_test()
        return
    if args.image_ref_from_tar:
        print(image_ref_from_tar(Path(args.image_ref_from_tar)))
        return
    if not args.input or not args.output_dir:
        raise SystemExit("평가에는 --input과 --output-dir가 필요합니다")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit은 양수여야 합니다")
    if args.expected_count < 1:
        raise SystemExit("--expected-count는 양수여야 합니다")
    if args.request_max_tokens < 1:
        raise SystemExit("--request-max-tokens는 양수여야 합니다")
    summary, metrics = run_evaluation(
        base_url=args.base_url,
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        expected_count=args.expected_count,
        readiness_timeout=args.readiness_timeout,
        request_timeout=args.request_timeout,
        request_max_tokens=args.request_max_tokens,
        expected_model=args.expected_model,
        image_ref=args.image_ref,
        container_name=args.container_name,
        docker_network=args.docker_network,
        gpu_spec=args.gpu_spec,
    )
    official = metrics["official"]["official_after_per_trait_half_up"]
    raw = metrics["official"]["raw_continuous"]
    print(json.dumps(summary["gates"], ensure_ascii=False, indent=2))
    print(
        f"official average: RMSE={official['rmse']:.6f}, Spearman={official['spearman']}"
    )
    raw_rmse = "N/A" if raw["rmse"] is None else f"{raw['rmse']:.6f}"
    print(f"raw HTTP average: RMSE={raw_rmse}, Spearman={raw['spearman']}")
    raise SystemExit(0 if summary["all_gates_passed"] else 1)


if __name__ == "__main__":
    main()
