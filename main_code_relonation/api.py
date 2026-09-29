from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

from .artifacts import sha256_text


NONTHINKING_TEMPLATE_KWARGS = {"enable_thinking": False}


def local_base_url(base_url: str) -> str:
    """Validate that teacher/Judge traffic cannot leave the local machine."""

    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("base_url은 http 또는 https URL이어야 합니다")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("teacher/Judge endpoint는 localhost만 사용할 수 있습니다")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base_url에는 인증 정보, query, fragment를 넣지 마십시오")
    return base_url.rstrip("/")


def _openai_url(base_url: str, resource: str) -> str:
    endpoint = local_base_url(base_url)
    prefix = endpoint if urlparse(endpoint).path.rstrip("/").endswith("/v1") else f"{endpoint}/v1"
    return f"{prefix}/{resource.lstrip('/')}"


def request_json(
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 900.0,
) -> dict[str, Any]:
    encoded = (
        None
        if payload is None
        else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    )
    request = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code}: response_chars={len(body)} "
            f"response_sha256={sha256_text(body)}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"local endpoint 연결 실패: {exc.reason}") from exc
    if not isinstance(value, dict):
        raise ValueError("OpenAI-compatible 응답은 JSON 객체여야 합니다")
    return value


def resolve_model_id(base_url: str, *, timeout: float = 30.0) -> str:
    endpoint = local_base_url(base_url)
    value = request_json(_openai_url(endpoint, "models"), timeout=timeout)
    models = value.get("data")
    if not isinstance(models, list) or not models:
        raise ValueError("/v1/models 응답에 data가 없습니다")
    model_id = models[0].get("id") if isinstance(models[0], dict) else None
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("/v1/models 응답에 data[0].id가 없습니다")
    return model_id


# 시도 번호별 sampling. 0번은 공식 결정적 설정이고, 재시도는 표본을 흔든다. greedy로 고정하면
# 재시도가 **글자 단위로 같은 출력**을 내므로 형식·QC 실패는 몇 번을 보내도 같은 이유로
# 실패한다(연산만 3배 쓴다). 어느 설정으로 통과했는지는 audit의 `sampling`에 남는다.
SAMPLING_ATTEMPTS: tuple[dict[str, Any], ...] = (
    {"temperature": 0.0, "top_p": 1.0, "seed": 42},
    {"temperature": 0.7, "top_p": 0.95, "seed": 43},
    {"temperature": 1.0, "top_p": 0.95, "seed": 44},
)


def sampling_for_attempt(index: int) -> dict[str, Any]:
    return SAMPLING_ATTEMPTS[min(max(index, 0), len(SAMPLING_ATTEMPTS) - 1)]


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    timeout: float = 900.0,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Call a local OpenAI-compatible server.

    기본값은 공식 결정적 설정(greedy, seed 42)이다. 호출자가 값을 바꿀 수 있게 열어 둔 이유는
    재시도 때문이다. greedy로 고정하면 재시도가 **글자 단위로 같은 출력**을 내므로, 형식이나
    QC로 실패한 건은 몇 번을 다시 보내도 같은 이유로 실패한다(연산만 3배 쓴다).
    """

    endpoint = local_base_url(base_url)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "chat_template_kwargs": NONTHINKING_TEMPLATE_KWARGS,
    }
    request_hash = sha256_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    started = time.perf_counter()
    response = request_json(
        _openai_url(endpoint, "chat/completions"), payload, timeout=timeout
    )
    response["_client_latency_seconds"] = time.perf_counter() - started
    response["_request_sha256"] = request_hash
    return response


def completion_parts(response: dict[str, Any]) -> tuple[str, str | None]:
    try:
        message = response["choices"][0]["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning")
        if reasoning is None:
            reasoning = message.get("reasoning_content")
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("OpenAI Chat Completions 응답 형식이 아닙니다") from exc
    if not isinstance(content, str):
        raise ValueError("choices[0].message.content가 문자열이 아닙니다")
    if reasoning is not None and not isinstance(reasoning, str):
        reasoning = str(reasoning)
    return content, reasoning


def response_audit(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else {}
    usage = response.get("usage")
    return {
        "response_id": response.get("id"),
        "response_model": response.get("model"),
        "finish_reason": first.get("finish_reason")
        if isinstance(first, dict)
        else None,
        "usage": dict(usage) if isinstance(usage, dict) else None,
        "latency_seconds": float(response.get("_client_latency_seconds", 0.0)),
        "request_sha256": response.get("_request_sha256"),
    }
