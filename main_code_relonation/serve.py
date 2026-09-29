from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Mapping, Protocol, Sequence, TYPE_CHECKING

from fastapi import FastAPI, HTTPException

from . import TRAITS
from .artifacts import sha256_text
from .prompts import build_messages
from .request_parser import OfficialRequestText, parse_single_user_messages
from .schema import canonical_judge, compact_judge_json, parse_generated_judge

if TYPE_CHECKING:
    from .config import RationaleConfig


LOGGER = logging.getLogger("main_code_relonation.serve")
MAX_REQUEST_TOKENS = 2048


class GenerationResultLike(Protocol):
    raw: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float


class GeneratorRuntimeLike(Protocol):
    """Minimum interface implemented by modeling.GeneratorRuntime."""

    config: "RationaleConfig"

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> GenerationResultLike: ...


class FixedScoreBackend(Protocol):
    """Future scorer boundary for fixed-score rationale generation.

    No lookup or pseudo-score implementation is supplied here. Once a deployable
    scorer is selected, it must predict from the two request spans at inference
    time and its scores can then condition a separate rationale generator.
    """

    def predict_scores(self, prompt: str, essay: str) -> Mapping[str, float]: ...


def _environment_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    if value not in {"0", "1"}:
        raise ValueError(f"{name}은 0 또는 1이어야 합니다")
    return value == "1"


def _runtime_config_from_environment() -> "RationaleConfig":
    """Load a recipe or construct the small joint-generation Docker recipe."""

    from .config import RationaleConfig, load_config

    recipe_path = os.getenv("RATIONALE_CONFIG")
    if recipe_path:
        config = load_config(recipe_path)
    else:
        use_adapter = _environment_flag("USE_ADAPTER", False)
        config = RationaleConfig(
            model_id=os.getenv("MODEL_PATH", "/opt/models/model"),
            model_revision=os.getenv("MODEL_REVISION", "main"),
            trust_remote_code=_environment_flag("TRUST_REMOTE_CODE", False),
            adapter_path=(
                os.getenv("ADAPTER_PATH", "/opt/models/adapter")
                if use_adapter
                else None
            ),
            load_in_4bit=_environment_flag("LOAD_IN_4BIT", False),
            torch_dtype=os.getenv("TORCH_DTYPE", "bfloat16"),
            max_length=int(os.getenv("MAX_LENGTH", "4096")),
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "512")),
            temperature=0.0,
            score_mode="joint",
        )

    updates: dict[str, Any] = {}
    if "MODEL_PATH" in os.environ:
        updates["model_id"] = os.environ["MODEL_PATH"]
    if "USE_ADAPTER" in os.environ or "ADAPTER_PATH" in os.environ:
        use_adapter = _environment_flag(
            "USE_ADAPTER", bool(os.getenv("ADAPTER_PATH"))
        )
        updates["adapter_path"] = (
            os.getenv("ADAPTER_PATH", "/opt/models/adapter")
            if use_adapter
            else None
        )
    if "MAX_LENGTH" in os.environ:
        updates["max_length"] = int(os.environ["MAX_LENGTH"])
    if "MAX_NEW_TOKENS" in os.environ:
        updates["max_new_tokens"] = int(os.environ["MAX_NEW_TOKENS"])
    if updates:
        config = config.with_updates(**updates)
    return config


def _load_runtime() -> GeneratorRuntimeLike:
    # Importing modeling loads torch/transformers and may allocate the full model,
    # so it deliberately happens only during FastAPI startup, never at module import.
    from .modeling import GeneratorRuntime

    return GeneratorRuntime(_runtime_config_from_environment())


def _served_model_name(runtime: GeneratorRuntimeLike) -> str:
    return os.getenv("SERVED_MODEL_NAME") or runtime.config.model_id


def _validate_runtime(runtime: GeneratorRuntimeLike) -> None:
    if runtime.config.score_mode != "joint":
        raise ValueError(
            "현재 Docker server는 generator-only joint mode만 지원합니다. "
            "fixed mode에는 실제 score backend 연결이 필요합니다"
        )
    if runtime.config.temperature != 0.0:
        raise ValueError("제출 runtime config.temperature는 0.0이어야 합니다")
    if runtime.config.max_new_tokens > MAX_REQUEST_TOKENS:
        raise ValueError("runtime max_new_tokens는 2048 이하여야 합니다")


def _request_max_tokens(
    payload: Mapping[str, Any], runtime: GeneratorRuntimeLike
) -> int:
    value = payload.get("max_tokens", runtime.config.max_new_tokens)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_tokens는 정수여야 합니다")
    if not 1 <= value <= MAX_REQUEST_TOKENS:
        raise ValueError("max_tokens는 1~2048 범위여야 합니다")
    return value


def _request_temperature(payload: Mapping[str, Any]) -> float:
    value = payload.get("temperature", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("temperature는 숫자여야 합니다")
    result = float(value)
    if result != 0.0:
        raise ValueError("이 제출 server는 temperature=0.0 요청만 지원합니다")
    return result


def _canonical_assistant_content(raw: str) -> str:
    parsed = parse_generated_judge(raw)
    judge = canonical_judge(
        {trait: parsed[trait]["score"] for trait in TRAITS},
        {trait: parsed[trait]["rationale"] for trait in TRAITS},
    )
    return compact_judge_json(judge)


def _request_log(
    request_id: str,
    parsed: OfficialRequestText,
    *,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "user_characters": len(parsed.user_content),
        "prompt_characters": len(parsed.prompt),
        "essay_characters": len(parsed.essay),
        "user_sha256": sha256_text(parsed.user_content),
        "prompt_sha256": sha256_text(parsed.prompt),
        "essay_sha256": sha256_text(parsed.essay),
    }


def create_app(runtime: GeneratorRuntimeLike | None = None) -> FastAPI:
    generation_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        active_runtime = runtime if runtime is not None else _load_runtime()
        _validate_runtime(active_runtime)
        application.state.runtime = active_runtime
        application.state.model_name = _served_model_name(active_runtime)
        application.state.ready = True
        LOGGER.info(
            "runtime_ready %s",
            json.dumps(
                {
                    "model": application.state.model_name,
                    "model_path": active_runtime.config.model_id,
                    "adapter_path": active_runtime.config.adapter_path,
                    "max_length": active_runtime.config.max_length,
                    "max_new_tokens": active_runtime.config.max_new_tokens,
                    "temperature": active_runtime.config.temperature,
                    "score_mode": active_runtime.config.score_mode,
                },
                ensure_ascii=False,
            ),
        )
        try:
            yield
        finally:
            application.state.ready = False

    application = FastAPI(lifespan=lifespan)
    application.state.ready = False

    @application.get("/health")
    async def health() -> dict[str, str]:
        if not application.state.ready:
            raise HTTPException(status_code=503, detail="model is loading")
        return {"status": "ok"}

    @application.get("/v1/models")
    async def models() -> dict[str, Any]:
        if not application.state.ready:
            raise HTTPException(status_code=503, detail="model is loading")
        return {
            "object": "list",
            "data": [
                {
                    "id": application.state.model_name,
                    "object": "model",
                    "owned_by": "submission",
                }
            ],
        }

    @application.post("/v1/chat/completions")
    async def chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
        if not application.state.ready:
            raise HTTPException(status_code=503, detail="model is loading")
        active_runtime: GeneratorRuntimeLike = application.state.runtime
        model_name = application.state.model_name
        try:
            requested_model = payload.get("model")
            if requested_model != model_name:
                raise ValueError(
                    f"model은 /v1/models의 id와 같아야 합니다: {model_name}"
                )
            messages = payload.get("messages")
            if not isinstance(messages, list):
                raise ValueError("messages는 배열이어야 합니다")
            parsed = parse_single_user_messages(messages)
            max_tokens = _request_max_tokens(payload, active_runtime)
            _request_temperature(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        log = _request_log(
            request_id,
            parsed,
            model=model_name,
            max_tokens=max_tokens,
        )
        model_messages = build_messages(
            parsed.prompt,
            parsed.essay,
            scores=None,
        )
        log["model_user_sha256"] = sha256_text(model_messages[0]["content"])
        started = time.perf_counter()
        LOGGER.info("request_start %s", json.dumps(log, ensure_ascii=False))
        try:
            with generation_lock:
                result = active_runtime.generate(
                    model_messages,
                    max_new_tokens=max_tokens,
                )
            content = _canonical_assistant_content(result.raw)
        except Exception as exc:
            preview_limit = int(os.getenv("RAW_ERROR_PREVIEW_CHARS", "500"))
            raw = getattr(locals().get("result"), "raw", "")
            LOGGER.exception(
                "request_failed %s",
                json.dumps(
                    {
                        **log,
                        "error": str(exc),
                        "raw_characters": len(raw),
                        "raw_preview": raw[:preview_limit],
                    },
                    ensure_ascii=False,
                ),
            )
            raise HTTPException(
                status_code=500,
                detail="generation or first-JSON parsing failed",
            ) from exc

        elapsed = time.perf_counter() - started
        LOGGER.info(
            "request_complete %s",
            json.dumps(
                {
                    **log,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "generation_seconds": result.latency_seconds,
                    "request_seconds": elapsed,
                    "raw_characters": len(result.raw),
                    "canonical_characters": len(content),
                },
                ensure_ascii=False,
            ),
        )
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
        }

    return application


app = create_app()
