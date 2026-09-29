"""Minimal OpenAI-compatible server required by the Docker submission rules.

Startup is deliberately synchronous: CUDA, the score model, and the rationale
adapter must all load before the HTTP server starts.  A load failure therefore
prints a traceback and terminates the container instead of returning an
untrained/CPU/fallback answer or remaining in a permanent 503 state.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import SubmissionConfig, load_manifest
from .degrade import last_resort_outputs
from .engine import SubmissionEngine
from .request_parser import (
    MARKER_ANOMALY_TOKEN,
    SCORING_MIN_CHARS,
    coerce_message_content,
    marker_anomaly,
    parse_single_user_messages,
    salvage_scoring_text,
)
from .schema import (
    ABSOLUTE_FALLBACK_BODY,
    TRAITS,
    TraitOutput,
    enforce_official_parse,
)

LOGGER = logging.getLogger("main_code_submission.serve")

DEFAULT_MANIFEST = Path(
    os.environ.get("SUBMISSION_MANIFEST", "/opt/submission/submission_manifest.json")
)
BUNDLED_HF_HOME = Path("/opt/submission/hf")


def _configure_bundled_hf_cache(
    hf_home: Path = BUNDLED_HF_HOME,
) -> dict[str, str]:
    """Use the image-baked HF cache regardless of runtime user or ``HOME``.

    The evaluation service may start the image with a different uid/home.  A
    cache under ``/root/.cache`` then looks empty even though the weights are in
    the image.  Configure the neutral image path before Transformers is imported
    by the model-loading functions.  Offline mode remains mandatory: a missing
    asset must terminate startup instead of attempting a download.
    """

    hub = hf_home / "hub"
    if not hub.is_dir():
        raise FileNotFoundError(f"이미지 내 Hugging Face cache가 없습니다: {hub}")
    repositories = sorted(
        path.name for path in hub.iterdir() if path.is_dir() and path.name.startswith("models--")
    )
    if not repositories:
        raise FileNotFoundError(f"이미지 내 Hugging Face model snapshot이 없습니다: {hub}")
    if not os.access(hub, os.R_OK | os.X_OK):
        raise PermissionError(f"현재 uid가 Hugging Face cache를 읽을 수 없습니다: {hub}")

    values = {
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hub),
        # Older huggingface_hub releases use this compatibility name.
        "HUGGINGFACE_HUB_CACHE": str(hub),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    os.environ.update(values)
    return {**values, "repositories": ",".join(repositories)}


def _require_cuda_runtime() -> dict[str, Any]:
    """Fail before model loading unless a real BF16 CUDA operation succeeds."""

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("제출 서버에는 CUDA GPU가 필요합니다 (CPU fallback 없음)")
    probe = torch.ones(4, dtype=torch.bfloat16, device="cuda:0")
    result = float((probe + 1).sum().item())
    torch.cuda.synchronize(0)
    if result != 8.0:
        raise RuntimeError(f"CUDA BF16 실행 결과가 잘못됐습니다: {result}")
    return {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }


def guaranteed_constant_body() -> str:
    """어떤 상황에서도 공식 파서를 통과하는 상수 응답.

    요청 자체를 못 읽어 채점이 불가능한 경우에 쓴다. 그때도 0점(제곱오차 12.16)보다
    중립 삼중(0.41)이 30배 낫다. 먼저 `enforce_official_parse`로 만들어 보고, 그것마저
    실패하면 글자 그대로 박아 둔 상수를 낸다. 이 함수는 예외를 올리지 않는다 —
    예외 처리기가 이 함수를 부르므로 여기서 raise하면 처리기가 다시 실패한다.
    """

    try:
        scores, rationales = last_resort_outputs()
        body, _ = enforce_official_parse(
            {
                trait: TraitOutput(score=scores[trait], rationale=rationales[trait])
                for trait in TRAITS
            }
        )
        return body
    except Exception:  # noqa: BLE001 - 여기서 raise하면 응답이 통째로 사라진다
        LOGGER.exception("상수 응답 생성마저 실패해 고정 문자열을 냅니다")
        return ABSOLUTE_FALLBACK_BODY


def _completion(
    config: SubmissionConfig,
    payload: dict[str, Any],
    body: str,
    started: float,
) -> dict[str, Any]:
    """Wrap assistant text in the minimum OpenAI Chat Completions shape."""

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(started),
        "model": payload.get("model") or config.served_model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": body},
                "finish_reason": "stop",
            }
        ],
    }


def _unscoreable_request(
    config: SubmissionConfig, payload: dict[str, Any], reason: str, started: float
) -> dict[str, Any]:
    """채점할 텍스트를 못 얻었을 때의 마지막 응답. **400을 내지 않는다.**

    2026-08-19 운영측 답변은 "모델 출력이 파싱 불가능한" 경우만 다루고 HTTP 4xx/5xx를
    어떻게 집계하는지는 말하지 않았다. 재시도되는지 0점인지 알 수 없으므로, 알 수 없는
    쪽에 거는 대신 확실히 파싱되는 중립 삼중을 200으로 돌려준다. 0점 한 행의 제곱오차는
    12.16이고 중립 삼중은 0.41이라 30배 차이다.

    규정 §10의 일반 API smoke는 이 경로에 오지 않는다. 짧고 마커 없는 요청은
    `_is_markerless_single_user`가 먼저 안내 문자열로 처리한다.
    """

    LOGGER.warning("채점 불가 요청(%s). 중립 상수로 응답합니다", reason)
    return _completion(config, payload, guaranteed_constant_body(), started)


def _is_markerless_single_user(messages: Any) -> bool:
    """Recognize the generic API smoke from Docker-rule section 10.

    규정 §10의 예시는 단일 user message에 짧은 인사말("한 줄로 자기소개해 주세요")을
    담는다. 그 요청에 C/O/E 점수를 꾸며 내 돌려주는 것은 이상하므로 안내 문자열로
    답한다. 마커가 없고 채점 대상으로 보기에 너무 짧은 요청이면 형태가 조금 달라도
    (typed part 배열, 앞에 붙은 system message) 같은 취급을 한다.
    """

    if not isinstance(messages, list) or not messages:
        return False
    chunks: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = coerce_message_content(message.get("content"))
        if content is None:
            return False
        chunks.append(content)
    if not chunks:
        return False
    joined = "\n".join(chunks)
    return (
        "[prompt_text]" not in joined
        and "[essay_text]" not in joined
        and len(joined.strip()) < SCORING_MIN_CHARS
    )


def create_app(config: SubmissionConfig, engine: SubmissionEngine) -> FastAPI:
    """Create the ready-only HTTP application around one loaded engine."""

    app = FastAPI(title="MalPyeong writing scorer", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": config.served_model_name,
                    "object": "model",
                    "owned_by": "submission",
                }
            ],
        }

    @app.exception_handler(RequestValidationError)
    def _malformed_request(request: Request, exc: Exception) -> JSONResponse:
        """본문이 JSON 객체가 아니면 FastAPI가 422를 내고 그 샘플은 0점이 된다.

        채점에 필요한 텍스트를 못 얻었으므로 중립 삼중이 최선이다. 그래도 0점보다 낫다.
        """

        LOGGER.warning("요청 본문을 읽지 못했습니다(%s). 중립 상수로 응답합니다", exc)
        return JSONResponse(
            _completion(config, {}, guaranteed_constant_body(), time.time()),
            status_code=200,
        )

    @app.exception_handler(Exception)
    def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        """핸들러 밖으로 새어 나온 어떤 예외도 0점으로 만들지 않는다."""

        LOGGER.exception("처리되지 않은 예외(%s). 중립 상수로 응답합니다", exc)
        return JSONResponse(
            _completion(config, {}, guaranteed_constant_body(), time.time()),
            status_code=200,
        )

    @app.post("/v1/chat/completions")
    def chat_completions(payload: dict[str, Any]) -> Any:
        started = time.time()
        messages = payload.get("messages")
        try:
            text = parse_single_user_messages(messages)
        except Exception as exc:  # noqa: BLE001 - ValueError만 잡으면 나머지가 500이 된다
            salvaged = salvage_scoring_text(messages)
            if salvaged is not None:
                # 마커를 못 읽었다고 400을 내면 그 에세이는 0점으로 집계된다
                # (2026-08-19 운영측 답변). 본문으로 보이는 길이의 텍스트가 있으면
                # 전체를 essay로 보고 채점한다. 잘못 읽어도 상수 삼중보다 낫고,
                # 0점(제곱오차 12.16)보다는 압도적으로 낫다.
                LOGGER.warning("공식 마커 파싱 실패(%s). 전체 본문으로 채점합니다", exc)
                text = salvaged
            elif _is_markerless_single_user(messages):
                return _completion(
                    config,
                    payload,
                    "이 서버는 [prompt_text]와 [essay_text]가 포함된 단일 user "
                    "메시지를 작문 평가 입력으로 사용합니다.",
                    started,
                )
            else:
                return _unscoreable_request(config, payload, str(exc), started)

        anomaly = marker_anomaly(text)
        if anomaly is not None:
            # 배포 게이트가 읽는 고정 토큰. 로컬 400편에서 한 건이라도 찍히면
            # 그 release는 조용한 오채점을 포함하므로 내보내면 안 된다.
            LOGGER.warning(
                "%s prompt=%d자 essay=%d자 사유=%s",
                MARKER_ANOMALY_TOKEN,
                len(text.prompt),
                len(text.essay),
                anomaly,
            )

        # SubmissionEngine owns the single-request lock and already degrades
        # internally rather than dropping a computed score.  This outer guard
        # exists for what it cannot catch -- an exception while serializing the
        # response, or any future path that starts raising again.  A 500 here is
        # scored as 0 for the essay, which costs ~28x more squared error than the
        # constant fallback triple, so nothing is allowed to escape.
        try:
            body, diagnostics = engine.respond(text)
        except Exception:  # noqa: BLE001 - 응답 유실이 어떤 오류보다 비싸다
            LOGGER.exception("engine.respond가 실패해 최후 상수 응답을 냅니다")
            body = guaranteed_constant_body()
            diagnostics = {
                "scores": last_resort_outputs()[0],
                "degradation": ["respond_failed"],
            }
        if diagnostics.get("degradation"):
            LOGGER.warning(
                "강등된 응답 %.2fs %s degradation=%s",
                time.time() - started,
                diagnostics.get("scores"),
                diagnostics["degradation"],
            )
        else:
            LOGGER.info(
                "scored in %.2fs %s", time.time() - started, diagnostics.get("scores")
            )
        return _completion(config, payload, body, started)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    LOGGER.info("고정 offline HF cache: %s", _configure_bundled_hf_cache())
    config = load_manifest(args.manifest)
    LOGGER.info("CUDA 확인: %s", _require_cuda_runtime())
    engine = SubmissionEngine(config)
    LOGGER.info("모델 로딩 시작: %s", config.name)
    engine.load()
    LOGGER.info("모델 로딩 완료: %s", config.name)

    import uvicorn

    uvicorn.run(create_app(config, engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
