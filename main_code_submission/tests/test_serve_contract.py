"""CPU tests for the minimum Docker/OpenAI server contract."""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from main_code_submission import serve  # noqa: E402
from main_code_submission.config import RationaleSpec, ScoreMember, SubmissionConfig
from main_code_submission.schema import TRAITS, _parse_model_output

PROMPT = "로봇세 도입에 대한 의견을 쓰시오."
ESSAY = "로봇세는 필요하다. 이유는 다음과 같다."


def build_official_user_content(prompt_text: str, essay_text: str) -> str:
    return f"지시문\n\n[prompt_text]\n{prompt_text}\n\n[essay_text]\n{essay_text}"


def _config() -> SubmissionConfig:
    return SubmissionConfig(
        name="serve_contract",
        score_members=(
            ScoreMember(
                name="m0",
                checkpoint=serve.Path("/nonexistent"),
                backbone_key="k0",
                parameters_billion=7.0,
            ),
        ),
        rationale=RationaleSpec(base_model="stub", adapter=None, enabled=False),
        served_model_name="malpyeong-writing-scorer",
    )


class _StubEngine:
    def __init__(self, config: SubmissionConfig) -> None:
        self.config = config
        self.respond_calls = 0

    def respond(self, text: object) -> tuple[str, dict[str, object]]:
        self.respond_calls += 1
        body = json.dumps(
            {trait: {"score": 3, "rationale": f"{trait} 근거"} for trait in TRAITS},
            ensure_ascii=False,
        )
        return body, {"scores": {trait: 3 for trait in TRAITS}}


def _official_request() -> dict[str, Any]:
    return {
        "model": "malpyeong-writing-scorer",
        "messages": [
            {
                "role": "user",
                "content": build_official_user_content(PROMPT, ESSAY),
            }
        ],
        "max_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "stop": ["Q:", "User:"],
    }


def test_ready_app_exposes_required_endpoints() -> None:
    engine = _StubEngine(_config())
    with TestClient(serve.create_app(_config(), engine)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        models = client.get("/v1/models")
        assert models.status_code == 200
        assert models.json()["data"][0]["id"] == "malpyeong-writing-scorer"

        response = client.post("/v1/chat/completions", json=_official_request())
        assert response.status_code == 200
        payload = response.json()
        assert payload["object"] == "chat.completion"
        assert payload["choices"][0]["message"]["role"] == "assistant"
        parsed = _parse_model_output(payload["choices"][0]["message"]["content"])
        assert parsed is not None
        assert all(parsed[trait]["score"] == 3 for trait in TRAITS)
        assert engine.respond_calls == 1


def test_markerless_rule_example_returns_openai_shape_without_fake_scores() -> None:
    engine = _StubEngine(_config())
    with TestClient(serve.create_app(_config(), engine)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "안녕하세요"}]},
        )
        assert response.status_code == 200
        body = response.json()["choices"][0]["message"]["content"]
        assert "[prompt_text]" in body and "[essay_text]" in body
        assert _parse_model_output(body) is None
        assert engine.respond_calls == 0


def test_unscoreable_official_request_returns_a_neutral_triple_not_a_400() -> None:
    """예전에는 400을 요구했다. `_unscoreable_request`와 같은 이유로 뒤집는다.

    2026-08-19 운영측 답변은 "모델 출력이 파싱 불가능한" 경우만 다루고 HTTP 4xx를
    어떻게 집계하는지는 말하지 않았다. 재시도인지 0점인지 모르는 쪽에 거는 대신
    확실히 파싱되는 중립 삼중을 200으로 낸다. 0점 한 행의 제곱오차는 12.16이고
    중립 삼중은 0.41이라 30배 차이다. (같은 정책 전환이
    `test_inference_error_still_returns_a_parseable_score_not_a_500`에는 이미
    반영돼 있었고 이 검사만 남아 있었다.)
    """

    engine = _StubEngine(_config())
    with TestClient(serve.create_app(_config(), engine)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "[prompt_text]\n주제만 있음"}]
            },
        )
        assert response.status_code == 200
        parsed = _parse_model_output(response.json()["choices"][0]["message"]["content"])
        assert parsed is not None
        assert all(1 <= int(parsed[trait]["score"]) <= 5 for trait in TRAITS)
        assert all(parsed[trait]["rationale"].strip() for trait in TRAITS)
        # 점수를 지어내려고 모델을 부르지는 않는다. 상수 응답이어야 한다.
        assert engine.respond_calls == 0


def test_inference_error_still_returns_a_parseable_score_not_a_500() -> None:
    """평가 서버에서 HTTP 500은 그 에세이 0점이다. 상수 점수보다 28배 비싸다.

    예전에는 여기서 500을 요구했다. 개발 중에는 실패를 드러내는 것이 옳지만 평가 서버
    위에서는 디버깅할 수단이 없고 대안이 확정 0점이라 정책을 뒤집었다. 실패는 응답을
    버리는 대신 로그와 ``degradation``에 남긴다.
    """

    class _FailingEngine(_StubEngine):
        def respond(self, text: object) -> tuple[str, dict[str, object]]:
            self.respond_calls += 1
            raise RuntimeError("sentinel-cuda-forward-failure")

    engine = _FailingEngine(_config())
    app = serve.create_app(_config(), engine)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/v1/chat/completions", json=_official_request())
        assert response.status_code == 200
        assert engine.respond_calls == 1
        parsed = _parse_model_output(response.json()["choices"][0]["message"]["content"])
        assert parsed is not None
        assert all(1 <= int(parsed[trait]["score"]) <= 5 for trait in TRAITS)
        assert all(parsed[trait]["rationale"].strip() for trait in TRAITS)
        assert client.get("/health").json() == {"status": "ok"}


def test_cuda_probe_requires_a_real_bf16_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    class _Scalar:
        def item(self) -> float:
            return 8.0

    class _Tensor:
        def __add__(self, value: int) -> "_Tensor":
            calls.append(("add", value))
            return self

        def sum(self) -> _Scalar:
            calls.append("sum")
            return _Scalar()

    monkeypatch.setattr(serve.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(serve.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(serve.torch.cuda, "get_device_name", lambda _device: "fake")
    monkeypatch.setattr(
        serve.torch.cuda, "get_device_capability", lambda _device: (8, 9)
    )
    monkeypatch.setattr(
        serve.torch,
        "ones",
        lambda *args, **kwargs: calls.append((args, kwargs)) or _Tensor(),
    )
    monkeypatch.setattr(
        serve.torch.cuda,
        "synchronize",
        lambda device: calls.append(("sync", device)),
    )

    report = serve._require_cuda_runtime()
    assert report["device"] == "fake"
    assert report["capability"] == [8, 9]
    assert ("sync", 0) in calls


def test_cuda_probe_rejects_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serve.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback 없음"):
        serve._require_cuda_runtime()


def test_bundled_hf_cache_ignores_runtime_home_and_forces_offline(
    tmp_path: serve.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hf_home = tmp_path / "neutral-hf"
    (hf_home / "hub" / "models--skt--A.X-4.0-Light").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path / "runtime-home"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "wrong-hf"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "wrong-hub"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")

    report = serve._configure_bundled_hf_cache(hf_home)

    expected_hub = str(hf_home / "hub")
    assert report["HF_HOME"] == str(hf_home)
    assert report["HF_HUB_CACHE"] == expected_hub
    assert report["HUGGINGFACE_HUB_CACHE"] == expected_hub
    assert report["HF_HUB_OFFLINE"] == "1"
    assert report["TRANSFORMERS_OFFLINE"] == "1"
    assert report["repositories"] == "models--skt--A.X-4.0-Light"
    assert serve.os.environ["HOME"] == str(tmp_path / "runtime-home")


def test_main_loads_engine_before_starting_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    config = _config()

    class _LoadedEngine(_StubEngine):
        def load(self) -> None:
            events.append("load")

    monkeypatch.setattr(serve, "load_manifest", lambda _path: config)
    monkeypatch.setattr(serve, "_configure_bundled_hf_cache", lambda: {})
    monkeypatch.setattr(serve, "_require_cuda_runtime", lambda: {"device": "fake"})
    monkeypatch.setattr(serve, "SubmissionEngine", _LoadedEngine)
    monkeypatch.setattr(sys, "argv", ["serve"])

    import uvicorn

    def fake_run(app: object, **kwargs: object) -> None:
        events.append("uvicorn")

    monkeypatch.setattr(uvicorn, "run", fake_run)
    serve.main()
    assert events == ["load", "uvicorn"]


def test_main_load_failure_never_starts_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()

    class _FailingLoadEngine(_StubEngine):
        def load(self) -> None:
            raise RuntimeError("sentinel-load-failure")

    monkeypatch.setattr(serve, "load_manifest", lambda _path: config)
    monkeypatch.setattr(serve, "_configure_bundled_hf_cache", lambda: {})
    monkeypatch.setattr(serve, "_require_cuda_runtime", lambda: {"device": "fake"})
    monkeypatch.setattr(serve, "SubmissionEngine", _FailingLoadEngine)
    monkeypatch.setattr(sys, "argv", ["serve"])

    import uvicorn

    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda *_args, **_kwargs: pytest.fail("uvicorn must not start"),
    )
    with pytest.raises(RuntimeError, match="sentinel-load-failure"):
        serve.main()
