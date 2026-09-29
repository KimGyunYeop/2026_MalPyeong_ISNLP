from __future__ import annotations

import asyncio
import json
import os
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import httpx
import pytest

# fastapi는 제출 이미지와 `.venv-vllm`에만 있다. 없는 환경에서 전체 suite가 collection error로
# 멈추지 않도록 이 모듈만 건너뛴다.
pytest.importorskip("fastapi")

from main_code_relonation.config import RationaleConfig  # noqa: E402
from main_code_relonation.prompts import build_messages  # noqa: E402
from main_code_relonation.request_parser import (  # noqa: E402
    parse_official_user_content,
    parse_single_user_messages,
)
from main_code_relonation.serve import create_app  # noqa: E402


@dataclass(frozen=True)
class FakeGenerationResult:
    raw: str
    prompt_tokens: int = 101
    completion_tokens: int = 37
    latency_seconds: float = 0.25


class FakeRuntime:
    def __init__(self, raw: str, *, score_mode: str = "joint") -> None:
        self.config = RationaleConfig(
            model_id="local-fake-model",
            max_length=8192,
            max_new_tokens=512,
            temperature=0.0,
            score_mode=score_mode,
        )
        self.raw = raw
        self.calls: list[tuple[list[dict[str, str]], int | None]] = []

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> FakeGenerationResult:
        self.calls.append((messages, max_new_tokens))
        return FakeGenerationResult(self.raw)


def valid_raw() -> str:
    return (
        "내부 추론은 응답에 포함하지 않는다.\n"
        '{"content":{"score":4,"rationale":"주장이 논제에 직접 대응하고 이유가 구체적이다.",'
        '"unused":"drop"},'
        '"organization":{"score":3,"rationale":"도입과 결론은 있으나 중간 전환이 다소 급하다."},'
        '"expression":{"score":4,"rationale":"문장이 대체로 자연스럽고 어휘가 적절하다."},'
        '"unused_top":"drop"}'
        "\n뒤 설명도 응답에 포함하지 않는다."
        '{"content":{"score":1,"rationale":"두 번째 JSON"}}'
    )


class RequestParserTest(unittest.TestCase):
    def test_last_markers_and_payload_bytes_are_preserved(self) -> None:
        prompt = "  논제 첫 줄\n논제 끝\t"
        essay = "\t 에세이 첫 줄  \n\n둘째 줄\r\n"
        content = (
            "지시문에서 [prompt_text]와 [essay_text]를 설명한다."
            "\n\n[prompt_text]\n이것은 앞의 예시"
            "\n\n[essay_text]\n앞의 예시 본문"
            f"\n\n[prompt_text]\n{prompt}"
            f"\n\n[essay_text]\n{essay}"
        )
        parsed = parse_official_user_content(content)
        self.assertEqual(parsed.prompt.encode("utf-8"), prompt.encode("utf-8"))
        self.assertEqual(parsed.essay.encode("utf-8"), essay.encode("utf-8"))
        self.assertEqual(parsed.user_content, content)

    def test_crlf_sections_preserve_crlf_payload(self) -> None:
        prompt = "질문\r\n계속"
        essay = "  글\r\n\r\n끝  "
        content = (
            "instruction\r\n\r\n[prompt_text]\r\n"
            + prompt
            + "\r\n\r\n[essay_text]\r\n"
            + essay
        )
        parsed = parse_official_user_content(content)
        self.assertEqual(parsed.prompt, prompt)
        self.assertEqual(parsed.essay, essay)

    def test_single_newline_marker_layout_is_accepted(self) -> None:
        content = "header\n[prompt_text]\n질문\n[essay_text]\n  답안  "
        parsed = parse_official_user_content(content)
        self.assertEqual(parsed.prompt, "질문")
        self.assertEqual(parsed.essay, "  답안  ")

    def test_extra_messages_fall_back_to_the_last_user_message(self) -> None:
        """규정은 단일 user message지만 거절하면 그 에세이가 0점이 된다.

        정책 근거는 `main_code_submission/NEVER_DISCARD_A_SCORE.md`에 있다. 두 폴더의
        `request_parser.py`는 같은 공식 계약을 공유하므로 함께 바뀐다.
        """

        content = "[prompt_text]\n질문\n\n[essay_text]\n글"
        parsed = parse_single_user_messages(
            [
                {"role": "user", "content": content},
                {"role": "user", "content": content},
            ]
        )
        self.assertEqual(parsed.essay, "글")

        # user message가 하나도 없으면 여전히 거절한다.
        with self.assertRaisesRegex(ValueError, "role"):
            parse_single_user_messages([{"role": "system", "content": content}])
        with self.assertRaises(ValueError):
            parse_single_user_messages(
                [
                    {"role": "system", "content": content},
                    {"role": "assistant", "content": content},
                ]
            )


class ServingContractTest(unittest.TestCase):
    def request(
        self,
        runtime: FakeRuntime,
        method: str,
        path: str,
        payload: dict | None = None,
    ) -> httpx.Response:
        async def execute() -> httpx.Response:
            application = create_app(runtime)
            async with application.router.lifespan_context(application):
                transport = httpx.ASGITransport(app=application)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    return await client.request(method, path, json=payload)

        return asyncio.run(execute())

    def payload(self, *, max_tokens: int = 512, temperature: float = 0.0):
        content = (
            "공식 채점 지시 전문\n\n[prompt_text]\n 논제 원문\t"
            "\n\n[essay_text]\n  에세이 원문\n둘째 줄  "
        )
        return {
            "model": "test-model",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
            "seed": 42,
            "stop": ["Q:", "User:"],
        }

    def test_required_endpoints_and_canonical_first_json(self) -> None:
        runtime = FakeRuntime(valid_raw())
        with patch.dict(os.environ, {"SERVED_MODEL_NAME": "test-model"}):
            self.assertEqual(
                self.request(runtime, "GET", "/health").json(),
                {"status": "ok"},
            )
            models = self.request(runtime, "GET", "/v1/models").json()
            self.assertEqual(models["data"][0]["id"], "test-model")
            payload = self.payload(max_tokens=2048)
            response = self.request(
                runtime, "POST", "/v1/chat/completions", payload
            )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["model"], "test-model")
        self.assertEqual(body["usage"]["total_tokens"], 138)
        content = body["choices"][0]["message"]["content"]
        expected = {
            "content": {
                "score": 4.0,
                "rationale": "주장이 논제에 직접 대응하고 이유가 구체적이다.",
            },
            "organization": {
                "score": 3.0,
                "rationale": "도입과 결론은 있으나 중간 전환이 다소 급하다.",
            },
            "expression": {
                "score": 4.0,
                "rationale": "문장이 대체로 자연스럽고 어휘가 적절하다.",
            },
        }
        self.assertEqual(
            content,
            json.dumps(expected, ensure_ascii=False, separators=(",", ":")),
        )
        self.assertNotIn("내부 추론", content)
        self.assertNotIn("unused", content)
        self.assertNotIn("두 번째 JSON", content)
        self.assertEqual(runtime.calls[0][1], 2048)
        expected_messages = build_messages(
            " 논제 원문\t",
            "  에세이 원문\n둘째 줄  ",
            scores=None,
        )
        self.assertEqual(runtime.calls[0][0], expected_messages)
        self.assertNotEqual(runtime.calls[0][0], payload["messages"])

    def test_default_max_tokens_comes_from_runtime_config(self) -> None:
        runtime = FakeRuntime(valid_raw())
        payload = self.payload()
        del payload["max_tokens"]
        with patch.dict(os.environ, {"SERVED_MODEL_NAME": "test-model"}):
            response = self.request(
                runtime, "POST", "/v1/chat/completions", payload
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(runtime.calls[0][1], 512)

    def test_invalid_sampling_or_request_shape_fails_before_generation(self) -> None:
        runtime = FakeRuntime(valid_raw())
        with patch.dict(os.environ, {"SERVED_MODEL_NAME": "test-model"}):
            hot = self.request(
                runtime,
                "POST",
                "/v1/chat/completions",
                self.payload(temperature=0.1),
            )
            too_long = self.request(
                runtime,
                "POST",
                "/v1/chat/completions",
                self.payload(max_tokens=2049),
            )
            multiple = self.payload()
            multiple["messages"].append(dict(multiple["messages"][0]))
            bad_messages = self.request(
                runtime,
                "POST",
                "/v1/chat/completions",
                multiple,
            )
        self.assertEqual(hot.status_code, 400)
        self.assertEqual(too_long.status_code, 400)
        # 마커가 있는 요청은 앞뒤에 무엇이 붙어도 채점한다. 400은 그 에세이 0점이고,
        # 그 비용이 상수 점수 대비 약 30배다 (NEVER_DISCARD_A_SCORE.md).
        self.assertEqual(bad_messages.status_code, 200)
        self.assertEqual(len(runtime.calls), 1)

    def test_invalid_generation_never_returns_raw_text(self) -> None:
        runtime = FakeRuntime("생각만 있고 JSON은 없음")
        with patch.dict(os.environ, {"SERVED_MODEL_NAME": "test-model"}):
            response = self.request(
                runtime,
                "POST",
                "/v1/chat/completions",
                self.payload(),
            )
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("생각만 있고", response.text)

    def test_fixed_mode_cannot_start_without_real_scorer(self) -> None:
        runtime = FakeRuntime(valid_raw(), score_mode="fixed")

        async def enter_lifespan() -> None:
            application = create_app(runtime)
            async with application.router.lifespan_context(application):
                pass

        with patch.dict(os.environ, {"SERVED_MODEL_NAME": "test-model"}):
            with self.assertRaisesRegex(ValueError, "score backend"):
                asyncio.run(enter_lifespan())


if __name__ == "__main__":
    unittest.main()
