from __future__ import annotations

import hashlib
import json
import math
import re
from types import SimpleNamespace

import pytest
import torch

from main_code_submission.prompts import build_rationale_messages
from main_code_submission.request_parser import (
    parse_official_user_content,
    parse_single_user_messages,
)
from main_code_submission.schema import (
    OFFICIAL_STOP_SEQUENCES,
    RATIONALE_CHAR_LIMIT,
    TRAITS,
    TraitOutput,
    _parse_model_output,
    build_response_json,
    clamp_score,
    official_evaluator_score,
    official_evaluator_scores,
    rationale_length_report,
    contains_stop_sequence,
    sanitize_rationale,
    verify_official_parse,
)

PROMPT = "로봇세 도입에 대한 자신의 의견을 논리적으로 제시하는 글을 쓰시오."
ESSAY = (
    " 최근 4차 산업혁명으로  인한 기술 발전에 따라\t로봇 기술을 쓰는 기업이 늘고 있다."
)


def build_official_user_content(prompt_text: str, essay_text: str) -> str:
    return f"지시문\n\n[prompt_text]\n{prompt_text}\n\n[essay_text]\n{essay_text}"


def rmse(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) / len(a))


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in order[start:end]:
            result[index] = rank
        start = end
    return result


def spearman(a: list[float], b: list[float]) -> float:
    left, right = _average_ranks(a), _average_ranks(b)
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator


class _MinimalEngineConfig:
    score_postprocess = "average_matched"

    def validate(self) -> "_MinimalEngineConfig":
        return self


def test_submission_engine_default_device_requires_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from main_code_submission.engine import SubmissionEngine

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback은 허용하지 않습니다"):
        SubmissionEngine(_MinimalEngineConfig())  # type: ignore[arg-type]


def test_submission_engine_explicit_cpu_remains_available_for_tests() -> None:
    from main_code_submission.engine import SubmissionEngine

    engine = SubmissionEngine(_MinimalEngineConfig(), device="cpu")  # type: ignore[arg-type]
    assert engine.device == torch.device("cpu")


def test_enabled_rationale_requires_a_trained_adapter(tmp_path) -> None:
    """base CausalLM만 올려 학습되지 않은 근거를 정상 출력할 수 없다."""

    from main_code_submission.config import RationaleSpec, ScoreMember, SubmissionConfig

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    cfg = SubmissionConfig(
        name="no-untrained-rationale",
        score_members=(
            ScoreMember(
                name="a",
                checkpoint=checkpoint,
                backbone_key="ax4_light",
                parameters_billion=7.2596,
            ),
        ),
        rationale=RationaleSpec(base_model="stub", adapter=None, enabled=True),
    )
    with pytest.raises(ValueError, match="학습된 rationale adapter"):
        cfg.validate()


def _outputs(**rationales: str) -> dict[str, TraitOutput]:
    return {
        trait: TraitOutput(score=3.25, rationale=rationales.get(trait, f"{trait} 근거"))
        for trait in TRAITS
    }


# --- 요청 파싱 --------------------------------------------------------------
def test_official_request_roundtrip_preserves_bytes() -> None:
    content = build_official_user_content(PROMPT, ESSAY)
    parsed = parse_single_user_messages([{"role": "user", "content": content}])
    # 마지막 섹션인 essay는 뒤에 아무것도 없으므로 byte 그대로 복원된다.
    assert parsed.essay == ESSAY
    # prompt는 구분자(`\n\n[essay_text]\n`) **앞까지**라 개행이 남지 않는다.
    # 예전에는 `.rstrip("\n")`으로 비교해 두 동작을 모두 통과시켰고, 그 사이에
    # 여기서 byte 단위로 고정해 parser가 whitespace를 정규화하지 못하게 한다.
    assert parsed.prompt == PROMPT
    assert "\t" in parsed.essay
    # 왕복 재조립까지 확인한다.
    assert build_official_user_content(parsed.prompt, parsed.essay) == content


def test_extra_messages_use_the_last_user_message_instead_of_failing() -> None:
    """규정은 단일 user message지만, 거절하면 그 에세이가 0점이 된다.

    HTTP 400은 평가 서버에서 무응답과 같고, 무응답 한 행의 제곱오차(≈12)는 상수 점수를
    낸 행(≈0.43)의 28배다. 두 공식 마커가 있는 요청은 앞뒤에 무엇이 붙어도 모호하지
    않으므로 마지막 user message를 쓴다.
    """

    content = build_official_user_content(PROMPT, ESSAY)
    parsed = parse_single_user_messages(
        [{"role": "system", "content": "x"}, {"role": "user", "content": content}]
    )
    assert parsed.essay == ESSAY

    # user message가 하나도 없으면 여전히 거절한다.
    with pytest.raises(ValueError):
        parse_single_user_messages([{"role": "assistant", "content": content}])
    with pytest.raises(ValueError):
        parse_single_user_messages(
            [{"role": "system", "content": "x"}, {"role": "assistant", "content": content}]
        )


def test_list_of_content_parts_is_accepted() -> None:
    """공식 OpenAI 클라이언트가 보내는 typed part 배열도 받는다."""

    content = build_official_user_content(PROMPT, ESSAY)
    parsed = parse_single_user_messages(
        [{"role": "user", "content": [{"type": "text", "text": content}]}]
    )
    assert parsed.prompt == PROMPT
    assert parsed.essay == ESSAY


def test_missing_section_markers_fail_closed() -> None:
    with pytest.raises(ValueError):
        parse_official_user_content("채점 지시문만 있고 섹션 마커가 없다")


def test_instruction_text_containing_markers_uses_last_occurrence() -> None:
    """지시문 안에 마커 문자열이 예시로 들어 있어도 마지막 실제 섹션을 잡아야 한다."""

    instruction = "[prompt_text]\n예시 지시문입니다\n\n[essay_text]\n예시 본문입니다"
    content = f"{instruction}\n\n[prompt_text]\n{PROMPT}\n\n[essay_text]\n{ESSAY}"
    parsed = parse_official_user_content(content)
    assert parsed.essay == ESSAY
    assert parsed.prompt == PROMPT


# --- 출력 JSON --------------------------------------------------------------
def test_response_is_top_level_three_traits_without_wrapper() -> None:
    body = build_response_json(_outputs())
    parsed = json.loads(body)
    assert set(parsed) == set(TRAITS)
    assert "judge" not in parsed and "essay_id" not in parsed
    assert _parse_model_output(body) is not None


def test_official_parser_accepts_decimal_scores_unchanged() -> None:
    outputs = {
        trait: TraitOutput(score=value, rationale="근거")
        for trait, value in zip(TRAITS, (3.4889, 2.75, 4.0125), strict=True)
    }
    body = build_response_json(outputs)
    parsed = _parse_model_output(body)
    assert parsed is not None
    assert parsed["content"]["score"] == pytest.approx(3.4889)
    assert parsed["expression"]["score"] == pytest.approx(4.0125)


def test_official_evaluator_half_up_is_separate_from_parser() -> None:
    parsed = {
        trait: {"score": value, "rationale": "근거"}
        for trait, value in zip(TRAITS, (1.5, 2.49, 4.5), strict=True)
    }
    # 공지의 parser는 값을 바꾸지 않지만, 2026-08-06 이후 평가 단계는 영역별 사사오입한다.
    assert official_evaluator_scores(parsed) == {
        "content": 2,
        "organization": 2,
        "expression": 5,
    }
    assert official_evaluator_score(3.5) == 4
    with pytest.raises(ValueError):
        official_evaluator_score(5.1)


def test_integer_scores_are_serialized_as_json_integers() -> None:
    body = build_response_json(
        {trait: TraitOutput(score=3.0, rationale="근거") for trait in TRAITS}
    )
    assert all(isinstance(item["score"], int) for item in json.loads(body).values())


def test_braces_in_rationale_would_break_parser_and_are_removed() -> None:
    hostile = 'essay에 "{" 문자가 나온다'
    assert "{" in hostile
    cleaned = sanitize_rationale(hostile)
    assert "{" not in cleaned and "}" not in cleaned
    body = build_response_json(_outputs(content=hostile))
    assert _parse_model_output(body) is not None
    # 정제하지 않은 원문을 그대로 넣으면 실제로 파서가 깨지는지 확인한다.
    broken = json.dumps(
        {t: {"score": 3.0, "rationale": hostile} for t in TRAITS}, ensure_ascii=False
    )
    assert _parse_model_output(broken) is None


def test_stop_sequences_are_removed_from_rationale() -> None:
    body = build_response_json(_outputs(organization="Q: 문항을 인용했다"))
    assert contains_stop_sequence(body) == []
    for stop in OFFICIAL_STOP_SEQUENCES:
        assert stop not in body


def test_whitespace_is_preserved_for_grounded_quotes() -> None:
    original = "첫 줄\n둘째  줄\t탭\r\n끝"
    assert sanitize_rationale(original) == original
    body = build_response_json(
        {trait: TraitOutput(score=3, rationale=original) for trait in TRAITS}
    )
    parsed = _parse_model_output(body)
    assert parsed is not None
    assert parsed["content"]["rationale"] == original


def test_clamp_only_bounds_and_never_rounds() -> None:
    assert clamp_score(3.4889) == pytest.approx(3.4889)
    assert clamp_score(0.5) == 1.0
    assert clamp_score(6.0) == 5.0
    with pytest.raises(ValueError):
        clamp_score(float("nan"))


def test_missing_trait_fails_closed() -> None:
    outputs = _outputs()
    del outputs["expression"]
    with pytest.raises(ValueError):
        build_response_json(outputs)


def test_verify_official_parse_detects_score_drift() -> None:
    body = build_response_json(_outputs())
    good = verify_official_parse(body, {t: 3.25 for t in TRAITS})
    assert good["parse_ok"] and good["score_parity"] and good["has_rationale"]
    bad = verify_official_parse(body, {t: 4.0 for t in TRAITS})
    assert bad["parse_ok"] and not bad["score_parity"]


# --- 근거 프롬프트 ----------------------------------------------------------
def test_rationale_messages_are_the_training_messages() -> None:
    """serving 프롬프트가 학습 프롬프트와 문자 단위로 같아야 한다.

    근거 어댑터는 `main_code_relonation`의 fixed-score message 위에서 학습된다. 여기서 문구를
    새로 쓰면 어댑터가 학습 때 본 적 없는 입력을 받는다.
    """

    from main_code_relonation.prompts import (
        baseline_prompt_template,
        build_messages,
    )

    parsed = parse_official_user_content(build_official_user_content(PROMPT, ESSAY))
    scores = {"content": 3.4889, "organization": 2.75, "expression": 4.0125}
    prompt_template = baseline_prompt_template()
    messages = build_rationale_messages(
        parsed,
        scores,
        prompt_template=prompt_template,
    )
    assert messages == build_messages(
        PROMPT,
        parsed.essay,
        scores=scores,
        prompt_template=prompt_template,
    )
    assert [item["role"] for item in messages] == ["user"]
    content = messages[0]["content"]
    assert ESSAY in content
    # 조건 표기는 학습과 같은 소수 둘째 자리 양자화를 거친다. 어댑터는 인간 점수 격자(0.1 /
    # 0.25)로 학습되므로, serving에서만 `3.4889`처럼 네 자리를 보여 주면 학습 분포 밖이다.
    # **제출 점수는 head 값 그대로**이고 여기서 바뀌는 것은 프롬프트 표기뿐이다
    # (`rationales()`는 문자열만 돌려준다).
    assert "content: 3.49" in content
    assert "expression: 4.01" in content
    assert "3.4889" not in content


def test_v1_rationale_messages_are_the_training_messages() -> None:
    from main_code_relonation.prompts import (
        RATIONALE_PROMPT_V1_PATH,
        build_messages,
        load_prompt_template,
    )

    parsed = parse_official_user_content(build_official_user_content(PROMPT, ESSAY))
    scores = {"content": 4.0, "organization": 3.0, "expression": 3.0}
    template = load_prompt_template(RATIONALE_PROMPT_V1_PATH)
    messages = build_rationale_messages(parsed, scores, prompt_template=template)
    assert messages == build_messages(
        PROMPT,
        ESSAY,
        scores=scores,
        prompt_template=template,
    )
    content = messages[0]["content"]
    assert '"content":{"score":4,' in content
    assert all(
        sentinel not in content
        for sentinel in (
            "<<FIXED_SCORE_LINES>>",
            "<<OUTPUT_SKELETON>>",
            "<<PROMPT_TEXT>>",
            "<<ESSAY_TEXT>>",
        )
    )


def test_rationale_messages_reject_out_of_range_scores() -> None:
    parsed = parse_official_user_content(build_official_user_content(PROMPT, ESSAY))
    with pytest.raises(ValueError):
        build_rationale_messages(
            parsed, {"content": 9.0, "organization": 3.0, "expression": 3.0}
        )


# --- 빈 deterministic generation 복구 --------------------------------------
class _GenerationTokenizer:
    eos_token_id = 2
    pad_token_id = 2

    def apply_chat_template(self, *_: object, **__: object) -> list[int]:
        return [10, 11, 12]

    def decode(self, ids: object, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        values = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)  # type: ignore[arg-type]
        if values == [99]:
            return json.dumps(
                {trait: {"rationale": f"{trait} 재생성 근거"} for trait in TRAITS},
                ensure_ascii=False,
            )
        return ""  # EOS만 생성된 첫 시도


class _GenerationModel:
    def __init__(self, outputs: list[int]) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> torch.Tensor:
        self.calls.append(kwargs)
        token = self.outputs[len(self.calls) - 1]
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        suffix = torch.tensor([[token]], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat((input_ids, suffix), dim=1)


class _AdapterContext:
    def __init__(self, ensemble: "_AdapterEnsemble", name: str) -> None:
        self.ensemble = ensemble
        self.name = name
        self.previous = ensemble.active

    def __enter__(self) -> None:
        self.ensemble.active = self.name

    def __exit__(self, *_: object) -> None:
        self.ensemble.active = self.previous


class _AdapterEnsemble:
    def __init__(self) -> None:
        self.active = "score"

    @property
    def active_adapter(self) -> str | None:
        return self.active

    @property
    def adapter_names(self) -> list[str]:
        return ["score"]

    def using(self, name: str) -> _AdapterContext:
        return _AdapterContext(self, name)


def _generation_engine(model: _GenerationModel):
    from main_code_relonation.prompts import baseline_prompt_template
    from main_code_submission.engine import SubmissionEngine

    engine = object.__new__(SubmissionEngine)
    engine.config = SimpleNamespace(
        rationale=SimpleNamespace(
            enabled=True,
            max_length=64,
            max_new_tokens=16,
            chat_template_kwargs={},
            rationale_prompt_text=baseline_prompt_template(),
        )
    )
    engine.device = torch.device("cpu")
    engine._rationale_model = model
    engine._rationale_tokenizer = _GenerationTokenizer()
    engine._rationale_ensemble = _AdapterEnsemble()
    engine._rationale_adapter_name = "rationale"
    return engine


def test_empty_rationale_completion_retries_once_with_minimum_tokens() -> None:
    """EOS-only 첫 결과만 복구하고, context 종료 시 score adapter를 복원한다."""

    model = _GenerationModel([2, 99])
    engine = _generation_engine(model)
    text = parse_official_user_content(build_official_user_content(PROMPT, ESSAY))

    result = engine.rationales(text, {trait: 3.0 for trait in TRAITS})

    assert result == {trait: f"{trait} 재생성 근거" for trait in TRAITS}
    assert len(model.calls) == 2
    assert "min_new_tokens" not in model.calls[0]
    assert model.calls[1]["min_new_tokens"] == 8
    assert engine._rationale_ensemble.active == "score"


def test_nonempty_rationale_completion_keeps_the_single_original_call() -> None:
    """정상 399건 경로에는 재시도 인자를 추가하거나 두 번째 생성을 하지 않는다."""

    model = _GenerationModel([99])
    engine = _generation_engine(model)
    text = parse_official_user_content(build_official_user_content(PROMPT, ESSAY))

    result = engine.rationales(text, {trait: 3.0 for trait in TRAITS})

    assert result == {trait: f"{trait} 재생성 근거" for trait in TRAITS}
    assert len(model.calls) == 1
    assert "min_new_tokens" not in model.calls[0]
    assert engine._rationale_ensemble.active == "score"


def test_rationale_generation_uses_the_prompt_text_pinned_in_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Docker generation must use adapter-bound text, not today's source default."""

    import main_code_submission.engine as engine_module

    model = _GenerationModel([99])
    engine = _generation_engine(model)
    pinned = engine.config.rationale.rationale_prompt_text
    text = parse_official_user_content(build_official_user_content(PROMPT, ESSAY))
    captured: dict[str, object] = {}

    def _messages(
        request: object,
        scores: object,
        *,
        prompt_template: str | None = None,
        skeleton_hint: str | None = None,
    ) -> list[dict[str, str]]:
        captured["request"] = request
        captured["scores"] = scores
        captured["prompt_template"] = prompt_template
        captured["skeleton_hint"] = skeleton_hint
        return [{"role": "user", "content": "pinned"}]

    monkeypatch.setattr(engine_module, "build_rationale_messages", _messages)
    engine.rationales(text, {trait: 3.0 for trait in TRAITS})

    assert captured["prompt_template"] == pinned


def test_rationale_generation_exception_restores_score_adapter() -> None:
    """생성 자체가 실패해도 다음 채점을 rationale adapter로 오염시키지 않는다."""

    class _FailingGenerationModel:
        def generate(self, **_: object) -> torch.Tensor:
            raise RuntimeError("generation failed")

    engine = _generation_engine(_FailingGenerationModel())  # type: ignore[arg-type]
    text = parse_official_user_content(build_official_user_content(PROMPT, ESSAY))

    with pytest.raises(RuntimeError, match="generation failed"):
        engine.rationales(text, {trait: 3.0 for trait in TRAITS})
    assert engine._rationale_ensemble.active_adapter == "score"


def test_rationale_generation_requires_an_active_score_adapter() -> None:
    engine = _generation_engine(_GenerationModel([99]))
    engine._rationale_ensemble.active = None
    text = parse_official_user_content(build_official_user_content(PROMPT, ESSAY))

    with pytest.raises(RuntimeError, match="활성 score adapter가 없습니다"):
        engine.rationales(text, {trait: 3.0 for trait in TRAITS})


def test_unescaped_inner_quote_salvages_all_traits_and_official_json() -> None:
    """cu128 실측 오류는 세 trait 고정 shape일 때만 회수하고 최종 JSON은 다시 escape한다."""

    from main_code_submission.engine import parse_rationale_completion

    malformed = """{"content":{"score":2.0,"rationale":"주장은 로봇세 반대이나 근거가 제한적이다."},"organization":{"score":2.0,"rationale":"서론과 결론은 있으나 전환이 갑작스럽다."},"expression":{"score":2.0,"rationale":"'로봇공학이론중에 "불편한 골짜기"'와 같은 비문이 존재한다."}}"""

    rationales = parse_rationale_completion(malformed)

    assert set(rationales) == set(TRAITS)
    assert '"불편한 골짜기"' in rationales["expression"]
    body = build_response_json(
        {trait: TraitOutput(score=2, rationale=rationales[trait]) for trait in TRAITS}
    )
    parsed = _parse_model_output(body)
    assert parsed is not None
    assert {trait: parsed[trait]["score"] for trait in TRAITS} == {
        trait: 2 for trait in TRAITS
    }
    assert '"불편한 골짜기"' in parsed["expression"]["rationale"]


def test_rationale_template_placeholder_is_not_accepted_as_a_rationale() -> None:
    from main_code_submission.engine import parse_rationale_completion

    body = json.dumps(
        {
            trait: {
                "score": 3,
                "rationale": f"<{trait} 근거, 두 문장 이내 180자 이내>",
            }
            for trait in TRAITS
        },
        ensure_ascii=False,
    )
    assert parse_rationale_completion(body) == {}

    # placeholder를 뱉은 trait만 버리고 정상 trait은 살린다. 예전에는 셋 다 버렸고,
    # 그 결과 respond()가 예외를 올려 **확정된 점수까지** 함께 사라졌다.
    swapped = json.dumps(
        {
            "content": {
                "score": 3,
                "rationale": "<organization 근거, 두 문장 이내 180자 이내>",
            },
            "organization": {"score": 3, "rationale": "조직 실제 근거"},
            "expression": {"score": 3, "rationale": "표현 실제 근거"},
        },
        ensure_ascii=False,
    )
    assert parse_rationale_completion(swapped) == {
        "organization": "조직 실제 근거",
        "expression": "표현 실제 근거",
    }


def test_partial_trait_shape_keeps_what_parsed() -> None:
    """세 개를 못 채워도 얻은 만큼 돌려준다. 나머지는 respond()가 template으로 메운다."""

    from main_code_submission.engine import parse_rationale_completion

    partial = (
        '{"content":{"score":2,"rationale":"내용 근거"},'
        '"organization":{"score":2,"rationale":"조직 근거"}}'
    )
    assert parse_rationale_completion(partial) == {
        "content": "내용 근거",
        "organization": "조직 근거",
    }

    # JSON 자체가 없으면 여전히 빈 dict다.
    assert parse_rationale_completion("근거를 만들 수 없습니다") == {}


# --- 응답 길이 ---------------------------------------------------------------
def test_long_rationale_is_never_truncated() -> None:
    """운영측 상한은 2048 token이고 LLM Judge는 길이가 아니라 구체성을 본다.

    서버가 근거를 자르면 구체성만 잃는다. 길이는 진단만 하고 응답은 그대로 내보낸다.
    """

    long_text = "이 글은 주장을 분명히 제시한다. " * 40
    body = build_response_json(_outputs(content=long_text))
    parsed = _parse_model_output(body)
    assert parsed is not None
    assert parsed["content"]["rationale"] == sanitize_rationale(long_text)


def test_length_report_flags_runaway_generation() -> None:
    outputs = _outputs(content="가" * (RATIONALE_CHAR_LIMIT + 1))
    report = rationale_length_report(outputs)
    assert report["over_rationale_limit"] == ["content"]
    assert report["max_chars"] == RATIONALE_CHAR_LIMIT + 1
    assert rationale_length_report(_outputs())["over_rationale_limit"] == []


# --- chat template 검증 -----------------------------------------------------
class _StubTokenizer:
    def __init__(self, template: str | None) -> None:
        self.chat_template = template


def _verify(template: str | None, expected_hash: str = "") -> object:
    """모델 없이 `_verify_chat_template`만 호출한다."""

    from main_code_submission.config import RationaleSpec
    from main_code_submission.engine import SubmissionEngine

    engine = object.__new__(SubmissionEngine)
    spec = RationaleSpec(
        base_model="stub", adapter=None, chat_template_sha256=expected_hash
    )
    SubmissionEngine._verify_chat_template(engine, _StubTokenizer(template), spec)
    return engine


def test_chat_template_hash_matches_the_training_hash_function() -> None:
    """학습이 기록한 해시와 서빙이 계산하는 해시가 같은 함수여야 한다."""

    from main_code_relonation.artifacts import sha256_text

    template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    engine = _verify(template, sha256_text(template))
    assert engine._rationale_template_sha256 == sha256_text(template)


def test_chat_template_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="chat template이 학습 때와 다릅니다"):
        _verify("어떤 template", "deadbeef")


def test_missing_chat_template_fails_closed() -> None:
    with pytest.raises(ValueError, match="chat_template이 없습니다"):
        _verify(None)


# --- 공식 지표 --------------------------------------------------------------
def test_official_metric_helpers_match_manual_values() -> None:
    predicted = [3.0, 4.0, 2.0]
    gold = [3.0, 3.0, 2.0]
    assert rmse(predicted, gold) == pytest.approx((1 / 3) ** 0.5)
    assert spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_official_metric_handles_ties_with_average_ranks() -> None:
    # score.average는 소수 2자리라 동점이 생긴다. 평균 순위 처리가 필요하다.
    assert spearman([1.0, 1.0, 2.0], [1.0, 1.0, 2.0]) == pytest.approx(1.0)


# --- VRAM 회계 (어댑터 공유) -------------------------------------------------
def _member(name: str, backbone_key: str, billions: float):
    from main_code_submission.config import ScoreMember
    from pathlib import Path

    return ScoreMember(
        name=name,
        checkpoint=Path("/nonexistent"),
        backbone_key=backbone_key,
        parameters_billion=billions,
    )


def _config(*members):
    from main_code_submission.config import RationaleSpec, SubmissionConfig

    return SubmissionConfig(
        name="vram",
        score_members=tuple(members),
        rationale=RationaleSpec(base_model="stub", adapter=None, enabled=False),
    )


def test_submission_config_rejects_non_official_postprocess() -> None:
    """최종 Docker는 검증된 Y6 후처리를 다른 방식으로 바꾸지 않는다."""

    from dataclasses import replace

    cfg = replace(
        _config(_member("a", "ax4_light", 7.2596)),
        score_postprocess="per_trait_round",
    )
    with pytest.raises(ValueError, match="score_postprocess.*average_matched"):
        cfg.validate()


def test_manifest_must_declare_average_matched_at_top_level(tmp_path) -> None:
    """설명용 extra가 아니라 기계적으로 검증되는 top-level 계약이어야 한다."""

    from main_code_submission.config import load_manifest

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    manifest = {
        "name": "metric-contract",
        "root": str(tmp_path),
        "score_members": [
            {
                "name": "m0",
                "checkpoint": "checkpoint",
                "backbone_key": "m0",
                "parameters_billion": 1.0,
            }
        ],
        "rationale": {"base_model": "stub", "enabled": False},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="score_postprocess.*average_matched"):
        load_manifest(path)

    manifest["score_postprocess"] = "average_matched"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_manifest(path).score_postprocess == "average_matched"


def test_docker_staging_preserves_required_score_postprocess() -> None:
    """Host에서 검증한 계약이 staged container manifest에서 사라지면 안 된다."""

    from pathlib import Path

    source = (Path(__file__).parents[1] / "build_image.sh").read_text(encoding="utf-8")
    assert 'raw.get("score_postprocess") != "average_matched"' in source
    assert '"score_postprocess": raw["score_postprocess"]' in source


def test_release_manifest_and_dockerfile_use_one_fixed_release() -> None:
    """Historical Y6 계약과 공통 Docker build 경로를 고정한다.

    ``submission_assets``는 마지막으로 staging한 release를 가리키므로 여기에서
    historical Y6와 같다고 가정하지 않는다. 현재 staging 계약은 아래의 별도
    테스트가 검증한다.
    """

    from pathlib import Path

    root = Path(__file__).parents[2]
    manifest = json.loads(
        (root / "main_code_submission/manifests/Y6_matched_fallback.json").read_text(
            encoding="utf-8"
        )
    )
    extra = manifest["extra"]
    assert manifest["score_postprocess"] == "average_matched"
    offline = extra["offline_validation"]
    assert offline["average_matched"] == {
        "rmse": pytest.approx(0.42231208838961737),
        "spearman": pytest.approx(0.7500157380037686),
    }
    checkpoint_config = json.loads(
        (
            Path(extra["source_run"])
            / extra["checkpoint_selection"]["directory"]
            / "config.json"
        ).read_text(encoding="utf-8")
    )
    assert checkpoint_config["score_postprocess"] == "average_matched"
    assert extra["expected_image_tag"] == "y6-cu128-offline-r6-20260811"
    assert "runtime_base_image" not in extra
    assert "expected_weights_mode" not in extra

    source = (root / "main_code_submission/build_image.sh").read_text(encoding="utf-8")
    assert '"${TAG}" != "${EXPECTED_IMAGE_TAG}"' in source
    assert "WEIGHTS_MODE" not in source
    assert "BASE_IMAGE" not in source
    dockerfile = (root / "main_code_submission/Dockerfile").read_text(encoding="utf-8")
    assert "FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime" in dockerfile
    assert "HF_HOME=/opt/submission/hf" in dockerfile
    assert "HF_HUB_CACHE=/opt/submission/hf/hub" in dockerfile
    assert "/root/.cache/huggingface" not in dockerfile
    assert "ARG BASE_IMAGE" not in dockerfile


def test_current_staged_manifest_uses_the_submitted_8seed_qwen_release() -> None:
    """Mutable staging은 2026-08-25 확정 제출본(qwen35 8seed + v4 Qwen rationale) 계약이다.

    후보 이름과 prompt 판본은 release마다 바뀌므로 상수로 박지 않는다. 대신 (1) 지표
    표면처럼 규정과 직결되는 것, (2) 이 release의 정체성(멤버 구성), (3) release가
    바뀌어도 항상 지켜져야 하는 불변식(선언 SHA == 실제 원문)을 검사한다.
    """

    from pathlib import Path

    root = Path(__file__).parents[2]
    manifest_path = root / "submission_assets/submission_manifest.json"
    if not manifest_path.exists():
        pytest.skip("submission_assets staging이 없습니다 (build 전이거나 fresh clone)")
    staged = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert staged["score_postprocess"] == "average_matched"
    assert staged["essay_surface"] == "official_raw"
    assert staged["integer_total_offset"] == 0

    members = staged["score_members"]
    assert len(members) == 8
    assert sorted(m["name"] for m in members) == [
        f"qwen35_9b_bbq35_e5_s{seed}_plateau" for seed in range(42, 50)
    ]
    for member in members:
        assert member["backbone_key"] == "qwen35_9b"
        assert member["weight"] == pytest.approx(0.125)
        assert member["parameters_billion"] == pytest.approx(9.6531)
        assert member["checkpoint"] == f"checkpoints/{member['name']}"

    extra = staged["extra"]
    # tag는 release마다 새로 발급되지만 형식과 immutability는 지켜져야 한다.
    assert re.fullmatch(
        r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", extra["expected_image_tag"]
    )
    assert re.fullmatch(r"[0-9a-f]{64}", extra["expected_http_prediction_sha256"])
    assert extra["rationale_selection"]["teacher_model"] == (
        "google/gemma-4-26B-A4B-it"
    )
    # 학습 행 수는 teacher QC 통과 수라 판본마다 다르다. 상수로 박으면 거짓
    # provenance를 통과시킨다. 양수인지만 본다.
    assert extra["rationale_selection"]["teacher_selected_rows"] > 0

    rationale = staged["rationale"]
    assert rationale["base_model"] == "Qwen/Qwen3.5-9B"
    assert rationale["adapter"] == "rationale_adapter"
    assert rationale["share_backbone_key"] == "qwen35_9b"
    assert rationale["load_in_4bit"] is False
    assert rationale["enabled"] is True
    # manifest가 적은 SHA와 실제 prompt 원문은 항상 일치해야 한다. 어긋나면 서빙이
    # 학습과 다른 prompt를 쓴다.
    assert re.fullmatch(r"[0-9a-f]{64}", rationale["rationale_prompt_sha256"])
    assert (
        hashlib.sha256(
            rationale["rationale_prompt_text"].encode("utf-8")
        ).hexdigest()
        == rationale["rationale_prompt_sha256"]
    )


def test_shared_backbone_counted_once() -> None:
    """같은 backbone_key 구성원은 backbone 한 벌만 센다.

    `engine.load()`가 `load_shared_backbone_checkpoints`로 실제로 한 벌만 올리기 때문이다.
    점수 동등성은 `main_code/tests/test_shared_backbone_ensemble.py`가 보장한다.
    """

    three_adapters = _config(
        _member("a", "tri_7b", 7.5269),
        _member("b", "tri_7b", 7.5269),
        _member("c", "tri_7b", 7.5269),
    )
    sizes = three_adapters.unique_backbones()
    assert len(sizes) == 1
    assert three_adapters.estimated_weights_gib() == pytest.approx(
        7.5269 * 2.0, abs=1e-6
    )
    assert three_adapters.vram_budget_report()["fits"]


def test_distinct_backbones_are_counted_separately() -> None:
    two = _config(_member("a", "ax4_light", 7.2596), _member("b", "kanana", 8.0303))
    assert len(two.unique_backbones()) == 2
    assert two.estimated_weights_gib() == pytest.approx(
        (7.2596 + 8.0303) * 2.0, abs=1e-6
    )


# --- 고정 Y6 입력 표면 ------------------------------------------------------
@pytest.mark.parametrize(
    "surface",
    ["official_gap_newline", "official_raw_kiwi_sentence_newline_v1", "canonical"],
)
def test_non_y6_surface_is_rejected(surface: str) -> None:
    """최종 image는 Y6의 official_raw 이외 연구 표면을 열지 않는다."""

    from dataclasses import replace

    cfg = replace(_config(_member("a", "ax4_light", 7.2596)), essay_surface=surface)
    with pytest.raises(ValueError, match="essay_surface"):
        cfg.validate()


def test_partial_ensemble_load_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """구성원 하나가 실패한 앙상블로 서빙하지 않는다.

    예전에는 남은 구성원으로 계속 서빙했다. 그러면 가중 평균의 분모가 달라져 검증한 앙상블과
    다른 점수가 400편에 실린다. 한 편의 근거 실패와 달리 이것은 **로딩 문제**이므로 기동을
    실패시킨다.
    """

    from main_code_submission.engine import SubmissionEngine

    def make_engine() -> SubmissionEngine:
        engine = SubmissionEngine.__new__(SubmissionEngine)
        engine.config = SimpleNamespace(
            score_members=[
                SimpleNamespace(backbone_key="ax4_light", name="ok"),
                SimpleNamespace(backbone_key="tri_7b", name="broken"),
            ],
            rationale=SimpleNamespace(enabled=False, share_backbone_key=None),
        )
        engine._members = []
        engine._ensembles = []
        engine._ensembles_by_backbone = {}
        engine._rationale_model = None
        engine._rationale_tokenizer = None
        engine._rationale_ensemble = None
        engine._rationale_adapter_name = None
        engine._loaded = False
        return engine

    good = SimpleNamespace(member=SimpleNamespace(backbone_key="ax4_light"))

    def _load_member(member: object) -> object:
        if getattr(member, "backbone_key", None) == "tri_7b":
            raise RuntimeError("CUDA out of memory")
        return good

    engine = make_engine()
    monkeypatch.setattr(engine, "_load_member", _load_member)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        SubmissionEngine.load(engine)
    assert engine._loaded is False


def test_shared_rationale_failure_fails_closed_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """공유 근거 로드 실패는 환경변수와 무관하게 기동을 막는다."""

    from main_code_submission.engine import SubmissionEngine

    # 실제 checkpoint 없이 load() 분기만 본다. `_config`는 존재하지 않는 경로를 쓰므로
    # validate()를 태우면 FileNotFoundError로 이 분기에 도달하지 못한다.
    def make_engine() -> SubmissionEngine:
        engine = SubmissionEngine.__new__(SubmissionEngine)
        engine.config = SimpleNamespace(
            score_members=[SimpleNamespace(backbone_key="ax4_light")],
            rationale=SimpleNamespace(enabled=True, share_backbone_key="ax4_light"),
        )
        engine._members = []
        engine._ensembles = []
        engine._ensembles_by_backbone = {}
        engine._rationale_model = object()
        engine._rationale_tokenizer = object()
        engine._rationale_ensemble = object()
        engine._rationale_adapter_name = "rationale"
        engine._loaded = False
        return engine

    loaded_member = SimpleNamespace(member=SimpleNamespace(backbone_key="ax4_light"))

    # rationale이 같은 backbone을 공유하므로 load()는 공유 group 경로를 탄다.
    def _fake_load_shared_group(group: object, **kwargs: object) -> list[object]:
        return [loaded_member]

    def _boom() -> None:
        raise RuntimeError("adapter key mismatch")

    engine = make_engine()
    monkeypatch.setattr(engine, "_load_shared_group", _fake_load_shared_group)
    monkeypatch.setattr(engine, "_load_rationale", _boom)
    with pytest.raises(RuntimeError, match="adapter key mismatch"):
        SubmissionEngine.load(engine)
    assert engine._loaded is False
