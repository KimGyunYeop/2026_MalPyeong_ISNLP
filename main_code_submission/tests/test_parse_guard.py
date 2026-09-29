"""나가는 응답이 공식 파서를 반드시 통과한다는 계약.

운영측 2026-08-19 답변: 파싱 불가한 출력은 2회 재호출 후에도 실패하면 그 샘플을
제외하지 않고 **0점**으로 집계한다. official train gold 기준 한 행을 0점으로 잃는
비용은 제곱오차 12.16, 상수 삼중으로 찍는 비용은 0.41이다.

따라서 이 테스트는 "보통 입력에서 잘 된다"가 아니라 **적대적 입력에서도 반드시
파싱되고 점수가 보존된다**를 검사한다. 파서는 공지에 실린 원본 코드를 그대로 쓴다.
"""

from __future__ import annotations

import json

import pytest

from main_code_submission.schema import (
    TRAITS,
    TraitOutput,
    _parse_model_output,
    enforce_official_parse,
    official_evaluator_scores,
)


ADVERSARIAL = {
    "중괄호": '이 글은 {서론}과 {결론}이 분리되어 있다.',
    "중괄호 불균형": '문장에 } 만 하나 있다.',
    "코드블록": "```json\n{\"content\": 1}\n```  라고 썼다.",
    "stop sequence": "Q: 이 주장은 타당한가? User: 그렇다.",
    "쌍따옴표": '필자는 "로봇세"라는 표현을 반복한다.',
    "역슬래시": "경로 C:\\Users\\test 를 언급한다.",
    "제어문자": "첫 줄이다.\n\t둘째 줄이다.\x00\x07",
    "JSON 통째로": '{"content": {"score": 5, "rationale": "가짜"}}',
    "중첩 중괄호": "{{{{{{{{{{",
    "빈 문자열": "",
    "공백만": "   \n\t  ",
    "매우 긴 근거": "가" * 5000,
    "이모지": "구성이 좋다 🎯🔥",
    "역방향 중괄호": "}{}{}{",
    "널 문자만": "\x00",
}


def _outputs(rationale: str, scores=(4.0, 3.0, 5.0)) -> dict[str, TraitOutput]:
    return {
        trait: TraitOutput(score=score, rationale=rationale)
        for trait, score in zip(TRAITS, scores, strict=True)
    }


@pytest.mark.parametrize("name", sorted(ADVERSARIAL))
def test_adversarial_rationale_still_parses_with_scores_intact(name: str) -> None:
    body, guard = enforce_official_parse(_outputs(ADVERSARIAL[name]))
    parsed = _parse_model_output(body)
    assert parsed is not None, (name, body[:200])
    # 공식 평가자가 읽는 정수 점수가 우리가 정한 값 그대로여야 한다.
    assert official_evaluator_scores(parsed) == {
        "content": 4,
        "organization": 3,
        "expression": 5,
    }
    for trait in TRAITS:
        assert isinstance(parsed[trait]["rationale"], str)
        assert parsed[trait]["rationale"].strip()
    assert guard["final_stage"] in {
        "as_generated",
        "strict_sanitized",
        "minimal_rationale",
        "neutral_constant",
    }


def test_clean_rationale_is_returned_untouched() -> None:
    clean = "필자는 서빙 로봇 사례를 들어 주장을 뒷받침한다."
    body, guard = enforce_official_parse(_outputs(clean))
    assert guard["repaired"] is False
    assert guard["final_stage"] == "as_generated"
    assert json.loads(body)["content"]["rationale"] == clean


def test_braces_are_replaced_not_dropped() -> None:
    body, _ = enforce_official_parse(_outputs("구조가 {서론}으로 시작한다."))
    text = json.loads(body)["content"]["rationale"]
    # 의미를 보존해야 한다. 지우면 근거가 어색해지고 Judge의 specificity가 깎인다.
    assert "｛서론｝" in text
    assert "{" not in text and "}" not in text


def test_empty_rationale_falls_back_but_keeps_scores() -> None:
    body, guard = enforce_official_parse(_outputs(""))
    parsed = _parse_model_output(body)
    assert guard["repaired"] is True
    assert official_evaluator_scores(parsed) == {
        "content": 4,
        "organization": 3,
        "expression": 5,
    }


def test_every_attempt_is_recorded_for_diagnosis() -> None:
    _, guard = enforce_official_parse(_outputs(""))
    stages = [item["stage"] for item in guard["attempts"]]
    # 어느 단계에서 왜 떨어졌는지 사후에 알 수 있어야 한다.
    assert stages[0] == "as_generated"
    assert len(stages) >= 2


def test_scores_are_never_lost_to_an_exception() -> None:
    # clamp_score가 예외를 던지는 값이라도 응답은 나가야 한다.
    broken = {
        trait: TraitOutput(score=float("nan"), rationale="정상 근거다.")
        for trait in TRAITS
    }
    body, guard = enforce_official_parse(broken)
    parsed = _parse_model_output(body)
    assert parsed is not None
    assert guard["final_stage"] == "neutral_constant"
    assert official_evaluator_scores(parsed) == {
        "content": 3,
        "organization": 3,
        "expression": 3,
    }


def test_guard_output_survives_the_published_preprocessing() -> None:
    """공식 파서는 백틱을 먼저 제거한다. 그 후에도 JSON이 살아 있어야 한다."""

    body, guard = enforce_official_parse(_outputs("```json 이라고 적혀 있다."))
    assert _parse_model_output(body) is not None
    # 백틱이 남아 있으면 공식 파서가 그것을 지우고, 평가자가 읽는 근거가 우리가 보낸
    # 것과 달라진다. 인용이 어긋나면 Judge의 groundedness가 깎인다.
    assert "```" not in body
    parsed = _parse_model_output(body)
    assert parsed["content"]["rationale"] == json.loads(body)["content"]["rationale"]
    assert guard["repaired"] is False   # 일반 정제로 이미 해결된다


# ── 근거 회수 사다리 ──────────────────────────────────────────────────────
from main_code_submission.engine import parse_rationale_completion  # noqa: E402

_GOOD = (
    '{"content":{"score":4,"rationale":"내용 근거다."},'
    '"organization":{"score":3,"rationale":"구성 근거다."},'
    '"expression":{"score":5,"rationale":"표현 근거다."}}'
)


def test_intact_json_recovers_all_three() -> None:
    assert set(parse_rationale_completion(_GOOD)) == set(TRAITS)


def test_unescaped_quote_is_salvaged() -> None:
    broken = _GOOD.replace("내용 근거다.", '필자는 "로봇세"를 말한다.')
    assert set(parse_rationale_completion(broken)) == set(TRAITS)


def test_brace_in_rationale_breaks_structure_but_loose_scan_recovers() -> None:
    """`_extract_first_json`은 문자열 내부 중괄호를 구분하지 못한다.

    예전에는 여기서 조기 반환해 회수 경로에 도달하지 못했다.
    """

    broken = _GOOD.replace("내용 근거다.", "구조가 {서론}이다.")[:-1]
    assert set(parse_rationale_completion(broken)) == set(TRAITS)


def test_missing_opening_brace_still_recovers() -> None:
    assert set(parse_rationale_completion(_GOOD[1:])) == set(TRAITS)


def test_partial_output_recovers_what_survived() -> None:
    truncated = '{"content":{"score":4,"rationale":"내용 근거다."},"organiz'
    assert set(parse_rationale_completion(truncated)) == {"content"}


def test_hopeless_output_returns_empty_without_raising() -> None:
    for raw in ("", "   ", "아무 구조 없는 문장이다.", "{{{{{{"):
        assert parse_rationale_completion(raw) == {}


def test_loose_scan_never_supplies_scores() -> None:
    """점수는 score head 값만 쓴다. 생성 JSON의 숫자를 신뢰하면 안 된다."""

    lying = _GOOD.replace('"score":4', '"score":1')[:-1]
    recovered = parse_rationale_completion(lying)
    assert all(isinstance(value, str) for value in recovered.values())


# ── 정규화 ────────────────────────────────────────────────────────────────
def test_response_is_reserialized_from_what_the_official_parser_read() -> None:
    body, guard = enforce_official_parse(_outputs("서빙 로봇 사례를 든다."))
    assert guard["normalized"] is True
    # 보내는 바이트열이 곧 공식 파서의 출력이라 해석이 갈릴 여지가 없다.
    assert json.loads(body) == _parse_model_output(body)


# ── 요청 파싱 실패도 0점이다 ───────────────────────────────────────────────
from main_code_submission.request_parser import (  # noqa: E402
    SCORING_MIN_CHARS,
    salvage_scoring_text,
)


def test_essay_length_request_without_markers_is_scored_not_rejected() -> None:
    """마커를 못 읽었다고 400을 내면 그 에세이는 0점으로 집계된다.

    운영측 2026-08-19 답변 5번. 본문으로 보이는 길이면 잘못 읽더라도 채점한다.
    """

    salvaged = salvage_scoring_text([{"role": "user", "content": "가" * 400}])
    assert salvaged is not None and len(salvaged.essay) == 400
    # 논제를 모르므로 비운다. 본문을 논제로 잘못 넣는 것보다 덜 왜곡한다.
    assert salvaged.prompt == ""


def test_short_generic_smoke_is_not_hijacked_into_scoring() -> None:
    """Docker 규정 §10의 일반 API smoke는 대화 응답을 받아야 한다."""

    assert salvage_scoring_text([{"role": "user", "content": "안녕하세요"}]) is None
    assert salvage_scoring_text([]) is None
    assert salvage_scoring_text(None) is None


def test_only_user_turns_are_salvaged() -> None:
    assert salvage_scoring_text([{"role": "assistant", "content": "가" * 400}]) is None
    merged = salvage_scoring_text(
        [{"role": "user", "content": "가" * 200}, {"role": "user", "content": "나" * 200}]
    )
    assert merged is not None and len(merged.essay) == 401  # 개행 하나로 이어 붙인다


def test_threshold_sits_between_smoke_and_essay_length() -> None:
    # 논증적 글은 규격상 1,000자 내외(±200자)다.
    assert 50 < SCORING_MIN_CHARS < 800


# ── 원문 대체는 산문일 때만 ────────────────────────────────────────────────
from main_code_submission.degrade import (  # noqa: E402
    prose_fallback_rationale,
    template_rationale,
)


def test_prose_completion_is_preferred_over_a_template() -> None:
    """모델이 JSON 대신 설명을 써 버린 경우, 그 설명이 template보다 낫다."""

    prose = "서론에서 문제 상황을 제시했고 본론의 근거가 구체적이다."
    assert prose_fallback_rationale(prose) == prose


def test_broken_json_is_not_smeared_into_one_traits_rationale() -> None:
    """깨진 JSON을 벗겨 넣으면 세 영역 값이 뒤섞인다.

    Judge의 domain_match는 "다른 영역 기준이 꽤 섞여 있음"을 2점 이하로 본다.
    그 경우에는 template이 정직하다.
    """

    debris = '{"content": {"rationale": "내용 근거"}, "organization": {"rationale": "조직 근거"}'
    assert prose_fallback_rationale(debris) == ""


def test_empty_or_whitespace_completion_yields_nothing() -> None:
    for raw in ("", "   \n\t", None):
        assert prose_fallback_rationale(raw) == ""


def test_fill_uses_prose_then_template() -> None:
    from main_code_submission.degrade import fill_missing_rationales

    scores = {trait: 3.0 for trait in TRAITS}
    prose = "이 글은 서론과 결론이 분명하고 근거가 구체적이다."
    filled, substituted = fill_missing_rationales({}, scores, raw_completion=prose)
    assert substituted == list(TRAITS)
    assert all(value == prose for value in filled.values())

    filled, _ = fill_missing_rationales({}, scores, raw_completion='{"content": 1}')
    assert filled["expression"] == template_rationale("expression", 3.0)


def test_absolute_fallback_string_parses_without_calling_anything() -> None:
    """예외 처리기가 부르는 마지막 바닥. 함수 호출이 없으므로 실패할 수 없어야 한다."""

    from main_code_submission.schema import ABSOLUTE_FALLBACK_BODY

    parsed = _parse_model_output(ABSOLUTE_FALLBACK_BODY)
    assert parsed is not None
    assert official_evaluator_scores(parsed) == {
        "content": 3,
        "organization": 3,
        "expression": 3,
    }
    for trait in TRAITS:
        assert parsed[trait]["rationale"].strip()
