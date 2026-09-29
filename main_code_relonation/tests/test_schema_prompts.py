from __future__ import annotations

import json

import pytest

from main_code_relonation.prompts import (
    PROMPT_SENTINELS,
    RATIONALE_PROMPT_V1_PATH,
    build_messages,
    load_prompt_template,
)
from main_code_relonation.schema import (
    canonicalize_with_fixed_scores,
    compact_judge_json,
    parse_generated_judge,
    rationale_audit,
)


def generated_json() -> str:
    return json.dumps(
        {
            "content": {"score": 1, "rationale": "주장과 이유를 구체적으로 제시한다."},
            "organization": {
                "score": 2,
                "rationale": "도입 뒤 논거와 결론이 이어진다.",
            },
            "expression": {"score": 3, "rationale": "문장이 대체로 자연스럽다."},
        },
        ensure_ascii=False,
    )


def test_single_user_message_preserves_official_surface_and_fixed_scores() -> None:
    prompt = "논제  문장"
    essay = "첫 문장  둘째 문장\n원문 줄바꿈"
    scores = {"content": 3.4000000000000004, "organization": 4.0, "expression": 2.75}
    messages = build_messages(prompt, essay, scores=scores)
    assert [message["role"] for message in messages] == ["user"]
    content = messages[0]["content"]
    assert content.endswith(essay)
    assert prompt in content
    # 소수 둘째 자리로 양자화해 표기한다. content 0.1 격자 / O·E 0.25 격자라 무손실이다.
    assert "content: 3.4" in content
    assert "organization: 4\n" in content
    assert "expression: 2.75" in content
    # 부동소수 잡음이 프롬프트에 새면 teacher가 그 숫자를 정수로 "정리"해 버린다.
    assert "3.4000000000000004" not in content


def test_fixed_score_prompt_overrides_the_integer_rule_with_a_skeleton() -> None:
    """정수 출력 규칙을 무효화하고 복사할 JSON을 그대로 보여줘야 한다.

    2026-08-06에 24편 전량이 실패했다. `OFFICIAL_INSTRUCTION`이 "모든 점수는 1~5 정수"를
    요구하고 예시도 정수였기 때문에, 산문으로 "그대로 복사하라"고만 해서는 teacher가 3.4를
    3으로 반올림했다.
    """

    scores = {"content": 3.4, "organization": 3.75, "expression": 4.0}
    content = build_messages("논제", "본문", scores=scores)[0]["content"]
    assert "'모든 점수는 1~5 정수'는 이 작업에 적용되지 않는다" in content
    assert '"content":{"score":3.4,' in content
    assert '"organization":{"score":3.75,' in content
    assert '"expression":{"score":4,' in content
    # 스켈레톤은 essay 직전, 즉 정수 규칙보다 뒤에 와야 한다.
    assert content.index("[출력 스켈레톤]") > content.index("모든 점수는 1~5 정수")


def test_generated_scores_are_ignored_when_fixed_scores_exist() -> None:
    parsed = parse_generated_judge(
        "설명 앞부분\n```json\n" + generated_json() + "\n```"
    )
    fixed = {"content": 3.125, "organization": 4.0, "expression": 2.75}
    judge = canonicalize_with_fixed_scores(parsed, fixed)
    assert [judge[trait]["score"] for trait in judge] == [3.125, 4.0, 2.75]
    assert list(judge) == ["content", "organization", "expression"]
    compact = compact_judge_json(judge)
    assert compact.index('"content"') < compact.index('"organization"')
    assert compact.index('"organization"') < compact.index('"expression"')


def test_integer_fixed_score_target_matches_v1_skeleton_lexically() -> None:
    template = load_prompt_template(RATIONALE_PROMPT_V1_PATH)
    messages = build_messages(
        "논제",
        "본문",
        scores={"content": 4.0, "organization": 3.0, "expression": 3.0},
        prompt_template=template,
    )
    assert '"content":{"score":4,' in messages[0]["content"]
    target = compact_judge_json(
        {
            trait: {"score": score, "rationale": f"{trait} 실제 근거"}
            for trait, score in zip(
                ("content", "organization", "expression"),
                (4.0, 3.0, 3.0),
                strict=True,
            )
        }
    )
    assert '"content":{"score":4,' in target
    assert '"score":4.0' not in target


def test_v1_sentinels_are_replaced_once_without_recursive_input_replacement() -> None:
    template = load_prompt_template(RATIONALE_PROMPT_V1_PATH)
    assert {sentinel: template.count(sentinel) for sentinel in PROMPT_SENTINELS} == {
        sentinel: 1 for sentinel in PROMPT_SENTINELS
    }
    ordinary = build_messages(
        "논제",
        "본문",
        scores={"content": 4, "organization": 3, "expression": 3},
        prompt_template=template,
    )[0]["content"]
    assert all(sentinel not in ordinary for sentinel in PROMPT_SENTINELS)

    prompt = f"논제 자료 {PROMPT_SENTINELS[3]}"
    essay = f"본문 자료 {PROMPT_SENTINELS[2]}"
    collision = build_messages(
        prompt,
        essay,
        scores={"content": 4, "organization": 3, "expression": 3},
        prompt_template=template,
    )[0]["content"]
    assert prompt in collision and essay in collision


def test_rationale_placeholder_echo_fails_quality_audit() -> None:
    judge = json.loads(generated_json())
    judge["content"]["rationale"] = "<content 근거, 두 문장 이내 180자 이내>"
    assert rationale_audit(judge, "본문")["template_placeholder_echo"] is True

    judge["content"]["rationale"] = "<organization 근거, 두 문장 이내 180자 이내>"
    judge["organization"]["rationale"] = "조직 실제 근거"
    assert rationale_audit(judge, "본문")["template_placeholder_echo"] is True


def test_public_parser_contract_uses_first_balanced_object() -> None:
    with pytest.raises(ValueError, match="content 객체"):
        parse_generated_judge('{"thought":"wrong first object"}\n' + generated_json())


def test_fixed_shape_recovers_only_unescaped_rationale_quotes() -> None:
    raw = (
        '{"content":{"score":4,"rationale":"본문의 "잊힐 권리"를 구체적으로 다룬다."},'
        '"organization":{"score":3,"rationale":"주장과 이유가 이어진다."},'
        '"expression":{"score":3,"rationale":""죽고 싶다."라는 표현을 사용한다."}}'
    )
    parsed = parse_generated_judge(raw)
    assert parsed["content"]["score"] == 4
    assert parsed["content"]["rationale"] == '본문의 "잊힐 권리"를 구체적으로 다룬다.'
    assert parsed["expression"]["rationale"] == '"죽고 싶다."라는 표현을 사용한다.'


def test_malformed_shape_is_not_broadly_repaired() -> None:
    raw = (
        '{"content":{"score":4,"rationale":"본문의 "표현"을 짚는다."},'
        '"expression":{"score":3,"rationale":"표현 근거"},'
        '"organization":{"score":3,"rationale":"조직 근거"}}'
    )
    with pytest.raises(ValueError, match="파싱할 수 없습니다"):
        parse_generated_judge(raw)


def test_rationale_brace_is_rejected_before_submission() -> None:
    value = json.loads(generated_json())
    value["content"]["rationale"] = "주장에 {근거}가 있다."
    parsed = parse_generated_judge(json.dumps(value, ensure_ascii=False))
    with pytest.raises(ValueError, match="중괄호"):
        canonicalize_with_fixed_scores(
            parsed, {"content": 3.0, "organization": 3.0, "expression": 3.0}
        )


def test_quoted_expression_error_must_exist_in_official_essay_surface() -> None:
    judge = json.loads(generated_json())
    judge["expression"]["rationale"] = "본문의 ‘없는 오타’가 어색하다."
    audit = rationale_audit(judge, "본문에는 실제 표현만 있다.")
    assert audit["all_quoted_spans_grounded"] is False
    assert audit["missing_quoted_spans"]["expression"] == ["없는 오타"]
