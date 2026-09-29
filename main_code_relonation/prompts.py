from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping

from . import TRAITS
from .schema import conditioning_scores, format_score


PROMPT_DIRECTORY = Path(__file__).with_name("prompts")
BASELINE_PROMPT_PATH = PROMPT_DIRECTORY / "baseline_prompt.txt"
# 2026-08-25 최종 제출본이 쓴 근거 프롬프트. baseline_*는 legacy adapter 호환 판정에
# 쓰이므로 건드리지 않고 제출본 상수를 따로 둔다.
SUBMITTED_PROMPT_PATH = PROMPT_DIRECTORY / "rationale_prompt_v4.txt"
RATIONALE_PROMPT_V1_PATH = PROMPT_DIRECTORY / "rationale_prompt_v1.txt"
DEFAULT_TRAINING_PROMPT_PATH = RATIONALE_PROMPT_V1_PATH
PROMPT_SENTINELS = (
    "<<FIXED_SCORE_LINES>>",
    "<<OUTPUT_SKELETON>>",
    "<<PROMPT_TEXT>>",
    "<<ESSAY_TEXT>>",
)
_PROMPT_SENTINEL_PATTERN = re.compile(
    "|".join(re.escape(value) for value in PROMPT_SENTINELS)
)

#: `<<OUTPUT_SKELETON>>`의 rationale 자리표시자 안에 들어가는 길이 지시.
#:
#: 스켈레톤은 프롬프트 **맨 끝**, 즉 생성 직전에 붙으므로 본문의 길이 규칙보다 강하게
#: 작동한다. v3는 본문에서 200~350자를 요구했는데 배포 실측이 148.8자였고, 이 문구가
#: 그 차이를 설명한다. 따라서 프롬프트 판본이 다른 길이를 요구하면 이 힌트도 함께
#: 바꿔야 한다. 기본값은 역사적 문자열 그대로이므로 v1/v2/v3 recipe의 렌더 결과와
#: `main_code_submission`의 기존 manifest 동작은 bit 단위로 보존된다.
DEFAULT_SKELETON_HINT = "두 문장 이내 180자 이내"
# 제출본 스켈레톤 지시. v4 프롬프트 본문이 요구하는 길이와 짝이다.
SUBMITTED_SKELETON_HINT = "6~9문장 450~540 tokens"


def skeleton_placeholder(trait: str, hint: str = DEFAULT_SKELETON_HINT) -> str:
    """스켈레톤 자리표시자 원문. 제출 engine의 placeholder 탐지와 공유한다."""

    return f"<{trait} 근거, {hint}>"


def prompt_template_sha256(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def _read_prompt_file(path: str | Path) -> str:
    # Repository text files end in one newline. It is not part of the model prompt.
    return Path(path).read_text(encoding="utf-8").rstrip("\n")


def validate_prompt_template(template: str) -> str:
    if not template:
        raise ValueError("rationale prompt template이 비었습니다")
    counts = {sentinel: template.count(sentinel) for sentinel in PROMPT_SENTINELS}
    bad = {key: value for key, value in counts.items() if value != 1}
    if bad:
        raise ValueError(
            f"rationale prompt sentinel은 각각 정확히 한 번 필요합니다: {bad}"
        )
    return template


def load_prompt_template(path: str | Path) -> str:
    return validate_prompt_template(_read_prompt_file(path))


def baseline_prompt_template() -> str:
    return load_prompt_template(BASELINE_PROMPT_PATH)


def submitted_prompt_template() -> str:
    """2026-08-25 제출본(v4)의 근거 프롬프트 원문."""

    return load_prompt_template(SUBMITTED_PROMPT_PATH)


def training_prompt_template() -> str:
    return load_prompt_template(DEFAULT_TRAINING_PROMPT_PATH)


def prompt_template_id(template: str) -> str:
    digest = prompt_template_sha256(template)
    if template == baseline_prompt_template():
        return "baseline_prompt"
    if template == training_prompt_template():
        return "rationale_prompt_v1"
    return f"custom_{digest[:12]}"


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


JOINT_SCORE_RULE = """[평균 점수 출력 보충 규칙]
- 학습 정답은 복수 인간 평가자의 평균이므로 소수일 수 있다.
- score는 1~5 범위의 숫자로 출력하고, 필요한 경우 소수값을 사용하라.
- 이 보충 규칙은 위의 일반적인 정수 출력 문구보다 우선한다."""


DETAILED_RUBRIC = """[세부 채점 준거]
content는 다음 다섯 관점을 종합한다.
- C1 문제 상황 제시: 논제를 공론화할 배경·필요와 관련 정보를 적절하고 충분히 제시하는가.
- C2 주장: 주장이 논제에 부합하며 글 전체에서 일관되고 뚜렷한가.
- C3 이유·근거의 적절성: 주장과 이유·근거의 추론 연결이 타당하고 설득력 있는가.
- C4 이유·근거의 충분성: 적절한 하위주장의 수와 각 뒷받침의 깊이가 충분한가.
- C5 다른 입장 고려: 반대·대안 입장을 다루고 비교·논박하여 깊게 전개하는가.

organization은 다음 두 관점을 종합한다.
- O1 글 전체 조직: 서론·본론·결론의 역할 구분과 정보 배열이 유기적인가.
- O2 의미 단위 내 조직: 각 문단 또는 의미 단위가 완결성·통일성·문장 간 일관성을 갖추는가.

expression은 다음 두 관점을 종합한다.
- E1 문장과 어휘: 문장이 자연스럽고 명료·효과적이며 어휘가 문맥에 적절한가.
- E2 어문 규범과 관습: 맞춤법·띄어쓰기·오탈자와 문어체·종결어미 일관성을 지키는가.

한 결함을 관련 없는 영역에 중복 반영하지 말고, 세부 준거 전체를 기계적으로 나열하지 말며
해당 점수를 가장 잘 설명하는 실제 강점과 약점을 골라 rationale에 적어라."""


RATIONALE_RULES = """[근거 생성 추가 원칙]
- 각 영역의 고정 점수를 변경하지 말고 그 점수에 부합하는 근거를 작성하라.
- 고정 점수는 소수일 수 있다. 위의 일반적인 정수 출력 규칙보다 고정값 복사 지시가 우선한다.
- 각 rationale은 두 문장 이내, 180자 이내로 작성하라.
- content는 실제 주장, 이유·근거와 논리 연결을 구체적으로 짚어라.
- organization은 실제 도입·논거 배열·전환·마무리 기능과 담화 표지를 짚어라.
- 원문의 줄바꿈과 문단 경계는 입력에서 유실될 수 있으므로 줄바꿈 부재나 보이는 문단 수를 근거로 감점하지 마라.
- expression에서 오류나 어색한 표현을 지적하면 essay_text에 실제로 존재하는 문자열만 언급하라.
- 에세이에 없는 통계, 사례, 표현, 맞춤법 오류를 만들지 마라.
- 세 영역의 기준을 서로 섞거나 상투적인 총평을 반복하지 마라."""


def official_user_message(prompt: str, essay: str) -> str:
    return (
        f"{OFFICIAL_INSTRUCTION}\n\n{JOINT_SCORE_RULE}\n\n"
        f"[prompt_text]\n{prompt}\n\n[essay_text]\n{essay}"
    )


def fixed_score_user_message(
    prompt: str,
    essay: str,
    scores: Mapping[str, Any],
    *,
    prompt_template: str | None = None,
    skeleton_hint: str = DEFAULT_SKELETON_HINT,
) -> str:
    """고정 점수를 조건으로 근거만 생성시키는 메시지.

    `OFFICIAL_INSTRUCTION`은 "모든 점수는 1~5 정수"를 요구하고 출력 예시도 정수다. 그 지시를
    여기서 **명시적으로 무효화하지 않으면** teacher가 3.4를 3으로 반올림한다. 산문 지시만으로는
    약해서, 실제 숫자가 박힌 출력 스켈레톤을 프롬프트 끝에 붙여 복사 대상을 못 박는다.
    """
    fixed = conditioning_scores(scores)
    score_lines = "\n".join(
        f"- {trait}: {format_score(fixed[trait])}" for trait in TRAITS
    )
    skeleton = (
        "{"
        + ",".join(
            f'"{trait}":{{"score":{format_score(fixed[trait])},'
            f'"rationale":"{skeleton_placeholder(trait, skeleton_hint)}"}}'
            for trait in TRAITS
        )
        + "}"
    )
    template = validate_prompt_template(
        prompt_template if prompt_template is not None else baseline_prompt_template()
    )
    replacements = {
        "<<FIXED_SCORE_LINES>>": score_lines,
        "<<OUTPUT_SKELETON>>": skeleton,
        "<<PROMPT_TEXT>>": prompt,
        "<<ESSAY_TEXT>>": essay,
    }
    return _PROMPT_SENTINEL_PATTERN.sub(
        lambda match: replacements[match.group(0)], template
    )


def build_messages(
    prompt: str,
    essay: str,
    *,
    scores: Mapping[str, Any] | None,
    prompt_template: str | None = None,
    skeleton_hint: str = DEFAULT_SKELETON_HINT,
) -> list[dict[str, str]]:
    if scores is None and prompt_template is not None:
        raise ValueError("custom rationale prompt는 fixed score mode에서만 사용합니다")
    if scores is None and skeleton_hint != DEFAULT_SKELETON_HINT:
        raise ValueError("skeleton_hint는 fixed score mode에서만 사용합니다")
    content = (
        official_user_message(prompt, essay)
        if scores is None
        else fixed_score_user_message(
            prompt,
            essay,
            scores,
            prompt_template=prompt_template,
            skeleton_hint=skeleton_hint,
        )
    )
    # The organizer's 2026-07-20 notice says the evaluator sends exactly one
    # user role containing instruction + prompt_text + essay_text.
    return [{"role": "user", "content": content}]
