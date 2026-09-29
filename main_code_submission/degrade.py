"""제출 중 어떤 실패가 나도 **점수를 버리지 않기 위한** 최후 수단.

평가 서버는 한 요청이 실패하면 그 에세이를 0점으로 집계한다. 우리 validation gold 기준
0점 한 행의 제곱오차는 평균 11.97이고, 상수 점수를 낸 행은 0.43이다. 즉 **한 행을 포기하는
비용이 그 행을 대충 찍는 비용의 28배**다. 400편 기준 3편만 0점이 되어도 mean-first RMSE는
0.4168 -> 0.5118로 무너진다(2026-08-15 리더보드 역산에서 실제로 관측된 크기다).

그래서 이 파일은 다음 순서를 강제한다.

1. 점수가 계산됐으면 근거가 어떻게 되든 **그 점수를 반드시 응답에 싣는다.**
2. 점수 계산까지 실패했으면 0점 대신 학습 분포의 중앙 상수 삼중을 낸다.
3. 어떤 경우에도 공식 파서가 읽을 수 있는 3-trait JSON을 200으로 돌려준다.

이 정책은 "실패를 숨기지 말라"는 개발 원칙과 의도적으로 다르다. 개발 중에는 조용한
fallback이 버그를 감추지만, 평가 서버 위에서는 디버깅할 수단이 없고 대안이 확정 0점이다.
대신 모든 강등은 `degradation` 진단과 서버 로그에 남겨 사후에 전부 셀 수 있게 한다.
"""

from __future__ import annotations

import re

from .schema import TRAITS

#: 점수 계산 자체가 실패했을 때 쓰는 상수 삼중.
#:
#: 공식 지표가 세 영역 **평균** 하나만 보므로 최적 상수는 gold 평균에 가장 가까운 평균을
#: 만드는 삼중이다. official train 2,000편의 영역별 평균은 content 3.278 / organization
#: 3.337 / expression 3.673이고, 각 영역을 사사오입한 (3, 3, 4)의 평균 3.333은 gold average
#: 평균 3.429와 가장 가깝다. 이 상수의 mean-first RMSE는 train 0.6368 / validation 0.6565로,
#: 같은 행을 0점으로 버릴 때의 3.4865 / 3.4600 대비 5.3배 낫다.
#:
#: 영역별로도 각 영역의 사사오입 중앙값이라, 만에 하나 지표가 영역별로 바뀌어도 합리적이다.
NEUTRAL_SUBMITTED_SCORES: dict[str, float] = {
    "content": 3.0,
    "organization": 3.0,
    "expression": 4.0,
}

_TRAIT_KO = {
    "content": "내용",
    "organization": "조직",
    "expression": "표현",
}

# 점수대별 한 문장. 근거 모델이 죽었을 때만 쓰이므로 essay를 인용하지 않는다. 인용하지
# 못하는 상황에서 인용한 척하면 LLM Judge의 groundedness에서 오히려 더 깎인다. 대신
# 응답에 실리는 정수 점수와 모순되지 않는 평가 언어만 쓴다.
_LEVEL_KO = {
    "content": {
        1: "주장과 근거의 연결이 거의 드러나지 않아 논지를 따라가기 어렵다",
        2: "주장은 제시되지만 이를 뒷받침하는 근거가 부족하고 논의가 피상적이다",
        3: "주장과 근거가 대체로 갖추어져 있으나 논거의 구체성과 깊이가 제한적이다",
        4: "주장이 분명하고 근거도 대체로 타당하여 논지가 설득력 있게 전개된다",
        5: "주장이 명확하고 근거가 구체적이며 반론까지 고려해 논의가 충실하다",
    },
    "organization": {
        1: "글 전체의 짜임이 잡히지 않아 논의의 흐름을 파악하기 어렵다",
        2: "도입과 마무리의 역할이 불분명하고 내용 사이의 연결이 매끄럽지 않다",
        3: "전체 구성은 갖추었으나 부분 사이의 연결과 비중 배분이 고르지 않다",
        4: "논의의 순서가 자연스럽고 각 부분이 전체 주제에 맞게 배치되어 있다",
        5: "도입에서 마무리까지 논의가 유기적으로 이어지고 구성이 매우 안정적이다",
    },
    "expression": {
        1: "문장 오류가 잦아 의미 전달 자체가 어려운 부분이 많다",
        2: "어휘와 문장 구조가 단조롭고 문법·표기 오류가 눈에 띈다",
        3: "의미 전달에는 무리가 없으나 표현이 다소 단조롭고 부분적인 오류가 있다",
        4: "어휘 선택이 적절하고 문장이 대체로 정확하여 읽기에 무리가 없다",
        5: "어휘가 정확하고 문장 구조가 다양하여 논의를 효과적으로 전달한다",
    },
}


#: template 근거를 **응답만 보고** 알아보는 정규식.
#:
#: 로컬 harness(`code_for_docker_check_otherserv/evaluate_http.py`)는 표준 라이브러리만 쓰는
#: self-contained 스크립트라 이 모듈을 import할 수 없다. 그래서 같은 문자열을 그쪽에도 복사해
#: 두고, `tests/test_no_score_loss.py`가 두 리터럴이 갈라지지 않았는지 매번 확인한다.
#:
#: 이 정규식에 걸리는 응답이 하나라도 있으면 그 release는 배포 금지다. 강등 자체는 평가
#: 서버에서 점수를 지키려고 넣은 것이지만, **로컬에서 강등이 나왔다면 그건 고칠 수 있는
#: 버그**이므로 그대로 내보내면 안 된다.
TEMPLATE_RATIONALE_PATTERN = r"^(내용|조직|표현) 영역은 [1-5]점 수준으로, .+\.$"


def template_rationale(trait: str, score: float) -> str:
    """확정 점수와 모순되지 않는 결정적 근거 문장.

    근거 생성이 실패한 trait에만 쓴다. 이 문장이 LLM Judge에서 낮은 점수를 받는 것은
    당연하지만, Judge 가중치는 10%이고 근거 하나를 비우면 응답 전체가 무효가 되어 그
    에세이의 RMSE·Spearman 기여까지 통째로 사라진다.
    """

    if trait not in TRAITS:
        raise ValueError(f"알 수 없는 trait: {trait!r}")
    level = min(5, max(1, int(round(float(score)))))
    return f"{_TRAIT_KO[trait]} 영역은 {level}점 수준으로, {_LEVEL_KO[trait][level]}."


# 완성문이 JSON 구조를 갖고 있는지 보는 최소 표지. 구조가 있으면 그 안의 값은
# 이미 회수 경로들이 trait별로 꺼내 갔고, 남은 trait에는 줄 것이 없다.
_JSON_SHAPE_RE = re.compile(r'[{}]|"\s*[a-zA-Z_]+\s*"\s*:')


def prose_fallback_rationale(raw_completion: str) -> str:
    """원문이 **산문**일 때만 근거로 되살린다.

    모델이 JSON 대신 그냥 설명을 써 버린 경우가 이 경로의 대상이다. 그때 원문은
    이 글에 대한 실제 서술이므로 template보다 낫다(Judge: "generic한 총평,
    상투적 표현, 템플릿형 설명은 낮게 평가하라").

    반대로 원문이 깨진 JSON이면 쓰지 않는다. 구조 문자를 벗겨 내면 세 영역의 값이
    한 문자열로 뒤섞여, 그걸 한 trait의 근거로 넣는 순간 Judge의 ``domain_match``가
    "다른 영역 기준이 섞여 있음"으로 감점된다. 그 경우에는 template이 정직하다.
    """

    if not isinstance(raw_completion, str) or not raw_completion.strip():
        return ""
    if _JSON_SHAPE_RE.search(raw_completion):
        return ""
    from .schema import _text_only

    return _text_only(raw_completion)


def fill_missing_rationales(
    rationales: dict[str, str],
    scores: dict[str, float],
    *,
    raw_completion: str = "",
) -> tuple[dict[str, str], list[str]]:
    """비어 있는 trait만 template으로 채우고 채운 목록을 함께 돌려준다.

    부분 성공을 살리는 것이 핵심이다. 세 근거 중 둘이 정상인데 하나가 비었다고 응답
    전체를 버리면 그 에세이의 점수까지 잃는다.
    """

    filled: dict[str, str] = {}
    substituted: list[str] = []
    for trait in TRAITS:
        value = rationales.get(trait)
        if isinstance(value, str) and value.strip():
            filled[trait] = value.strip()
        else:
            # template보다 **모델이 실제로 쓴 문장**이 낫다. Judge는 "generic한 총평,
            # 상투적 표현, 템플릿형 설명은 낮게 평가하라"고 명시한다. 구조 회수에
            # 전부 실패해도 원문에는 이 글에 대한 서술이 남아 있다.
            salvaged = prose_fallback_rationale(raw_completion)
            filled[trait] = salvaged or template_rationale(trait, scores[trait])
            substituted.append(trait)
    return filled, substituted


def last_resort_outputs() -> tuple[dict[str, float], dict[str, str]]:
    """점수 경로까지 죽었을 때 쓰는 (점수, 근거) 한 쌍."""

    scores = dict(NEUTRAL_SUBMITTED_SCORES)
    rationales = {trait: template_rationale(trait, scores[trait]) for trait in TRAITS}
    return scores, rationales
