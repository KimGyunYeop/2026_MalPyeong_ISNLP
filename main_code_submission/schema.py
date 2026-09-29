"""공식 파서 재현과 안전한 출력 JSON 조립.

`_extract_first_json` / `_parse_model_output`은 2026-07-15 공지에 실린 운영측 코드를 그대로
옮긴 것이다. 우리 응답이 그 함수를 통과하는지 로컬에서 먼저 검사하기 위해 존재하며,
동작을 개선하거나 관대하게 만들지 않는다.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Optional

TRAITS = ("content", "organization", "expression")

# 평가 서버가 넘기는 stop sequence. rationale에 들어가면 생성이 끊겨 JSON이 깨진다.
OFFICIAL_STOP_SEQUENCES = ("Q:", "User:")

# 출력 길이 상한은 운영측 FAQ가 명시했다: "모델 출력은 최대 2048 token까지 생성됩니다".
# Docker 규정 §5의 `max_tokens: 512`는 오케스트레이터가 넘기는 값이지만 FAQ의 2048이 실제
# 상한이므로 이를 기준으로 둔다. 그리고 LLM Judge는 길이가 아니라 구체성·정합성을 본다
# ("단순히 길다고 유리한 것은 아닙니다"). 따라서 근거를 **자르지 않는다.** 자르면 오히려
# 구체성을 잃고, 문장 중간에서 끊기면 품질만 떨어진다.
#
# 아래 두 값은 진단 전용이다. 넘으면 harness가 보고하지만 응답을 변형하지 않는다.
RATIONALE_CHAR_LIMIT = 600
# 2048 token을 한국어 기준으로 보수적으로 환산한 진단 임계다(1 token당 약 1.5자).
RESPONSE_CHAR_BUDGET = 3000

# 코드가 아니라 **글자 그대로** 박아 둔 마지막 바닥. 어떤 함수도 부르지 않으므로
# 실패할 수 있는 부분이 없다. 테스트가 이 문자열이 공식 파서를 통과하는지 검사한다.
ABSOLUTE_FALLBACK_BODY = (
    '{"content": {"score": 3, "rationale": "본문 전반의 내용 전개를 근거로 중간 수준으로 판단하였다."}, '
    '"organization": {"score": 3, "rationale": "글 전체 구조와 문단 연결을 근거로 중간 수준으로 판단하였다."}, '
    '"expression": {"score": 3, "rationale": "문장과 어휘 사용을 근거로 중간 수준으로 판단하였다."}}'
)


# --- 운영측 공지 코드 (수정 금지) -------------------------------------------------
def _extract_first_json(text: str) -> Optional[str]:
    """중괄호 균형을 맞춰 첫 번째 JSON 객체 문자열만 추출."""

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_model_output(raw: str) -> Optional[dict[str, Any]]:
    """모델 출력 JSON 파싱. 실패 시 None 반환."""

    text = re.sub(r"```(?:json)?", "", raw.strip()).replace("```", "").strip()
    json_str = _extract_first_json(text)
    if json_str is None:
        return None
    try:
        parsed = json.loads(json_str)
        required = {"content", "organization", "expression"}
        if required.issubset(parsed.keys()):
            if all(
                isinstance(parsed[d], dict) and "score" in parsed[d] for d in required
            ):
                return parsed
    except (json.JSONDecodeError, Exception):
        pass
    return None


# --- 우리 출력 쪽 -----------------------------------------------------------------
def strip_unencodable(text: str) -> str:
    """UTF-8로 직렬화할 수 없는 문자를 지운다.

    tokenizer가 깨진 바이트열을 lone surrogate(U+D800~U+DFFF)로 디코드하면
    ``json.dumps``와 공식 파서는 통과하지만, 응답을 UTF-8로 인코딩하는 순간
    ``UnicodeEncodeError``가 나서 그 요청이 HTTP 500이 된다. 500은 그 에세이
    0점이고(제곱오차 12.16 대 0.41), 이 실패는 응답 조립이 전부 끝난 **뒤**에
    ASGI 계층에서 터지므로 아래의 어떤 강등 사다리도 잡지 못한다.

    현재 Qwen2Tokenizer는 깨진 바이트를 U+FFFD로 바꾸므로 실제로는 도달하지
    않는다(랜덤 토큰 4,000회 디코드에서 0건). 그러나 그건 방어가 아니라 우연이고,
    tokenizer를 바꾸면 되살아나는 구멍이라 출력 경계에서 닫는다.
    """

    if not isinstance(text, str):
        text = str(text)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return text


def sanitize_rationale(text: str) -> str:
    """공식 파서와 stop sequence를 깨뜨리는 문자만 바꾼다.

    - `{`/`}`: 공식 파서는 문자열 내부를 구분하지 못하는 depth 카운터라 중괄호 한 글자가
      균형 계산을 깨뜨린다. 대응하는 전각 괄호로 바꿔 의미를 보존한다.
    - `Q:` / `User:`: 평가 서버 stop sequence다. 걸리면 생성이 중단되어 JSON이 잘린다.

    공백·탭·개행과 다른 제어문자는 ``json.dumps``가 안전하게 escape한다. 이를 접거나
    삭제하면 essay의 정확한 연속 부분문자열이던 인용이 제출 응답에서 달라져 LLM Judge의
    groundedness를 해칠 수 있으므로 그대로 보존한다.
    """

    if not isinstance(text, str):
        text = str(text)
    cleaned = strip_unencodable(text).replace("{", "｛").replace("}", "｝")
    # 공식 파서는 `json.loads` 전에 백틱과 코드펜스를 **삭제**한다. 파싱은 살아남지만
    # 평가자가 읽는 근거 문구가 우리가 의도한 것과 달라진다. 인용이 원문과 어긋나면
    # LLM Judge의 groundedness가 깎이므로, 삭제당하는 대신 우리가 치환한다.
    cleaned = cleaned.replace("`", "'")
    for stop in OFFICIAL_STOP_SEQUENCES:
        # 콜론 뒤에 폭 없는 구분을 넣지 않고 문자 자체를 바꿔 stop 매칭을 피한다.
        cleaned = cleaned.replace(stop, stop[:-1] + "：")
    return cleaned.strip()


def rationale_length_report(outputs: dict[str, "TraitOutput"]) -> dict[str, Any]:
    """근거 길이를 **보고만** 한다. 자르지 않는다.

    운영측 FAQ가 상한을 2048 token으로 밝혔고, LLM Judge는 길이가 아니라 구체성·정합성·
    essay 충실도를 본다. 따라서 서버가 근거를 잘라 구체성을 깎을 이유가 없다. 대신 비정상적으로
    긴 출력(생성 폭주, 반복 루프)은 진단해야 하므로 harness가 이 값을 gate로 본다.
    """

    lengths = {trait: len(outputs[trait].rationale) for trait in TRAITS}
    return {
        "lengths": lengths,
        "max_chars": max(lengths.values(), default=0),
        "over_rationale_limit": [
            trait for trait, size in lengths.items() if size > RATIONALE_CHAR_LIMIT
        ],
    }


def clamp_score(value: float) -> float:
    """서버 응답을 만들 때 1~5 범위만 보장한다.

    실제 제출 engine은 이 함수 전에 ``average_matched``를 적용한다. 이 낮은 수준의
    serializer는 parser 회귀검사에서 실수 입력도 다룰 수 있도록 범위만 검사한다.
    """

    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"score가 유한한 수가 아닙니다: {value!r}")
    return min(5.0, max(1.0, number))


def official_evaluator_score(value: Any) -> int:
    """2026-08-06 공지의 모델 예측 점수 사사오입을 재현한다.

    공식 parser 공지 코드는 ``score`` 필드의 존재만 확인하고 값을 바꾸지 않는다. 이후
    평가 단계에서 모델의 영역별 실수 점수를 사사오입한다. 점수 계약 자체가 1~5이므로
    범위 밖 값이나 비유한 값은 로컬 검증에서 즉시 실패시킨다. 양수 범위의 사사오입은
    ``floor(x + 0.5)``이며, Python ``round``의 banker rounding을 쓰지 않는다.
    """

    number = float(value)
    if not math.isfinite(number) or not 1.0 <= number <= 5.0:
        raise ValueError(f"공식 평가 점수는 1~5의 유한한 수여야 합니다: {value!r}")
    return int(math.floor(number + 0.5))


def official_evaluator_scores(parsed: dict[str, Any]) -> dict[str, int]:
    """공식 parser가 반환한 세 영역 점수에 평가기의 사사오입을 적용한다."""

    return {trait: official_evaluator_score(parsed[trait]["score"]) for trait in TRAITS}


@dataclass(frozen=True)
class TraitOutput:
    score: float
    rationale: str


def build_response_json(outputs: dict[str, TraitOutput]) -> str:
    """서버가 그대로 반환할 최상위 3-trait JSON 문자열을 만든다.

    `essay_id`나 `judge` wrapper를 쓰지 않는다. 공식 파서는 응답의 첫 balanced JSON에서
    최상위 key를 읽으므로 wrapper가 있으면 즉시 파싱 실패한다.
    """

    missing = [trait for trait in TRAITS if trait not in outputs]
    if missing:
        raise ValueError(f"출력에 빠진 trait이 있습니다: {missing}")
    payload: dict[str, dict[str, Any]] = {}
    for trait in TRAITS:
        score = clamp_score(outputs[trait].score)
        # JSON에는 3.0보다 3을 써서 과제기술서의 "1~5 정수" 계약을 문자 표면에서도
        # 만족시킨다. 진짜 소수 후보는 그대로 보존하므로 parser 단위 테스트도 가능하다.
        serialized_score: int | float = int(score) if score.is_integer() else score
        payload[trait] = {
            "score": serialized_score,
            "rationale": sanitize_rationale(outputs[trait].rationale),
        }
    # ensure_ascii=False로 한국어를 그대로 두고, 코드블록이나 앞말 없이 JSON만 낸다.
    return json.dumps(payload, ensure_ascii=False)


def verify_official_parse(
    raw: str, expected_scores: dict[str, float], *, tolerance: float = 1e-9
) -> dict[str, Any]:
    """우리 응답이 공식 파서를 통과하고 점수가 보존되는지 검사한다."""

    parsed = _parse_model_output(raw)
    if parsed is None:
        return {"parse_ok": False, "score_parity": False, "reason": "공식 파서 실패"}
    drift: dict[str, float] = {}
    for trait in TRAITS:
        got = float(parsed[trait]["score"])
        want = clamp_score(expected_scores[trait])
        if abs(got - want) > tolerance:
            drift[trait] = got - want
    has_rationale = all(
        isinstance(parsed[trait].get("rationale"), str)
        and parsed[trait]["rationale"].strip()
        for trait in TRAITS
    )
    return {
        "parse_ok": True,
        "score_parity": not drift,
        "score_drift": drift,
        "has_rationale": has_rationale,
        "reason": None if not drift else "score가 보존되지 않았습니다",
    }


# 공식 파서를 깨뜨릴 수 있는 문자만 골라낸다. `_extract_first_json`은 문자열 내부를
# 구분하지 못하는 depth 카운터라 중괄호 한 글자가 균형 계산을 무너뜨린다. 나머지는
# `json.dumps`가 escape하므로 건드리지 않는다(인용을 원문 그대로 보존해야 한다).
_PARSER_BREAKING = {"{": "｛", "}": "｝"}
# 마지막 단계에서 쓰는 근거. 상수라 어떤 입력에도 파싱이 보장된다.
FALLBACK_RATIONALE = "본문에서 확인한 근거에 따라 해당 영역 점수를 부여하였다."
_MINIMAL_RATIONALE = FALLBACK_RATIONALE


def _strict_sanitize(text: str) -> str:
    """공격적 정제. 파서를 깨뜨릴 수 있는 것과 제어문자를 모두 없앤다."""

    if not isinstance(text, str):
        text = str(text)
    text = strip_unencodable(text)
    for bad, good in _PARSER_BREAKING.items():
        text = text.replace(bad, good)
    for stop in OFFICIAL_STOP_SEQUENCES:
        text = text.replace(stop, stop[:-1] + "：")
    text = text.replace("`", "'")
    text = "".join(" " if ord(ch) < 0x20 else ch for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def _text_only(text: str, *, limit: int = RATIONALE_CHAR_LIMIT) -> str:
    """구조 문자를 전부 버리고 읽을 수 있는 본문만 남긴다.

    상수 근거로 덮어쓰기 **직전** 단계다. Judge는 "generic한 총평, 상투적 표현,
    템플릿형 설명은 낮게 평가하라"고 명시하므로, 내용을 통째로 버리는 것은 마지막
    수단이어야 한다. 한글·영숫자·기본 문장부호만 남기면 파서를 깨뜨릴 문자가 남지
    않으면서도 구체적인 서술은 보존된다.
    """

    if not isinstance(text, str):
        text = str(text)
    kept = [
        ch if (ch.isalnum() or ch.isspace() or ch in ".,;:!?()·~-'“”‘’") else " "
        for ch in text
    ]
    cleaned = re.sub(r"\s+", " ", "".join(kept)).strip()
    return cleaned[:limit].strip()


def enforce_official_parse(
    outputs: dict[str, "TraitOutput"],
) -> tuple[str, dict[str, Any]]:
    """공식 파서를 **직접 돌려** 통과하는 응답만 내보낸다.

    운영측 2026-08-19 답변: 파싱 불가한 출력은 2회 재호출 후에도 실패하면 그 샘플을
    제외하지 않고 **0점**으로 집계한다. official train gold 기준 한 행을 0점으로 잃는
    비용은 제곱오차 12.16이고 상수 삼중으로 찍는 비용은 0.41이다. 30배 차이다.
    그래서 여기서는 "잘 만들어 보내고 로그를 남긴다"가 아니라 **나갈 문자열을 공식
    파서에 먹여 보고 통과할 때만 반환한다.**

    사다리는 아래로 갈수록 정보를 잃지만 점수는 절대 잃지 않는다. 마지막 단계는 상수
    근거를 쓰므로 입력과 무관하게 파싱이 보장된다.
    """

    def _rewrite(transform: Any) -> dict[str, TraitOutput]:
        return {
            trait: TraitOutput(
                score=outputs[trait].score,
                rationale=transform(outputs[trait].rationale) or _MINIMAL_RATIONALE,
            )
            for trait in TRAITS
        }

    # 아래로 갈수록 근거의 형태만 깎고 **내용은 최대한 늦게** 버린다.
    # 점수는 어느 단계에서도 그대로다.
    stages: list[tuple[str, dict[str, TraitOutput]]] = [
        ("as_generated", outputs),
        # 파서를 깨뜨리는 문자만 치환한다. 내용은 그대로 남는다.
        ("strict_sanitized", _rewrite(_strict_sanitize)),
        # 구조 문자를 전부 버리고 읽을 수 있는 본문만 남긴다. 여전히 원문 기반이다.
        ("text_only", _rewrite(_text_only)),
        # 여기서부터 내용을 포기한다. Judge가 템플릿형을 감점하므로 마지막 수단이다.
        ("minimal_rationale", _rewrite(lambda _: _MINIMAL_RATIONALE)),
    ]
    expected = {trait: outputs[trait].score for trait in TRAITS}
    attempts: list[dict[str, Any]] = []
    for name, candidate in stages:
        try:
            body = build_response_json(candidate)
        except Exception as error:  # noqa: BLE001 - 응답을 죽이지 않는 것이 목적이다
            attempts.append({"stage": name, "parse_ok": False, "reason": repr(error)})
            continue
        report = verify_official_parse(body, expected)
        # 파싱만으로는 부족하다. 공식 파서의 전처리(백틱·코드펜스 삭제)를 거친 뒤에도
        # 근거 문구가 우리가 보낸 것과 같아야 평가자가 읽는 내용이 우리 의도와 일치한다.
        intended = {
            trait: sanitize_rationale(candidate[trait].rationale) for trait in TRAITS
        }
        parsed = _parse_model_output(body)
        text_parity = parsed is not None and all(
            parsed[trait].get("rationale") == intended[trait] for trait in TRAITS
        )
        # 파서를 통과해도 UTF-8로 못 내보내면 ASGI 계층에서 500이 된다. 그 실패는
        # 이 사다리 **밖**에서 터지므로 여기서 미리 확인해야 한다.
        try:
            body.encode("utf-8")
            encodable = True
        except UnicodeEncodeError:
            encodable = False
        report = {**report, "text_parity": text_parity, "encodable": encodable}
        attempts.append({"stage": name, **report})
        if (
            report["parse_ok"]
            and report["score_parity"]
            and report["has_rationale"]
            and text_parity
            and encodable
        ):
            # 공식 파서가 **실제로 읽어낸 객체**를 다시 직렬화해서 내보낸다.
            # 이렇게 하면 우리가 보내는 바이트열이 곧 파서의 출력이므로, 중괄호 균형
            # 계산이나 백틱 전처리에서 해석이 갈릴 여지가 남지 않는다. 정규화 결과가
            # 같은 객체로 다시 읽히는지(고정점)까지 확인하고, 아니면 원문을 보낸다.
            normalized = json.dumps(parsed, ensure_ascii=False)
            reparsed = _parse_model_output(normalized)
            if reparsed != parsed:
                # 검증에 실패한 문자열을 그냥 내보내지 않는다. 다음 단계로 내려가
                # 더 단순한 근거로 다시 만든다. 이 사다리의 전제는 "확인된 것만
                # 내보낸다"이고, 여기서 예외를 두면 그 전제가 무너진다.
                attempts[-1] = {
                    **attempts[-1],
                    "normalized": False,
                    "reason": "정규화 결과가 같은 객체로 다시 읽히지 않습니다",
                }
                continue
            return normalized, {
                "final_stage": name,
                "repaired": name != "as_generated",
                "normalized": True,
                "attempts": attempts,
            }
    # 여기까지 왔다는 것은 점수 자체가 직렬화 불가라는 뜻이다. 점수를 버리는 대신
    # 중립 정수로 대체해서라도 응답을 낸다. 0점 처리(제곱오차 12.16)보다 항상 낫다.
    fallback = {
        trait: TraitOutput(score=3.0, rationale=_MINIMAL_RATIONALE) for trait in TRAITS
    }
    body = build_response_json(fallback)
    report = verify_official_parse(body, {trait: 3.0 for trait in TRAITS})
    attempts.append({"stage": "neutral_constant", **report})
    if not (report["parse_ok"] and report["score_parity"]):
        # 여기까지 실패하는 것은 파이썬 표준 라이브러리가 깨진 수준이다. 그래도
        # 예외를 올려 응답을 잃는 대신, 호출자가 자기 바닥으로 내려가게 둔다.
        raise AssertionError("상수 응답조차 공식 파서를 통과하지 못했습니다")
    return body, {
        "final_stage": "neutral_constant",
        "repaired": True,
        "normalized": False,
        "attempts": attempts,
    }


def contains_stop_sequence(raw: str) -> list[str]:
    """응답 전체에서 평가 서버 stop sequence를 찾는다."""

    return [stop for stop in OFFICIAL_STOP_SEQUENCES if stop in raw]
