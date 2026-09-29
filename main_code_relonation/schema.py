from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping

from . import TRAITS


def essay_id(row: Mapping[str, Any]) -> str:
    value = row.get("essay_id", row.get("id"))
    if not isinstance(value, str) or not value:
        raise ValueError("essay_id/id가 필요합니다")
    return value


def prompt_text(row: Mapping[str, Any]) -> str:
    value = row.get("prompt_text", row.get("prompt", row.get("question")))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"prompt_text가 필요합니다: {essay_id(row)}")
    return value


def official_raw_essay(row: Mapping[str, Any]) -> str:
    surfaces = row.get("essay_surfaces")
    if isinstance(surfaces, Mapping) and "official_raw" in surfaces:
        value = surfaces["official_raw"]
    elif "essay_text" in row:
        value = row["essay_text"]
    else:
        value = row.get("essay")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"essay_text/official_raw가 필요합니다: {essay_id(row)}")
    return value


def validate_score(value: Any, trait: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{trait} score는 숫자여야 합니다: {value!r}")
    number = float(value)
    if not math.isfinite(number) or not 1.0 <= number <= 5.0:
        raise ValueError(f"{trait} score가 1~5 범위가 아닙니다: {number}")
    return number


def validate_scores(scores: Mapping[str, Any]) -> dict[str, float]:
    return {trait: validate_score(scores.get(trait), trait) for trait in TRAITS}


# 고정 점수를 프롬프트에 넣을 때 쓰는 소수 자릿수.
CONDITIONING_DECIMALS = 2


def conditioning_scores(scores: Mapping[str, Any]) -> dict[str, float]:
    """고정 점수를 프롬프트에 넣을 수 있는 형태로 양자화한다.

    라벨은 이산 격자 위에 있다. content는 정수 10개(준거 5 x 평가자 2)의 평균이라 0.1 격자,
    organization과 expression은 정수 4개의 평균이라 0.25 격자다. 따라서 소수 둘째 자리
    반올림은 인간 점수에 대해 **무손실**이다.

    이걸 하지 않으면 `3.4000000000000004` 같은 부동소수 잡음이 그대로 프롬프트에 들어가
    두 가지가 동시에 깨진다. (1) teacher가 그 숫자를 "정리"해 정수로 반올림한다.
    (2) 모델이 `3.4`를 정확히 복사해도 `3.4 != 3.4000000000000004`라 복사 검증이 실패한다.
    2026-08-06에 24편 전량이 이 두 이유로 실패했다. 양자화한 값 하나를 프롬프트·검증·정답에
    모두 쓴다.
    """
    return {
        trait: round(value, CONDITIONING_DECIMALS)
        for trait, value in validate_scores(scores).items()
    }


def format_score(value: float) -> str:
    """프롬프트에 쓸 표기. `3.4000000000000004` -> `3.4`, `4.0` -> `4`."""
    text = f"{round(float(value), CONDITIONING_DECIMALS):.{CONDITIONING_DECIMALS}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def human_scores(row: Mapping[str, Any]) -> dict[str, float]:
    value = row.get("human_score", row.get("score"))
    if not isinstance(value, Mapping):
        raise ValueError(f"human score가 필요합니다: {essay_id(row)}")
    return validate_scores(value)


def score_prediction_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        key = essay_id(row)
        value = row.get("scores")
        if not isinstance(value, Mapping):
            judge = row.get("judge")
            if isinstance(judge, Mapping):
                value = {
                    trait: (
                        judge.get(trait, {}).get("score")
                        if isinstance(judge.get(trait), Mapping)
                        else None
                    )
                    for trait in TRAITS
                }
        if not isinstance(value, Mapping):
            raise ValueError(f"score_predictions row에 scores가 없습니다: {key}")
        if key in result:
            raise ValueError(f"중복 score prediction ID: {key}")
        result[key] = validate_scores(value)
    return result


def _extract_first_json_official(text: str) -> str | None:
    """Mirror the organizer's public brace-balancing parser exactly."""

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for index, character in enumerate(text[start:], start):
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _decode_unescaped_quote_string(raw: str) -> str | None:
    """Decode one JSON string whose inner double quotes were not escaped.

    The rationale model occasionally copies a quoted essay phrase as ``"..."``
    inside the JSON string.  The submission runtime already recovers exactly
    this error from the fixed three-trait shape.  Research inference uses the
    same narrow recovery so a Docker-valid completion is not discarded here.
    """

    escaped: list[str] = []
    backslashes = 0
    for character in raw:
        if character == '"' and backslashes % 2 == 0:
            escaped.append("\\")
        escaped.append(character)
        if character == "\\":
            backslashes += 1
        else:
            backslashes = 0
    try:
        value = json.loads('"' + "".join(escaped) + '"')
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, str) else None


def _salvage_fixed_shape_judge(candidate: str) -> dict[str, dict[str, Any]] | None:
    """Recover only unescaped rationale quotes from the fixed C/O/E shape.

    This is deliberately not a general malformed-JSON repair.  Trait order,
    keys, score syntax, and object endings must all match the requested output;
    otherwise the completion remains an error and follows the normal retry
    path.
    """

    if not candidate.startswith("{") or not candidate.endswith("}"):
        return None

    trait_matches: list[re.Match[str]] = []
    for trait in TRAITS:
        matches = list(re.finditer(rf'"{re.escape(trait)}"\s*:\s*\{{', candidate))
        if len(matches) != 1:
            return None
        trait_matches.append(matches[0])
    if [match.start() for match in trait_matches] != sorted(
        match.start() for match in trait_matches
    ):
        return None
    if candidate[1 : trait_matches[0].start()].strip():
        return None

    result: dict[str, dict[str, Any]] = {}
    for index, (trait, trait_marker) in enumerate(
        zip(TRAITS, trait_matches, strict=True)
    ):
        end = (
            trait_matches[index + 1].start()
            if index + 1 < len(trait_matches)
            else len(candidate)
        )
        segment = candidate[trait_marker.start() : end]
        score_markers = list(re.finditer(r'"score"\s*:', segment))
        rationale_markers = list(re.finditer(r'"rationale"\s*:\s*"', segment))
        if len(score_markers) != 1 or len(rationale_markers) != 1:
            return None
        score_marker = score_markers[0]
        rationale_marker = rationale_markers[0]
        if score_marker.start() >= rationale_marker.start():
            return None
        opening_end = trait_marker.end() - trait_marker.start()
        if segment[opening_end : score_marker.start()].strip():
            return None
        score_surface = segment[score_marker.end() : rationale_marker.start()]
        score_match = re.fullmatch(
            r"\s*(-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*",
            score_surface,
        )
        if score_match is None:
            return None
        try:
            score = json.loads(score_match.group(1))
        except json.JSONDecodeError:
            return None

        closing = re.search(
            r'"\s*\}\s*,\s*$' if index + 1 < len(TRAITS) else r'"\s*\}\s*\}\s*$',
            segment,
        )
        if closing is None:
            return None
        raw_rationale = segment[rationale_marker.end() : closing.start()]
        rationale = _decode_unescaped_quote_string(raw_rationale)
        if rationale is None or not rationale.strip():
            return None
        result[trait] = {"score": score, "rationale": rationale}
    return result if len(result) == len(TRAITS) else None


def parse_generated_judge(
    raw: str,
    *,
    require_rationales: bool = True,
    require_numeric_scores: bool = True,
) -> dict[str, dict[str, Any]]:
    text = re.sub(r"```(?:json)?", "", raw.strip()).replace("```", "").strip()
    extracted = _extract_first_json_official(text)
    if extracted is None:
        raise ValueError("첫 balanced JSON 객체가 없습니다")
    try:
        value = json.loads(extracted)
    except json.JSONDecodeError as exc:
        value = _salvage_fixed_shape_judge(extracted)
        if value is None:
            raise ValueError(f"첫 JSON 객체를 파싱할 수 없습니다: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("생성 JSON의 최상위 값은 객체여야 합니다")

    result: dict[str, dict[str, Any]] = {}
    for trait in TRAITS:
        item = value.get(trait)
        if not isinstance(item, dict) or "score" not in item:
            raise ValueError(f"{trait} 객체와 score가 필요합니다")
        if require_numeric_scores:
            validate_score(item.get("score"), trait)
        rationale = item.get("rationale")
        if require_rationales and (
            not isinstance(rationale, str) or not rationale.strip()
        ):
            raise ValueError(f"{trait} rationale이 비었습니다")
        result[trait] = dict(item)
    return result


def canonical_judge(
    scores: Mapping[str, Any], rationales: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    fixed = validate_scores(scores)
    result: dict[str, dict[str, Any]] = {}
    for trait in TRAITS:
        rationale = rationales.get(trait)
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"{trait} rationale이 비었습니다")
        if "{" in rationale or "}" in rationale:
            raise ValueError(f"{trait} rationale에 중괄호를 사용할 수 없습니다")
        result[trait] = {
            "score": fixed[trait],
            "rationale": rationale.strip(),
        }
    return result


def canonicalize_with_fixed_scores(
    parsed: Mapping[str, Mapping[str, Any]], scores: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    rationales = {trait: parsed.get(trait, {}).get("rationale") for trait in TRAITS}
    if not all(isinstance(value, str) for value in rationales.values()):
        raise ValueError("생성 결과에 세 rationale이 모두 필요합니다")
    return canonical_judge(scores, rationales)  # type: ignore[arg-type]


def compact_judge_json(judge: Mapping[str, Any]) -> str:
    # Prompt skeleton uses ``4`` rather than ``4.0`` and explicitly asks the model
    # to copy the score token verbatim.  Keep fractional legacy scores as-is, but
    # serialize integer-valued scores as JSON integers so the supervised assistant
    # target does not contradict that instruction.
    normalized: dict[str, dict[str, Any]] = {}
    for trait in TRAITS:
        item = judge.get(trait)
        if not isinstance(item, Mapping):
            raise ValueError(f"{trait} judge 객체가 필요합니다")
        score = validate_score(item.get("score"), trait)
        normalized[trait] = dict(item)
        normalized[trait]["score"] = int(score) if score.is_integer() else score
    return json.dumps(
        normalized, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def rationale_audit(
    judge: Mapping[str, Mapping[str, Any]], essay: str
) -> dict[str, Any]:
    texts = [str(judge[trait].get("rationale", "")).strip() for trait in TRAITS]
    forbidden_organization = (
        "문단 구분이 없다",
        "한 문단으로",
        "줄바꿈이 없다",
        "문단이 나뉘지",
    )
    template_placeholders = tuple(
        f"<{trait} 근거, 두 문장 이내 180자 이내>" for trait in TRAITS
    )
    quoted_spans: dict[str, list[str]] = {}
    missing_quoted_spans: dict[str, list[str]] = {}
    quote_patterns = (
        r"[‘']([^‘'\n]{2,80})[’']",
        r'[“"]([^“"\n]{2,80})[”"]',
    )
    for trait, rationale in zip(TRAITS, texts, strict=True):
        spans: list[str] = []
        for pattern in quote_patterns:
            spans.extend(re.findall(pattern, rationale))
        spans = list(dict.fromkeys(span.strip() for span in spans if span.strip()))
        quoted_spans[trait] = spans
        missing_quoted_spans[trait] = [span for span in spans if span not in essay]
    return {
        "rationale_characters": {
            trait: len(texts[index]) for index, trait in enumerate(TRAITS)
        },
        "all_nonempty": all(texts),
        "all_distinct": len(set(texts)) == len(TRAITS),
        "template_placeholder_echo": any(
            placeholder in rationale
            for rationale in texts
            for placeholder in template_placeholders
        ),
        "organization_surface_claim": any(
            phrase in texts[1] for phrase in forbidden_organization
        ),
        "quoted_spans": quoted_spans,
        "missing_quoted_spans": missing_quoted_spans,
        "all_quoted_spans_grounded": not any(missing_quoted_spans.values()),
        "essay_characters": len(essay),
    }
