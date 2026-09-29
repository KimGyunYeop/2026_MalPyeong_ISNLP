"""공식 단일 user message 파싱.

`main_code_relonation/request_parser.py`에서 그대로 가져왔다. 공식 입출력 계약은 폴더 사이
공유 범위이고, Docker 이미지를 self-contained로 유지하려고 import 대신 복사했다. 두 파일이
갈라지면 안 되므로 계약이 바뀌면 둘을 함께 고친다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class OfficialRequestText:
    """The two scorer-visible text spans from the organizer's user message."""

    prompt: str
    essay: str
    user_content: str


def _section_markers(content: str, label: str, *, start: int = 0, end: int | None = None):
    """Return every ``(separator_start, payload_start)`` for an exact marker.

    The organizer serializes a section as a blank line, a bracketed label, and a
    line break.  Slicing around that exact delimiter preserves every character in
    the prompt and essay payloads; in particular, this function never strips or
    normalizes whitespace.  CRLF is accepted for local Docker contract tests too.
    """

    limit = len(content) if end is None else end
    found: dict[int, tuple[int, int]] = {}
    for line_break in ("\n", "\r\n"):
        for separator in (line_break + line_break, line_break):
            delimiter = f"{separator}[{label}]{line_break}"
            cursor = start
            while True:
                index = content.find(delimiter, cursor, limit)
                if index < 0:
                    break
                label_start = index + len(separator)
                previous = found.get(label_start)
                # 같은 label을 1-줄바꿈과 2-줄바꿈 구분자가 모두 설명하면 더 긴
                # 공식 구분자가 separator를 갖는다(더 작은 separator_start).
                if previous is None or index < previous[0]:
                    found[label_start] = (index, index + len(delimiter))
                cursor = index + 1

        # A marker may be the first section in a minimal local request.
        initial = f"[{label}]{line_break}"
        if start == 0 and content.startswith(initial) and len(initial) <= limit:
            found.setdefault(0, (0, len(initial)))

    return [found[key] for key in sorted(found)]


def _last_section_marker(
    content: str, label: str, *, before: int | None = None
) -> tuple[int, int]:
    """Backward-compatible accessor for the last marker of ``label``."""

    markers = _section_markers(content, label, end=before)
    if not markers:
        raise ValueError(f"마지막 [{label}] 섹션을 찾을 수 없습니다")
    return markers[-1]


def parse_official_user_content(content: str) -> OfficialRequestText:
    """Split the final prompt/essay sections without changing their bytes.

    Marker selection is deliberately asymmetric.

    ``[prompt_text]``  -> the **last** occurrence.  The fixed grading instruction
    that precedes it is organizer-controlled and may legitimately repeat the
    label (the 과제기술서 output example does), so the closest one to the payload
    wins.

    ``[essay_text]``   -> the **first** occurrence *after* that prompt marker.
    The essay body is arbitrary participant-written text and may contain the
    literal line ``[essay_text]``.  Taking the last one would then silently score
    only the tail after the injected marker and push the real body into
    ``prompt``: a 200 response, a clean parse, no degradation flag, and a wrong
    score that no output-side gate can detect.  On the official single-marker
    request both rules select the same span, so this changes nothing for a
    well-formed request.

    Empty payloads are returned as-is instead of raising.  A request that carries
    both markers is unambiguous even when a section is blank, and raising here
    became an HTTP 400 that the evaluator scores as 0 for the essay.
    """

    if not isinstance(content, str):
        raise ValueError("user content는 문자열이어야 합니다")

    prompt_markers = _section_markers(content, "prompt_text")
    if not prompt_markers:
        raise ValueError("마지막 [prompt_text] 섹션을 찾을 수 없습니다")

    # 뒤에서부터 훑되, 실제로 뒤에 essay 마커가 따라오는 prompt 마커만 채택한다.
    # 에세이 본문이 "[prompt_text]"를 포함하면 그 마커 뒤에는 essay 마커가 없으므로
    # 자동으로 건너뛰고 진짜 구조를 되찾는다.
    pair: tuple[int, int, int] | None = None
    for _, candidate_start in reversed(prompt_markers):
        following = _section_markers(content, "essay_text", start=candidate_start)
        if following:
            pair = (candidate_start, *following[0])
            break
    if pair is None:
        raise ValueError("마지막 [essay_text] 섹션을 찾을 수 없습니다")
    prompt_start, essay_marker_start, essay_start = pair

    prompt = content[prompt_start:essay_marker_start]
    essay = content[essay_start:]
    return OfficialRequestText(prompt=prompt, essay=essay, user_content=content)


#: 로컬 배포 게이트가 container.log에서 찾는 고정 토큰. 문자열이 바뀌면
#: `run_docker_check.sh`의 게이트도 함께 바꿔야 한다.
MARKER_ANOMALY_TOKEN = "REQUEST_MARKER_ANOMALY"


def marker_anomaly(text: "OfficialRequestText") -> str | None:
    """추출한 payload가 의심스러우면 사람이 읽을 설명을 돌려준다.

    이 경로의 위험은 예외가 아니라 **조용한 오채점**이다. 마커가 본문 안에
    끼어들면 응답은 200이고 공식 파서도 통과하며 강등도 기록되지 않는데
    채점 대상만 다른 글이 된다. 출력만 보는 게이트로는 절대 잡을 수 없으므로
    입력 쪽에서 표시해 둔다.
    """

    reasons: list[str] = []
    for label in ("prompt_text", "essay_text"):
        extra = text.essay.count(f"[{label}]")
        if extra:
            reasons.append(f"essay 안에 [{label}] {extra}회")
    if text.prompt.count("[essay_text]"):
        reasons.append("prompt 안에 [essay_text]")
    if not text.essay.strip():
        reasons.append("essay payload가 비어 있음")
    if not text.prompt.strip():
        reasons.append("prompt payload가 비어 있음")
    return "; ".join(reasons) if reasons else None


def coerce_message_content(content: Any) -> str | None:
    """Accept the two content shapes an OpenAI-compatible client can send.

    The notice specifies a plain string, and that is what the evaluator has been
    observed to send.  Official OpenAI clients may instead send a list of typed
    parts.  Rejecting that shape would return HTTP 400, and a rejected request is
    scored as 0 for the essay, so accept it when the text is unambiguous.
    """

    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, Mapping) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
            else:
                return None
        if chunks:
            return "".join(chunks)
    return None


def parse_single_user_messages(
    messages: Sequence[Mapping[str, Any]],
) -> OfficialRequestText:
    """Validate the notice's one-user-message request contract and split it.

    The strict contract is checked first and is what every observed request has
    matched.  When it does not match, fall back to the last user message instead
    of raising: an HTTP 400 costs the whole essay, and a request carrying the two
    official markers is unambiguous regardless of what surrounds it.
    """

    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise ValueError("messages에는 정확히 하나의 user message가 필요합니다")
    if len(messages) == 1:
        message = messages[0]
        if not isinstance(message, Mapping) or message.get("role") != "user":
            raise ValueError("messages[0].role은 user여야 합니다")
        content = coerce_message_content(message.get("content"))
        if content is None:
            raise ValueError("messages[0].content는 문자열이어야 합니다")
        return parse_official_user_content(content)

    for message in reversed(messages):
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = coerce_message_content(message.get("content"))
        if content is None:
            continue
        return parse_official_user_content(content)
    raise ValueError("messages에는 정확히 하나의 user message가 필요합니다")


# 채점 요청으로 볼 최소 길이. 이 값 **미만**이고 마커도 없으면 Docker 규정 §10의
# 일반 API smoke로 보고 안내 문자열을 낸다. 즉 이 임계 아래는 점수가 나가지 않으므로
# 진짜 에세이가 여기 걸리면 그 편은 0점이다.
#
# 실측(2026-08-20): 공식 validation 400편 본문 길이 min=800 / p1=801, train 11,600편
# min=757. 12,000편 중 300자 미만은 0편이다. 그래도 임계를 757의 1/5 수준으로 내려
# 여유를 크게 잡는다. §10 예시("한 줄로 자기소개해 주세요")는 20자 안팎이라 150자
# 아래에서도 안전하게 구분된다. 채점 실패의 비용(제곱오차 12.16)이 smoke 요청에
# 점수를 돌려주는 어색함보다 압도적으로 크다.
SCORING_MIN_CHARS = 150


def salvage_scoring_text(messages: Any) -> OfficialRequestText | None:
    """마커를 못 읽었지만 채점해야 할 만큼 긴 본문이 있으면 그것으로 입력을 만든다.

    운영측이 보내는 형식은 `[prompt_text]` / `[essay_text]` 마커가 있는 단일 user
    메시지다. 그 계약이 어긋나는 상황은 원래 없어야 하지만, 어긋났을 때 400을 내면
    그 샘플이 0점으로 집계된다. 짧은 일반 smoke 요청과 구분해 긴 것만 채점한다.
    """

    if not isinstance(messages, list) or not messages:
        return None
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = coerce_message_content(message.get("content"))
        if content:
            parts.append(content)
    if not parts:
        return None
    joined = "\n".join(parts).strip()
    if len(joined) < SCORING_MIN_CHARS:
        return None
    # prompt는 비운다. 마커를 못 읽은 상황이라 어디까지가 논제인지 알 수 없고,
    # 본문을 논제로 잘못 넣는 것보다 비우는 쪽이 채점 입력을 덜 왜곡한다.
    return OfficialRequestText(prompt="", essay=joined, user_content=joined)
