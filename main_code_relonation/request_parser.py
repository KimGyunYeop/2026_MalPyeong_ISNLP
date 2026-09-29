from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class OfficialRequestText:
    """The two scorer-visible text spans from the organizer's user message."""

    prompt: str
    essay: str
    user_content: str


def _last_section_marker(
    content: str, label: str, *, before: int | None = None
) -> tuple[int, int]:
    """Return ``(separator_start, payload_start)`` for the last exact marker.

    The organizer serializes a section as a blank line, a bracketed label, and a
    line break.  Slicing around that exact delimiter preserves every character in
    the prompt and essay payloads; in particular, this function never strips or
    normalizes whitespace.  CRLF is accepted for local Docker contract tests too.
    """

    end = len(content) if before is None else before
    candidates: list[tuple[int, int, int]] = []
    for line_break in ("\n", "\r\n"):
        for separator in (line_break + line_break, line_break):
            delimiter = f"{separator}[{label}]{line_break}"
            index = content.rfind(delimiter, 0, end)
            if index >= 0:
                label_start = index + len(separator)
                candidates.append(
                    (label_start, index, index + len(delimiter))
                )

        # A marker may be the first section in a minimal local request.
        initial = f"[{label}]{line_break}"
        if content.startswith(initial) and len(initial) <= end:
            candidates.append((0, 0, len(initial)))

    if not candidates:
        raise ValueError(f"마지막 [{label}] 섹션을 찾을 수 없습니다")
    # Pick the last label occurrence. If both one- and two-newline delimiters
    # describe that same label, the longer official delimiter owns the separator.
    _, separator_start, payload_start = max(
        candidates, key=lambda item: (item[0], -item[1])
    )
    return separator_start, payload_start


def parse_official_user_content(content: str) -> OfficialRequestText:
    """Split the final prompt/essay sections without changing their bytes."""

    if not isinstance(content, str):
        raise ValueError("user content는 문자열이어야 합니다")

    essay_marker_start, essay_start = _last_section_marker(content, "essay_text")
    _, prompt_start = _last_section_marker(
        content, "prompt_text", before=essay_marker_start
    )
    if prompt_start >= essay_marker_start:
        raise ValueError("[prompt_text]는 [essay_text]보다 앞에 있어야 합니다")

    prompt = content[prompt_start:essay_marker_start]
    essay = content[essay_start:]
    if not prompt:
        raise ValueError("prompt_text가 비었습니다")
    if not essay:
        raise ValueError("essay_text가 비었습니다")
    return OfficialRequestText(prompt=prompt, essay=essay, user_content=content)


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
