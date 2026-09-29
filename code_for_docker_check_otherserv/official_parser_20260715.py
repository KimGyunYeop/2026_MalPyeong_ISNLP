"""2026-07-15 운영측 공지에 공개된 모델 출력 JSON parser.

공지의 두 함수 본문은 동작을 고치거나 관대하게 만들지 않고 그대로 둔다. 이 파일의
imports와 module docstring만 독립 실행을 위해 덧붙였다.
"""

import json
import re
from typing import Any, Dict, Optional


def _extract_first_json(text: str) -> Optional[str]:
    """중괄호 균형을 맞춰 첫 번째 JSON 객체 문자열만 추출."""

    start = text.find('{')

    if start == -1:

        return None

    depth = 0

    for i, ch in enumerate(text[start:], start):

        if ch == '{':

            depth += 1

        elif ch == '}':

            depth -= 1

            if depth == 0:

                return text[start:i + 1]

    return None


def _parse_model_output(raw: str) -> Optional[Dict[str, Any]]:
    """모델 출력 JSON 파싱. 실패 시 None 반환."""

    text = re.sub(r"```(?:json)?", "", raw.strip()).replace("```", "").strip()

    json_str = _extract_first_json(text)

    if json_str is None:

        return None

    try:

        parsed = json.loads(json_str)

        required = {"content", "organization", "expression"}

        if required.issubset(parsed.keys()):

            if all(isinstance(parsed[d], dict) and "score" in parsed[d] for d in required):

                return parsed

    except (json.JSONDecodeError, Exception):

        pass

    return None
