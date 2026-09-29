"""근거 생성 프롬프트. 학습 쪽 정의를 그대로 재사용한다.

근거 어댑터는 `main_code_relonation`이 학습한다. 그 학습은
`prompts.build_messages(..., scores=고정점수)`가 만든 **단일 user message**를 tokenizer의
`apply_chat_template`으로 감싼 토큰열에서 assistant span만 supervise한다.

따라서 서빙도 정확히 같은 message와 같은 chat template을 써야 한다. 이 파일이 프롬프트
문구를 새로 쓰면 어댑터는 학습 때 본 적 없는 입력을 받고, 형식 준수와 근거 품질이 조용히
무너진다. 이 프로젝트는 이미 같은 종류의 train/serve 표면 불일치로 organization RMSE가
`.54 -> 1.03`으로 붕괴한 전례가 있다. 그래서 여기서는 위임만 한다.

현행 제출 계약이 점수 구현을 공유하는 것과 같은 이유로 근거 prompt 구현도 학습 폴더에서
직접 공유한다.
"""

from __future__ import annotations

from typing import Any, Mapping

from main_code_relonation.prompts import DEFAULT_SKELETON_HINT, build_messages

from .request_parser import OfficialRequestText


def build_rationale_messages(
    text: OfficialRequestText,
    scores: Mapping[str, float],
    *,
    prompt_template: str | None = None,
    skeleton_hint: str = DEFAULT_SKELETON_HINT,
) -> list[dict[str, str]]:
    """학습과 동일한 고정 점수 조건 message를 만든다.

    점수 head의 원점수에서 최종 확정한 제출 정수를 넘긴다. 근거 학습은 fixed-score
    conditioning으로 수행됐으므로 응답에 실제로 실리는 점수를 조건으로 써야 한다.
    메시지 형식과 표기 처리는 `build_messages`의 학습 코드에 그대로 위임한다.
    """

    return build_messages(
        text.prompt,
        text.essay,
        scores=dict(scores),
        prompt_template=prompt_template,
        skeleton_hint=skeleton_hint,
    )


def render_for_diagnostics(messages: list[dict[str, str]]) -> str:
    """chat template 없이 사람이 읽을 수 있게 이어 붙인다. 추론에는 쓰지 않는다."""

    return "\n\n".join(f"<{item['role']}>\n{item['content']}" for item in messages)


def rationale_message_contract() -> dict[str, Any]:
    """serving이 재사용하는 학습 계약을 진단 기록으로 남긴다."""

    return {
        "source": "main_code_relonation.prompts.build_messages",
        "score_mode": "fixed",
        "roles": ["user"],
        "chat_template": "tokenizer.apply_chat_template(add_generation_prompt=True)",
    }
