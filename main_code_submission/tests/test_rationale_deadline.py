"""근거 생성 벽시계 예산의 계약.

왜 이게 필요한가: `NEVER_DISCARD_A_SCORE.md` §6이 지목한 **남은 최대 위험**은
`_inference_lock` 직렬화와 평가 서버의 미공개 요청 timeout이다. 근거 생성은 첫
시도 + 재시도 2회까지 갈 수 있어 최악의 경우 편당 3배가 걸린다. 그 편이 timeout
되면 우리 잘못 없이 0점(제곱오차 12.16)이고, 근거 세 개를 template으로 잃는 비용은
그 편 Judge 점수뿐(종합 가중치 10%)이다. 30배 차이라 예산을 거는 쪽이 항상 옳다.

기본값은 0(무제한)이라 이 파일이 없던 시절 동작은 bit-exact 그대로다.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import torch

from main_code_submission.config import RationaleSpec
from main_code_submission.request_parser import parse_official_user_content
from main_code_submission.schema import TRAITS

def build_official_user_content(prompt_text: str, essay_text: str) -> str:
    return f"지시문\n\n[prompt_text]\n{prompt_text}\n\n[essay_text]\n{essay_text}"


PROMPT = "로봇세 도입에 대한 자신의 생각을 쓰시오."
ESSAY = "첫 문장. 두 번째 문장. 세 번째 문장."
COMPLETE = json.dumps(
    {trait: {"rationale": f"{trait} 근거"} for trait in TRAITS}, ensure_ascii=False
)


class _Tokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
        return [5, 6, 7]

    def encode(self, text, add_special_tokens=False):
        return [5] * max(1, len(text) // 4)

    def decode(self, ids, skip_special_tokens=True):
        values = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
        if values and values[-1] == 99:
            return COMPLETE
        if values and values[-1] == 55:
            # 세 개 중 하나만 담긴 부분 결과.
            return json.dumps({"content": {"rationale": "content 근거"}}, ensure_ascii=False)
        return ""


class _SlowModel:
    """호출마다 `seconds_per_call`을 소모하는 가짜 생성기."""

    def __init__(self, outputs: list[int], seconds_per_call: float) -> None:
        self.outputs = outputs
        self.seconds_per_call = seconds_per_call
        self.calls: list[dict[str, object]] = []
        self.clock = 0.0

    def generate(self, **kwargs: object) -> torch.Tensor:
        self.calls.append(kwargs)
        self.clock += self.seconds_per_call
        token = self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)]
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        suffix = torch.tensor([[token]], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat((input_ids, suffix), dim=1)


class _Context:
    def __init__(self, ensemble, name):
        self.ensemble, self.name, self.previous = ensemble, name, ensemble.active

    def __enter__(self):
        self.ensemble.active = self.name

    def __exit__(self, *_):
        self.ensemble.active = self.previous


class _Ensemble:
    def __init__(self) -> None:
        self.active = "score"

    @property
    def active_adapter(self):
        return self.active

    @property
    def adapter_names(self):
        return ["score"]

    def using(self, name):
        return _Context(self, name)


def _engine(model: _SlowModel, deadline_seconds: float, monkeypatch):
    from main_code_relonation.prompts import baseline_prompt_template
    from main_code_submission.engine import SubmissionEngine

    engine = object.__new__(SubmissionEngine)
    engine.config = SimpleNamespace(
        rationale=SimpleNamespace(
            enabled=True,
            max_length=64,
            max_new_tokens=16,
            chat_template_kwargs={},
            rationale_prompt_text=baseline_prompt_template(),
            deadline_seconds=deadline_seconds,
        )
    )
    engine.device = torch.device("cpu")
    engine._rationale_model = model
    engine._rationale_tokenizer = _Tokenizer()
    engine._rationale_ensemble = _Ensemble()
    engine._rationale_adapter_name = "rationale"

    # 실제로 자지 않고 가짜 시계를 쓴다. 테스트가 예산만큼 느려지면 아무도 안 돌린다.
    monkeypatch.setattr(
        "main_code_submission.engine.time.monotonic", lambda: model.clock
    )
    return engine


def _text():
    return parse_official_user_content(build_official_user_content(PROMPT, ESSAY))


def test_zero_deadline_keeps_the_historical_unlimited_ladder(monkeypatch) -> None:
    """기본값에서는 재시도 사다리가 예전 그대로 3단계까지 간다."""

    model = _SlowModel([2, 3, 99], seconds_per_call=100.0)
    engine = _engine(model, 0.0, monkeypatch)
    result = engine.rationales(_text(), {trait: 3.0 for trait in TRAITS})
    assert len(model.calls) == 3
    assert result == {trait: f"{trait} 근거" for trait in TRAITS}
    # 무제한 경로는 stopping_criteria를 아예 달지 않는다.
    assert "stopping_criteria" not in model.calls[0]


def test_exhausted_budget_skips_the_first_retry(monkeypatch) -> None:
    model = _SlowModel([2, 99], seconds_per_call=10.0)
    engine = _engine(model, 5.0, monkeypatch)
    result = engine.rationales(_text(), {trait: 3.0 for trait in TRAITS})
    # 첫 호출만 하고 예산이 끝나 재시도를 하지 않는다.
    assert len(model.calls) == 1
    assert result == {}


def test_exhausted_budget_skips_the_sampled_retry(monkeypatch) -> None:
    model = _SlowModel([2, 55, 99], seconds_per_call=4.0)
    engine = _engine(model, 5.0, monkeypatch)
    result = engine.rationales(_text(), {trait: 3.0 for trait in TRAITS})
    # 4초 + 4초 = 8초 > 5초라 표본추출 단계는 생략된다.
    assert len(model.calls) == 2
    # **얻은 만큼은 반드시 돌려준다.** 이게 없으면 예산이 근거를 전부 버린다.
    assert result == {"content": "content 근거"}


def test_generous_budget_still_completes_the_ladder(monkeypatch) -> None:
    model = _SlowModel([2, 3, 99], seconds_per_call=1.0)
    engine = _engine(model, 60.0, monkeypatch)
    result = engine.rationales(_text(), {trait: 3.0 for trait in TRAITS})
    assert len(model.calls) == 3
    assert result == {trait: f"{trait} 근거" for trait in TRAITS}


def test_budget_attaches_a_stopping_criterion_to_generation(monkeypatch) -> None:
    """한 번의 긴 생성도 예산 안에서 멈춰야 한다. 재시도 게이트만으로는 부족하다."""

    model = _SlowModel([99], seconds_per_call=1.0)
    engine = _engine(model, 30.0, monkeypatch)
    engine.rationales(_text(), {trait: 3.0 for trait in TRAITS})
    criteria = model.calls[0].get("stopping_criteria")
    assert criteria is not None and len(criteria) == 1
    # 예산이 남아 있으면 멈추지 않는다.
    assert criteria[0](torch.tensor([[1]]), None) is False
    # 시계를 예산 너머로 밀면 즉시 멈춘다.
    model.clock = 1_000.0
    assert criteria[0](torch.tensor([[1]]), None) is True


def test_spec_rejects_a_negative_budget() -> None:
    import pytest

    with pytest.raises(ValueError, match="deadline_seconds"):
        RationaleSpec(base_model="x", adapter=None, deadline_seconds=-1.0)


def test_spec_default_is_unlimited() -> None:
    assert RationaleSpec(base_model="x", adapter=None).deadline_seconds == 0.0
