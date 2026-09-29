"""제출 응답에서 **점수를 버리는 경로가 하나도 남지 않았음**을 test로 고정한다.

평가 서버는 실패한 요청을 그 에세이 0점으로 집계한다. validation gold 기준 한 행을 0점으로
잃는 비용(제곱오차 11.97)은 그 행에 상수 점수를 낸 비용(0.43)의 28배이고, 400편 중 3편만
그렇게 잃어도 mean-first RMSE가 0.4168 -> 0.5118로 무너진다.

여기 있는 test는 전부 "무엇이 실패해도 공식 파서가 읽을 수 있는 점수가 나온다"를 검사한다.
새 실패 경로를 추가할 때 이 파일을 먼저 깨뜨려 보라.
"""

from __future__ import annotations

import json
import math
import threading
from types import SimpleNamespace

import pytest
import torch

from main_code.postprocess import ScorePostprocessor
from main_code_submission.degrade import (
    NEUTRAL_SUBMITTED_SCORES,
    fill_missing_rationales,
    last_resort_outputs,
    template_rationale,
)
from main_code_submission.engine import SubmissionEngine
from main_code_submission.request_parser import parse_official_user_content
from main_code_submission.schema import TRAITS, _parse_model_output

from .test_submission_contract import (
    ESSAY,
    PROMPT,
    _AdapterEnsemble,
    _GenerationModel,
    _GenerationTokenizer,
    build_official_user_content,
)
import pathlib

# 대회 데이터는 배포가 제한되어 저장소에 없다. 데이터가 있는 환경에서만 돈다.
_DATA_ROOT = pathlib.Path(__file__).resolve().parents[2] / "main_code/datasets/processed_dataset"
pytestmark = pytest.mark.skipif(
    not (_DATA_ROOT / "train.jsonl").is_file(),
    reason="대회 데이터(main_code/datasets/processed_dataset)가 없는 환경",
)


TEXT = parse_official_user_content(build_official_user_content(PROMPT, ESSAY))


def _engine(
    *,
    rationale_model: object,
    score_fn,
    rationale_enabled: bool = True,
    max_length: int = 64,
) -> SubmissionEngine:
    from main_code_relonation.prompts import baseline_prompt_template

    engine = object.__new__(SubmissionEngine)
    engine.config = SimpleNamespace(
        rationale=SimpleNamespace(
            enabled=rationale_enabled,
            max_length=max_length,
            max_new_tokens=16,
            chat_template_kwargs={},
            rationale_prompt_text=baseline_prompt_template(),
        )
    )
    engine.device = torch.device("cpu")
    engine._members = [object()]
    engine._inference_lock = threading.Lock()
    engine._score_postprocessor = ScorePostprocessor("average_matched")
    engine._rationale_model = rationale_model
    engine._rationale_tokenizer = _GenerationTokenizer()
    engine._rationale_ensemble = _AdapterEnsemble()
    engine._rationale_adapter_name = "rationale"
    engine.score = score_fn  # type: ignore[method-assign]
    return engine


def _official(body: str) -> dict:
    parsed = _parse_model_output(body)
    assert parsed is not None, "공식 파서가 응답을 읽지 못했습니다"
    return parsed


# --- 근거가 죽어도 점수는 산다 -------------------------------------------------
def test_rationale_generation_failure_still_returns_the_computed_score() -> None:
    """가장 비쌌던 경로. 예전에는 RuntimeError -> HTTP 500 -> 그 에세이 0점이었다."""

    class _Failing:
        def generate(self, **_: object) -> torch.Tensor:
            raise RuntimeError("generation failed")

    engine = _engine(
        rationale_model=_Failing(),
        score_fn=lambda text: {"content": 4.4, "organization": 3.6, "expression": 2.5},
    )

    body, diagnostics = engine.respond(TEXT)

    parsed = _official(body)
    assert {trait: parsed[trait]["score"] for trait in TRAITS} == {
        "content": 4,
        "organization": 4,
        "expression": 3,
    }
    assert all(parsed[trait]["rationale"].strip() for trait in TRAITS)
    assert diagnostics["degradation"] == ["rationale_failed"]
    # 근거가 죽어도 score adapter는 복원되어 다음 요청이 오염되지 않는다.
    assert engine._rationale_ensemble.active_adapter == "score"


def test_partial_rationale_keeps_generated_traits_and_templates_only_the_rest() -> None:
    """세 개 중 둘만 파싱돼도 그 둘은 그대로 쓰고 하나만 template으로 메운다."""

    class _PartialTokenizer(_GenerationTokenizer):
        def decode(self, ids: object, *, skip_special_tokens: bool) -> str:
            return json.dumps(
                {
                    "content": {"rationale": "내용 실제 근거"},
                    "organization": {"rationale": "조직 실제 근거"},
                },
                ensure_ascii=False,
            )

    engine = _engine(
        rationale_model=_GenerationModel([99, 99, 99]),
        score_fn=lambda text: {trait: 3.0 for trait in TRAITS},
    )
    engine._rationale_tokenizer = _PartialTokenizer()

    body, diagnostics = engine.respond(TEXT)

    parsed = _official(body)
    assert parsed["content"]["rationale"] == "내용 실제 근거"
    assert parsed["organization"]["rationale"] == "조직 실제 근거"
    assert parsed["expression"]["rationale"] == template_rationale("expression", 3.0)
    assert diagnostics["substituted_rationales"] == ["expression"]
    assert {trait: parsed[trait]["score"] for trait in TRAITS} == {t: 3 for t in TRAITS}


def test_unrecoverable_rationale_escalates_to_a_sampled_retry() -> None:
    """greedy 재시도는 같은 토큰을 돌려준다. 마지막 한 번은 실제로 다른 경로여야 한다."""

    class _AlwaysMalformed(_GenerationTokenizer):
        def decode(self, ids: object, *, skip_special_tokens: bool) -> str:
            return "근거를 만들 수 없습니다"

    model = _GenerationModel([99, 99, 99])
    engine = _engine(
        rationale_model=model, score_fn=lambda text: {t: 3.0 for t in TRAITS}
    )
    engine._rationale_tokenizer = _AlwaysMalformed()

    body, diagnostics = engine.respond(TEXT)

    assert len(model.calls) == 3
    assert "min_new_tokens" not in model.calls[0]
    assert model.calls[1]["min_new_tokens"] == 8
    assert model.calls[2]["do_sample"] is True  # 표본추출로만 다른 토큰이 나온다
    parsed = _official(body)
    assert {trait: parsed[trait]["score"] for trait in TRAITS} == {t: 3 for t in TRAITS}
    assert set(diagnostics["substituted_rationales"]) == set(TRAITS)


def test_normal_path_still_makes_exactly_one_generation_call() -> None:
    """정상 경로에는 재시도도, template 대체도, 강등 기록도 없어야 한다."""

    model = _GenerationModel([99])
    engine = _engine(
        rationale_model=model, score_fn=lambda text: {t: 3.0 for t in TRAITS}
    )

    body, diagnostics = engine.respond(TEXT)

    assert len(model.calls) == 1
    assert diagnostics["degradation"] == []
    assert diagnostics["substituted_rationales"] == []
    parsed = _official(body)
    assert parsed["content"]["rationale"] == "content 재생성 근거"


# --- 점수 경로가 죽어도 0점은 내지 않는다 ---------------------------------------
def test_score_failure_falls_back_to_the_constant_triple_not_a_500() -> None:
    def _boom(text: object) -> dict[str, float]:
        raise RuntimeError("scorer exploded")

    engine = _engine(rationale_model=_GenerationModel([99]), score_fn=_boom)

    body, diagnostics = engine.respond(TEXT)

    parsed = _official(body)
    assert {trait: parsed[trait]["score"] for trait in TRAITS} == {
        trait: int(value) for trait, value in NEUTRAL_SUBMITTED_SCORES.items()
    }
    assert all(parsed[trait]["rationale"].strip() for trait in TRAITS)
    assert diagnostics["degradation"] == ["score_failed"]


def test_non_finite_score_is_replaced_instead_of_raising() -> None:
    """NaN 하나가 clamp_score 예외로 응답 전체를 날리던 경로."""

    engine = _engine(
        rationale_model=_GenerationModel([99]),
        score_fn=lambda text: {
            "content": float("nan"),
            "organization": 3.0,
            "expression": 4.0,
        },
    )

    body, _ = engine.respond(TEXT)

    parsed = _official(body)
    assert all(math.isfinite(float(parsed[trait]["score"])) for trait in TRAITS)
    assert parsed["organization"]["score"] == 3


def test_rationale_disabled_config_responds_instead_of_raising() -> None:
    engine = _engine(
        rationale_model=_GenerationModel([99]),
        score_fn=lambda text: {t: 3.0 for t in TRAITS},
        rationale_enabled=False,
    )

    body, diagnostics = engine.respond(TEXT)

    assert diagnostics["degradation"] == ["rationale_disabled"]
    parsed = _official(body)
    assert all(parsed[trait]["rationale"].strip() for trait in TRAITS)


# --- 예산 초과 ------------------------------------------------------------------
def test_over_budget_prompt_truncates_the_essay_instead_of_dropping_the_score() -> None:
    """예전에는 ValueError를 던져 이미 확정된 점수까지 함께 버렸다."""

    class _LengthAwareTokenizer(_GenerationTokenizer):
        """essay 길이에 비례하는 토큰 수를 흉내 낸다."""

        def apply_chat_template(self, messages, **__: object) -> list[int]:
            content = "".join(message["content"] for message in messages)
            return [10] * max(1, len(content) // 4)

        def decode(self, ids: object, *, skip_special_tokens: bool) -> str:
            return json.dumps(
                {trait: {"rationale": f"{trait} 근거"} for trait in TRAITS},
                ensure_ascii=False,
            )

    long_text = parse_official_user_content(
        build_official_user_content(PROMPT, ESSAY * 200)
    )
    model = _GenerationModel([99])
    engine = _engine(
        rationale_model=model,
        score_fn=lambda text: {t: 3.0 for t in TRAITS},
        max_length=512,
    )
    engine._rationale_tokenizer = _LengthAwareTokenizer()

    prompt_ids, truncated = engine._rationale_prompt_ids(long_text, {t: 3.0 for t in TRAITS})
    assert truncated > 0, "긴 essay가 잘리지 않았습니다"
    assert len(prompt_ids) <= 512 - 16

    body, diagnostics = engine.respond(long_text)
    parsed = _official(body)
    assert {trait: parsed[trait]["score"] for trait in TRAITS} == {t: 3 for t in TRAITS}
    assert diagnostics["degradation"] == []


# --- adapter 복원 ---------------------------------------------------------------
def test_failed_adapter_restore_is_forced_back_before_raising() -> None:
    """복원 실패는 다음 요청의 점수를 오염시킨다. 예외만 올리면 오염이 멈추지 않는다."""

    class _StickyEnsemble(_AdapterEnsemble):
        """`using` 종료 시 복원을 놓치지만 `activate`에는 응답하는 ensemble."""

        def using(self, name: str):
            ensemble = self

            class _Ctx:
                def __enter__(self) -> None:
                    ensemble.active = name

                def __exit__(self, *_: object) -> None:
                    pass  # 복원 누락

            return _Ctx()

        def activate(self, name: str) -> None:
            self.active = name

    engine = _engine(
        rationale_model=_GenerationModel([99]),
        score_fn=lambda text: {t: 3.0 for t in TRAITS},
    )
    engine._rationale_ensemble = _StickyEnsemble()

    body, diagnostics = engine.respond(TEXT)

    assert engine._rationale_ensemble.active_adapter == "score"
    assert diagnostics["degradation"] == []
    _official(body)


# --- HTTP 계층의 마지막 방어선 --------------------------------------------------
def test_server_returns_a_parseable_200_even_if_respond_raises() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from main_code_submission.serve import create_app

    class _Exploding:
        def respond(self, text: object):
            raise RuntimeError("respond exploded")

    config = SimpleNamespace(served_model_name="test-model")
    client = TestClient(create_app(config, _Exploding()), raise_server_exceptions=False)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": build_official_user_content(PROMPT, ESSAY)}
            ]
        },
    )

    assert response.status_code == 200
    parsed = _official(response.json()["choices"][0]["message"]["content"])
    assert {trait: parsed[trait]["score"] for trait in TRAITS} == {
        trait: int(value) for trait, value in NEUTRAL_SUBMITTED_SCORES.items()
    }


# --- 상수 삼중이 실제로 0점보다 나은지 ------------------------------------------
def test_constant_triple_beats_a_zero_by_a_wide_margin_on_real_gold() -> None:
    """`NEUTRAL_SUBMITTED_SCORES`가 임의의 값이 아니라 gold 분포에서 고른 값임을 고정한다."""

    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    train = next(
        path
        for path in (root / "datasets").iterdir()
        if path.name.endswith("_train.jsonl")
    )
    gold = [
        float(json.loads(line)["score"]["average"])
        for line in train.open(encoding="utf-8")
    ]
    constant = sum(NEUTRAL_SUBMITTED_SCORES.values()) / 3.0

    constant_mse = sum((constant - value) ** 2 for value in gold) / len(gold)
    zero_mse = sum(value**2 for value in gold) / len(gold)

    assert constant_mse < 0.45
    assert zero_mse / constant_mse > 25, (
        "한 행을 0점으로 잃는 비용이 상수 점수 비용의 25배 미만이면 이 정책의 근거가 약해진다"
    )

    # 정수 삼중 중 mean-first 기준 최적에서 벗어나지 않았는지도 함께 고정한다.
    best = min(
        sum(((c + o + e) / 3.0 - value) ** 2 for value in gold) / len(gold)
        for c in range(1, 6)
        for o in range(1, 6)
        for e in range(1, 6)
    )
    assert constant_mse == pytest.approx(best, abs=1e-9)


def test_fill_missing_rationales_never_leaves_an_empty_trait() -> None:
    scores = {"content": 2.0, "organization": 5.0, "expression": 4.0}
    filled, substituted = fill_missing_rationales({"content": "  "}, scores)

    assert set(filled) == set(TRAITS)
    assert all(filled[trait].strip() for trait in TRAITS)
    assert substituted == list(TRAITS)
    assert "5점" in filled["organization"]


def test_every_template_rationale_matches_the_release_gate_pattern() -> None:
    """모든 template 문장이 harness의 탐지 정규식에 걸려야 배포 gate가 실제로 작동한다."""

    import re

    from main_code_submission.degrade import TEMPLATE_RATIONALE_PATTERN

    for trait in TRAITS:
        for score in (1, 2, 3, 4, 5):
            sentence = template_rationale(trait, score)
            assert re.match(TEMPLATE_RATIONALE_PATTERN, sentence), sentence

    # 실제 모델 근거는 걸리지 않아야 한다(오탐 방지).
    assert not re.match(
        TEMPLATE_RATIONALE_PATTERN, "주장이 분명하고 근거가 구체적으로 제시되었다."
    )


def test_harness_copy_of_the_pattern_has_not_drifted() -> None:
    """self-contained harness는 import를 못 하므로 리터럴 동일성을 여기서 고정한다."""

    from pathlib import Path

    from main_code_submission.degrade import TEMPLATE_RATIONALE_PATTERN

    import ast

    source = (
        Path(__file__).resolve().parents[2]
        / "code_for_docker_check_otherserv/evaluate_http.py"
    ).read_text(encoding="utf-8")
    literals = [
        ast.literal_eval(node.value)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "TEMPLATE_RATIONALE_PATTERN"
            for target in node.targets
        )
    ]
    assert literals == [TEMPLATE_RATIONALE_PATTERN], (
        "harness의 TEMPLATE_RATIONALE_PATTERN이 degrade.py와 갈라졌습니다. "
        "강등된 응답이 로컬 gate를 조용히 통과하게 됩니다."
    )


def test_last_resort_outputs_pass_the_official_parser() -> None:
    from main_code_submission.schema import TraitOutput, build_response_json

    scores, rationales = last_resort_outputs()
    body = build_response_json(
        {
            trait: TraitOutput(score=scores[trait], rationale=rationales[trait])
            for trait in TRAITS
        }
    )
    parsed = _official(body)
    assert all(isinstance(parsed[trait]["score"], int) for trait in TRAITS)
