"""미학습 문항 보류와 문항별 지표의 계약.

왜 이 축이 필요한가
    2025 최종보고서 표 V-6: 채점 데이터 4,000편은 Q1 654 / Q2 189 / Q3 314 /
    Q11 1,512 / Q12 1,331편이다. Q11·Q12는 우리 학습(Q1~Q9)에 없고 71.1%를 차지한다.
    그런데 공개 validation 400편은 학습에서 **본** 9문항의 새 에세이라 그 격차를
    측정하지 못한다. 게다가 그 400편의 문항 구성은 우리 학습 편수에 비례해서
    (Q1 25편 … Q5 51편) essay micro 평균으로 고르면 우리 validation 구성에만 맞춘
    모델을 고른다.

    `unseen_prompt_holdout`은 문항을 학습에서 완전히 빼서 validation의 같은 문항을
    "본 적 없는 문항" 평가로 만든다. `prompt_macro_*`/`worst_prompt_*`는 문항 구성에
    불변인 저울이다.

위험한 종류의 변경이므로 세 가지를 못 박는다.
  1. 기본값(빈 문자열)에서 학습 행 선택이 **한 행도** 바뀌지 않는다.
  2. 보류한 문항이 학습 행에서 완전히 사라진다(파생 row 포함 누수 없음).
  3. 보류가 없으면 `unseen_prompt_*` 키를 **만들지 않는다** — 있는 척하면 선택이 조용히 틀어진다.
"""

from __future__ import annotations

import numpy as np
import pytest

from main_code.config import BEST_CHECKPOINT_METRICS, RegressionConfig
from main_code.datasets import (
    exclude_unseen_prompt_rows,
    parse_unseen_prompt_holdout,
    prompt_group_id,
)
from main_code.train import compute_trainer_metrics, prompt_group_metrics


def config(**updates: object) -> RegressionConfig:
    base = RegressionConfig(model_id="skt/A.X-4.0-Light")
    return base.with_updates(**updates) if updates else base


def rows(prompts: list[str]) -> list[dict]:
    return [{"prompt_num": p, "id": f"e{index}"} for index, p in enumerate(prompts)]


# --- 설정 계약 ---------------------------------------------------------------
def test_disabled_by_default() -> None:
    assert config().unseen_prompt_holdout == ""
    assert parse_unseen_prompt_holdout("") == ()


def test_default_keeps_every_training_row() -> None:
    original = rows([f"Q{i}" for i in range(1, 10)])
    kept = exclude_unseen_prompt_rows(original, parse_unseen_prompt_holdout(""))
    assert [r["id"] for r in kept] == [r["id"] for r in original]


def test_holdout_removes_exactly_those_prompts() -> None:
    original = rows(["Q1", "Q5", "Q5", "Q9", "Q2"])
    kept = exclude_unseen_prompt_rows(original, ("Q5",))
    assert [r["prompt_num"] for r in kept] == ["Q1", "Q9", "Q2"]


def test_multiple_prompts_and_whitespace() -> None:
    assert parse_unseen_prompt_holdout(" Q8 , Q9 ") == ("Q8", "Q9")
    kept = exclude_unseen_prompt_rows(rows(["Q8", "Q9", "Q1"]), ("Q8", "Q9"))
    assert [r["prompt_num"] for r in kept] == ["Q1"]


def test_duplicate_and_empty_entries_rejected() -> None:
    with pytest.raises(ValueError, match="중복"):
        config(unseen_prompt_holdout="Q5,Q5")
    with pytest.raises(ValueError, match="빈 항목"):
        config(unseen_prompt_holdout="Q5,")


def test_unseen_metric_requires_a_holdout() -> None:
    """보류 없이 unseen 지표로 고르면 아무 값도 없어 조용히 잘못 고른다."""

    with pytest.raises(ValueError, match="unseen_prompt_holdout"):
        config(best_checkpoint_metric="unseen_prompt_rmse")
    assert config(
        unseen_prompt_holdout="Q5", best_checkpoint_metric="unseen_prompt_rmse"
    ).best_checkpoint_metric == "unseen_prompt_rmse"


def test_new_metrics_are_selectable() -> None:
    for name in (
        "prompt_macro_rmse",
        "worst_prompt_rmse",
        "prompt_macro_spearman",
        "unseen_prompt_rmse",
        "unseen_prompt_spearman",
    ):
        assert name in BEST_CHECKPOINT_METRICS


# --- 문항 id 계약 ------------------------------------------------------------
def test_prompt_group_id_is_readable_and_stable() -> None:
    assert prompt_group_id({"prompt_num": "Q1"}) == 1
    assert prompt_group_id({"prompt_num": "Q12"}) == 12
    assert prompt_group_id({"prompt_num": "12"}) == 12
    assert prompt_group_id({"prompt_num": ""}) == -1
    # 형식이 달라도 같은 문자열은 항상 같은 값이어야 한다(프로세스 간에도).
    first = prompt_group_id({"prompt_num": "특수문항-A"})
    assert first == prompt_group_id({"prompt_num": "특수문항-A"})
    assert first != prompt_group_id({"prompt_num": "특수문항-B"})


# --- 지표 계약 ---------------------------------------------------------------
def _sample(count: int = 360):
    generator = np.random.default_rng(43)
    groups = np.repeat(np.arange(1, 10), count // 9)
    truth = np.clip(generator.normal(3.4, 0.65, count), 1, 5)
    prediction = np.clip(truth + generator.normal(0, 0.4, count), 1, 5)
    return prediction, truth, groups


def test_macro_is_the_mean_of_per_prompt_rmse() -> None:
    prediction, truth, groups = _sample()
    metrics = prompt_group_metrics(prediction, truth, groups)
    expected = np.mean(
        [
            np.sqrt(np.mean((prediction[groups == key] - truth[groups == key]) ** 2))
            for key in sorted(set(groups.tolist()))
        ]
    )
    assert metrics["prompt_macro_rmse"] == pytest.approx(float(expected))
    assert metrics["worst_prompt_rmse"] >= metrics["prompt_macro_rmse"]


def test_macro_is_invariant_to_prompt_mix() -> None:
    """이게 이 지표를 쓰는 이유다. micro는 구성이 바뀌면 값이 바뀐다."""

    prediction, truth, groups = _sample()
    keep = (groups != 5) | (np.cumsum(groups == 5) <= 4)  # Q5를 4편만 남긴다
    macro_full = prompt_group_metrics(prediction, truth, groups)["prompt_macro_rmse"]
    macro_thin = prompt_group_metrics(
        prediction[keep], truth[keep], groups[keep]
    )["prompt_macro_rmse"]
    micro_full = float(np.sqrt(np.mean((prediction - truth) ** 2)))
    micro_thin = float(
        np.sqrt(np.mean((prediction[keep] - truth[keep]) ** 2))
    )
    # Q5가 5편 미만이 되면 macro 집계에서 빠지므로 남은 8문항 평균과 같아야 한다.
    assert abs(macro_thin - macro_full) < abs(micro_thin - micro_full) + 1e-9


def test_unseen_keys_absent_without_holdout() -> None:
    prediction, truth, groups = _sample()
    metrics = prompt_group_metrics(prediction, truth, groups)
    assert "unseen_prompt_rmse" not in metrics
    assert "unseen_prompt_spearman" not in metrics


def test_unseen_keys_restrict_to_the_held_out_prompts() -> None:
    prediction, truth, groups = _sample()
    metrics = prompt_group_metrics(prediction, truth, groups, ("Q5", "Q9"))
    mask = np.isin(groups, [5, 9])
    expected = float(np.sqrt(np.mean((prediction[mask] - truth[mask]) ** 2)))
    assert metrics["unseen_prompt_rmse"] == pytest.approx(expected)


def test_small_prompt_groups_are_skipped() -> None:
    """5편 미만 문항의 RMSE는 잡음이라 macro/worst를 흔든다."""

    prediction = np.array([3.0, 3.0, 4.0, 2.0, 3.0, 3.0, 4.0])
    truth = np.array([3.1, 2.9, 4.2, 2.1, 3.3, 2.8, 3.9])
    groups = np.array([1, 1, 1, 1, 1, 2, 2])  # 문항 2는 2편뿐
    metrics = prompt_group_metrics(prediction, truth, groups)
    only_first = float(
        np.sqrt(np.mean((prediction[groups == 1] - truth[groups == 1]) ** 2))
    )
    assert metrics["prompt_macro_rmse"] == pytest.approx(only_first)


# --- Trainer 배선 계약 -------------------------------------------------------
class _Prediction:
    def __init__(self, predictions, label_ids) -> None:
        self.predictions, self.label_ids = predictions, label_ids


def _trait_sample(count: int = 360):
    generator = np.random.default_rng(43)
    labels = np.clip(generator.normal(3.4, 0.65, (count, 3)), 1, 5)
    scores = np.clip(labels + generator.normal(0, 0.4, (count, 3)), 1, 5)
    groups = np.repeat(np.arange(1, 10), count // 9)
    return scores, labels, groups


def test_three_item_label_payload_produces_prompt_metrics() -> None:
    scores, labels, groups = _trait_sample()
    result = compute_trainer_metrics(
        _Prediction(scores, (labels, labels.mean(axis=1), groups)),
        unseen_prompt_holdout=("Q5",),
    )
    assert "prompt_macro_rmse" in result
    assert "unseen_prompt_rmse" in result


def test_two_item_label_payload_still_works() -> None:
    """과거 caller와 unit test는 (labels, average_labels)만 준다."""

    scores, labels, _ = _trait_sample()
    result = compute_trainer_metrics(_Prediction(scores, (labels, labels.mean(axis=1))))
    assert "official_matched_rmse" in result
    assert "prompt_macro_rmse" not in result
    assert "unseen_prompt_rmse" not in result


def test_label_payload_of_unexpected_length_is_rejected() -> None:
    scores, labels, groups = _trait_sample()
    with pytest.raises(ValueError, match="label_ids"):
        compute_trainer_metrics(
            _Prediction(scores, (labels, labels.mean(axis=1), groups, groups))
        )


def test_collator_emits_prompt_group_ids() -> None:
    """지표가 문항을 못 보면 위 모든 계약이 무의미하다."""

    import inspect

    from main_code import datasets as datasets_module

    source = inspect.getsource(datasets_module.RegressionCollator)
    assert '"prompt_group_ids"' in source


def test_forward_accepts_prompt_group_ids() -> None:
    """Trainer는 batch의 모든 key를 forward로 넘긴다. 안 받으면 학습이 죽는다."""

    import inspect

    from main_code.models import RegressionScorer

    assert "prompt_group_ids" in inspect.signature(RegressionScorer.forward).parameters
