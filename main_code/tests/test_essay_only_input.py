"""문항 지문을 뺀 입력(`essay_only_v1`)의 계약.

왜 이 축을 재는가
    2026-08-20 진단: 문항은 target을 거의 설명하지 못한다. 문항별 평균 예측기의
    공식 400편 RMSE는 0.6470으로 전체 평균 예측기(0.6533)와 사실상 같다. 그런데
    지문은 baseline_v1 입력의 19.4%(평균 333자 / 1,715자)를 차지하고 pooling="mean"
    은 그 token을 essay token과 같은 무게로 평균한다. 문항이 9개뿐이라 같은 문항
    글 1,300여 편이 이 성분을 글자 그대로 공유한다.

    부작용도 있다. content 준거 C1(문제 상황 제시)과 C2(주장)는 정의상 문항에
    상대적이고 content는 우리 최약 영역이다(r=0.687). 그래서 이건 개선이 보장된
    변경이 아니라 **재야 하는 가설**이다. 이 파일은 그 실험이 딴 것을 재지 않도록
    입력 계약만 못 박는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from main_code.config import INPUT_FORMATS, RegressionConfig
from main_code.datasets import (
    ESSAY_ONLY_INPUT_TEMPLATE,
    INPUT_TEMPLATE,
    format_input_segments,
)
import pathlib

# 대회 데이터는 배포가 제한되어 저장소에 없다. 데이터가 있는 환경에서만 돈다.
_DATA_ROOT = pathlib.Path(__file__).resolve().parents[2] / "main_code/datasets/processed_dataset"
pytestmark = pytest.mark.skipif(
    not (_DATA_ROOT / "train.jsonl").is_file(),
    reason="대회 데이터(main_code/datasets/processed_dataset)가 없는 환경",
)


ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE = ROOT / "main_code/datasets/processed_dataset/train.jsonl"


def config(**updates: object) -> RegressionConfig:
    base = RegressionConfig(model_id="skt/A.X-4.0-Light").with_updates(
        essay_surface="official_raw",
        pooling="mean",
        training_mode="lora_only",
        input_format="baseline_v1",
    )
    return base.with_updates(**updates) if updates else base


def row() -> dict:
    with SAMPLE.open(encoding="utf-8") as handle:
        return json.loads(handle.readline())


def test_format_is_registered() -> None:
    assert "essay_only_v1" in INPUT_FORMATS


def test_prompt_text_is_absent_from_the_input() -> None:
    sample = row()
    segments = format_input_segments(
        sample, config(input_format="essay_only_v1"), essay_surface="official_raw"
    )
    assert "[prompt_text]" not in segments.text
    assert sample["prompt"][:40] not in segments.text


def test_essay_is_still_the_final_span() -> None:
    """essay_span이 어긋나면 문단경계 supervision과 essay pooling이 조용히 틀린다."""

    sample = row()
    segments = format_input_segments(
        sample, config(input_format="essay_only_v1"), essay_surface="official_raw"
    )
    start, end = segments.essay_span
    assert segments.text[start:end] == sample["essay_surfaces"]["official_raw"]
    assert end == len(segments.text)
    assert segments.scoring_end == len(segments.text)


def test_baseline_still_contains_the_prompt() -> None:
    """기존 경로가 바뀌지 않았음을 같은 파일에서 확인한다."""

    sample = row()
    segments = format_input_segments(
        sample, config(), essay_surface="official_raw"
    )
    assert "[prompt_text]" in segments.text
    assert sample["prompt"][:40] in segments.text


def test_input_is_shorter_by_about_the_prompt_length() -> None:
    sample = row()
    long = format_input_segments(sample, config(), essay_surface="official_raw")
    short = format_input_segments(
        sample, config(input_format="essay_only_v1"), essay_surface="official_raw"
    )
    saved = len(long.text) - len(short.text)
    # 절약분은 지문 길이 ± 헤더 문구 차이다. 지문의 절반보다 크면 충분하다.
    assert saved > len(sample["prompt"]) * 0.5


def test_row_without_a_prompt_still_formats() -> None:
    """지문 필드가 없는 행에서도 동작해야 한다. prompt_text(row)를 부르면 KeyError다."""

    sample = {key: value for key, value in row().items() if key != "prompt"}
    segments = format_input_segments(
        sample, config(input_format="essay_only_v1"), essay_surface="official_raw"
    )
    assert segments.text.endswith(sample["essay_surfaces"]["official_raw"])


def test_paragraph_boundary_multitask_is_allowed() -> None:
    """c02는 문단경계 loss 0.1을 쓴다. 이게 막히면 arm이 두 축을 동시에 바꾼다.

    2026-09-29: 최종 제출이 ABCD로 바뀌면서 organization_pooling 기본값이
    paragraph_mean이 됐다. 문단경계 multi-task는 essay_mask를 요구하는 두 축과
    배타이므로(collator가 두 supervision을 동시에 만들지 못한다), 이 게이트를
    단독으로 검증하려면 organization_pooling을 shared로 명시해야 한다.
    """

    spec = config(
        input_format="essay_only_v1",
        organization_pooling="shared",
        paragraph_boundary_loss_weight=0.1,
    )
    assert spec.input_format == "essay_only_v1"
    assert spec.paragraph_boundary_loss_weight == 0.1


def test_other_formats_still_rejected_by_the_boundary_gate() -> None:
    with pytest.raises(ValueError, match="paragraph boundary"):
        config(input_format="source_aware_v1", paragraph_boundary_loss_weight=0.1)


def test_template_has_no_prompt_placeholder() -> None:
    assert "{prompt}" in INPUT_TEMPLATE
    assert "{prompt}" not in ESSAY_ONLY_INPUT_TEMPLATE
    assert ESSAY_ONLY_INPUT_TEMPLATE.rstrip().endswith("{essay}")


def test_unknown_format_is_rejected() -> None:
    with pytest.raises(ValueError, match="input_format"):
        config(input_format="essay_only_v2")


# --- 지문 dropout (2026-08-20) -----------------------------------------------
#
# 왜 전량 제거와 따로 재는가: content 준거 C1·C2는 정의상 논제에 상대적이고
# content는 우리 최약 영역이다(r=0.687). 지문을 아예 없애면 그쪽이 상할 수 있다.
# dropout은 baseline_v1(p=0)과 essay_only_v1(p=1) 사이의 연속 축이라, 미학습 문항의
# 손해와 content 손해를 맞바꾸는 지점을 찾을 수 있다.
def test_dropout_is_off_by_default() -> None:
    assert config().prompt_dropout_probability == 0.0


def test_dropout_probability_range() -> None:
    for value in (0.0, 0.25, 1.0):
        assert config(prompt_dropout_probability=value).prompt_dropout_probability == value
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError, match="prompt_dropout_probability"):
            config(prompt_dropout_probability=bad)


def test_dropout_requires_a_format_with_a_known_prompt_slot() -> None:
    # 문단경계 게이트가 먼저 걸리지 않도록 끄고, 지문 위치가 다른 format을 준다.
    with pytest.raises(ValueError, match="prompt_dropout_probability"):
        config(
            input_format="source_aware_v1",
            paragraph_boundary_loss_weight=0.0,
            prompt_dropout_probability=0.3,
        )


def test_marked_row_renders_without_the_prompt() -> None:
    from main_code.datasets import _TrainingSurfaceRow

    sample = row()
    spec = config(prompt_dropout_probability=0.5)
    marked = _TrainingSurfaceRow(sample, essay_surface="official_raw", drop_prompt=True)
    segments = format_input_segments(marked, spec, essay_surface="official_raw")
    assert "[prompt_text]" not in segments.text
    start, end = segments.essay_span
    assert segments.text[start:end] == sample["essay_surfaces"]["official_raw"]


def test_unmarked_row_keeps_the_prompt_even_when_dropout_is_on() -> None:
    """확률은 dataset이 뽑는다. format 함수는 표시만 본다."""

    from main_code.datasets import _TrainingSurfaceRow

    sample = row()
    spec = config(prompt_dropout_probability=0.9)
    marked = _TrainingSurfaceRow(sample, essay_surface="official_raw", drop_prompt=False)
    assert "[prompt_text]" in format_input_segments(
        marked, spec, essay_surface="official_raw"
    ).text
    # 표시가 아예 없는 평가 경로 row도 지문을 유지해야 한다.
    assert "[prompt_text]" in format_input_segments(
        sample, spec, essay_surface="official_raw"
    ).text


def test_validation_split_never_drops_the_prompt() -> None:
    """평가에서 지문이 빠지면 학습/추론 계약이 달라져 조용히 틀린다."""

    import collections

    from main_code.datasets import EssayRegressionDataset

    sample = row()
    dataset = EssayRegressionDataset.__new__(EssayRegressionDataset)
    dataset.config = config(prompt_dropout_probability=1.0)
    dataset.split = "validation"
    dataset.surface_view_counts = collections.Counter()
    dataset.rows = [sample]
    selected = dataset[0]
    assert not getattr(selected, "drop_prompt", False)
    assert "[prompt_text]" in format_input_segments(
        selected, dataset.config, essay_surface="official_raw"
    ).text


def test_train_split_drops_at_the_configured_rate() -> None:
    """p를 바꿨는데 비율이 안 바뀌면 배선이 끊긴 것이다."""

    import collections
    import random

    from main_code.datasets import EssayRegressionDataset

    sample = row()
    for probability, low, high in ((0.0, 0.0, 0.0), (0.3, 0.2, 0.4), (1.0, 1.0, 1.0)):
        dataset = EssayRegressionDataset.__new__(EssayRegressionDataset)
        dataset.config = config(prompt_dropout_probability=probability)
        dataset.split = "train"
        dataset.surface_view_counts = collections.Counter()
        dataset.rows = [sample]
        random.seed(43)
        dropped = sum(
            bool(getattr(dataset[0], "drop_prompt", False)) for _ in range(600)
        )
        assert low <= dropped / 600 <= high, (probability, dropped)
