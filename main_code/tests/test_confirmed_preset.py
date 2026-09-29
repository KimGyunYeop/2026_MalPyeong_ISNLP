"""현재 final, 역사적 Y1, frozen baseline의 3층 설정 계약을 고정한다.

`RegressionConfig()`와 no-config train은 ``confirmed_final.json``과 같다. 역사적 Y1과
pre-Y1 baseline은 각각 완전한 preset을 명시할 때만 선택된다. 일반 부분 JSON은 현재 final을
상속한다.

과거 실패: preset이 `_note`로 확정 근거를 파일 안에 적었는데 `with_updates`가 dataclass
field가 아니라며 `unknown config fields: ['_note']`로 즉시 죽었다. `--help`는 config 로드
전에 나가기 때문에 정상으로 보였다.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from main_code.config import RegressionConfig, legacy_baseline_config, load_config
from main_code.train import parse_args, resolved_config

MAIN_CODE = Path(__file__).resolve().parents[1]
CONFIGS = MAIN_CODE / "configs"
FINAL = CONFIGS / "confirmed_final.json"
Y1 = CONFIGS / "confirmed_y1.json"
BASELINE = CONFIGS / "baseline.json"
# 2026-08-25 최종 제출본의 채점 멤버. 8 seed는 --seed만 다르므로 s42의 resolved
# config가 계열 전체의 레시피다. 이전 확정본(c02/A.X-4.0-Light)은 provenance로
# configs/confirmed_c02_ax4.json에 보존한다.
# 2026-09-29: 최종 제출은 A+B가 아니라 ABCD(= A 분포 head + B soft-Spearman +
# C 평가자 2인 보조 + D 문단 구간 평균)로 확정됐다. 제출 빌드는 다른 서버에서
# 만들었으므로 이 레포에는 같은 recipe의 재실험 run을 근거로 둔다.
FINAL_SOURCE = (
    MAIN_CODE
    / "results/qwen_result_analysis/p1_plain_baseline_s42_v1"
    / "c4_ABCD_s42/qwen35_9b/5e33de7af0c8/resolved_config.json"
)
FINAL_SHA256 = "968f91c0e1087061ce8551e9516088abbd0d8fcc79ffb301a683b1d11d3c380a"


def _normalize(value: object) -> object:
    """JSON은 tuple을 list로 저장하므로 비교 전에 같은 모양으로 만든다."""

    if isinstance(value, tuple):
        return [list(item) if isinstance(item, (list, tuple)) else item for item in value]
    return value


def _executable_json_fields(path: Path) -> set[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return {key for key in value if not key.startswith("_")}


def test_final_and_historical_presets_are_complete() -> None:
    expected_fields = {field.name for field in dataclasses.fields(RegressionConfig)}
    additive_noop_fields = {
        key
        for key, value in dataclasses.asdict(RegressionConfig()).items()
        if key.startswith("quantization_")
        or key.startswith("quantized_")
        or key
        in {
            "checkpoint_retention",
            "secondary_checkpoint_metric",
            "ordinal_steps",
            "ordinal_blend_weight",
            "ordinal_loss_weight",
            # 근거 pooling 축. 기본 pooling="mean"에서는 세 값이 읽히지도 않아
            # preset의 실행 의미가 그대로다. preset JSON을 고치면 c02 resolved
            # config와의 동일성 검사가 깨지므로 여기서 예외로 둔다.
            "analysis_pool_weight",
            "analysis_pool_learnable",
            "analysis_pool_per_trait",
            # 2026-08-20 꼬리 가중 축. 기본 tail_weight_target_sigma=0.0이면
            # sample_tail_weights가 None을 돌려주고 loss 호출부가 element_weights
            # 없이 부르던 경로를 그대로 타므로 historical preset의 실행 의미가
            # 바뀌지 않는다. FINAL preset은 c02 resolved config와 byte 동일해야
            # 하므로 JSON을 고치는 대신 여기서 예외로 둔다.
            "tail_weight_target_sigma",
            "tail_weight_reference_mean",
            "tail_weight_reference_sigma",
            "tail_weight_max",
            # 2026-08-20 미학습 문항 보류 / 지문 dropout / 문항 적대적 학습.
            # 각각 빈 문자열, 0.0, 0.0에서 학습 행 선택·입력 문자열·parameter 수가
            # 전부 그대로이므로 historical preset의 실행 의미가 바뀌지 않는다.
            "unseen_prompt_holdout",
            "prompt_dropout_probability",
            "prompt_adversary_weight",
            "prompt_adversary_classes",
            # pooled 정규화. "none"에서 연산이 추가되지 않는다.
            "pooled_normalization",
            # 문항 GroupDRO. step_size=0에서 가중치가 만들어지지 않는다.
            "group_dro_step_size",
            # clip 임계는 기존 하드코딩 1.0과 같은 기본값이고, 수준 prototype은
            # weight=0에서 buffer가 읽히지 않으므로 preset의 실행 의미가 그대로다.
            "max_grad_norm",
            "level_prototype_weight",
            "level_prototype_count",
            "level_prototype_momentum",
            "level_prototype_temperature",
        }
    }
    # 데이터 경로는 방법이 아니라 실행 환경이다. 2026-09-29 GitHub 공개를 위해
    # FINAL에서 뺐다 — resolved config에 박혀 있던 절대경로가 다른 머신에서는
    # 존재하지 않는 경로를 가리키기 때문이다. dataclass 기본값이 __file__ 기준으로
    # 계산하므로 빼두면 어느 체크아웃에서도 올바른 경로를 얻는다.
    environment_path_fields = {"dataset_root", "extended_data_dir"}

    # FINAL은 제출본 resolved config이므로 위 경로 두 개를 뺀 실행 필드를 **전부**
    # 갖는다. Y1/BASELINE은 그 필드들이 생기기 전에 고정된 역사적 preset이라
    # additive no-op만 빠질 수 있다.
    assert isinstance(load_config(FINAL), RegressionConfig)
    assert _executable_json_fields(FINAL) == expected_fields - environment_path_fields
    for path in (Y1, BASELINE):
        assert isinstance(load_config(path), RegressionConfig)
        # Y1/BASELINE은 동결된 역사적 preset이라 경로 필드를 그대로 갖는다.
        assert _executable_json_fields(path) == expected_fields - additive_noop_fields


def test_final_preset_is_the_submitted_qwen_resolved_config() -> None:
    assert hashlib.sha256(FINAL.read_bytes()).hexdigest() == FINAL_SHA256
    if FINAL_SOURCE.is_file():
        # 재실험 run은 output_dir 등 run 고유 필드가 달라 byte 동일성을 요구하지
        # 않는다. 대신 ABCD를 정의하는 네 축이 정확히 같은지 확인한다.
        source = json.loads(FINAL_SOURCE.read_text(encoding="utf-8"))
        final = json.loads(FINAL.read_text(encoding="utf-8"))
        for field in (
            "score_head",
            "listwise_loss",
            "detail_head_mode",
            "detail_rater_set_loss_weight",
            "detail_expected_loss_weight",
            "organization_pooling",
        ):
            assert final[field] == source[field], field


def test_dataclass_and_no_config_train_are_the_confirmed_final() -> None:
    expected = dataclasses.asdict(load_config(FINAL))
    assert dataclasses.asdict(RegressionConfig().validate()) == expected
    args = Namespace(config=None, baseline=False, model=None)
    assert dataclasses.asdict(resolved_config(args)) == expected


def test_historical_y1_is_explicit_and_unchanged() -> None:
    y1 = load_config(Y1)
    assert y1 != load_config(FINAL)
    assert y1.score_head == "regression"
    assert y1.distribution_loss_weight == 0.0
    assert y1.distribution_label_smoothing == 0.0
    assert y1.detail_final_source == "criterion"
    assert y1.trait_average_loss_weight == 0.5
    assert y1.best_checkpoint_metric == "official_rmse"
    assert y1.seed == 42
    args = Namespace(config=str(Y1), baseline=False, model=None)
    assert resolved_config(args) == y1


def test_frozen_legacy_baseline_is_explicit_and_neutral() -> None:
    baseline = load_config(BASELINE)
    assert baseline == legacy_baseline_config()
    assert baseline.training_mode == "head_only"
    assert baseline.detail_head_mode == "none"
    assert baseline.detail_final_source == "direct"
    assert baseline.paragraph_boundary_loss_weight == 0.0
    assert baseline.listwise_loss == "none"
    assert baseline.listwise_loss_weight == 0.0
    assert baseline.trait_average_loss_weight == 0.0
    assert (baseline.lora_r, baseline.lora_alpha) == (16, 32)
    assert baseline.lora_include_mlp is False


def test_partial_config_always_inherits_final_defaults(tmp_path: Path) -> None:
    path = tmp_path / "partial.json"
    path.write_text(
        json.dumps({"schema_version": 1, "batch_size": 16}), encoding="utf-8"
    )
    loaded = load_config(path)
    assert loaded == RegressionConfig().with_updates(batch_size=16)
    assert loaded.model_id == "Qwen/Qwen3.5-9B"
    assert loaded.model_slug == "qwen35_9b"
    assert loaded.score_head == "distribution"
    assert loaded.distribution_loss_weight == 1.0
    assert loaded.distribution_label_smoothing == 0.0
    assert loaded.detail_head_mode == "rater_set"
    assert loaded.detail_final_source == "direct"
    assert loaded.organization_pooling == "paragraph_mean"
    assert loaded.listwise_loss == "soft_spearman"
    assert loaded.trait_average_loss_weight == 0.0
    assert loaded.paragraph_boundary_loss_weight == 0.0
    assert (loaded.max_train_steps, loaded.eval_steps) == (1104, 64)
    assert loaded.seed == 42
    assert (loaded.lora_r, loaded.lora_alpha) == (32, 64)
    assert loaded.lora_include_mlp is True
    assert loaded.best_checkpoint_metric == "official_matched_rmse"
    assert loaded.seed == 42


def test_retired_model_cache_path_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "old_checkpoint_config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_cache_dir": "/obsolete/project/.cache/huggingface/hub",
            }
        ),
        encoding="utf-8",
    )
    loaded = load_config(path)
    assert loaded == RegressionConfig().validate()
    assert "model_cache_dir" not in {field.name for field in dataclasses.fields(loaded)}


def test_baseline_alias_resolves_the_frozen_preset() -> None:
    args = Namespace(config=None, baseline=True, model=None)
    assert resolved_config(args) == load_config(BASELINE)


def test_baseline_and_config_cli_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--baseline",
            "--config",
            str(FINAL),
            "--output-dir",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2


def test_underscore_keys_are_human_comments_not_fields(tmp_path: Path) -> None:
    """`_`로 시작하는 key는 무시하고, 진짜 오타는 여전히 실패해야 한다."""

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "model_id": "skt/A.X-4.0-Light",
                "training_mode": "lora_only",
                "_note": "사람이 읽는 확정 근거",
                "_provenance": {"run": "y1_f4_rank_metric"},
            }
        ),
        encoding="utf-8",
    )
    assert load_config(path).training_mode == "lora_only"

    path.write_text(
        json.dumps({"model_id": "skt/A.X-4.0-Light", "poolingg": "mean"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown config fields"):
        load_config(path)


def test_final_preset_pins_every_confirmed_axis() -> None:
    """확정 구조의 축이 preset에 실제로 들어 있는지 값으로 확인한다.

    여기 나열한 값이 `METHOD_CONFIRMED.md`가 설명하는 바로 그 구조다. preset을 고치면 이
    test가 먼저 깨지므로 문서와 코드가 조용히 갈라지지 않는다.
    """

    config = load_config(FINAL)
    assert config.model_id == "Qwen/Qwen3.5-9B"
    assert config.model_slug == "qwen35_9b"
    # 표현 추출
    assert (config.pooling, config.layer_aggregation) == ("mean", "last")
    assert config.normalize_features is False
    # ABCD: 최종 score는 direct 5-class head가 만들고, 익명 평가자 2인 branch
    # 18개 분류기가 9준거를 보조감독한다(요소 C).
    assert (config.detail_head_mode, config.detail_final_source) == (
        "rater_set",
        "direct",
    )
    # 요소 D: organization만 이중 공백 단서로 나눈 문단 평균을 쓴다.
    assert config.organization_pooling == "paragraph_mean"
    assert config.score_head == "distribution"
    assert config.categorical_readout == "expectation"
    # 손실 구성
    assert config.mse_loss_weight == 1.0
    assert config.distribution_loss_weight == 1.0
    assert config.distribution_label_smoothing == 0.0
    # rater_set은 단일 재평가자 행의 unmatched branch를 묶을 expected 손실을
    # 요구하므로 두 가중이 함께 켜진다(G5-1 + G5-3).
    assert config.detail_expected_loss_weight == 0.25
    assert config.detail_rater_set_loss_weight == 0.25
    assert config.trait_average_loss_weight == 0.0
    assert (config.listwise_loss, config.listwise_loss_weight) == ("soft_spearman", 0.2)
    assert config.ranking_target == "per_trait"
    assert config.paragraph_boundary_loss_weight == 0.0
    assert config.pairwise_loss == "none"
    # 순위 로스는 공식 400편 전체 순위를 겨냥하므로 batch가 문항에 묶이면 안 된다.
    assert config.inbatch_sampling == "random"
    # 학습 조건
    assert (config.lora_r, config.lora_alpha) == (32, 64)
    assert config.lora_include_mlp is True
    assert (config.max_train_steps, config.eval_steps) == (1104, 64)
    assert config.essay_surface == "official_raw"
    assert config.best_checkpoint_metric == "official_matched_rmse"
    # 제출 후처리
    assert config.score_postprocess == "average_matched"
    assert config.seed == 42


def test_expanded_y1_differs_from_original_y1_only_in_lora_defaults() -> None:
    """보존한 expanded-Y1과 원래 Y1 artifact의 차이는 LoRA 세 축뿐이다.

    run 폴더가 없는 환경(컨테이너, 새 클론)에서는 건너뛴다.
    """

    resolved_path = (
        Path(__file__).resolve().parents[1]
        / "results/new_proposed/ax4light_final_combo_s1288_official_raw_v1"
        / "y1_f4_rank_metric/ax4_light/5f475fbd3458/resolved_config.json"
    )
    if not resolved_path.is_file():
        pytest.skip(f"y1 run artifact가 없습니다: {resolved_path}")

    preset = dataclasses.asdict(load_config(Y1))
    # Historical artifacts may contain retired compatibility fields such as the
    # old project-local model_cache_dir. Compare through the loader so only the
    # current executable schema is part of the y1 contract.
    resolved = dataclasses.asdict(load_config(resolved_path))
    mismatched = sorted(
        key
        for key in set(preset) | set(resolved)
        if _normalize(preset.get(key)) != _normalize(resolved.get(key))
    )
    assert mismatched == ["lora_alpha", "lora_include_mlp", "lora_r"]
    assert (preset["lora_r"], preset["lora_alpha"], preset["lora_include_mlp"]) == (
        32,
        64,
        True,
    )
    assert (
        resolved["lora_r"],
        resolved["lora_alpha"],
        resolved["lora_include_mlp"],
    ) == (16, 32, False)
