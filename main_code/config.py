from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any


# Package paths and prepared data profiles -----------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PACKAGE_ROOT / "datasets"
RAW_DATA_DIR = DATA_ROOT / "raw_dataset"
MODEL_CATALOG_PATH = PACKAGE_ROOT / "candidiate_model_list" / "all_models.tsv"
LEGACY_BASELINE_CONFIG_PATH = PACKAGE_ROOT / "configs" / "baseline.json"

# NIKL 전체에서 공개 validation만 뺀 full이 기본 학습 데이터다. leaked는
# 누수 진단 전용이므로 RegressionConfig의 명시적인 허용 없이는 사용할 수 없다.
DEFAULT_PRIMARY_DATA_PROFILE = "full"
PRIMARY_DATA_PROFILES = {
    "full": DATA_ROOT / "processed_dataset",
    "official": DATA_ROOT / "processed_dataset_only_official_competition",
    "validation_leaked": DATA_ROOT / "processed_dataset_validation_leaked",
}
# 조금 더 긴 이름도 외부 코드에서 읽기 좋도록 같은 immutable 경로를 노출한다.
PRIMARY_DATASET_PROFILES = PRIMARY_DATA_PROFILES

AIHUB_DATASETS = {
    "aihub24_essay": DATA_ROOT / "processed_dataset_aihub_external_에세이",
    "aihub25_descriptive": DATA_ROOT / "processed_dataset_aihub_external_서술",
    "aihub26_essay": DATA_ROOT / "processed_dataset_aihub_external_논술",
    "aihub27_topic": DATA_ROOT / "processed_dataset_aihub_external_주제별",
}
NIKL_SUMMARY_DATASETS = {
    "nikl24_summary": DATA_ROOT / "processed_dataset_nikl_external_summary_2024",
    "nikl25_argumentative_summary": (
        DATA_ROOT / "processed_dataset_nikl_external_argumentative_summary_2025"
    ),
    "nikl25_cooperative_summary": (
        DATA_ROOT / "processed_dataset_nikl_external_cooperative_summary_2025"
    ),
}
EXTERNAL_DATASETS = {**AIHUB_DATASETS, **NIKL_SUMMARY_DATASETS}
AIHUB_DATASET_ALIASES = tuple(AIHUB_DATASETS)
EXTERNAL_DATASET_ALIASES = tuple(EXTERNAL_DATASETS)
# 새 학습에서는 NIKL이 primary지만, 이전 checkpoint config의 inference는 계속
# 열 수 있도록 과거 external 이름만 validation 단계에서 허용한다.
LEGACY_EXTERNAL_DATASET_ALIASES = ("nikl_grading",)


def primary_data_dir(profile: str = DEFAULT_PRIMARY_DATA_PROFILE) -> Path:
    try:
        return PRIMARY_DATA_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(
            f"unknown primary data profile={profile!r}; "
            f"choices={tuple(PRIMARY_DATA_PROFILES)}"
        ) from exc


def external_data_dir(alias: str) -> Path:
    try:
        return EXTERNAL_DATASETS[alias]
    except KeyError as exc:
        raise ValueError(
            f"unknown external dataset={alias!r}; choices={EXTERNAL_DATASET_ALIASES}"
        ) from exc


# Experiment option names ----------------------------------------------------
TRAITS = ("content", "organization", "expression")
DETAIL_CRITERIA_BY_TRAIT = (
    ("content_1", "content_2", "content_3", "content_4", "content_5"),
    ("organization_1", "organization_2"),
    ("expression_1", "expression_2"),
)
DETAIL_CRITERIA = tuple(
    criterion
    for trait_criteria in DETAIL_CRITERIA_BY_TRAIT
    for criterion in trait_criteria
)
TRAINING_MODES = ("head_only", "lora_only", "two_stage")
POOLING_MODES = (
    "mean",
    "last",
    "first",
    "essay_mean",
    "essay_attention",
    # 분석/근거 접미사가 붙는 dual-view 입력용. "mean"은 전 토큰을 균등 평균하므로
    # 접미사가 토큰 비중(2026-08-18 실측 17~21%)만큼만 기여하고 그중 실제 판단
    # 정보는 5% 아래로 묻힌다. 이 두 모드는 접미사 구간(`analysis_mask==1`)과 나머지
    # 채점 구간을 각각 pooling한 뒤 `analysis_pool_weight` 지분으로 섞어 접미사에
    # 토큰 수와 무관한 지분을 준다. 접미사가 없는 view는 지분이 0으로 강제되어
    # "mean"과 bit-exact하게 같아지므로 두 view가 같은 head를 공유할 수 있다.
    #
    # 구간 식별에 essay_mask가 아니라 별도 `analysis_mask`를 쓰는 이유: c02는
    # paragraph boundary head를 켜고 있고 RegressionCollator의 분기가 배타적이라
    # 그 경우 essay_mask 자체가 batch에 없다. 접미사 길이를 아는 것은 접미사를
    # 붙인 collator뿐이므로 그쪽이 명시 mask를 낸다.
    "analysis_mix",
    # analysis_mix의 접미사 평균을 trait별 attention 가중평균으로 바꾼다. projection을
    # 0으로 초기화하므로 첫 forward는 균등 가중, 즉 analysis_mix와 정확히 같다.
    # 근거 DSL은 기준별 구획이 있어 content/organization/expression이 서로 다른
    # 구획을 봐야 하는데 평평한 평균으로는 그 선택이 불가능하다.
    "analysis_attention_mix",
)

# 접미사 구간을 별도 지분으로 섞는 pooling 모드. 코드 여러 곳에서 같은 집합을 봐야 한다.
ANALYSIS_MIX_POOLINGS = ("analysis_mix", "analysis_attention_mix")
LAYER_AGGREGATION_MODES = ("last", "last_n_mean", "scalar_mix")
# independent와 independent_mlp는 trait 사이를 전혀 섞지 않는다. 둘의 차이는 head가
# 선형 1층인지 GELU를 낀 2층인지 하나뿐이다. weighted_mixed는 1층 head의 최종 점수를,
# mixed_head_v2는 2층 head의 중간 표현을 trait 사이에서 섞는다.
HEAD_TYPES = ("independent", "independent_mlp", "weighted_mixed", "mixed_head_v2")
# trait 사이를 섞지 않는 head. 평균-대비 재파라미터화처럼 trait별 출력의 의미가
# 고정되어야 하는 옵션이 이 집합을 요구한다.
UNMIXED_HEAD_TYPES = ("independent", "independent_mlp")
PROMPT_HEAD_MODES = ("none", "bias", "average")
PROMPT_HEAD_TRAITS = ("organization", "all")
ORGANIZATION_POOLING_MODES = (
    "shared",
    "essay_mean",
    "first_middle_last",
    # first_middle_last는 essay token을 위치상 3등분한 근사다. paragraph_mean은
    # 공식 입력에서 결정적으로 보이는 이중공백 cue로 실제 문단을 나누고 문단별
    # 평균의 평균을 쓴다. 문단 cue가 없는 글은 essay_mean으로 되돌아간다.
    "paragraph_mean",
    "sentence_transition",
)
SCORE_PARAMETERIZATIONS = ("traits", "average_plus_contrast")
BACKBONE_TYPES = ("auto", "decoder", "encoder")
# 라벨은 전부 이산 격자 위에 있다(train 11,600편 전수 확인, 위탈 0).
#   평가자 x 세부지표 = 정수 1~5          -> 5개 class
#   세부지표(2인 평균) = 1~5의 0.5 간격    -> 9개 class
#   trait content(5준거 x 2인 평균) = 0.1  -> 41개 class
#   trait organization/expression(2준거 x 2인) = 0.25 -> 17개 class
# 따라서 어떤 head 단위를 쓰든 "그 단위의 네이티브 격자"로 분류할 수 있다.
TRAIT_SCORE_STEPS = {"content": 0.1, "organization": 0.25, "expression": 0.25}
CRITERION_SCORE_STEP = 0.5
RATER_SCORE_STEP = 1.0
SCORE_MIN, SCORE_MAX = 1.0, 5.0

# regression은 bounded scalar, distribution은 정수 1~5의 5-class,
# trait_native_distribution은 각 trait의 네이티브 격자(41/17/17)로 분류한다.
# regression_ordinal은 bounded regression과 누적 ordinal 확률을 함께 내고 두
# continuous prediction을 섞는다. 경쟁자 artifact에서 확인한 16-step 구조를
# 독립 재현하되, 기존 head의 checkpoint key/forward는 전혀 바꾸지 않는다.
SCORE_HEADS = (
    "regression",
    "distribution",
    "trait_native_distribution",
    "regression_ordinal",
)
# 분류 head가 만든 확률을 점수로 바꾸는 방법이다. 세 값 모두 같은 head를 쓰고
# 읽는 방식만 다르므로 head 개수 축과 완전히 직교한다.
#   expectation            sum_k p_k * v_k. 연속값이고 미분 가능하다(기존 동작).
#   argmax                 격자 위의 한 점. 미분 불가라 CE loss가 반드시 켜져 있어야 한다.
#   gumbel_straight_through 순전파는 argmax와 같은 격자 점, 역전파는 softmax gradient다.
#                          즉 격자 위 예측을 유지하면서 downstream MSE까지 학습된다.
CATEGORICAL_READOUTS = ("expectation", "argmax", "gumbel_straight_through")
# 2026-08-06 공지로 채점이 정수 반올림 기준이 됐다. train label을 미리 정수로 바꿔
# train/serve 분포를 맞추는 실험용 옵션이다. 기본 none은 실수 label 그대로다.
TRAIN_LABEL_ROUNDINGS = ("none", "integer")
DETAIL_HEAD_MODES = (
    "none",
    "scalar",
    # One five-class distribution per criterion.  The target is the empirical
    # distribution of the one or two official integer ratings.
    "categorical",
    # One nine-class distribution per criterion on 1, 1.5, ..., 5.
    "halfstep_categorical",
    # Two anonymous full-rater branches, each with nine five-class heads.
    # Their two nine-score vectors are matched permutation-invariantly.
    "rater_set",
    # 같은 18개 단위를 분류가 아니라 bounded scalar로 낸다. rater_set과 이 값의
    # 차이는 출력 형태 하나뿐이라 "head 개수 x 출력 형태" 격자가 완성된다.
    "rater_set_scalar",
)
# 익명 평가자 2인 branch를 쓰는 detail mode. 순서 없는 매칭 경로를 공유한다.
RATER_SET_HEAD_MODES = ("rater_set", "rater_set_scalar")
# 확률을 내는 detail mode. categorical_readout이 적용되는 대상이다.
CATEGORICAL_DETAIL_HEAD_MODES = ("categorical", "halfstep_categorical", "rater_set")
DETAIL_FINAL_SOURCES = ("direct", "criterion")
# essay_only_v1은 문항 지문을 입력에서 뺀다. 근거는 datasets.py의
# ESSAY_ONLY_INPUT_TEMPLATE 주석에 있다.
POOLED_NORMALIZATIONS = ("none", "layernorm")
INPUT_FORMATS = (
    "baseline_v1",
    "source_aware_v1",
    "rubric_conditioned_v1",
    "essay_only_v1",
)
RUBRIC_PROFILES = ("none", "short_3trait_v1", "full_9criterion_1to5_v1")
CRITERION_READOUTS = ("shared", "textual_anchor_residual")
ESSAY_SURFACES = (
    "canonical",
    "flat",
    "official_raw",
    "official_gap_newline",
    "official_raw_kiwi_sentence_newline_v1",
)
PAIRWISE_LOSSES = ("none", "soft_ranknet", "hard_ranknet", "hinge")
LISTWISE_LOSSES = ("none", "soft_rank_mse", "soft_spearman", "listnet")
INBATCH_SAMPLING_MODES = ("random", "same_question")
# 순위 로스가 무엇의 순위를 맞출지 고른다. 공식 지표는 essay별 세 trait 평균 점수 하나의
# 순위이므로 trait_average가 지표와 정렬된 쪽이다. per_trait는 trait마다 따로 순위를
# 맞춘 뒤 평균하는 기존 동작이다.
RANKING_TARGETS = ("per_trait", "trait_average")
# 연속 score head 위에 실제 제출 정수 decoder를 얹는 quantization-aware objective.
# ``none``은 역사 실험과 현재 실행 중인 실험에서 완전한 no-op이다.
QUANTIZATION_RULES = ("none", "independent_half_up", "average_matched")
QUANTIZATION_SURROGATES = ("soft", "straight_through", "expected_risk")
QUANTIZED_TARGET_RULES = (
    "raw",
    "same_as_prediction",
    "independent_half_up",
    "average_matched",
)
QUANTIZED_ERROR_FORMS = ("mse", "rmse")
CHECKPOINT_RETENTION_POLICIES = ("legacy_core", "selected_pair")
# 제출 점수 후처리 규칙. 실제 구현과 설명은 postprocess.py에 있다. 순환 import를 피하려고
# 이름만 여기 둔다.
SCORE_POSTPROCESS_RULES = ("none", "per_trait_round", "average_matched")
DEFAULT_SCORE_POSTPROCESS = "average_matched"
METRIC_SURFACES = ("raw_continuous", "independent_half_up", "average_matched")
METRIC_AGGREGATIONS = ("trait_macro", "mean_first", "pooled")
METRIC_NAMES = ("rmse", "spearman")
SURFACE_CHECKPOINT_METRICS = tuple(
    f"{surface}_{aggregation}_{metric}"
    for surface in METRIC_SURFACES
    for aggregation in METRIC_AGGREGATIONS
    for metric in METRIC_NAMES
)
BEST_CHECKPOINT_METRICS = (
    "rmse",
    "spearman",
    # 공식 지표(essay별 세 trait 평균 하나)와 2026-08-06 공지의 정수 반올림 지표다.
    # 위 rmse/spearman은 trait별 지표의 산술평균이라 리더보드 정의와 다르다.
    "official_rmse",
    "official_spearman",
    "official_matched_rmse",
    # 위와 같은 예측을 **리더보드 라벨 분포로 옮겨** 잰 RMSE. 2026-08-20 진단에서
    # 로컬 400편(정답 SD 0.653)과 채점 집합(추정 SD 0.77)이 다르다는 것이 확인됐다.
    # 근거와 상수는 main_code/distribution_shift.py에 있다.
    "official_matched_rmse_shifted",
    # 문항 구성에 불변인 지표들. 공개 validation 400편의 문항 구성은 우리 학습 편수에
    # 비례하는데(Q1 25편 … Q5 51편) 2025 채점 데이터는 Q11 1,512 / Q12 1,331편으로
    # 완전히 다르다(신규 두 문항이 71.1%). essay micro 평균으로 고르면 우리 validation의
    # 구성에만 맞춘 모델을 고르게 된다. macro는 구성에 불변이고 worst는 약한 문항이
    # 하나도 없는 모델을 고른다.
    "prompt_macro_rmse",
    "worst_prompt_rmse",
    "prompt_macro_spearman",
    # unseen_prompt_holdout으로 학습에서 뺀 문항만 모은 부분집합의 지표. 미학습 문항
    # 성능을 **학습 중에** 보고 고를 수 있게 한다.
    "unseen_prompt_rmse",
    "unseen_prompt_spearman",
    # 최종 정수 C/O/E를 영역별로 평가한 뒤 평균하는 RMSE와, 같은 정수 C/O/E를
    # essay별 평균낸 Spearman. 2026-08-13 혼합 평가 가설의 exact checkpoint pair다.
    "submitted_trait_macro_rmse",
    "official_matched_spearman",
    "content_rmse",
    "organization_rmse",
    "expression_rmse",
    "content_spearman",
    "organization_spearman",
    "expression_spearman",
) + SURFACE_CHECKPOINT_METRICS
SECONDARY_CHECKPOINT_METRICS = ("auto", "none") + BEST_CHECKPOINT_METRICS
LORA_SCHEDULER_SCOPES = ("global", "joint")
DATASET_SCHEDULES = (
    "competition_only",
    "external_only",
    "mixed",
    "alternating",
    "pretrain_then_competition",
)
EXTENDED_SOURCE_SAMPLING_MODES = ("uniform", "proportional")
EXTERNAL_ORGANIZATION_LABEL_POLICIES = (
    "official",
    "aihub24_highschool_structural_v1",
    "aihub24_structural_aihub26_persuasion_v1",
)
ORGANIZATION_AUGMENTATIONS = (
    "none",
    "paragraph_order_high_confidence_v1",
    "sentence_order_high_confidence_v1",
)
EXTERNAL_SCORE_ALIGNMENTS = ("none", "quantile_to_competition_v1")

CONFIG_CHOICES = {
    "score_postprocess": SCORE_POSTPROCESS_RULES,
    "training_mode": TRAINING_MODES,
    "pooling": POOLING_MODES,
    "layer_aggregation": LAYER_AGGREGATION_MODES,
    "head_type": HEAD_TYPES,
    "prompt_head_mode": PROMPT_HEAD_MODES,
    "prompt_head_traits": PROMPT_HEAD_TRAITS,
    "organization_pooling": ORGANIZATION_POOLING_MODES,
    "backbone_type": BACKBONE_TYPES,
    "score_head": SCORE_HEADS,
    "categorical_readout": CATEGORICAL_READOUTS,
    "train_label_rounding": TRAIN_LABEL_ROUNDINGS,
    "score_parameterization": SCORE_PARAMETERIZATIONS,
    "detail_head_mode": DETAIL_HEAD_MODES,
    "detail_final_source": DETAIL_FINAL_SOURCES,
    "input_format": INPUT_FORMATS,
    "rubric_profile": RUBRIC_PROFILES,
    "criterion_readout": CRITERION_READOUTS,
    "essay_surface": ESSAY_SURFACES,
    "lr_scheduler_type": ("constant", "linear", "cosine"),
    "lora_scheduler_scope": LORA_SCHEDULER_SCOPES,
    "pairwise_loss": PAIRWISE_LOSSES,
    "listwise_loss": LISTWISE_LOSSES,
    "inbatch_sampling": INBATCH_SAMPLING_MODES,
    "ranking_target": RANKING_TARGETS,
    "quantization_rule": QUANTIZATION_RULES,
    "quantization_surrogate": QUANTIZATION_SURROGATES,
    "quantized_target_rule": QUANTIZED_TARGET_RULES,
    "quantized_error_form": QUANTIZED_ERROR_FORMS,
    "checkpoint_retention": CHECKPOINT_RETENTION_POLICIES,
    "best_checkpoint_metric": BEST_CHECKPOINT_METRICS,
    "secondary_checkpoint_metric": SECONDARY_CHECKPOINT_METRICS,
    "dataset_schedule": DATASET_SCHEDULES,
    "extended_source_sampling": EXTENDED_SOURCE_SAMPLING_MODES,
    "external_organization_label_policy": EXTERNAL_ORGANIZATION_LABEL_POLICIES,
    "external_score_alignment": EXTERNAL_SCORE_ALIGNMENTS,
    "organization_augmentation": ORGANIZATION_AUGMENTATIONS,
    "primary_data_profile": tuple(PRIMARY_DATA_PROFILES),
}
CONFIG_CLI_ALIASES = {"use_qlora": ("--qlora",)}
CONFIG_CLI_EXCLUDED = {
    "schema_version",
    "model_id",
    "model_slug",
    "model_source_run",
    "prompt_registry",
    "detail_rater_registry",
    # Old checkpoint JSON may contain this field.  New training has one clear
    # package-local dataset_root, so exposing a second ignored CLI path would
    # be misleading.
    "extended_data_dir",
}


def lora_target_modules(include_mlp: bool) -> list[str]:
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if include_mlp:
        targets += ["gate_proj", "up_proj", "down_proj"]
    return targets


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    trust_remote_code: bool = True
    large: bool = False
    backbone_type: str = "decoder"
    slug: str = ""
    revision: str = "main"
    parameters_b: float = 0.0
    source_run: str = "decoder"
    enabled: bool = True
    source_quantization: str = "unquantized"
    download_excludes: str = ""
    note: str = ""

    @property
    def repo_id(self) -> str:
        """TSV 용어인 repo_id와 기존 model_id를 둘 다 지원한다."""

        return self.model_id


# Model catalog ---------------------------------------------------------------
MODEL_CATALOG_COLUMNS = (
    "enabled",
    "slug",
    "repo_id",
    "revision",
    "parameters_b",
    "source_run",
    "parse_rate",
    "overall_rmse",
    "overall_spearman",
    "valid_only_rmse",
    "valid_only_spearman",
    "source_quantization",
    "download_excludes",
    "note",
)


def _catalog_backbone_type(source_run: str, repo_id: str) -> str:
    """catalog의 용도 표기를 scorer가 쓰는 encoder/decoder로 바꾼다."""

    if source_run.startswith("encoder-decoder") or source_run == "encoder":
        return "encoder"
    if source_run.startswith("decoder"):
        return "decoder"
    # embedding/reranker에는 양방향 encoder와 causal LM이 함께 들어 있다.
    decoder_style = (
        repo_id.startswith("Qwen/Qwen3-Embedding-")
        or repo_id.startswith("Qwen/Qwen3-Reranker-")
        or repo_id.startswith("kakaocorp/kanana-nano-")
        or repo_id == "microsoft/harrier-oss-v1-27b"
    )
    if source_run in {"embedding", "reranker"}:
        return "decoder" if decoder_style else "encoder"
    return "auto"


def load_model_catalog(
    path: str | Path = MODEL_CATALOG_PATH,
    *,
    enabled_only: bool = False,
) -> tuple[ModelSpec, ...]:
    """주석 header를 가진 14열 TSV를 파일 순서 그대로 읽는다."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"model catalog가 없습니다: {source}")
    specs: list[ModelSpec] = []
    seen_slugs: set[str] = set()
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        columns = line.split("\t")
        if len(columns) != len(MODEL_CATALOG_COLUMNS):
            raise ValueError(
                f"{source}:{line_number}: TSV 열은 "
                f"{len(MODEL_CATALOG_COLUMNS)}개여야 합니다"
            )
        row = dict(zip(MODEL_CATALOG_COLUMNS, columns, strict=True))
        if row["enabled"] not in {"0", "1"}:
            raise ValueError(f"{source}:{line_number}: enabled는 0 또는 1이어야 합니다")
        enabled = row["enabled"] == "1"
        if enabled_only and not enabled:
            continue
        slug = row["slug"].strip()
        model_id = row["repo_id"].strip()
        revision = row["revision"].strip()
        if not slug or not model_id or not revision:
            raise ValueError(
                f"{source}:{line_number}: slug/repo_id/revision은 비어 있을 수 없습니다"
            )
        if slug in seen_slugs:
            raise ValueError(f"{source}:{line_number}: 중복 model slug={slug!r}")
        try:
            parameters_b = float(row["parameters_b"])
        except ValueError as exc:
            raise ValueError(
                f"{source}:{line_number}: parameters_b가 숫자가 아닙니다"
            ) from exc
        if not math.isfinite(parameters_b) or parameters_b < 0:
            raise ValueError(
                f"{source}:{line_number}: parameters_b는 유한한 음이 아닌 수여야 합니다"
            )
        seen_slugs.add(slug)
        specs.append(
            ModelSpec(
                model_id=model_id,
                # 이 catalog는 사용자가 검토한 checkpoint 목록이며, 기존 TSV
                # runner도 모든 항목에 remote code 허용을 명시했다.
                trust_remote_code=True,
                large=parameters_b >= 16.0,
                backbone_type=_catalog_backbone_type(row["source_run"], model_id),
                slug=slug,
                revision=revision,
                parameters_b=parameters_b,
                source_run=row["source_run"],
                enabled=enabled,
                source_quantization=row["source_quantization"],
                download_excludes=(
                    "" if row["download_excludes"] == "-" else row["download_excludes"]
                ),
                note=row["note"],
            )
        )
    if not specs:
        qualifier = "enabled " if enabled_only else ""
        raise ValueError(f"{source}: {qualifier}model이 없습니다")
    return tuple(specs)


MODEL_CATALOG = load_model_catalog()
MODEL_CATALOG_BY_SLUG = {spec.slug: spec for spec in MODEL_CATALOG}

# 과거 결과 script가 사용하던 짧은 이름은 그대로 유지한다. 새 실험은 TSV slug를
# 우선 사용하고, 여기의 alias는 예전 명령을 깨지 않기 위한 호환 계층이다.
LEGACY_MODEL_REGISTRY = {
    "midm": ModelSpec(
        "K-intelligence/Midm-2.0-Base-Instruct",
        trust_remote_code=True,
        slug="midm",
    ),
    "gemma_e4b": ModelSpec("google/gemma-4-E4B-it", slug="gemma_e4b"),
    "gemma12": ModelSpec("google/gemma-4-12B-it", slug="gemma12"),
    "qwen35_9b": ModelSpec("Qwen/Qwen3.5-9B", slug="qwen35_9b"),
    "ax40_light": ModelSpec("skt/A.X-4.0-Light", slug="ax40_light"),
    "kormo10b": ModelSpec(
        "KORMo-Team/KORMo-10B-sft",
        trust_remote_code=True,
        slug="kormo10b",
    ),
    "hyperclovax14": ModelSpec(
        "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B",
        slug="hyperclovax14",
    ),
    "tri7b": ModelSpec("trillionlabs/Tri-7B", slug="tri7b"),
    "tri21": ModelSpec(
        "trillionlabs/Tri-21B",
        trust_remote_code=True,
        large=True,
        slug="tri21",
        parameters_b=21.0,
    ),
    "mistral24": ModelSpec(
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        large=True,
        slug="mistral24",
        parameters_b=24.0,
    ),
    "gemma26": ModelSpec(
        "google/gemma-4-26B-A4B-it",
        large=True,
        slug="gemma26",
        parameters_b=26.0,
    ),
    "gemma31": ModelSpec(
        "google/gemma-4-31B-it-qat-q4_0-unquantized",
        large=True,
        slug="gemma31",
        parameters_b=31.0,
    ),
    "exaone33": ModelSpec(
        "LGAI-EXAONE/EXAONE-4.5-33B",
        trust_remote_code=True,
        large=True,
        slug="exaone33",
        parameters_b=33.0,
    ),
    "exaone45_33b_bf16": ModelSpec(
        "LGAI-EXAONE/EXAONE-4.5-33B",
        trust_remote_code=True,
        large=True,
        slug="exaone45_33b_bf16",
        parameters_b=33.0,
    ),
    "bge_m3": ModelSpec("BAAI/bge-m3", backbone_type="encoder", slug="bge_m3"),
    "klue_roberta_large": ModelSpec(
        "klue/roberta-large",
        backbone_type="encoder",
        slug="klue_roberta_large",
    ),
}

# 동일 slug가 있으면 폴더 안 TSV를 최종 기준으로 삼는다. legacy entry는 TSV에
# 없는 과거 별칭만 보완하며 model_source_run/revision provenance를 덮지 않는다.
MODEL_REGISTRY = {**LEGACY_MODEL_REGISTRY, **MODEL_CATALOG_BY_SLUG}


def enabled_model_specs(
    path: str | Path = MODEL_CATALOG_PATH,
) -> tuple[ModelSpec, ...]:
    return load_model_catalog(path, enabled_only=True)


def resolve_model(value: str) -> ModelSpec:
    """TSV slug, legacy alias, repo id 순서로 하나의 checkpoint를 찾는다."""

    if value in MODEL_REGISTRY:
        return MODEL_REGISTRY[value]
    for spec in MODEL_CATALOG:
        if value == spec.model_id:
            return spec
    for spec in LEGACY_MODEL_REGISTRY.values():
        if value == spec.model_id:
            return spec

    # 예전 결과 폴더명이 `tri7b_suffix`처럼 alias 뒤에 실험명을 붙인 경우도
    # 지원한다. 직접 전달한 Hugging Face repo id는 위에서 먼저 exact match한다.
    parts = value.split("_")
    for index in range(len(parts)):
        alias = "_".join(parts[: index + 1])
        if alias in MODEL_REGISTRY:
            return MODEL_REGISTRY[alias]
    if "/" in value:
        return ModelSpec(value, backbone_type="auto", slug=value)
    choices = ", ".join(MODEL_REGISTRY)
    raise ValueError(f"지원하지 않는 모델입니다: {value!r}; aliases={choices}")


@dataclass(frozen=True)
class RegressionConfig:
    schema_version: int = 1
    model_id: str = 'Qwen/Qwen3.5-9B'
    # train.py가 TSV slug를 resolve한 뒤 아래 provenance도 checkpoint에 저장한다.
    # 빈 slug/source_run은 과거 config를 그대로 읽기 위한 호환값이다.
    model_slug: str = 'qwen35_9b'
    model_revision: str = "main"
    model_source_run: str = 'decoder'
    # 기본 backbone은 검토된 local catalog 항목이며 remote model code가 필요하다.
    # 과거 baseline의 False 값은 configs/baseline.json에 별도로 고정한다.
    trust_remote_code: bool = True
    # decoder는 last token, encoder는 masked mean을 주로 쓰지만 모델마다 다르다.
    # 따라서 backbone 종류와 pooling은 분리해서 명시적으로 저장한다.
    backbone_type: str = "decoder"
    training_mode: str = 'lora_only'
    use_qlora: bool = False
    input_format: str = "baseline_v1"
    # rubric_conditioned_v1의 두 실험 축이다. none/shared는 neutral 기본값이라
    # 기존 input format과 checkpoint의 동작을 바꾸지 않는다. full은 공식 9개
    # 준거와 1~5 경계를 입력에 적고, textual_anchor_residual은 criterion별
    # readout anchor를 zero-gated residual로 detail head에만 전달한다.
    rubric_profile: str = "none"
    criterion_readout: str = "shared"
    # canonical은 원천 corpus에서 복원한 문단 경계를 유지한다. flat은 과거
    # 실험 재현을 위해 모든 whitespace를 한 칸으로 접는다. 실제 대회 JSONL은
    # raw paragraph form을 바로 이어 붙여 이중 공백/tab까지 보존하므로 official_raw가
    # 정확한 배포 surface다. official_gap_newline은 고정밀 이중공백 cue만 newline으로
    # 드러내고, official_raw_kiwi_sentence_newline_v1은 원문을 보존한 채 Kiwi가 찾은
    # 문장 사이에만 newline을 추가하는 deployable ablation이다.
    essay_surface: str = 'official_raw'
    # 학습 row만 이 확률로 원천에서 복원한 canonical 문단 view를 보여 준다.
    # 나머지 학습 view와 validation/inference는 essay_surface를 그대로 쓴다.
    # 0은 neutral 기본값이다.
    train_canonical_surface_probability: float = 0.0
    pooling: str = "mean"
    # ANALYSIS_MIX_POOLINGS에서 접미사(분석) 구간이 차지하는 지분.
    # 0.0이면 "mean"과 같고 1.0이면 접미사만 본다. learnable=True면 초기값이다.
    analysis_pool_weight: float = 0.5
    # 지분을 학습 대상으로 둘지. scalar 1~3개뿐이라 과적합 위험이 없고, 근거가
    # 쓸모없으면 스스로 0으로 내려가 baseline으로 환원된다. 하한이 막힌 설정이라
    # 지분 sweep에 GPU를 쓰는 대신 이쪽을 기본값으로 둔다.
    analysis_pool_learnable: bool = True
    # 지분을 trait별로 분리한다. 근거 DSL은 기준별 구획이 있어 항목마다 유효한
    # 지분이 다를 수 있다. True면 pooled가 [B,3,H]가 된다(essay_attention과 동일 경로).
    analysis_pool_per_trait: bool = False
    normalize_features: bool = False
    # --- pooled 표현 정규화 방식 (2026-08-20) ----------------------------------
    # "none"이면 아무것도 하지 않는다(기존 동작).
    # "layernorm"이면 parameterless LayerNorm을 pooled 벡터에 건다.
    #
    # 왜 `normalize_features`(L2)와 따로 두는가
    #   L2 정규화는 벡터를 단위 구면에 올려 **크기 정보를 통째로 버린다**. LayerNorm은
    #   차원별 평균·분산만 맞추고 상대 구조를 남긴다. 서로 다른 연산이다.
    #
    # 왜 이 축을 재는가
    #   c02의 trainer_state.json에서 grad_norm이 5.841~19.662인데 max_grad_norm=1.0이다.
    #   기록된 모든 step이 6~20배로 clip되므로 설정한 학습률이 step마다 다른 배율로
    #   축소되고 cosine 스케줄이 사실상 작동하지 않는다. pooled 크기를 줄이면 head의
    #   기울기가 작아져 clip이 지배하지 않게 된다.
    #
    # parameterless라 state_dict가 바뀌지 않는다 → 기존 checkpoint 구조와 호환된다.
    pooled_normalization: str = "none"
    # --- 문항 GroupDRO (2026-08-20) --------------------------------------------
    # 지수 기울기 상승으로 **현재 손실이 큰 문항**의 가중치를 올린다(Sagawa et al.).
    # 0이면 완전히 꺼져 기존 run과 bit-exact 동일하다.
    #
    # 왜 고정 재가중(tail_weight)과 다른가
    #   tail_weight는 라벨 값으로 가중치를 미리 정했고 실패했다. GroupDRO는 학습 중
    #   관측된 그룹 손실로 가중치를 갱신하므로 어느 문항이 어려운지 스스로 찾는다.
    #
    # 왜 문항으로 그룹을 나누는가
    #   우리 문항별 RMSE가 Q6 0.3236 ~ Q4 0.5020으로 벌어져 있고, 숨은 평가셋의 문항
    #   구성은 우리 validation과 전혀 다르다(2025 자료의 71%가 Q11·Q12). 평균이 아니라
    #   최악 문항을 좋게 만드는 것이 구성 변화에 강하다. 새 선택 지표
    #   `worst_prompt_rmse`가 재는 양을 목적함수가 직접 최적화하게 된다.
    group_dro_step_size: float = 0.0
    # --- gradient clipping 임계 (2026-08-20) ------------------------------------
    # c02의 trainer_state.json에서 grad_norm이 5.841~19.662인데 이 값이 1.0이었다.
    # 기록된 모든 step이 6~20배로 clip되므로 설정한 학습률이 step마다 다른 배율로
    # 축소되고 cosine 스케줄이 사실상 작동하지 않는다. 1.0은 기존 run과 동일한
    # 기본값이므로 재현이 깨지지 않는다.
    max_grad_norm: float = 1.0
    # --- 수준 prototype contrastive (2026-08-21) --------------------------------
    # 라벨 평균으로 수준 bin을 만들고 각 수준의 prototype을 EMA로 유지하면서, essay
    # 표현이 자기 수준 prototype에 가까워지도록 prototypical cross-entropy를 건다.
    #
    # 왜 이 축이 살아남는가
    #   공식 지표는 세 연속값의 **합** 하나만 본다(2026-08-21 부록에서 100% 확인).
    #   영역 배분을 재조정하는 구조는 지표가 못 본다. 이 축은 "이 글이 전체적으로 어느
    #   수준인가"의 표현을 다루므로 합에 직접 작용하고, 추론 비용이 0이다.
    #
    # 0이면 완전히 꺼져 기존 run과 bit-exact 동일하다. prototype은 학습 전용 상태라
    # checkpoint에서 제외한다(TRAINING_ONLY_STATE_PREFIXES).
    level_prototype_weight: float = 0.0
    level_prototype_count: int = 5
    level_prototype_momentum: float = 0.9
    level_prototype_temperature: float = 1.0
    # 기본값은 기존 코드와 완전히 같은 마지막 layer다. P1에서 효과가 있었던
    # last-N 평균/scalar mix는 실험할 때만 CLI로 켠다.
    layer_aggregation: str = "last"
    last_n_layers: int = 4
    attention_pool_dim: int = 256
    head_type: str = "independent"
    # 문제별 head는 공용 head를 대체하지 않고 함께 사용한다. average는
    # 공용/문제별 bounded score(또는 확률)를 prompt_head_weight로 평균한다.
    # registry는 train row에서 만들고 checkpoint config에 그대로 저장한다.
    prompt_head_mode: str = "none"
    prompt_head_traits: str = "all"
    prompt_head_weight: float = 0.5
    prompt_registry: tuple[tuple[str, str], ...] = ()
    # organization만 essay의 앞/중간/뒤 또는 인접 문장 전이 표현을 사용한다.
    # shared가 기존 baseline이며 다른 두 trait 표현은 항상 그대로 둔다.
    # 2026-09-29: 최종 제출(ABCD)은 organization만 이중 공백 단서로 나눈 문단
    # 평균을 쓴다(요소 D, G2-3). 학습되는 스칼라 게이트가 0에서 시작하므로 문단
    # 단서가 없는 글은 shared와 bit-exact하게 같아진다.
    organization_pooling: str = 'paragraph_mean'
    # 최종 c02는 direct 5-class 확률의 기댓값을 score로 사용한다. 역사적 Y1의
    # regression head는 configs/confirmed_y1.json에 완전하게 고정한다.
    score_head: str = "distribution"
    # 활성화된 분류 head의 확률을 점수로 바꾸는 방법이다. head 개수(3/9/18)와
    # 무관하게 같은 규칙이 적용된다. 기본 expectation은 기존 동작과 같다.
    categorical_readout: str = "expectation"
    # gumbel_straight_through에서만 쓰는 온도다. 작을수록 one-hot에 가깝다.
    gumbel_temperature: float = 1.0
    # train row의 trait label만 사사오입 정수로 바꾼다. validation label과 공식 gold는
    # 절대 바꾸지 않는다. 기본 none이 기존 동작이다.
    train_label_rounding: str = "none"
    # 공식 지표는 essay당 세 trait 평균 점수 하나에서 계산된다. traits는 기존처럼
    # 세 head가 각자 trait 점수를 내고 평균은 파생값이다. average_plus_contrast는
    # 같은 세 head를 (평균, 대비1, 대비2)로 재해석해 평균을 직접 학습하는
    # 파라미터로 만든다. 세 trait의 편차 합이 0이므로 (C+O+E)/3 == 평균 head가
    # 항등으로 성립한다. 새 parameter나 checkpoint key를 만들지 않는다.
    score_parameterization: str = "traits"
    distribution_loss_weight: float = 1.0
    distribution_label_smoothing: float = 0.0
    # regression_ordinal 전용. ordinal_steps개의 누적 확률은
    # 1 + (4 / steps) * sum(sigmoid(logit))으로 연속 점수가 된다. 최종 점수는
    # ordinal_blend_weight * ordinal + 나머지 * bounded regression이다.
    # 기본 ordinal loss=0은 역사 config/checkpoint에 대한 additive no-op이다.
    ordinal_steps: int = 16
    ordinal_blend_weight: float = 0.7
    ordinal_loss_weight: float = 0.0
    # 세부 채점 기준은 content 5개, organization 2개, expression 2개다.
    # none은 기존 3-trait 모델과 checkpoint key를 그대로 유지한다. scalar는
    # 기준마다 bounded score 하나, categorical은 공식 정수 rating의 1~5 확률,
    # halfstep_categorical은 공식 평균의 1~5(0.5 간격) 확률을 낸다. rater_set은
    # 익명 평가자 두 명 각각에 9개의 1~5 분류 head를 두고 평가자 순서 없이 학습한다.
    # 모든 mode는 기존 direct 3-trait branch를 보존하며, 아래 final source가 direct
    # 출력과 criterion 5/2/2 집계 중 최종 score를 명시적으로 고른다.
    # 2026-09-29: 최종 제출(ABCD)은 익명 평가자 2인 branch를 보조 감독으로 쓴다
    # (요소 C, G5-3). 단일 재평가자 행을 묶으려면 detail_expected_loss_weight > 0이
    # 함께 필요하므로 G5-1의 실수 점수 손실도 같이 켠다.
    detail_head_mode: str = 'rater_set'
    # direct는 기존 3-trait branch, criterion은 9개 expected score의 5/2/2
    # 평균을 최종 [B,3] score와 primary trait MSE/ranking에 사용한다.
    detail_final_source: str = 'direct'
    detail_expected_loss_weight: float = 0.25
    detail_distribution_loss_weight: float = 0.0
    detail_halfstep_loss_weight: float = 0.0
    detail_rater_set_loss_weight: float = 0.25
    detail_hierarchy_loss_weight: float = 0.0
    detail_rater_loss_weight: float = 0.0
    # 원천 canonical 문단은 정답을 만드는 데만 쓰고 scorer에는 official_raw만
    # 보여 준다. 문장 다음 token hidden에서 문단 경계 여부를 함께 예측해 같은
    # LoRA backbone에 보조 gradient를 주는 joint multi-task loss다. 0이면 head,
    # collator, checkpoint key를 만들지 않아 기존 score baseline과 정확히 같다.
    paragraph_boundary_loss_weight: float = 0.0
    # (source_dataset, 실제 평가자 문자열 ID)의 안정적인 정렬이다. source가
    # 다른 동일 ID의 충돌을 막고 validation/test 추론 입력에는 사용하지 않는다.
    # CLI에서 임의 문자열 tuple을 받지 않고 train data registry builder가 채운다.
    detail_rater_registry: tuple[tuple[str, str], ...] = ()
    mixed_head_weight: float = 0.8
    head_hidden_size: int = 256
    max_length: int = 4096
    batch_size: int = 32
    gradient_accumulation: int = 1
    epochs: int = 15
    head_epochs: int = 3
    joint_epochs: int = 12
    head_learning_rate: float = 2e-4
    lora_learning_rate: float = 4e-05
    # LoRA+의 B/A learning-rate ratio다. 1은 기존처럼 모든 adapter parameter를
    # 한 optimizer group에서 같은 LR로 학습한다. 1보다 크면 A는
    # lora_learning_rate, B는 그 배수의 LR을 사용한다.
    lora_plus_lr_ratio: float = 1.0
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    # global은 기존처럼 head/LoRA group이 전체 학습 step의 scheduler를 공유한다.
    # joint는 two-stage에서 LoRA group만 joint 시작부터 별도 warmup/schedule을 쓴다.
    lora_scheduler_scope: str = "global"
    weight_decay: float = 0.0
    # 최종 c02와 역사적 expanded-Y1이 공통으로 쓰는 all-linear LoRA 용량이다.
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_include_mlp: bool = True
    # 비어 있으면 새 LoRA를 만든다. 경로를 주면 그 adapter weight만 trainable하게
    # 불러오고 score head는 현재 config에 맞춰 새로 초기화한다. 외부 3-trait
    # preadapt 뒤 NIKL H2 head를 새로 학습하는 단순한 representation transfer용이다.
    initial_lora_adapter: str = ""
    # 기존 LoRA 학습은 checkpointing을 항상 켰다. 48GB에서 여유가 있는
    # 7B~14B 모델은 이를 끄고 물리 batch를 키우면 같은 objective를 더 빨리
    # 계산할 수 있으므로 재현 가능한 명시 옵션으로 노출한다.
    gradient_checkpointing: bool = True
    # auto면 backbone 종류에 맞는 attention projection을 찾는다. 필요할 때
    # "query,value"처럼 comma-separated module suffix를 직접 지정할 수 있다.
    lora_targets: str = "auto"
    eval_every_epoch: bool = True
    # 0이면 기존 epoch 학습을 그대로 사용한다. 양수이면 이 값은 확장 데이터의
    # "main phase" optimizer step 수이며, final_competition_epochs가 있으면 그 뒤에
    # 대회 데이터 전용 step을 덧붙인다. 최신 full-NIKL same-question runner는
    # 한 pass 약 730 step을 기준으로 5-pass/8-pass 예산을 명시한다.
    max_train_steps: int = 1104
    # step mode의 two-stage 경계와 평가 주기. 0이면 train row/batch에서 각각
    # head_epochs와 대회 데이터 한 epoch에 해당하는 step 수로 자동 계산한다.
    head_warmup_steps: int = 0
    eval_steps: int = 64
    # 모든 데이터 경로는 main_code/datasets 아래의 profile/alias로 해석한다.
    dataset_root: str = str(DATA_ROOT)
    primary_data_profile: str = DEFAULT_PRIMARY_DATA_PROFILE
    # 0은 전체 사용이다. 양수이면 seed로 문제별 비율을 보존해 뽑으므로
    # --limit처럼 validation까지 잘라 버리지 않고 데이터 크기 비교에 쓸 수 있다.
    competition_train_limit: int = 0
    external_train_limit: int = 0
    # 비어 있으면 external source 전체를 쓴다. 값이 있으면 prepared row의
    # metadata.purpose와 정확히 같은 글만 남긴다(예: "설득"). 논증형 대회에
    # 설명/정서 글까지 섞는 효과와 외부 점수 사전학습 효과를 분리하기 위한 옵션이다.
    external_purpose_filter: str = ""
    # 비어 있으면 끈다. trait를 지정하면 external source의 같은 문제 묶음 중
    # 서로 다른 본문과 서로 다른 해당 trait 점수가 모두 존재하는 묶음만 남긴다.
    # 순위 학습에서 동일 본문 충돌이나 전부 동점인 batch를 명시적으로 제거한다.
    external_rankable_trait: str = ""
    # AIHub24의 저장 O는 문단 구조 외에 일관성·분량과 가변 가중치를 섞는다.
    # 별도 정책을 켠 실험에서만 고등학생+활성 구조준거 행을 고르고 대회 O1/O2에
    # 가까운 두 구조점수 평균으로 재구성한다. A24+A26 pack 정책은 A26에서 설득
    # 목적만 남기되 전문가 2인의 기존 O는 바꾸지 않는다. 기본 official은 원형이다.
    external_organization_label_policy: str = "official"
    # 외부 corpus의 점수 척도는 대회와 다르다(AIHub는 고득점 편향이 크다). 절대
    # MSE는 그 차이를 그대로 학습하고 hard_ranknet은 절대값을 전부 버린다.
    # quantile_to_competition_v1은 그 중간으로, source별 경험 분위수를 대회 train
    # label 분포에 사상해 순서를 보존한 채 척도만 맞춘다. train external row에만
    # 적용하고 validation 원문·label은 건드리지 않는다.
    external_score_alignment: str = "none"
    # 대회 train의 문단/문장 순서 synthetic pair를 온라인으로 만드는 O-only
    # augmentation이다. 별도 processed essay 폴더를 만들지 않는다.
    organization_augmentation: str = "none"
    # -1은 기존 full pool을 그대로 쓴다. 0 이상이면 official train은 모두
    # 보존하고 origin_pool_extra만 두 원평가자의 9개 준거 절대차 합으로 거른다.
    origin_extra_max_rater_disagreement: float = -1.0
    # --- 미학습 문항 평가용 보류 (2026-08-20) ---------------------------------
    # 쉼표로 구분한 prompt_num 목록. 여기 적힌 문항은 **학습에서 완전히 빠지고**
    # validation의 같은 문항 에세이가 "한 번도 본 적 없는 문항" 평가가 된다.
    #
    # 왜 데이터 파이프라인에 넣는가
    #   2025 최종보고서 표 V-6에서 채점 데이터 4,000편의 71.1%가 우리 학습에 없는
    #   문항(Q11 1,512 / Q12 1,331편)임을 확인했다. 그런데 공개 validation 400편은
    #   학습에서 **본** 9문항의 새 에세이라 이 격차를 측정할 수 없다. 임시로
    #   `--train-file`에 필터 파일을 넘겨서 재 봤지만, 그 방식은 매번 잊을 수 있고
    #   run 사이 비교가 깨진다. 설정으로 두면 resolved_config.json에 남고 모든
    #   run이 같은 규칙으로 미학습 문항 지표를 갖는다.
    #
    # 빈 문자열이면 완전히 꺼지고 기존 run과 bit-exact 동일하다. 최종 제출 모델은
    # 이 값을 비워 전체 데이터로 다시 학습한다 — 보류는 개발용 저울이다.
    unseen_prompt_holdout: str = ""
    # --- 지문 dropout (2026-08-20) --------------------------------------------
    # 학습 시 이 확률로 문항 지문을 입력에서 뺀다. validation/inference는 절대
    # 건드리지 않는다.
    #
    # 왜 전량 제거(essay_only_v1)와 따로 두는가
    #   content 준거 C1(문제 상황 제시)·C2(주장)는 정의상 논제에 상대적이라 지문을
    #   아예 안 보면 원리적으로 판단할 수 없다. 반면 organization·expression은 글
    #   자체의 성질이다. 지금 아키텍처는 pooled 표현 하나로 세 영역을 다 내므로
    #   trait마다 다른 입력을 주려면 별도 forward가 필요하다.
    #
    #   dropout은 그 사이의 연속 축이다. p=0이면 baseline_v1과 bit-exact 같고,
    #   p=1이면 essay_only_v1과 같다. 중간값에서는 "지문이 있으면 쓰되 없어도
    #   채점할 수 있는" 표현을 배운다. 2025 채점 데이터의 71%가 우리 학습에 없는
    #   Q11·Q12이므로, 미학습 문항에서 지문이 오히려 해가 된다면 이쪽이 전량
    #   제거보다 손실이 작을 수 있다.
    prompt_dropout_probability: float = 0.0
    # --- C7 문항 적대적 학습 (2026-08-20) --------------------------------------
    # pooled 표현에서 **어느 문항의 답안인지** 못 맞히게 만든다. 작은 판별기를 붙이고
    # gradient를 뒤집는(DANN) 방식이다.
    #
    # 왜 prompt dropout과 별개인가
    #   dropout은 **입력**에서 지문 문자열을 뺀다. 그런데 에세이 본문 자체가 주제
    #   어휘로 문항을 강하게 드러내므로(문항 9개, 각 1,300여 편) 지문을 빼도 표현은
    #   여전히 문항별로 갈린다. 적대적 항은 **표현** 수준에서 그 정보를 지운다.
    #   두 축은 서로 보완적이고 함께 켤 수 있다.
    #
    # 위험: content 준거 C1(문제 상황 제시)·C2(주장)는 논제 대응을 보므로 완전
    # 불변은 해롭다. 그래서 가중치를 작게 두고 `unseen_prompt_rmse`로 재야 한다.
    # 0이면 판별기를 아예 만들지 않아 parameter 수와 graph가 기존과 동일하다.
    prompt_adversary_weight: float = 0.0
    # 판별기 출력 class 수. prompt_group_id(Qn -> n)를 그대로 색인으로 쓰므로
    # 등장할 수 있는 문항 번호의 상한이면 된다. 2025 자료의 Q12까지 여유 있게 덮는다.
    prompt_adversary_classes: int = 32
    # Legacy checkpoint JSON load only. New CLI does not expose this field.
    extended_data_dir: str = str(DATA_ROOT)
    extended_datasets: str = ""
    dataset_schedule: str = "competition_only"
    # full NIKL의 공개 official_train 2,000편을 target source로, 나머지
    # origin_pool_extra 9,600편을 pretraining source로 분리한다. 기본 false에서는
    # 지금까지처럼 두 source를 하나의 primary pool로 섞는다.
    split_primary_sources: bool = False
    extended_source_sampling: str = "uniform"
    # 비어 있으면 uniform/proportional을 사용한다. 필요할 때
    # "aihub26_essay=3,aihub27_topic=2"처럼 source 비율을 직접 지정한다.
    extended_source_weights: str = ""
    competition_mix_ratio: float = 0.5
    competition_every_n_steps: int = 2
    final_competition_epochs: int = 0
    # 모델 선택/성능 비교에서는 train row와 validation row가 ID 또는 정규화 본문으로
    # 하나라도 겹치면 즉시 중단한다. validation까지 학습하는 최종 refit처럼 평가
    # metric을 사용하지 않는 명시적인 최종 학습에서만 CLI로 이 보호를 해제한다.
    allow_validation_overlap: bool = False
    # validation_leaked profile은 일반 overlap 허용과 별도로 한 번 더 동의해야 한다.
    allow_validation_leaked: bool = False
    # 공식 validation은 400편뿐이라 arm 사이 차이를 재는 짝지은 표준오차가 약 .005다.
    # head/목적함수의 실제 효과 크기(.002~.005)가 그 아래라 400편만으로는 원리적으로
    # 판정할 수 없다. 이 값을 켜면 origin_pool_extra에서 그만큼을 학습에서 빼
    # validation에 붙여 표준오차를 1/sqrt(1+N/400)로 줄인다. official_train 2,000편은
    # 공식 validation과 같은 분포이므로 건드리지 않고 전부 학습에 남긴다. 0이면 기존과
    # 완전히 같다. 지표는 source_split별로도 나뉘어 기록되므로 공식 400편 값은 그대로
    # 비교할 수 있다.
    validation_holdout_size: int = 0
    # 최종 c02도 primary trait MSE 비중 1을 유지하면서 아래 보조항을 함께 쓴다.
    mse_loss_weight: float = 1.0
    # --- 평가셋 라벨 분포 이동 보정 (2026-08-20) --------------------------------
    # 운영측이 "예측값이 0점으로 처리된 샘플은 없다"를 확인해 주면서 유실 가설이
    # 죽었다. 남은 설명은 하나다: 리더보드 RMSE 0.5191 / Spearman 0.7340은 로컬
    # 400편(정답 SD 0.653)에서의 0.4168 / 0.7597과 **같은 모델, 같은 순위 능력**으로
    # 정답 SD가 더 넓은 집합을 채점했을 때 나오는 값이다.
    #   RMSE^2 = σ_p^2 + σ_g^2 - 2·r·σ_p·σ_g  에 로컬 상수(σ_p=0.494, r=0.770)를
    #   그대로 넣고 σ_g만 0.653 -> 0.79로 바꾸면 0.5168이 나온다(실측 0.5191).
    # Spearman이 0.7597 -> 0.7340으로 거의 그대로인 것이 이 해석의 근거다. 미학습
    # 문항 때문이라면 순위 능력이 같이 무너져야 하는데 그러지 않았다.
    #
    # 우리 예측 SD는 0.494로 정답 SD의 0.755배뿐이고, 영역 점수 1,200개 중 3점과
    # 4점이 1,095개(91%)다. 학습 라벨 자체가 평균 3.40 / SD 0.617에 몰려 있어
    # (2.0 미만 1.5%, 4.5 초과 2.8%) 꼬리를 배울 기회가 없었다.
    #
    # 사후 확장은 이미 반증했다. average_matched 뒤에 평균 주변으로 λ배를 하면
    # λ>1 전 구간에서 RMSE와 Spearman이 **같이** 나빠진다(λ=1.05 이미 0.4228).
    # 그래서 추론이 아니라 **학습 시점에** 꼬리 표본의 비중을 올린다.
    #
    #   w(y) = exp( (y-μ)²/(2σ_ref²) − (y-μ)²/(2σ_target²) ) · (σ_ref/σ_target)
    #
    # 라벨 분포 N(μ, σ_ref²)를 N(μ, σ_target²)로 옮기는 중요도 가중치 그대로다.
    # (σ_ref/σ_target) 배는 E[w]=1을 만드는 닫힌 해라서 batch 구성에 흔들리지 않는다.
    # target_sigma = 0이면 완전히 꺼지고 기존 run과 bit-exact 동일하다.
    tail_weight_target_sigma: float = 0.0
    tail_weight_reference_mean: float = 3.398
    tail_weight_reference_sigma: float = 0.617
    # 상한이 없으면 최저점 소수 표본이 batch gradient를 독점한다. 4.0은 y=1.5에서
    # σ_target=0.90일 때 걸리는 값이다.
    tail_weight_max: float = 4.0
    # 세 trait의 절대점수 loss 비중. 기본 1/1/1은 기존 elementwise mean과 같다.
    content_loss_weight: float = 1.0
    organization_loss_weight: float = 1.0
    expression_loss_weight: float = 1.0
    pairwise_loss: str = "none"
    pairwise_loss_weight: float = 0.0
    pairwise_temperature: float = 0.5
    pairwise_gap_weighted: bool = False
    listwise_loss: str = 'soft_spearman'
    listwise_loss_weight: float = 0.2
    listwise_temperature: float = 0.5
    # ranking만 organization에 적용하는 실험을 위해 main loss weight와 분리한다.
    ranking_content_weight: float = 1.0
    ranking_organization_weight: float = 1.0
    ranking_expression_weight: float = 1.0
    # 리더보드 지표가 trait별 RMSE/Spearman의 평균이 아니라 세 trait 평균 점수 1개에서
    # 계산된 것으로 보이는 2026-08-05 진단에 대응하는 auxiliary objective다. 기존
    # per-trait loss는 그대로 두고 essay별 평균 점수의 오차만 추가로 줄인다. trait 오차의
    # 공분산까지 벌하므로 per-trait MSE만으로는 얻을 수 없는 축이다. 최종 c02는 0.25이고
    # 역사적 Y1의 0.5와 frozen legacy baseline의 0은 각 완전 preset에 고정한다.
    trait_average_loss_weight: float = 0.0
    # 같은 평균 점수에 대한 in-batch pairwise ranking. 정의 C의 Spearman을 직접 겨냥한다.
    trait_average_pairwise_weight: float = 0.0
    # 순위 로스(pairwise/listwise)가 볼 대상이다. 기본 per_trait가 기존 동작이다.
    ranking_target: str = "per_trait"
    # 실제 제출 정수 표면을 학습 loss에 넣는 opt-in 축이다. 모든 weight가 0이고
    # rule=none인 기본값은 기존 c02/Y6 학습 graph를 전혀 호출하지 않는다.
    quantization_rule: str = "none"
    # soft: 125-state 기대 점수, straight_through: hard forward/soft backward,
    # expected_risk: 125개 정수 상태의 squared cost 기대값을 정확히 계산한다.
    quantization_surrogate: str = "soft"
    # raw가 공식 계약이다. 나머지는 사용자가 요청한 rounded-label ablation이며,
    # gold 반올림이 평가 규정으로 확정됐다는 뜻이 아니다.
    quantized_target_rule: str = "raw"
    quantized_error_form: str = "rmse"
    quantized_trait_loss_weight: float = 0.0
    quantized_mean_loss_weight: float = 0.0
    quantized_pooled_loss_weight: float = 0.0
    quantized_trait_rank_weight: float = 0.0
    quantized_mean_rank_weight: float = 0.0
    quantized_pooled_rank_weight: float = 0.0
    quantization_temperature: float = 0.5
    quantization_final_temperature: float = 0.15
    quantization_allocation_temperature: float = 0.5
    quantization_final_allocation_temperature: float = 0.15
    quantized_rank_temperature: float = 0.5
    quantized_loss_start_step: int = 0
    quantized_loss_ramp_steps: int = 0
    quantization_anneal_steps: int = 0
    inbatch_sampling: str = "random"
    best_checkpoint_metric: str = 'official_matched_rmse'
    # legacy_core는 기존 7개 core sibling을 그대로 보존한다. 새 quant stage만
    # selected_pair를 켜 primary+secondary+final 세 벌로 디스크 사용을 제한한다.
    checkpoint_retention: str = "legacy_core"
    secondary_checkpoint_metric: str = "auto"
    # 모델 원점수를 제출 점수로 바꾸는 규칙. 연구 산출물과 서빙이 **같은 함수**를 쓰게
    # 하려고 config 축으로 뒀다. 기본값은 공식 지표 최적인 평균 정합 정수 삼중이다.
    # 원점수는 어떤 규칙에서도 그대로 저장되므로 나중에 다른 규칙으로 재측정할 수 있다.
    score_postprocess: str = DEFAULT_SCORE_POSTPROCESS
    seed: int = 42

    def validate(self) -> "RegressionConfig":
        """Validate five visible option groups and return this immutable config."""

        self._validate_model_and_architecture()
        self._validate_training()
        self._validate_data()
        self._validate_losses()
        self._validate_flags()
        return self

    def _validate_model_and_architecture(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported schema_version={self.schema_version}")
        resolve_model(self.model_id)
        if not self.model_revision.strip():
            raise ValueError("model_revision must not be empty")
        if self.model_slug and not self.model_slug.strip():
            raise ValueError("model_slug must be empty or non-whitespace")
        if self.training_mode not in TRAINING_MODES:
            raise ValueError(f"training_mode choices={TRAINING_MODES}")
        if self.backbone_type not in BACKBONE_TYPES:
            raise ValueError(f"backbone_type choices={BACKBONE_TYPES}")
        if self.pooling not in POOLING_MODES:
            raise ValueError(f"pooling choices={POOLING_MODES}")
        analysis_weight = float(self.analysis_pool_weight)
        if not math.isfinite(analysis_weight) or not 0.0 <= analysis_weight <= 1.0:
            raise ValueError("analysis_pool_weight는 [0,1] 범위여야 합니다")
        if self.pooling in ANALYSIS_MIX_POOLINGS and self.analysis_pool_learnable:
            # logit 변환이 유한해야 한다. 0/1은 학습 가능한 지분의 초기값이 될 수 없다.
            if not 0.0 < analysis_weight < 1.0:
                raise ValueError(
                    "analysis_pool_learnable=True면 analysis_pool_weight는 (0,1) "
                    "열린 구간이어야 합니다 (logit 초기값이 유한해야 함)"
                )
        if self.pooling not in ANALYSIS_MIX_POOLINGS and self.analysis_pool_per_trait:
            raise ValueError(
                "analysis_pool_per_trait는 "
                f"pooling={ANALYSIS_MIX_POOLINGS} 에서만 의미가 있습니다"
            )
        if self.layer_aggregation not in LAYER_AGGREGATION_MODES:
            raise ValueError(f"layer_aggregation choices={LAYER_AGGREGATION_MODES}")
        if self.input_format not in INPUT_FORMATS:
            raise ValueError(f"input_format choices={INPUT_FORMATS}")
        if self.rubric_profile not in RUBRIC_PROFILES:
            raise ValueError(f"rubric_profile choices={RUBRIC_PROFILES}")
        if self.criterion_readout not in CRITERION_READOUTS:
            raise ValueError(f"criterion_readout choices={CRITERION_READOUTS}")
        rubric_conditioned = self.input_format == "rubric_conditioned_v1"
        if not rubric_conditioned and (
            self.rubric_profile != "none" or self.criterion_readout != "shared"
        ):
            raise ValueError(
                "rubric_profile/criterion_readout의 non-default 값은 "
                "input_format='rubric_conditioned_v1'에서만 사용할 수 있습니다"
            )
        if rubric_conditioned:
            if self.rubric_profile == "none":
                raise ValueError(
                    "rubric_conditioned_v1에는 versioned rubric_profile이 필요합니다"
                )
            if self.backbone_type != "decoder":
                raise ValueError("rubric_conditioned_v1은 decoder backbone만 지원합니다")
            # RC tokenizer가 만드는 mask는 "base prefix = 1, readout suffix = 0" 하나뿐이다.
            # essay/문장/문단 mask를 만들지 않으므로 그것을 요구하는 pooling은 쓸 수 없다.
            # 자의적 제한이 아니라 _tokenize_rubric_conditioned의 실제 출력 한계다.
            if self.pooling != "mean":
                raise ValueError(
                    "rubric_conditioned_v1의 tokenizer는 essay mask를 만들지 않으므로 "
                    f"pooling={self.pooling!r}를 지원하지 않습니다 (pooling='mean'만 가능)"
                )
            if self.organization_pooling != "shared":
                raise ValueError(
                    "rubric_conditioned_v1의 tokenizer는 문장/문단 mask를 만들지 않으므로 "
                    f"organization_pooling={self.organization_pooling!r}를 지원하지 "
                    "않습니다 (organization_pooling='shared'만 가능)"
                )
            # detail head는 여기서 강제하지 않는다. "입력에 채점 기준 텍스트를 넣는다"와
            # "9개 anchor에서 지표별로 읽어낸다"는 분리 가능한 두 축이고, 확정 구조인
            # 3-trait 정수 5-class direct와도 결합할 수 있어야 한다. 실제 정합성은
            # detail_final_source / criterion_readout의 일반 gate가 이미 검사한다.
            if (
                self.criterion_readout == "textual_anchor_residual"
                and self.detail_final_source != "criterion"
            ):
                raise ValueError(
                    "textual_anchor_residual은 9개 anchor를 최종 점수 경로에 쓰므로 "
                    "detail_final_source='criterion'이 필요합니다"
                )
        if self.essay_surface not in ESSAY_SURFACES:
            raise ValueError(f"essay_surface choices={ESSAY_SURFACES}")
        if not 0 <= self.train_canonical_surface_probability <= 1:
            raise ValueError("train_canonical_surface_probability must be in [0, 1]")
        if self.train_canonical_surface_probability > 0 and self.essay_surface == "canonical":
            raise ValueError(
                "train_canonical_surface_probability > 0 requires a non-canonical "
                "deployment essay_surface"
            )
        if self.head_type not in HEAD_TYPES:
            raise ValueError(f"head_type choices={HEAD_TYPES}")
        if self.prompt_head_mode not in PROMPT_HEAD_MODES:
            raise ValueError(f"prompt_head_mode choices={PROMPT_HEAD_MODES}")
        if self.prompt_head_traits not in PROMPT_HEAD_TRAITS:
            raise ValueError(f"prompt_head_traits choices={PROMPT_HEAD_TRAITS}")
        if self.organization_pooling not in ORGANIZATION_POOLING_MODES:
            raise ValueError(
                f"organization_pooling choices={ORGANIZATION_POOLING_MODES}"
            )
        if (
            self.organization_pooling == "paragraph_mean"
            and self.essay_surface != "official_raw"
        ):
            # flat은 모든 whitespace를 한 칸으로 접어 문단 cue를 지우고, canonical은
            # 재현 불가능한 원천 경계를 쓴다. 배포 계약 표면에서만 의미가 있다.
            raise ValueError(
                "paragraph_mean은 이중공백 문단 cue가 남아 있는 "
                "essay_surface='official_raw'에서만 사용할 수 있습니다"
            )
        if self.prompt_head_mode != "none" and self.head_type != "independent":
            raise ValueError(
                "문제별 head는 먼저 head_type=independent에서만 지원합니다"
            )
        if not 0 <= self.prompt_head_weight <= 1:
            raise ValueError("prompt_head_weight must be in [0, 1]")
        try:
            normalized_registry = tuple(
                (str(prompt_number), str(prompt))
                for prompt_number, prompt in self.prompt_registry
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "prompt_registry item은 (prompt_num, prompt_text)여야 합니다"
            ) from exc
        if len(normalized_registry) != len(set(normalized_registry)):
            raise ValueError("prompt_registry에는 중복 문제가 없어야 합니다")
        if self.score_head not in SCORE_HEADS:
            raise ValueError(f"score_head choices={SCORE_HEADS}")
        if self.score_head == "regression_ordinal":
            if self.head_type not in UNMIXED_HEAD_TYPES:
                raise ValueError(
                    "regression_ordinal은 trait를 섞지 않는 head가 필요합니다: "
                    f"head_type choices={UNMIXED_HEAD_TYPES}"
                )
            if self.prompt_head_mode != "none":
                raise ValueError(
                    "regression_ordinal 첫 구현은 prompt_head_mode='none'만 "
                    "지원합니다"
                )
            if self.distribution_loss_weight != 0:
                raise ValueError(
                    "regression_ordinal에서는 사용되지 않는 "
                    "distribution_loss_weight를 0으로 명시해야 합니다"
                )
            if self.ordinal_loss_weight <= 0:
                raise ValueError(
                    "regression_ordinal에는 ordinal_loss_weight > 0이 필요합니다"
                )
        if self.detail_head_mode not in DETAIL_HEAD_MODES:
            raise ValueError(f"detail_head_mode choices={DETAIL_HEAD_MODES}")
        if self.detail_final_source not in DETAIL_FINAL_SOURCES:
            raise ValueError(f"detail_final_source choices={DETAIL_FINAL_SOURCES}")
        if self.detail_final_source == "criterion" and self.detail_head_mode == "none":
            raise ValueError(
                "detail_final_source='criterion'에는 활성 detail_head_mode가 필요합니다"
            )
        if self.train_label_rounding not in TRAIN_LABEL_ROUNDINGS:
            raise ValueError(f"train_label_rounding choices={TRAIN_LABEL_ROUNDINGS}")
        if self.categorical_readout not in CATEGORICAL_READOUTS:
            raise ValueError(f"categorical_readout choices={CATEGORICAL_READOUTS}")
        if self.gumbel_temperature <= 0:
            raise ValueError("gumbel_temperature must be positive")
        # score_head는 direct branch만 바꾼다. 최종 점수를 criterion 집계에서 내면
        # 이 축은 결과에 전혀 나타나지 않아 "차이 없는 두 run"이 만들어진다.
        if self.score_head != "regression" and self.detail_final_source != "direct":
            raise ValueError(
                f"score_head={self.score_head!r}는 direct branch만 바꾸므로 "
                "detail_final_source='direct'가 필요합니다"
            )
        final_head_is_categorical = (
            self.detail_head_mode in CATEGORICAL_DETAIL_HEAD_MODES
            if self.detail_final_source == "criterion"
            else self.score_head in {"distribution", "trait_native_distribution"}
        )
        if self.categorical_readout != "expectation":
            if not final_head_is_categorical:
                raise ValueError(
                    "categorical_readout은 최종 점수를 내는 head가 분류일 때만 "
                    "사용할 수 있습니다"
                )
            if self.head_type not in UNMIXED_HEAD_TYPES:
                # weighted_mixed/mixed_head_v2는 확률이나 중간 표현을 trait 사이에서
                # 섞는다. 섞인 뒤에는 Gumbel이 쓸 logits이 남지 않고 argmax도 의미가
                # 흐려지므로 첫 gate에서 expectation으로 고정한다.
                raise ValueError(
                    "expectation이 아닌 categorical_readout은 trait를 섞지 않는 "
                    f"head가 필요합니다: head_type choices={UNMIXED_HEAD_TYPES}"
                )
            if self.categorical_readout == "argmax":
                # argmax는 점수 경로의 gradient를 끊는다. cross entropy가 없으면
                # head가 전혀 학습되지 않으므로 조용한 실패 대신 여기서 막는다.
                cross_entropy_weight = (
                    self.detail_distribution_loss_weight
                    + self.detail_halfstep_loss_weight
                    + self.detail_rater_set_loss_weight
                    if self.detail_final_source == "criterion"
                    else self.distribution_loss_weight
                )
                if cross_entropy_weight <= 0:
                    raise ValueError(
                        "categorical_readout='argmax'는 점수 경로가 미분 불가하므로 "
                        "해당 분류 loss weight가 양수여야 합니다. gradient가 필요하면 "
                        "'gumbel_straight_through'를 쓰세요"
                    )
        if self.score_parameterization not in SCORE_PARAMETERIZATIONS:
            raise ValueError(
                f"score_parameterization choices={SCORE_PARAMETERIZATIONS}"
            )
        if self.score_parameterization == "average_plus_contrast":
            # 세 head 출력을 (평균, 대비1, 대비2)로 재해석하는 방식이므로 trait마다
            # 5개 logit을 내는 distribution head나 trait 사이를 섞는 head는 쓸 수
            # 없다. 최종 점수가 이 branch에서 나오지 않으면 재해석이 조용히
            # 무시되므로 detail_final_source='direct'도 함께 요구한다.
            if self.score_head != "regression":
                raise ValueError(
                    "average_plus_contrast는 score_head='regression'만 지원합니다"
                )
            if self.head_type not in UNMIXED_HEAD_TYPES:
                raise ValueError(
                    "average_plus_contrast는 trait를 섞지 않는 head가 필요합니다: "
                    f"head_type choices={UNMIXED_HEAD_TYPES}"
                )
            if self.prompt_head_mode != "none":
                raise ValueError(
                    "첫 average_plus_contrast gate는 prompt_head_mode='none'으로 "
                    "고정합니다"
                )
            if self.detail_final_source != "direct":
                raise ValueError(
                    "average_plus_contrast는 최종 점수를 이 branch에서 내야 하므로 "
                    "detail_final_source='direct'가 필요합니다"
                )
        try:
            normalized_raters = tuple(
                (str(source_dataset), str(rater_id))
                for source_dataset, rater_id in self.detail_rater_registry
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "detail_rater_registry item은 (source_dataset, evaluator_id)여야 합니다"
            ) from exc
        if any(
            not source_dataset.strip() or not rater_id.strip()
            for source_dataset, rater_id in normalized_raters
        ):
            raise ValueError(
                "detail_rater_registry의 source_dataset/evaluator_id는 비어 있을 수 없습니다"
            )
        if len(normalized_raters) != len(set(normalized_raters)):
            raise ValueError("detail_rater_registry에는 중복 ID가 없어야 합니다")
        if not 0 <= self.mixed_head_weight <= 1:
            raise ValueError("mixed_head_weight must be in [0, 1]")

    def _validate_training(self) -> None:
        for name in (
            "max_length",
            "batch_size",
            "gradient_accumulation",
            "epochs",
            "head_epochs",
            "joint_epochs",
            "lora_r",
            "lora_alpha",
            "head_hidden_size",
            "last_n_layers",
            "attention_pool_dim",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        for name in (
            "max_train_steps",
            "head_warmup_steps",
            "eval_steps",
            "final_competition_epochs",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.max_train_steps == 0 and any(
            (self.head_warmup_steps, self.eval_steps, self.final_competition_epochs)
        ):
            raise ValueError(
                "head_warmup_steps/eval_steps/final_competition_epochs는 "
                "max_train_steps > 0인 step mode에서만 사용합니다"
            )
        if self.competition_every_n_steps < 1:
            raise ValueError("competition_every_n_steps must be positive")
        for name in (
            "head_learning_rate",
            "lora_learning_rate",
            "lora_plus_lr_ratio",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.lr_scheduler_type not in {"constant", "linear", "cosine"}:
            raise ValueError(
                "lr_scheduler_type choices=('constant', 'linear', 'cosine')"
            )
        if self.lora_scheduler_scope not in LORA_SCHEDULER_SCOPES:
            raise ValueError(f"lora_scheduler_scope choices={LORA_SCHEDULER_SCOPES}")
        if self.lora_scheduler_scope == "joint" and self.training_mode != "two_stage":
            raise ValueError(
                "lora_scheduler_scope='joint'는 training_mode='two_stage'에서만 "
                "사용할 수 있습니다"
            )
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not self.lora_targets.strip():
            raise ValueError("lora_targets must not be empty")
        if self.initial_lora_adapter and self.training_mode == "head_only":
            raise ValueError(
                "initial_lora_adapter는 lora_only 또는 two_stage에서만 사용할 수 있습니다"
            )
        if self.paragraph_boundary_loss_weight > 0:
            if self.training_mode == "head_only":
                raise ValueError(
                    "paragraph boundary multi-task는 shared backbone을 학습하는 "
                    "lora_only 또는 two_stage가 필요합니다"
                )
            # boundary head는 essay span의 token hidden만 소비한다. essay_only_v1도
            # 같은 essay를 마지막 span으로 두고 같은 essay_surface를 쓰므로 supervision
            # 자체가 동일하다. 지문 유무는 boundary label과 무관하다.
            if self.input_format not in {"baseline_v1", "essay_only_v1"}:
                raise ValueError(
                    "paragraph boundary multi-task는 input_format="
                    "'baseline_v1' 또는 'essay_only_v1'이어야 합니다"
                )
            if self.essay_surface != "official_raw":
                raise ValueError(
                    "paragraph boundary multi-task는 scorer 입력을 official_raw로 "
                    "고정합니다"
                )
            if self.train_canonical_surface_probability != 0:
                raise ValueError(
                    "paragraph boundary multi-task는 canonical을 label에만 쓰므로 "
                    "train_canonical_surface_probability=0이어야 합니다"
                )
            # boundary head 자체는 pooling과 무관하다. token hidden만 소비하므로
            # pooling=mean/first/last 어느 쪽과도 결합할 수 있고, 그래서 이전의
            # `pooling != "mean"` 금지는 근거가 없었다.
            #
            # 다만 `RegressionCollator.__call__`의 분기가 배타적이다.
            # `elif self.include_paragraph_boundaries:`가 `elif use_essay_mask:`보다
            # 앞이라 두 조건이 동시에 참이면 essay_mask를 만드는 분기에 도달하지 못하고
            # `padded_essay_masks`가 정의되지 않은 채 참조된다. 따라서 essay_mask를
            # 요구하는 두 축만 정확히 막는다. 두 분기를 합치면 이 제약도 풀 수 있다.
            if self.pooling.startswith("essay_"):
                raise ValueError(
                    "paragraph boundary multi-task는 essay_mask가 필요한 "
                    f"pooling={self.pooling!r}와 함께 쓸 수 없습니다 "
                    "(collator가 두 supervision을 동시에 만들지 못한다)"
                )
            if self.organization_pooling != "shared":
                raise ValueError(
                    "paragraph boundary multi-task는 essay_mask가 필요한 "
                    f"organization_pooling={self.organization_pooling!r}와 함께 쓸 수 "
                    "없습니다 (collator가 두 supervision을 동시에 만들지 못한다)"
                )

    def _validate_data(self) -> None:
        if self.dataset_schedule not in DATASET_SCHEDULES:
            raise ValueError(f"dataset_schedule choices={DATASET_SCHEDULES}")
        if self.extended_source_sampling not in EXTENDED_SOURCE_SAMPLING_MODES:
            raise ValueError(
                "extended_source_sampling choices=" f"{EXTENDED_SOURCE_SAMPLING_MODES}"
            )
        if self.primary_data_profile not in PRIMARY_DATA_PROFILES:
            raise ValueError(
                f"primary_data_profile choices={tuple(PRIMARY_DATA_PROFILES)}"
            )
        if self.primary_data_profile == "validation_leaked" and not (
            self.allow_validation_overlap and self.allow_validation_leaked
        ):
            raise ValueError(
                "validation_leaked profile에는 --allow-validation-overlap과 "
                "--allow-validation-leaked가 모두 필요합니다"
            )
        if self.score_postprocess not in SCORE_POSTPROCESS_RULES:
            raise ValueError(f"score_postprocess choices={SCORE_POSTPROCESS_RULES}")
        if self.validation_holdout_size < 0:
            raise ValueError("validation_holdout_size는 0 이상이어야 합니다")
        if self.validation_holdout_size and self.primary_data_profile != "full":
            raise ValueError(
                "validation_holdout_size는 origin_pool_extra가 있는 full profile에서만 "
                f"쓸 수 있습니다: primary_data_profile={self.primary_data_profile!r}"
            )
        if self.paragraph_boundary_loss_weight > 0 and (
            self.primary_data_profile != "full"
            or self.dataset_schedule != "competition_only"
            or self.extended_datasets.strip()
            or self.organization_augmentation != "none"
            or self.split_primary_sources
        ):
            raise ValueError(
                "첫 paragraph boundary multi-task gate는 clean full NIKL의 "
                "competition_only schedule로 고정합니다"
            )
        if not self.dataset_root.strip():
            raise ValueError("dataset_root must not be empty")
        for name in ("competition_train_limit", "external_train_limit"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        agreement_threshold = self.origin_extra_max_rater_disagreement
        if not math.isfinite(agreement_threshold) or (
            agreement_threshold != -1.0 and agreement_threshold < 0
        ):
            raise ValueError(
                "origin_extra_max_rater_disagreement must be -1 (off) or non-negative"
            )

        selected_external = {
            name for name in self.extended_datasets.replace(",", " ").split() if name
        }
        uses_organization_augmentation = self.organization_augmentation != "none"
        if (
            self.external_organization_label_policy
            not in EXTERNAL_ORGANIZATION_LABEL_POLICIES
        ):
            raise ValueError(
                "external_organization_label_policy choices="
                f"{EXTERNAL_ORGANIZATION_LABEL_POLICIES}"
            )
        if self.organization_augmentation not in ORGANIZATION_AUGMENTATIONS:
            raise ValueError(
                f"organization_augmentation choices={ORGANIZATION_AUGMENTATIONS}"
            )
        if self.external_score_alignment not in EXTERNAL_SCORE_ALIGNMENTS:
            raise ValueError(
                f"external_score_alignment choices={EXTERNAL_SCORE_ALIGNMENTS}"
            )
        if self.external_score_alignment != "none" and not selected_external:
            raise ValueError(
                "external_score_alignment에는 extended_datasets가 필요합니다"
            )
        if (
            self.external_organization_label_policy
            == "aihub24_highschool_structural_v1"
            and "aihub24_essay" not in selected_external
        ):
            raise ValueError(
                "aihub24_highschool_structural_v1에는 "
                "extended_datasets에 aihub24_essay가 필요합니다"
            )
        if (
            self.external_organization_label_policy
            == "aihub24_structural_aihub26_persuasion_v1"
            and selected_external != {"aihub24_essay", "aihub26_essay"}
        ):
            raise ValueError(
                "aihub24_structural_aihub26_persuasion_v1에는 "
                "extended_datasets=aihub24_essay,aihub26_essay만 사용합니다"
            )
        if self.external_rankable_trait not in {"", *TRAITS}:
            raise ValueError(
                "external_rankable_trait choices=" f"{('', *TRAITS)}"
            )
        if self.external_rankable_trait and not selected_external:
            raise ValueError(
                "external_rankable_trait에는 extended_datasets가 필요합니다"
            )
        if self.external_rankable_trait and self.external_train_limit:
            raise ValueError(
                "external_rankable_trait은 문제 묶음을 보존해야 하므로 "
                "external_train_limit과 함께 사용할 수 없습니다"
            )
        allowed_external = set(EXTERNAL_DATASET_ALIASES) | set(
            LEGACY_EXTERNAL_DATASET_ALIASES
        )
        unknown_external = selected_external - allowed_external
        if unknown_external:
            raise ValueError(
                f"unknown external datasets={sorted(unknown_external)}; "
                f"choices={tuple(sorted(allowed_external))}"
            )
        if self.split_primary_sources:
            if self.primary_data_profile != "full":
                raise ValueError(
                    "split_primary_sources는 full NIKL profile에서만 사용할 수 있습니다"
                )
            if selected_external:
                raise ValueError(
                    "split_primary_sources gate에는 별도 external dataset을 함께 쓰지 않습니다"
                )
            if self.dataset_schedule == "competition_only":
                raise ValueError(
                    "split_primary_sources에는 mixed, alternating 또는 "
                    "pretrain_then_competition schedule이 필요합니다"
                )
        if selected_external and self.essay_surface in {
            "official_raw",
            "official_gap_newline",
            "official_raw_kiwi_sentence_newline_v1",
        }:
            raise ValueError(
                "official_raw 계열은 NIKL competition 전용입니다. external dataset에는 "
                "동일한 raw surface 계약이 없으므로 별도 input policy를 먼저 정의하세요."
            )
        if not 0 < self.competition_mix_ratio <= 1:
            raise ValueError("competition_mix_ratio must be in (0, 1]")
        if (
            self.dataset_schedule == "competition_only"
            and (self.extended_datasets.strip() or uses_organization_augmentation)
        ):
            raise ValueError(
                "확장 데이터를 쓰려면 dataset_schedule을 external_only, mixed, "
                "alternating, pretrain_then_competition 중 하나로 지정하세요"
            )
        if (
            self.dataset_schedule != "competition_only"
            and not self.extended_datasets.strip()
            and not self.split_primary_sources
            and not uses_organization_augmentation
        ):
            raise ValueError(
                "확장 데이터 schedule에는 extended_datasets, split_primary_sources "
                "또는 organization_augmentation이 필요합니다"
            )
        if self.dataset_schedule == "external_only" and self.split_primary_sources:
            raise ValueError("external_only는 split_primary_sources와 함께 쓰지 않습니다")
        if self.dataset_schedule == "external_only" and self.final_competition_epochs:
            raise ValueError("external_only에는 final_competition_epochs를 쓰지 않습니다")
        if (
            self.dataset_schedule == "pretrain_then_competition"
            and self.final_competition_epochs < 1
        ):
            raise ValueError(
                "pretrain_then_competition에는 final_competition_epochs >= 1이 필요합니다"
            )
        if self.dataset_schedule != "competition_only" and self.max_train_steps < 1:
            raise ValueError(
                "확장 데이터 학습은 재현 가능한 step schedule을 위해 "
                "max_train_steps >= 1이 필요합니다"
            )
        if self.dataset_schedule == "mixed" and self.competition_mix_ratio >= 1:
            raise ValueError(
                "mixed는 competition/external을 모두 쓰도록 "
                "competition_mix_ratio < 1이 필요합니다"
            )
        if self.dataset_schedule == "mixed" and self.max_train_steps < 2:
            raise ValueError(
                "mixed에서 competition/external step을 하나씩 쓰려면 "
                "max_train_steps >= 2가 필요합니다"
            )
        if self.dataset_schedule == "alternating":
            if self.competition_every_n_steps < 2:
                raise ValueError(
                    "alternating은 external/competition을 모두 쓰도록 "
                    "competition_every_n_steps >= 2가 필요합니다"
                )
            if self.max_train_steps < self.competition_every_n_steps:
                raise ValueError(
                    "alternating main phase에 competition step이 하나 이상 있도록 "
                    "max_train_steps >= competition_every_n_steps가 필요합니다"
                )

    def _validate_losses(self) -> None:
        if self.pairwise_loss not in PAIRWISE_LOSSES:
            raise ValueError(f"pairwise_loss choices={PAIRWISE_LOSSES}")
        if self.listwise_loss not in LISTWISE_LOSSES:
            raise ValueError(f"listwise_loss choices={LISTWISE_LOSSES}")
        if self.ranking_target not in RANKING_TARGETS:
            raise ValueError(f"ranking_target choices={RANKING_TARGETS}")
        if self.quantization_rule not in QUANTIZATION_RULES:
            raise ValueError(f"quantization_rule choices={QUANTIZATION_RULES}")
        if self.quantization_surrogate not in QUANTIZATION_SURROGATES:
            raise ValueError(
                f"quantization_surrogate choices={QUANTIZATION_SURROGATES}"
            )
        if self.quantized_target_rule not in QUANTIZED_TARGET_RULES:
            raise ValueError(
                f"quantized_target_rule choices={QUANTIZED_TARGET_RULES}"
            )
        if self.quantized_error_form not in QUANTIZED_ERROR_FORMS:
            raise ValueError(
                f"quantized_error_form choices={QUANTIZED_ERROR_FORMS}"
            )
        if self.checkpoint_retention not in CHECKPOINT_RETENTION_POLICIES:
            raise ValueError(
                f"checkpoint_retention choices={CHECKPOINT_RETENTION_POLICIES}"
            )
        if self.inbatch_sampling not in INBATCH_SAMPLING_MODES:
            raise ValueError(f"inbatch_sampling choices={INBATCH_SAMPLING_MODES}")
        if self.best_checkpoint_metric not in BEST_CHECKPOINT_METRICS:
            raise ValueError(
                f"best_checkpoint_metric choices={BEST_CHECKPOINT_METRICS}"
            )
        if self.secondary_checkpoint_metric not in SECONDARY_CHECKPOINT_METRICS:
            raise ValueError(
                "secondary_checkpoint_metric choices="
                f"{SECONDARY_CHECKPOINT_METRICS}"
            )
        if (
            self.secondary_checkpoint_metric not in {"auto", "none"}
            and self.secondary_checkpoint_metric == self.best_checkpoint_metric
        ):
            raise ValueError("secondary checkpoint metric은 primary와 달라야 합니다")
        for name in (
            "mse_loss_weight",
            "pairwise_loss_weight",
            "listwise_loss_weight",
            "distribution_loss_weight",
            "ordinal_loss_weight",
            "detail_expected_loss_weight",
            "detail_distribution_loss_weight",
            "detail_halfstep_loss_weight",
            "detail_rater_set_loss_weight",
            "detail_hierarchy_loss_weight",
            "detail_rater_loss_weight",
            "paragraph_boundary_loss_weight",
            "content_loss_weight",
            "organization_loss_weight",
            "expression_loss_weight",
            "ranking_content_weight",
            "ranking_organization_weight",
            "ranking_expression_weight",
            "trait_average_loss_weight",
            "trait_average_pairwise_weight",
            "quantized_trait_loss_weight",
            "quantized_mean_loss_weight",
            "quantized_pooled_loss_weight",
            "quantized_trait_rank_weight",
            "quantized_mean_rank_weight",
            "quantized_pooled_rank_weight",
            "tail_weight_target_sigma",
            "tail_weight_reference_sigma",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "pairwise_temperature",
            "listwise_temperature",
            "quantization_temperature",
            "quantization_final_temperature",
            "quantization_allocation_temperature",
            "quantization_final_allocation_temperature",
            "quantized_rank_temperature",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.ordinal_steps <= 0:
            raise ValueError("ordinal_steps must be positive")
        if not math.isfinite(self.ordinal_blend_weight) or not (
            0.0 <= self.ordinal_blend_weight <= 1.0
        ):
            raise ValueError("ordinal_blend_weight must be finite and in [0, 1]")
        if self.ordinal_loss_weight > 0 and self.score_head != "regression_ordinal":
            raise ValueError(
                "ordinal_loss_weight는 score_head='regression_ordinal'에서만 "
                "사용할 수 있습니다"
            )
        for name in (
            "quantized_loss_start_step",
            "quantized_loss_ramp_steps",
            "quantization_anneal_steps",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        quantized_weights = (
            self.quantized_trait_loss_weight,
            self.quantized_mean_loss_weight,
            self.quantized_pooled_loss_weight,
            self.quantized_trait_rank_weight,
            self.quantized_mean_rank_weight,
            self.quantized_pooled_rank_weight,
        )
        if self.quantization_rule == "none" and any(
            weight > 0 for weight in quantized_weights
        ):
            raise ValueError("quantized loss를 사용하려면 quantization_rule을 켜야 합니다")
        if not any(
            (
                self.mse_loss_weight > 0,
                self.score_head == "distribution" and self.distribution_loss_weight > 0,
                self.score_head == "regression_ordinal"
                and self.ordinal_loss_weight > 0,
                self.detail_head_mode != "none"
                and any(
                    weight > 0
                    for weight in (
                        self.detail_expected_loss_weight,
                        self.detail_distribution_loss_weight,
                        self.detail_halfstep_loss_weight,
                        self.detail_rater_set_loss_weight,
                        self.detail_hierarchy_loss_weight,
                        self.detail_rater_loss_weight,
                    )
                ),
                self.pairwise_loss != "none" and self.pairwise_loss_weight > 0,
                self.listwise_loss != "none" and self.listwise_loss_weight > 0,
                self.paragraph_boundary_loss_weight > 0,
                any(weight > 0 for weight in quantized_weights),
            )
        ):
            raise ValueError("at least one training loss must have a positive weight")
        detail_weights = (
            self.detail_expected_loss_weight,
            self.detail_distribution_loss_weight,
            self.detail_halfstep_loss_weight,
            self.detail_rater_set_loss_weight,
            self.detail_hierarchy_loss_weight,
            self.detail_rater_loss_weight,
        )
        if self.detail_head_mode == "none" and any(
            weight > 0 for weight in detail_weights
        ):
            raise ValueError("detail loss를 사용하려면 detail_head_mode를 켜야 합니다")
        if (
            self.detail_head_mode != "none"
            and self.detail_final_source == "direct"
            and not any(weight > 0 for weight in detail_weights)
        ):
            raise ValueError(
                "direct final의 활성 detail head에는 detail loss weight가 하나 이상 "
                "필요합니다"
            )
        if (
            self.detail_final_source == "criterion"
            and self.detail_hierarchy_loss_weight > 0
        ):
            raise ValueError(
                "detail hierarchy loss는 detail_final_source='direct'에서만 사용합니다"
            )
        if (
            self.detail_head_mode != "categorical"
            and self.detail_distribution_loss_weight > 0
        ):
            raise ValueError(
                "detail_distribution_loss_weight는 detail_head_mode='categorical'에서만 "
                "사용할 수 있습니다"
            )
        if (
            self.detail_head_mode != "halfstep_categorical"
            and self.detail_halfstep_loss_weight > 0
        ):
            raise ValueError(
                "detail_halfstep_loss_weight는 "
                "detail_head_mode='halfstep_categorical'에서만 사용할 수 있습니다"
            )
        if (
            self.detail_head_mode not in RATER_SET_HEAD_MODES
            and self.detail_rater_set_loss_weight > 0
        ):
            raise ValueError(
                "detail_rater_set_loss_weight는 detail_head_mode="
                f"{RATER_SET_HEAD_MODES}에서만 사용할 수 있습니다"
            )
        if (
            self.detail_head_mode in RATER_SET_HEAD_MODES
            and self.detail_rater_set_loss_weight > 0
            and self.detail_expected_loss_weight <= 0
        ):
            raise ValueError(
                "rater_set loss에는 단일 재평가자 행의 unmatched branch를 묶는 "
                "detail_expected_loss_weight > 0이 필요합니다"
            )
        if self.detail_rater_loss_weight > 0:
            if self.detail_head_mode != "categorical":
                raise ValueError(
                    "detail_rater_loss_weight는 detail_head_mode='categorical'에서만 "
                    "사용할 수 있습니다"
                )
        ranking_enabled = (
            self.pairwise_loss != "none" and self.pairwise_loss_weight > 0
        ) or (self.listwise_loss != "none" and self.listwise_loss_weight > 0)
        if ranking_enabled and self.batch_size < 2:
            raise ValueError("in-batch ranking loss requires batch_size >= 2")
        if self.mse_loss_weight > 0 and not any(
            weight > 0
            for weight in (
                self.content_loss_weight,
                self.organization_loss_weight,
                self.expression_loss_weight,
            )
        ):
            raise ValueError(
                "MSE를 사용할 때 trait loss weight가 하나는 양수여야 합니다"
            )
        if ranking_enabled and not any(
            weight > 0
            for weight in (
                self.ranking_content_weight,
                self.ranking_organization_weight,
                self.ranking_expression_weight,
            )
        ):
            raise ValueError(
                "ranking을 사용할 때 ranking trait weight가 하나는 양수여야 합니다"
            )
        if self.pairwise_loss == "hard_ranknet" and self.pairwise_gap_weighted:
            raise ValueError(
                "hard_ranknet은 점수 간격을 사용하지 않으므로 "
                "pairwise_gap_weighted를 함께 사용할 수 없습니다"
            )

    def _validate_flags(self) -> None:
        if not isinstance(self.eval_every_epoch, bool):
            raise ValueError("eval_every_epoch must be boolean")
        if not isinstance(self.gradient_checkpointing, bool):
            raise ValueError("gradient_checkpointing must be boolean")
        if not isinstance(self.pairwise_gap_weighted, bool):
            raise ValueError("pairwise_gap_weighted must be boolean")
        if not isinstance(self.normalize_features, bool):
            raise ValueError("normalize_features must be boolean")
        if not isinstance(self.allow_validation_overlap, bool):
            raise ValueError("allow_validation_overlap must be boolean")
        if not isinstance(self.allow_validation_leaked, bool):
            raise ValueError("allow_validation_leaked must be boolean")
        if not isinstance(self.split_primary_sources, bool):
            raise ValueError("split_primary_sources must be boolean")
        if not 0 <= self.distribution_label_smoothing < 1:
            raise ValueError("distribution_label_smoothing must be in [0, 1)")
        holdout = self.unseen_prompt_holdout.strip()
        if holdout:
            items = [item.strip() for item in holdout.split(",")]
            if any(not item for item in items):
                raise ValueError("unseen_prompt_holdout에 빈 항목이 있습니다")
            if len(set(items)) != len(items):
                raise ValueError("unseen_prompt_holdout에 중복 문항이 있습니다")
        if self.best_checkpoint_metric.startswith("unseen_prompt_") and not holdout:
            raise ValueError(
                "unseen_prompt_* 지표로 고르려면 unseen_prompt_holdout이 필요합니다"
            )
        if self.tail_weight_target_sigma > 0:
            if self.tail_weight_reference_sigma <= 0:
                raise ValueError(
                    "tail_weight_target_sigma를 쓰려면 tail_weight_reference_sigma > 0"
                )
            if self.tail_weight_max < 1:
                raise ValueError("tail_weight_max는 1 이상이어야 합니다")
        if not math.isfinite(self.tail_weight_reference_mean):
            raise ValueError("tail_weight_reference_mean must be finite")
        if not math.isfinite(self.tail_weight_max) or self.tail_weight_max <= 0:
            raise ValueError("tail_weight_max must be finite and positive")
        if not 0 <= self.prompt_dropout_probability <= 1:
            raise ValueError("prompt_dropout_probability must be in [0, 1]")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
        if not math.isfinite(self.level_prototype_weight) or (
            self.level_prototype_weight < 0
        ):
            raise ValueError("level_prototype_weight must be finite and non-negative")
        if self.level_prototype_count < 2:
            raise ValueError("level_prototype_count는 2 이상이어야 합니다")
        if not 0 <= self.level_prototype_momentum < 1:
            raise ValueError("level_prototype_momentum must be in [0, 1)")
        if self.level_prototype_temperature <= 0:
            raise ValueError("level_prototype_temperature must be positive")
        if self.level_prototype_weight > 0 and self.training_mode == "head_only":
            raise ValueError(
                "수준 prototype은 backbone을 학습하는 lora_only 또는 two_stage가 "
                "필요하다 (head만 학습하면 표현이 바뀌지 않는다)"
            )
        if not math.isfinite(self.group_dro_step_size) or self.group_dro_step_size < 0:
            raise ValueError("group_dro_step_size must be finite and non-negative")
        if self.group_dro_step_size > 0 and self.tail_weight_target_sigma > 0:
            raise ValueError(
                "group_dro_step_size와 tail_weight_target_sigma는 함께 쓸 수 없다 "
                "(두 재가중이 같은 element_weights 경로를 다퉈 무엇을 재는지 흐려진다)"
            )
        if self.pooled_normalization not in POOLED_NORMALIZATIONS:
            raise ValueError(f"pooled_normalization choices={POOLED_NORMALIZATIONS}")
        if self.pooled_normalization != "none" and self.normalize_features:
            raise ValueError(
                "pooled_normalization과 normalize_features(L2)는 함께 쓸 수 없다 "
                "(두 정규화가 겹쳐 무엇을 재는지 흐려진다)"
            )
        if not math.isfinite(self.prompt_adversary_weight) or (
            self.prompt_adversary_weight < 0
        ):
            raise ValueError("prompt_adversary_weight must be finite and non-negative")
        if self.prompt_adversary_classes < 2:
            raise ValueError("prompt_adversary_classes는 2 이상이어야 합니다")
        if self.prompt_adversary_weight > 0 and self.training_mode == "head_only":
            raise ValueError(
                "문항 적대적 학습은 backbone을 학습하는 lora_only 또는 two_stage가 "
                "필요하다 (head만 학습하면 표현이 바뀌지 않는다)"
            )
        if self.prompt_dropout_probability > 0 and self.input_format not in {
            "baseline_v1",
            "essay_only_v1",
        }:
            raise ValueError(
                "prompt_dropout_probability는 baseline_v1 또는 essay_only_v1에서만 "
                "정의된다 (다른 format은 지문 위치가 다르다)"
            )

    def stage_plan(self) -> list[tuple[str, int]]:
        if self.training_mode == "head_only":
            return [("head", self.epochs)]
        if self.training_mode == "lora_only":
            return [("joint", self.epochs)]
        return [("head", self.head_epochs), ("joint", self.joint_epochs)]

    def with_updates(self, **updates: Any) -> "RegressionConfig":
        known = {field.name for field in fields(self)}
        unknown = set(updates) - known
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        return replace(self, **updates).validate()


# Config persistence and automatic CLI --------------------------------------
def _read_config_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config JSON must be an object")
    return value


def _normalize_config_values(value: dict[str, Any]) -> dict[str, Any]:
    """JSON payload를 dataclass update에 사용할 값으로 정규화한다."""

    # `_`로 시작하는 key는 사람이 읽는 주석이다. `configs/`의 확정 구조를
    # 주입하는 preset은 "이 값이 왜 확정인지"를 파일 안에 적어 두어야 유용한데, dataclass
    # field가 아니라 `with_updates`가 `unknown config fields`로 즉시 죽었다. 오타를 잡는
    # unknown-field 검사는 그대로 두고 예약 metadata/주석 key만 통과시킨다.
    value = {key: item for key, item in value.items() if not key.startswith("_")}
    # 과거 checkpoint에는 프로젝트 내부 Hugging Face cache의 절대 경로가 저장됐다.
    # cache 위치는 config/CLI 계약이 아니며 Transformers의 사용자 기본값을 그대로 쓴다.
    value.pop("model_cache_dir", None)
    # Checkpoints created before the local model catalog did not store these
    # provenance fields. Fill only missing values; an explicit False for
    # trust_remote_code remains authoritative.
    if value.get("model_id"):
        spec = resolve_model(str(value["model_id"]))
        if not value.get("model_slug"):
            value["model_slug"] = spec.slug
        if not value.get("model_source_run"):
            value["model_source_run"] = spec.source_run
        if not value.get("model_revision"):
            value["model_revision"] = spec.revision
        if "trust_remote_code" not in value:
            value["trust_remote_code"] = spec.trust_remote_code
    # JSON은 tuple을 list로 저장하므로 prompt head parameter의 의미를 정하는
    # registry 순서를 immutable nested tuple로 복원한다.
    if "prompt_registry" in value:
        registry = value["prompt_registry"]
        if not isinstance(registry, list):
            raise ValueError("prompt_registry must be a JSON list")
        normalized = tuple(
            (str(item[0]), str(item[1]))
            for item in registry
            if isinstance(item, (list, tuple)) and len(item) == 2
        )
        if len(normalized) != len(registry):
            raise ValueError(
                "prompt_registry item은 [prompt_num, prompt_text]여야 합니다"
            )
        value["prompt_registry"] = normalized
    if "detail_rater_registry" in value:
        registry = value["detail_rater_registry"]
        if not isinstance(registry, list):
            raise ValueError("detail_rater_registry must be a JSON list")
        normalized = tuple(
            (str(item[0]), str(item[1]))
            for item in registry
            if isinstance(item, (list, tuple)) and len(item) == 2
        )
        if len(normalized) != len(registry):
            raise ValueError(
                "detail_rater_registry item은 [source_dataset, evaluator_id]여야 합니다"
            )
        value["detail_rater_registry"] = normalized
    return value


def legacy_baseline_config() -> RegressionConfig:
    """명시적 ``--baseline``용 완전 고정 pre-y1 대조군을 읽는다."""

    value = _normalize_config_values(_read_config_object(LEGACY_BASELINE_CONFIG_PATH))
    known = {item.name for item in fields(RegressionConfig)}
    # 2026-08-13에 추가된 quantization/checkpoint-policy 필드는 기본값이 기존
    # graph/artifact policy와 정확히 같은 additive no-op이다. 역사 preset 파일의
    # byte fingerprint를 바꾸지 않고 이 필드들만 현재 기본으로 보완한다.
    additive_noop_fields = {
        "quantization_rule",
        "quantization_surrogate",
        "quantized_target_rule",
        "quantized_error_form",
        "quantized_trait_loss_weight",
        "quantized_mean_loss_weight",
        "quantized_pooled_loss_weight",
        "quantized_trait_rank_weight",
        "quantized_mean_rank_weight",
        "quantized_pooled_rank_weight",
        "quantization_temperature",
        "quantization_final_temperature",
        "quantization_allocation_temperature",
        "quantization_final_allocation_temperature",
        "quantized_rank_temperature",
        "quantized_loss_start_step",
        "quantized_loss_ramp_steps",
        "quantization_anneal_steps",
        "checkpoint_retention",
        "secondary_checkpoint_metric",
        "ordinal_steps",
        "ordinal_blend_weight",
        "ordinal_loss_weight",
        # 2026-08-18에 추가된 근거 pooling 축. 기본값 pooling="mean"에서는 세 값이
        # 전부 읽히지 않으므로 legacy preset의 의미도 graph도 바뀌지 않는다.
        "analysis_pool_weight",
        "analysis_pool_learnable",
        "analysis_pool_per_trait",
        # 2026-08-20에 추가된 꼬리 가중 축. 기본 tail_weight_target_sigma=0.0에서
        # sample_tail_weights가 None을 돌려주고 호출부가 기존 인자 없이 부르던
        # 경로를 그대로 타므로 legacy preset의 loss graph가 바뀌지 않는다.
        "tail_weight_target_sigma",
        "tail_weight_reference_mean",
        "tail_weight_reference_sigma",
        "tail_weight_max",
        # 2026-08-20 미학습 문항 보류 축. 빈 문자열에서 학습 행 선택이 그대로이므로
        # historical preset의 실행 의미가 바뀌지 않는다.
        "unseen_prompt_holdout",
        # 2026-08-20 지문 dropout. p=0에서 입력 문자열이 그대로다.
        "prompt_dropout_probability",
        # 2026-08-20 문항 적대적 학습. weight=0에서 판별기를 만들지 않으므로
        # parameter 수와 loss graph가 기존과 동일하다.
        "prompt_adversary_weight",
        "prompt_adversary_classes",
        # 2026-08-20 pooled 정규화. "none"에서 연산이 추가되지 않는다.
        "pooled_normalization",
        # 2026-08-20 문항 GroupDRO. step_size=0에서 가중치가 만들어지지 않는다.
        "group_dro_step_size",
        # 2026-08-20 clip 임계. 기본 1.0이 기존 하드코딩 값과 같다.
        "max_grad_norm",
        # 2026-08-21 수준 prototype. weight=0에서 buffer가 읽히지 않는다.
        "level_prototype_weight",
        "level_prototype_count",
        "level_prototype_momentum",
        "level_prototype_temperature",
    }
    missing = known - set(value) - additive_noop_fields
    unknown = set(value) - known
    if missing or unknown:
        raise ValueError(
            "legacy baseline은 현재 config schema를 완전히 고정해야 합니다: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return RegressionConfig().with_updates(**value)


def load_config(path: str | Path) -> RegressionConfig:
    value = _normalize_config_values(_read_config_object(path))
    # 일반 config의 생략 필드는 언제나 현재 final 기본값을 상속한다. 역사적 Y1/baseline은
    # 완전한 preset을 --config 또는 --baseline으로 명시했을 때만 사용한다.
    return RegressionConfig().with_updates(**value)


def save_config(config: RegressionConfig, path: str | Path) -> None:
    config.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """RegressionConfig의 scalar field를 CLI override로 자동 노출한다.

    새 scalar 설정은 dataclass field만 추가하면 ``--field-name``과
    ``--field_name``으로 사용할 수 있다. model alias와 schema version은 별도 처리한다.
    """

    defaults = RegressionConfig()
    for item in fields(defaults):
        name = item.name
        if name in CONFIG_CLI_EXCLUDED:
            continue
        default = getattr(defaults, name)
        dashed = f"--{name.replace('_', '-')}"
        underscored = f"--{name}"
        option_strings = list(CONFIG_CLI_ALIASES.get(name, (dashed,)))
        if name not in CONFIG_CLI_ALIASES and underscored != dashed:
            option_strings.append(underscored)
        kwargs: dict[str, Any] = {
            "dest": name,
            "default": argparse.SUPPRESS,
        }
        if isinstance(default, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = type(default)
            if name in CONFIG_CHOICES:
                kwargs["choices"] = CONFIG_CHOICES[name]
        parser.add_argument(*option_strings, **kwargs)


def config_updates_from_namespace(namespace: argparse.Namespace) -> dict[str, Any]:
    """명시적으로 전달된 config CLI 값만 반환한다."""

    values = vars(namespace)
    return {
        item.name: values[item.name]
        for item in fields(RegressionConfig)
        if item.name not in CONFIG_CLI_EXCLUDED and item.name in values
    }
