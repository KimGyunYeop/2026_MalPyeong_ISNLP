from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import RegressionConfig
from .utils import average_matched_integer_scores


TRAITS = ("content", "organization", "expression")
SCOPES = ("overall", *TRAITS)
RUNNER_SUITES = {"basemodel", "proposed", "new_proposed", "data", "best"}
LOCAL_PATH_CONFIGS = {"dataset_root", "extended_data_dir"}

# CSV 앞부분은 자주 비교하는 값만 고정한다. 새 방법론 설정은 아래에
# ``config.<name>`` 열로 자동 추가되므로 집계 코드를 매번 수정할 필요가 없다.
RUN_FIELDS = (
    "rank",
    "suite",
    "base_name",
    "case_name",
    "description",
    "config_id",
    "config_diff",
    "base_args",
    "override_args",
    "inference_args",
    "status",
    "train_status",
    "infer_status",
    "warnings",
    "model_slug",
    "model_id",
    "model_revision",
    "backbone_type",
    "training_mode",
    "seed",
    "checkpoint_type",
    "inference_essay_surface",
    "infer_batch_size",
    "infer_load_in_4bit",
    "infer_max_length",
    "config_source",
    "primary_data_profile",
    "dataset_schedule",
    "extended_datasets",
    "data_signature",
    "data_summary",
    "primary_train_sha256",
    "primary_validation_sha256",
    "external_train_sha256s",
    "training_rows",
    "competition_training_rows",
    "extended_training_rows",
    "training_source_counts",
    "validation_rows",
    "validation_overlap_id_count",
    "validation_overlap_text_count",
    "batch_size",
    "gradient_accumulation",
    "max_length",
    "epochs",
    "max_train_steps",
    "planned_steps",
    "actual_steps",
    "score_side_parameter_count",
    "score_side_delta_l2",
    "score_side_max_abs_delta",
    "score_side_changed",
    "lora_parameter_count",
    "lora_delta_l2",
    "lora_max_abs_delta",
    "lora_changed",
    "train_runtime_seconds",
    "train_samples_per_second",
    "train_steps_per_second",
    "peak_gpu_memory_reserved_gib",
    "peak_gpu_memory_allocated_gib",
    "best_checkpoint_metric_name",
    "best_checkpoint_metric_value",
    "best_checkpoint_epoch",
    "best_checkpoint_step",
    "prediction_records",
    "metric_count",
)
INFERENCE_METRIC_FIELDS = tuple(
    f"{prefix}{scope}_{metric}"
    for prefix in ("", "prompt_macro_")
    for scope in SCOPES
    for metric in ("rmse", "spearman")
)
# 운영진 확정 정의(essay별 세 trait 평균 1개 vs score.average, 전체 1회). CSV에도 반드시
# 있어야 한다. 없으면 machine-readable 산출물에서 공식 지표를 읽을 수 없고, markdown만
# 맞고 CSV는 정의 A만 담는 불일치가 생긴다.
OFFICIAL_METRIC_FIELDS = (
    # 새 중심 schema. raw는 모델 실수 C/O/E 평균, submitted는 average_matched 정수
    # 삼중 평균이며 둘 다 같은 score.average gold를 쓴다.
    "raw_average_rmse",
    "raw_average_spearman",
    "submitted_average_rmse",
    "submitted_average_spearman",
    "official_gold_source",
    "official_metric_count",
    "surface_class",
    # 과거 분석 코드/MD가 읽던 최소 alias. expected_* 같은 제3의 반올림 표면은 제거한다.
    "official_rmse",
    "official_spearman",
    "candidate_rmse",
    "candidate_spearman",
)
HISTORY_METRIC_FIELDS = tuple(
    field
    for scope in SCOPES
    for field in (
        f"best_{scope}_rmse",
        f"spearman_at_best_{scope}_rmse",
        f"best_{scope}_rmse_epoch",
        f"best_{scope}_rmse_step",
        f"best_{scope}_rmse_stage",
        f"best_{scope}_spearman",
        f"rmse_at_best_{scope}_spearman",
        f"best_{scope}_spearman_epoch",
        f"best_{scope}_spearman_step",
        f"best_{scope}_spearman_stage",
        f"last_eval_{scope}_rmse",
        f"last_eval_{scope}_spearman",
    )
)
# ``eval/`` remains the sole source for ranking and completion status.  The
# opposite-metric checkpoint evaluation is diagnostic and therefore lives in
# additive columns appended after every pre-existing core field.
ALTERNATE_CHECKPOINT_FIELDS = (
    "alternate_infer_status",
    "alternate_infer_exit_code",
    "alternate_checkpoint_metric_name",
    "alternate_checkpoint_metric_value",
    "alternate_checkpoint_epoch",
    "alternate_checkpoint_step",
    "alternate_checkpoint_type",
    "alternate_prediction_records",
    "alternate_metric_count",
)
ALTERNATE_INFERENCE_METRIC_FIELDS = tuple(
    f"alternate_{prefix}{scope}_{metric}"
    for prefix in ("", "prompt_macro_")
    for scope in SCOPES
    for metric in ("rmse", "spearman")
)
ALTERNATE_OFFICIAL_FIELDS = (
    "alternate_raw_average_rmse",
    "alternate_raw_average_spearman",
    "alternate_submitted_average_rmse",
    "alternate_submitted_average_spearman",
    "alternate_official_rmse",
    "alternate_official_spearman",
    "alternate_candidate_rmse",
    "alternate_candidate_spearman",
    "alternate_official_gold_source",
)
ALTERNATE_FIELDS = (
    *ALTERNATE_CHECKPOINT_FIELDS,
    *ALTERNATE_INFERENCE_METRIC_FIELDS,
    *ALTERNATE_OFFICIAL_FIELDS,
)
CORE_FIELDS = (
    *RUN_FIELDS,
    *OFFICIAL_METRIC_FIELDS,
    *INFERENCE_METRIC_FIELDS,
    *HISTORY_METRIC_FIELDS,
    "last_eval_epoch",
    "last_eval_step",
    "last_eval_stage",
    "relative_run_dir",
    "run_dir",
    *ALTERNATE_FIELDS,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent / "results"
    parser = argparse.ArgumentParser(
        description="모든 regression 실험 산출물을 하나의 CSV와 Markdown으로 집계"
    )
    parser.add_argument("--results-root", default=str(root))
    parser.add_argument(
        "--suite", help="한 suite만 선택 (basemodel/proposed/data/best)"
    )
    parser.add_argument(
        "--base",
        help=(
            "한 BASE_NAME(라운드)만 선택한다. 지정하면 기본 출력이 "
            "<results-root>/<suite>/<base>/results.{md,csv}가 된다"
        ),
    )
    parser.add_argument("--csv", "--csv-output", dest="csv")
    parser.add_argument("--markdown", "--markdown-output", dest="markdown")
    # 이전 baseline shell이 전달하던 인자다. 계획 행은 더 이상 만들지 않지만,
    # 오래된 명령이 즉시 깨지지 않도록 입력만 받아 무시한다.
    parser.add_argument("--model-tsv", help=argparse.SUPPRESS)
    return parser.parse_args()


def read_json(path: Path, warnings: list[str]) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"{path.name} 읽기 실패: {exc}")
        return {}
    if not isinstance(value, dict):
        warnings.append(f"{path.name}의 JSON root가 object가 아님")
        return {}
    return value


def read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(compact_json(value).encode("utf-8")).hexdigest()


def discover_run_dirs(results_root: Path) -> list[Path]:
    """실제 산출물 세 종류의 parent union으로 run directory를 찾는다."""

    run_dirs = {path.parent for path in results_root.rglob("run.json")}
    run_dirs.update(
        path.parent.parent for path in results_root.rglob("eval/metrics.json")
    )

    # 보통 experiment.json은 run 안에 있다. suite/case 공통 metadata로 상위에
    # 하나만 둔 경우에는 이미 발견한 하위 run에 상속하고 별도 행은 만들지 않는다.
    experiment_dirs = [path.parent for path in results_root.rglob("experiment.json")]
    for directory in experiment_dirs:
        has_descendant_run = any(
            directory != run_dir and directory in run_dir.parents
            for run_dir in run_dirs
        )
        if not has_descendant_run:
            run_dirs.add(directory)
    return sorted(run_dirs)


def nearest_experiment(
    run_dir: Path, results_root: Path, warnings: list[str]
) -> tuple[dict[str, Any], Path | None]:
    current = run_dir
    while current == results_root or results_root in current.parents:
        path = current / "experiment.json"
        if path.is_file():
            metadata = read_json(path, warnings)
            nested_metadata = metadata.get("experiment")
            if isinstance(nested_metadata, dict):
                metadata = {**metadata, **nested_metadata}
            return metadata, path
        if current == results_root:
            break
        current = current.parent
    return {}, None


def load_config(run_dir: Path, warnings: list[str]) -> tuple[dict[str, Any], str]:
    """새 resolved config를 우선하고 두 checkpoint 형식을 순서대로 지원한다."""

    candidates = (
        (run_dir / "resolved_config.json", "resolved_config"),
        (run_dir / "best_checkpoint/config.json", "best_checkpoint"),
        (run_dir / "checkpoint/config.json", "checkpoint"),
    )
    for path, source in candidates:
        if path.is_file():
            config = read_json(path, warnings)
            if config:
                return config, source
    warnings.append("config 없음")
    return {}, ""


def read_history(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    invalid = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        warnings.append(f"train_log.jsonl 읽기 실패: {exc}")
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if isinstance(value, dict):
            records.append(value)
        else:
            invalid += 1
    if invalid:
        warnings.append(f"train_log.jsonl 손상 행 {invalid}개")
    return records


def count_jsonl(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        return sum(
            bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()
        )
    except OSError:
        return None


def metric_key(scope: str, metric: str) -> str:
    return f"eval_{scope}_{metric}"


def summarize_history(row: dict[str, Any], records: list[dict[str, Any]]) -> None:
    eval_records = [
        record
        for record in records
        if finite(record.get("eval_overall_rmse")) is not None
    ]
    if not eval_records:
        return

    last = eval_records[-1]
    row.update(
        last_eval_epoch=last.get("epoch"),
        last_eval_step=last.get("step"),
        last_eval_stage=last.get("stage"),
    )
    for scope in SCOPES:
        rmse_name = metric_key(scope, "rmse")
        spearman_name = metric_key(scope, "spearman")
        candidates = [
            record for record in eval_records if finite(record.get(rmse_name))
        ]
        if not candidates:
            continue

        best_rmse = min(
            candidates,
            key=lambda record: (
                float(record[rmse_name]),
                -(
                    finite(record.get(spearman_name))
                    if finite(record.get(spearman_name)) is not None
                    else -math.inf
                ),
            ),
        )
        row.update(
            {
                f"best_{scope}_rmse": best_rmse.get(rmse_name),
                f"spearman_at_best_{scope}_rmse": best_rmse.get(spearman_name),
                f"best_{scope}_rmse_epoch": best_rmse.get("epoch"),
                f"best_{scope}_rmse_step": best_rmse.get("step"),
                f"best_{scope}_rmse_stage": best_rmse.get("stage"),
                f"last_eval_{scope}_rmse": last.get(rmse_name),
                f"last_eval_{scope}_spearman": last.get(spearman_name),
            }
        )

        spearman_candidates = [
            record for record in candidates if finite(record.get(spearman_name))
        ]
        if not spearman_candidates:
            continue
        best_spearman = max(
            spearman_candidates,
            key=lambda record: (
                float(record[spearman_name]),
                -float(record[rmse_name]),
            ),
        )
        row.update(
            {
                f"best_{scope}_spearman": best_spearman.get(spearman_name),
                f"rmse_at_best_{scope}_spearman": best_spearman.get(rmse_name),
                f"best_{scope}_spearman_epoch": best_spearman.get("epoch"),
                f"best_{scope}_spearman_step": best_spearman.get("step"),
                f"best_{scope}_spearman_stage": best_spearman.get("stage"),
            }
        )


def summarize_metrics(
    row: dict[str, Any], metrics: dict[str, Any], *, prefix: str = ""
) -> None:
    if not metrics:
        return
    row[f"{prefix}metric_count"] = metrics.get("count")
    for scope in SCOPES:
        block = (
            metrics.get("overall")
            if scope == "overall"
            else nested(metrics, "traits", scope)
        )
        if isinstance(block, dict):
            row[f"{prefix}{scope}_rmse"] = block.get("rmse")
            row[f"{prefix}{scope}_spearman"] = block.get("spearman")

        macro = (
            nested(metrics, "prompt_macro", "overall")
            if scope == "overall"
            else nested(metrics, "prompt_macro", "traits", scope)
        )
        if isinstance(macro, dict):
            row[f"{prefix}prompt_macro_{scope}_rmse"] = macro.get("rmse")
            row[f"{prefix}prompt_macro_{scope}_spearman"] = macro.get("spearman")

    # v2는 공식 gold에 대한 두 표면을 한 block에 둔다. 과거 artifact는 아래 legacy alias로
    # 읽어 같은 canonical 열에 올린다.
    official = metrics.get("official")
    raw = nested(official, "raw_continuous")
    submitted = nested(official, "average_matched")
    gold_source = official.get("gold_source") if isinstance(official, dict) else None
    if not isinstance(raw, dict):
        raw = metrics.get("trait_average")
        if isinstance(raw, dict) and gold_source is None:
            gold_source = raw.get("gold_source")
    if not isinstance(submitted, dict):
        submitted = nested(
            metrics, "trait_average_rounded", "average_matched_integer"
        )

    if isinstance(raw, dict):
        row[f"{prefix}raw_average_rmse"] = raw.get("rmse")
        row[f"{prefix}raw_average_spearman"] = raw.get("spearman")
        # legacy analysis alias
        row[f"{prefix}official_rmse"] = raw.get("rmse")
        row[f"{prefix}official_spearman"] = raw.get("spearman")
    if isinstance(submitted, dict):
        row[f"{prefix}submitted_average_rmse"] = submitted.get("rmse")
        row[f"{prefix}submitted_average_spearman"] = submitted.get("spearman")
        # legacy analysis alias
        row[f"{prefix}candidate_rmse"] = submitted.get("rmse")
        row[f"{prefix}candidate_spearman"] = submitted.get("spearman")
    row[f"{prefix}official_gold_source"] = gold_source


def official_metrics_from_predictions(
    eval_dir: Path, warnings: list[str]
) -> dict[str, Any] | None:
    """저장된 예측에서 공식 지표를 직접 계산한다.

    공식 지표가 확정되기 전에 만든 artifact의 `metrics.json`에는 `trait_average` key가 없다.
    그렇다고 표에서 `-`로 비워 두면 정의 A로 정렬된 과거 행과 정의 C를 가진 최근 행이 한 표에
    섞여 **행끼리 비교할 수 없는 표**가 된다. 예측 파일이 그대로 남아 있으므로 다시 계산한다.

    학습을 다시 하지 않는다. 새 score surface block과 과거 ``scores`` alias를 모두 읽고,
    raw 연속 평균과 average_matched 정수 평균을 같은 ``score.average``에 대해 계산한다.
    """

    predictions_path = eval_dir / "score_predictions.jsonl"
    if not predictions_path.is_file():
        return None
    gold = _official_average_gold(eval_dir, warnings)
    if not gold:
        return None

    pairs: list[tuple[float, float, float]] = []
    try:
        # ``str.splitlines``는 JSON string 안의 U+2028/U+2029도 행 경계로 잘못 자른다.
        # JSONL의 실제 구분자인 LF만 따르도록 file iterator를 쓴다.
        for line in predictions_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            record = json.loads(line)
            raw_scores = nested(
                record, "score_surfaces", "raw_continuous", "scores"
            )
            if not isinstance(raw_scores, dict):
                raw_scores = record.get("scores")
            if not isinstance(raw_scores, dict):
                continue
            try:
                raw_values = np.asarray(
                    [[float(raw_scores[trait]) for trait in TRAITS]],
                    dtype=np.float64,
                )
            except (KeyError, TypeError, ValueError):
                continue
            matched_scores = nested(
                record, "score_surfaces", "average_matched", "scores"
            )
            if isinstance(matched_scores, dict):
                try:
                    matched_values = np.asarray(
                        [[float(matched_scores[trait]) for trait in TRAITS]],
                        dtype=np.float64,
                    )
                except (KeyError, TypeError, ValueError):
                    matched_values = average_matched_integer_scores(raw_values)
            else:
                # 과거 artifact에는 canonical 제출 block이 없으므로 검증된 단일 변환으로
                # 재구성한다. ``submitted_scores``가 다른 실험 rule일 가능성은 배제한다.
                matched_values = average_matched_integer_scores(raw_values)
            essay_id = str(record.get("essay_id") or record.get("id") or "")
            truth = gold.get(essay_id)
            if truth is not None:
                pairs.append(
                    (float(raw_values.mean()), float(matched_values.mean()), truth)
                )
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"공식 지표 재계산 실패 {predictions_path.name}: {exc}")
        return None
    if len(pairs) < 2:
        return None

    raw_values = [item[0] for item in pairs]
    submitted_values = [item[1] for item in pairs]
    truth_values = [item[2] for item in pairs]
    raw = {
        "rmse": math.sqrt(
            sum((prediction - truth) ** 2 for prediction, _, truth in pairs)
            / len(pairs)
        ),
        "spearman": _spearman(raw_values, truth_values),
    }
    submitted = {
        "rmse": math.sqrt(
            sum((prediction - truth) ** 2 for _, prediction, truth in pairs)
            / len(pairs)
        ),
        "spearman": _spearman(submitted_values, truth_values),
    }
    return {
        "count": len(pairs),
        "gold_source": "score_average_recomputed",
        "raw_continuous": raw,
        "average_matched": submitted,
        # 과거 caller가 읽던 raw 공식 지표 alias.
        "rmse": raw["rmse"],
        "spearman": raw["spearman"],
    }


_OFFICIAL_GOLD_CACHE: dict[str, dict[str, float]] = {}


def _official_average_gold(
    eval_dir: Path, warnings: list[str]
) -> dict[str, float]:
    """해당 추론이 실제 사용한 입력의 ``score.average``를 읽는다.

    processed detail validation은 세 trait을 다시 평균한 full-precision 값을 갖지만, 운영측
    공식 입력은 소수 둘째 자리의 ``score.average``를 별도로 보존한다. 전역 processed
    validation을 쓰면 400편 중 317편의 gold가 바뀌고, holdout-1600 추론은 400편으로
    잘려 버린다. 따라서 각 eval manifest가 기록한 정확한 input을 source of truth로 쓴다.
    """

    input_path: Path | None = None
    manifest_path = eval_dir / "inference_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded_input = manifest.get("input") if isinstance(manifest, dict) else None
        if recorded_input:
            input_path = Path(str(recorded_input)).expanduser()
            if not input_path.is_absolute():
                input_path = (eval_dir / input_path).resolve()
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        warnings.append(f"공식 추론 manifest 로드 실패: {exc!r}")

    if input_path is None:
        # manifest가 없는 아주 오래된 artifact만 repository root의 공식 입력으로 복구한다.
        repository_root = Path(__file__).resolve().parents[1]
        candidates = sorted((repository_root / "datasets").glob("*_validation.jsonl"))
        if len(candidates) == 1:
            input_path = candidates[0]
        else:
            warnings.append(
                "공식 gold 입력을 하나로 결정할 수 없습니다: "
                f"manifest={manifest_path}, candidates={len(candidates)}"
            )
            return {}

    cache_key = str(input_path.resolve())
    if cache_key in _OFFICIAL_GOLD_CACHE:
        return _OFFICIAL_GOLD_CACHE[cache_key]
    gold: dict[str, float] = {}
    try:
        # 일부 에세이에 U+2028 line separator가 들어 있다. ``splitlines``는 이를 물리적
        # JSONL 행으로 오인하므로 newline 기준 file iterator만 사용한다.
        for line in input_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            record = json.loads(line)
            average = (record.get("score") or {}).get("average")
            if average is not None:
                record_id = record.get("id") or record.get("essay_id")
                if record_id is not None:
                    gold[str(record_id)] = float(average)
    except Exception as exc:  # noqa: BLE001 - gold가 없으면 재계산만 건너뛴다
        warnings.append(f"공식 gold 로드 실패 {input_path}: {exc!r}")
    _OFFICIAL_GOLD_CACHE[cache_key] = gold
    return gold


def _spearman(left: list[float], right: list[float]) -> float:
    """`scipy.stats.spearmanr`과 같은 average-rank 처리(운영측 확인 §1)."""

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values)
        start = 0
        while start < len(order):
            end = start
            while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
                end += 1
            shared = (start + end) / 2.0 + 1.0
            for index in order[start : end + 1]:
                result[index] = shared
            start = end + 1
        return result

    ranked_left, ranked_right = ranks(left), ranks(right)
    mean_left = sum(ranked_left) / len(ranked_left)
    mean_right = sum(ranked_right) / len(ranked_right)
    numerator = sum(
        (a - mean_left) * (b - mean_right)
        for a, b in zip(ranked_left, ranked_right, strict=True)
    )
    denominator = math.sqrt(
        sum((a - mean_left) ** 2 for a in ranked_left)
        * sum((b - mean_right) ** 2 for b in ranked_right)
    )
    return numerator / denominator if denominator else float("nan")


def alternate_evaluation_fields(
    run_dir: Path, config: dict[str, Any], warnings: list[str]
) -> dict[str, Any]:
    """Read the opposite selection metric's inference without affecting primary state.

    New training writes both metric-specific checkpoint siblings.  The runner
    evaluates only the sibling opposite ``best_checkpoint_metric`` because the
    selected legacy ``best_checkpoint/`` is already represented by ``eval/``.
    Old runs and interrupted runs keep explicit blank values in every additive
    field.
    """

    fields = {name: None for name in ALTERNATE_FIELDS}
    selected_metric = config.get("best_checkpoint_metric")
    alternate_metric = {
        "rmse": "spearman",
        "spearman": "rmse",
    }.get(selected_metric)
    if alternate_metric is None:
        return fields

    checkpoint_dir = run_dir / f"best_checkpoint_{alternate_metric}"
    eval_dir = run_dir / f"eval_best_{alternate_metric}"
    metrics_path = eval_dir / "metrics.json"
    secondary_exit = read_int(run_dir / "secondary_infer_exit_code.txt")
    explicit_status = read_text(run_dir / "status.txt")
    status_context = read_json(run_dir / "status_context.json", warnings)
    checkpoint_exists = (checkpoint_dir / "config.json").is_file()
    metrics_exists = metrics_path.is_file()
    failed_without_eval = (
        explicit_status == "infer_failed"
        and not eval_dir.is_dir()
        and (
            checkpoint_exists
            or nested(status_context, "phase") == "secondary_infer"
        )
    )
    if secondary_exit not in (None, 0):
        fields["alternate_infer_status"] = "failed"
        fields["alternate_infer_exit_code"] = secondary_exit
        warnings.append(f"secondary infer exit={secondary_exit}")
    elif failed_without_eval:
        fields["alternate_infer_status"] = "failed"
        fields["alternate_infer_exit_code"] = secondary_exit
        warnings.append("secondary inference 실패: eval directory 없음")
    elif checkpoint_exists and metrics_exists:
        fields["alternate_infer_status"] = "complete"
        fields["alternate_infer_exit_code"] = secondary_exit
    elif checkpoint_exists or secondary_exit is not None or eval_dir.is_dir():
        fields["alternate_infer_status"] = "incomplete"
        fields["alternate_infer_exit_code"] = secondary_exit

    if not checkpoint_exists or not metrics_exists:
        if metrics_exists and not checkpoint_exists:
            warnings.append(
                f"secondary eval은 있으나 checkpoint 없음: {checkpoint_dir.name}"
            )
        return fields

    metrics = read_json(metrics_path, warnings)
    if not metrics:
        return fields
    selection = read_json(checkpoint_dir / "selection.json", warnings)
    manifest = read_json(eval_dir / "inference_manifest.json", warnings)
    predictions = count_jsonl(eval_dir / "predictions.jsonl")
    metric_count = metrics.get("count")
    if (
        predictions is not None
        and metric_count is not None
        and predictions != metric_count
    ):
        warnings.append(
            "secondary metric count "
            f"{metric_count} != predictions {predictions} ({eval_dir.name})"
        )

    manifest_checkpoint = manifest.get("checkpoint")
    if manifest_checkpoint and Path(str(manifest_checkpoint)).name != checkpoint_dir.name:
        warnings.append(
            "secondary inference checkpoint 불일치: "
            f"{Path(str(manifest_checkpoint)).name} != {checkpoint_dir.name}"
        )

    fields.update(
        alternate_checkpoint_metric_name=(
            selection.get("metric") or f"eval_overall_{alternate_metric}"
        ),
        alternate_checkpoint_metric_value=selection.get("metric_value"),
        alternate_checkpoint_epoch=selection.get("epoch"),
        alternate_checkpoint_step=selection.get("global_step"),
        alternate_checkpoint_type=checkpoint_dir.name,
        alternate_prediction_records=predictions,
    )
    summarize_metrics(fields, metrics, prefix="alternate_")
    if (
        fields.get("alternate_raw_average_rmse") is None
        or fields.get("alternate_submitted_average_rmse") is None
    ):
        recomputed = official_metrics_from_predictions(eval_dir, warnings)
        if recomputed is not None:
            raw = recomputed["raw_continuous"]
            submitted = recomputed["average_matched"]
            fields["alternate_raw_average_rmse"] = raw["rmse"]
            fields["alternate_raw_average_spearman"] = raw["spearman"]
            fields["alternate_submitted_average_rmse"] = submitted["rmse"]
            fields["alternate_submitted_average_spearman"] = submitted["spearman"]
            # 최소 legacy alias: 과거 CSV reader가 즉시 깨지지 않게만 유지한다.
            fields["alternate_official_rmse"] = raw["rmse"]
            fields["alternate_official_spearman"] = raw["spearman"]
            fields["alternate_candidate_rmse"] = submitted["rmse"]
            fields["alternate_candidate_spearman"] = submitted["spearman"]
            fields["alternate_official_gold_source"] = recomputed["gold_source"]
    return fields


def runtime_seconds(run: dict[str, Any]) -> float | None:
    value = finite(nested(run, "trainer", "train_metrics", "train_runtime"))
    if value is not None:
        return value
    stages = run.get("trainer_stages")
    if not isinstance(stages, list):
        return None
    values = [
        finite(nested(stage, "train_metrics", "train_runtime"))
        for stage in stages
        if isinstance(stage, dict)
    ]
    available = [value for value in values if value is not None]
    return sum(available) if available else None


def file_hash(manifest: Any, split: str) -> str:
    value = nested(manifest, "files", split, "sha256")
    return str(value) if value else ""


def data_fields(config: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    primary = run.get("primary_data_manifest")
    primary = primary if isinstance(primary, dict) else {}
    external = run.get("prepared_data_manifest")
    external = external if isinstance(external, dict) else {}

    external_hashes: dict[str, str] = {}
    datasets = external.get("datasets")
    if isinstance(datasets, dict):
        for name, manifest in datasets.items():
            digest = file_hash(manifest, "train")
            if digest:
                external_hashes[str(name)] = digest

    train_files = run.get("train_files")
    file_names = (
        [Path(str(path)).name for path in train_files]
        if isinstance(train_files, list)
        else []
    )
    source_counts = run.get("training_source_counts")
    source_counts = source_counts if isinstance(source_counts, dict) else {}
    profile = (
        config.get("primary_data_profile") or run.get("primary_data_profile") or ""
    )
    extended_names = (
        config.get("extended_datasets") or run.get("extended_datasets") or ""
    )
    if isinstance(extended_names, list):
        extended_names = ",".join(str(name) for name in extended_names)
    schedule = config.get("dataset_schedule") or "competition_only"

    signature_payload = {
        "profile": profile,
        "primary_train": file_hash(primary, "train"),
        "primary_validation": file_hash(primary, "validation"),
        "external": external_hashes,
        "train_files": file_names,
        "training_rows": run.get("training_rows"),
        "source_counts": source_counts,
        "schedule": schedule,
        "extended_datasets": extended_names,
    }
    has_data = any(
        value not in (None, "", [], {}) for value in signature_payload.values()
    )
    # 새 train.py는 실제 선택된 row 자체의 fingerprint를 기록한다. profile manifest는
    # 전체 pool을 나타내므로 --limit 같은 data-size 실험에서는 이 값을 우선한다.
    recorded_signature = run.get("data_signature")
    recorded_sha256 = (
        recorded_signature.get("sha256")
        if isinstance(recorded_signature, dict)
        else None
    )
    summary = str(profile or "competition")
    if extended_names:
        summary += f"+{extended_names}({schedule})"
    if run.get("training_rows") is not None:
        summary += f" n={run['training_rows']}"

    return {
        "primary_data_profile": profile,
        "dataset_schedule": schedule,
        "extended_datasets": extended_names,
        "data_signature": (
            str(recorded_sha256)
            if recorded_sha256
            else canonical_hash(signature_payload) if has_data else ""
        ),
        "data_summary": summary,
        "primary_train_sha256": file_hash(primary, "train"),
        "primary_validation_sha256": file_hash(primary, "validation"),
        "external_train_sha256s": external_hashes,
        "training_source_counts": source_counts,
    }


def semantic_config_id(config: dict[str, Any], data_signature: str) -> str:
    portable = {
        key: value for key, value in config.items() if key not in LOCAL_PATH_CONFIGS
    }
    return canonical_hash({"config": portable, "data": data_signature})[:12]


def config_diff(metadata: dict[str, Any], config: dict[str, Any]) -> str:
    explicit = metadata.get("config_diff")
    if explicit is not None:
        return explicit if isinstance(explicit, str) else compact_json(explicit)
    if "override_args" in metadata:
        overrides = metadata.get("override_args") or []
        if isinstance(overrides, str):
            try:
                overrides = json.loads(overrides)
            except json.JSONDecodeError:
                return overrides
        return compact_json(overrides)
    # JSON config의 list와 dataclass의 tuple이 같은 의미일 때 차이로 잡히지 않게
    # 한 번 JSON round-trip해 type을 정규화한다.
    defaults = json.loads(compact_json(asdict(RegressionConfig())))
    changed = {
        key: value
        for key, value in config.items()
        if key not in defaults or value != defaults[key]
    }
    return compact_json(changed)


def experiment_option(metadata: dict[str, Any], name: str) -> Any:
    """experiment metadata의 base/override CLI에서 간단한 scalar 값을 찾는다."""

    tokens: list[str] = []
    for key in ("base_args", "override_args"):
        value = metadata.get(key)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                continue
        if isinstance(value, list):
            tokens.extend(str(item) for item in value)
    option = f"--{name.replace('_', '-')}"
    for index in range(len(tokens) - 1, -1, -1):
        if tokens[index] != option:
            continue
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            return tokens[index + 1]
        return True
    return None


def legacy_case_name(relative: Path) -> str:
    parts = relative.parts
    if len(parts) >= 3:
        return "/".join(parts[:-2])
    if len(parts) == 2:
        return parts[0]
    return parts[0] if parts else "."


def checkpoint_type(manifest: dict[str, Any]) -> str:
    explicit = manifest.get("checkpoint_type")
    if explicit:
        return str(explicit)
    checkpoint = manifest.get("checkpoint")
    if not checkpoint:
        return ""
    name = Path(str(checkpoint)).name
    if name == "checkpoint":
        return "final_checkpoint"
    return name


def phase_status(
    *,
    explicit: str,
    train_exit: int | None,
    infer_exit: int | None,
    run: dict[str, Any],
    metrics: dict[str, Any],
) -> tuple[str, str, str]:
    if train_exit not in (None, 0):
        train_status = "failed"
    elif run:
        train_status = "complete"
    elif explicit in {"train_failed", "model_not_cached", "running"}:
        train_status = explicit
    else:
        train_status = "unknown"

    if metrics:
        infer_status = "complete"
    elif infer_exit not in (None, 0) or explicit == "infer_failed":
        infer_status = "failed"
    elif run:
        infer_status = "incomplete"
    else:
        infer_status = "not_started"

    # 평가 metric은 사용 가능한 최종 산출물이므로 stale train exit/status보다 우선한다.
    if metrics:
        status = "complete"
    elif explicit in {"complete", "trained"}:
        status = "infer_incomplete"
    elif explicit:
        status = explicit
    elif run:
        status = "infer_incomplete"
    else:
        status = "not_started"
    return status, train_status, infer_status


def summarize_run(
    run_dir: Path, results_root: Path, suite_hint: str | None
) -> dict[str, Any]:
    warnings: list[str] = []
    metadata, _ = nearest_experiment(run_dir, results_root, warnings)
    config, config_source = load_config(run_dir, warnings)
    run = read_json(run_dir / "run.json", warnings)
    embedded_metadata = run.get("experiment")
    if isinstance(embedded_metadata, dict):
        # 독립 experiment.json을 우선하되 run.json에 저장된 동일 metadata도
        # 중단/이동된 과거 산출물의 fallback으로 사용한다.
        metadata = {**embedded_metadata, **metadata}
    metrics_path = run_dir / "eval/metrics.json"
    metrics = read_json(metrics_path, warnings)
    manifest = read_json(run_dir / "eval/inference_manifest.json", warnings)
    selection = read_json(run_dir / "best_checkpoint/selection.json", warnings)
    history = read_history(run_dir / "train_log.jsonl", warnings)

    try:
        relative = run_dir.relative_to(results_root)
    except ValueError:
        relative = Path(run_dir.name)

    # 공통 results root에서는 첫 경로가 suite다. `--suite proposed`로 집계할 때
    # metadata가 없는 다른 suite의 legacy run까지 proposed로 오인하지 않는다.
    path_suite = relative.parts[0] if relative.parts else ""
    inferred_suite = path_suite if path_suite in RUNNER_SUITES else ""
    suite = str(
        metadata.get("suite") or inferred_suite or suite_hint or results_root.name
    )
    base_name = str(metadata.get("base_name") or results_root.name)
    case_name = str(metadata.get("case_name") or legacy_case_name(relative))
    model_slug = str(
        metadata.get("model_slug")
        or config.get("model_slug")
        or metadata.get("model_argument")
        or run_dir.name
    )

    data = data_fields(config, run)
    generated_config_id = semantic_config_id(config, str(data["data_signature"]))
    config_id = str(metadata.get("config_id") or generated_config_id)

    train_exit = read_int(run_dir / "train_exit_code.txt")
    infer_exit = read_int(run_dir / "infer_exit_code.txt")
    explicit_status = read_text(run_dir / "status.txt")
    status, train_status, infer_status = phase_status(
        explicit=explicit_status,
        train_exit=train_exit,
        infer_exit=infer_exit,
        run=run,
        metrics=metrics,
    )
    if metrics and train_exit not in (None, 0):
        warnings.append(f"metric은 있으나 train_exit={train_exit}")

    predictions = count_jsonl(run_dir / "eval/predictions.jsonl")
    metric_count = metrics.get("count") if metrics else None
    if (
        predictions is not None
        and metric_count is not None
        and predictions != metric_count
    ):
        warnings.append(f"metric count {metric_count} != predictions {predictions}")

    trainer = run.get("trainer")
    trainer = trainer if isinstance(trainer, dict) else {}
    selected_metric_name = selection.get("metric") or trainer.get(
        "best_checkpoint_metric_name"
    )
    selected_metric_value = selection.get("metric_value")
    if selected_metric_value is None:
        selected_metric_value = trainer.get("best_checkpoint_metric")

    row: dict[str, Any] = {
        "suite": suite,
        "base_name": base_name,
        "case_name": case_name,
        "description": metadata.get("description", ""),
        "config_id": config_id,
        "config_diff": config_diff(metadata, config),
        "base_args": metadata.get("base_args", []),
        "override_args": metadata.get("override_args", []),
        "inference_args": metadata.get("inference_args", []),
        "status": status,
        "train_status": train_status,
        "infer_status": infer_status,
        "model_slug": model_slug,
        "model_id": config.get("model_id")
        or manifest.get("model_id")
        or metadata.get("model_argument")
        or "",
        "model_revision": config.get("model_revision", ""),
        "backbone_type": config.get("backbone_type", ""),
        "training_mode": config.get("training_mode")
        or experiment_option(metadata, "training_mode")
        or "",
        "seed": config.get("seed"),
        "checkpoint_type": checkpoint_type(manifest),
        # 추론 때 실제로 쓴 입력 표면. 학습 표면과 다를 수 있어(canonical 학습 -> raw 평가)
        # 표면 등급 판정에 둘 다 필요하다.
        "inference_essay_surface": manifest.get("essay_surface"),
        "infer_batch_size": manifest.get("batch_size"),
        "infer_load_in_4bit": manifest.get("load_in_4bit"),
        "infer_max_length": manifest.get("max_length"),
        "config_source": config_source,
        "training_rows": run.get("training_rows"),
        "competition_training_rows": run.get("competition_training_rows"),
        "extended_training_rows": run.get("extended_training_rows"),
        "validation_rows": run.get("validation_rows"),
        "validation_overlap_id_count": nested(
            run, "validation_overlap_guard", "identifier_overlap_count"
        ),
        "validation_overlap_text_count": nested(
            run, "validation_overlap_guard", "essay_text_overlap_count"
        ),
        "batch_size": config.get("batch_size"),
        "gradient_accumulation": config.get("gradient_accumulation"),
        "max_length": config.get("max_length"),
        "epochs": config.get("epochs"),
        "max_train_steps": config.get("max_train_steps"),
        "planned_steps": trainer.get("planned_steps"),
        "actual_steps": trainer.get("actual_steps"),
        "score_side_parameter_count": nested(
            trainer, "parameter_updates", "score_side", "parameter_count"
        ),
        "score_side_delta_l2": nested(
            trainer, "parameter_updates", "score_side", "delta_l2"
        ),
        "score_side_max_abs_delta": nested(
            trainer, "parameter_updates", "score_side", "max_abs_delta"
        ),
        "score_side_changed": nested(
            trainer, "parameter_updates", "score_side", "changed"
        ),
        "lora_parameter_count": nested(
            trainer, "parameter_updates", "lora", "parameter_count"
        ),
        "lora_delta_l2": nested(
            trainer, "parameter_updates", "lora", "delta_l2"
        ),
        "lora_max_abs_delta": nested(
            trainer, "parameter_updates", "lora", "max_abs_delta"
        ),
        "lora_changed": nested(
            trainer, "parameter_updates", "lora", "changed"
        ),
        "train_runtime_seconds": runtime_seconds(run),
        "train_samples_per_second": nested(
            trainer, "train_metrics", "train_samples_per_second"
        ),
        "train_steps_per_second": nested(
            trainer, "train_metrics", "train_steps_per_second"
        ),
        "peak_gpu_memory_reserved_gib": (
            float(run["peak_gpu_memory_reserved_bytes"]) / (1024**3)
            if isinstance(run.get("peak_gpu_memory_reserved_bytes"), (int, float))
            else None
        ),
        "peak_gpu_memory_allocated_gib": (
            float(run["peak_gpu_memory_allocated_bytes"]) / (1024**3)
            if isinstance(run.get("peak_gpu_memory_allocated_bytes"), (int, float))
            else None
        ),
        "best_checkpoint_metric_name": selected_metric_name,
        "best_checkpoint_metric_value": selected_metric_value,
        "best_checkpoint_epoch": selection.get("epoch")
        or trainer.get("best_checkpoint_epoch"),
        "best_checkpoint_step": selection.get("global_step")
        or trainer.get("best_checkpoint_global_step"),
        "prediction_records": predictions,
        "relative_run_dir": relative.as_posix(),
        "run_dir": str(run_dir.resolve()),
        **data,
    }
    summarize_metrics(row, metrics)
    # 두 공식 표면은 예측 파일에서 **항상 다시 계산한다.** 두 가지 이유가 있다.
    #
    # 1. 공식 지표가 확정되기 전 artifact의 `metrics.json`에는 raw 평균 block이 없고,
    #    average_matched 표면도 없다. 두 값을 같은 규칙으로 복원해야 세대가 다른 행을
    #    비교할 수 있다.
    # 2. `metrics.json`은 추론 당시 float64 내부 예측으로 계산했고 `score_predictions.jsonl`은
    #    직렬화된 값이다. 차이는 rho 약 1e-5 수준이지만 한 표에 두 정밀도 경로가 섞이는 것보다
    #    한 경로로 통일하는 것이 낫다.
    #
    # 예측 파일이 없으면 `metrics.json`에서 읽은 값을 그대로 둔다.
    recomputed = official_metrics_from_predictions(run_dir / "eval", warnings)
    if recomputed is not None:
        raw = recomputed["raw_continuous"]
        submitted = recomputed["average_matched"]
        row["raw_average_rmse"] = raw["rmse"]
        row["raw_average_spearman"] = raw["spearman"]
        row["submitted_average_rmse"] = submitted["rmse"]
        row["submitted_average_spearman"] = submitted["spearman"]
        # 과거 결과 reader/checkpoint 분석 notebook이 읽던 이름은 alias로만 남긴다.
        row["official_rmse"] = raw["rmse"]
        row["official_spearman"] = raw["spearman"]
        row["candidate_rmse"] = submitted["rmse"]
        row["candidate_spearman"] = submitted["spearman"]
        row["official_gold_source"] = recomputed["gold_source"]
        row["official_metric_count"] = recomputed["count"]
    summarize_history(row, history)
    row.update(alternate_evaluation_fields(run_dir, config, warnings))

    for key, value in config.items():
        row[f"config.{key}"] = value
    row["warnings"] = "; ".join(warnings)
    return row


def sort_and_rank(rows: list[dict[str, Any]]) -> None:
    # 현재 실제 제출 표면인 average_matched 정수 삼중 평균을 먼저 본다. canonical field가
    # 없는 과거 행은 최소 legacy alias와 raw/trait 진단값 순서로 내려가 표에서 사라지지 않게
    # 한다.
    def tier(row: dict[str, Any]) -> int:
        """표면 등급 우선 정렬. `contract`만 제출 성능이므로 위로 올린다."""

        return {
            "contract": 0,
            "derived": 1,
            "unknown": 2,
            "leaked": 3,
            "label_leak": 4,
        }.get(surface_class(row), 2)

    def primary(row: dict[str, Any]) -> float:
        for key in (
            "submitted_average_rmse",
            "candidate_rmse",
            "raw_average_rmse",
            "official_rmse",
            "overall_rmse",
        ):
            value = finite(row.get(key))
            if value is not None:
                return value
        return math.inf

    def secondary(row: dict[str, Any]) -> float:
        for key in (
            "submitted_average_spearman",
            "candidate_spearman",
            "raw_average_spearman",
            "official_spearman",
            "overall_spearman",
        ):
            value = finite(row.get(key))
            if value is not None:
                return -value
        return math.inf

    rows.sort(
        key=lambda row: (
            row.get("status") != "complete",
            # 표면 등급이 지표보다 먼저다. canonical 문단 구조를 본 run은 공식 RMSE가 `.41`로
            # 리더보드 1위보다 좋아 보이지만 제출 환경에서 재현할 수 없다. 그것이 순위 1위로
            # 올라오면 표를 읽는 사람이 최고 성능으로 오해한다.
            tier(row),
            primary(row),
            secondary(row),
            str(row.get("suite")),
            str(row.get("base_name")),
            str(row.get("case_name")),
            str(row.get("model_slug")),
        )
    )
    rank = 0
    for row in rows:
        # CSV에도 등급을 남긴다. markdown만 맞고 CSV는 등급이 없으면 자동 분석이 leaked 행을
        # 제출 성능으로 착각한다.
        row["surface_class"] = surface_class(row)
        rankable = any(
            finite(row.get(key)) is not None
            for key in (
                "submitted_average_rmse",
                "raw_average_rmse",
                "overall_rmse",
            )
        )
        if row.get("status") == "complete" and rankable:
            rank += 1
            row["rank"] = rank
        else:
            row["rank"] = ""


def collect(results_root: Path, suite: str | None = None) -> list[dict[str, Any]]:
    rows = [
        summarize_run(run_dir, results_root, suite)
        for run_dir in discover_run_dirs(results_root)
    ]
    if suite is not None:
        rows = [row for row in rows if row.get("suite") == suite]
    sort_and_rank(rows)
    return rows


def csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return compact_json(value)
    if isinstance(value, bool):
        return int(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    dynamic_fields = sorted(
        {key for row in rows for key in row if key.startswith("config.")}
    )
    fields = [*CORE_FIELDS, *dynamic_fields]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_cell(row.get(field)) for field in fields})


def fmt(value: Any, digits: int = 4) -> str:
    number = finite(value)
    return "-" if number is None else f"{number:.{digits}f}"


def md_cell(value: Any) -> str:
    text = str(value if value not in (None, "") else "-")
    return text.replace("|", "\\|").replace("\n", " ")


def trait_rmse(row: dict[str, Any], scope: str) -> str:
    """Return the sole trait-level metric shown in the front-facing table."""

    return fmt(row.get(f"{scope}_rmse"))


def progress(row: dict[str, Any]) -> str:
    value = fmt(row.get("best_overall_rmse"))
    if value == "-":
        return value
    step = finite(row.get("best_overall_rmse_step"))
    epoch = finite(row.get("best_overall_rmse_epoch"))
    if step is not None:
        return f"{value} (s{step:.0f})"
    if epoch is not None:
        return f"{value} (e{epoch:.0f})"
    return value


def short_data(row: dict[str, Any]) -> str:
    summary = str(row.get("data_summary") or "-")
    return summary if len(summary) <= 48 else summary[:45] + "..."


# 실제 대회 입력은 `paragraph[].form`을 구분자 없이 이어 붙인 official_raw다. 아래 등급이 다른
# 행의 점수를 같은 순위표에서 비교하면 안 된다.
DEPLOYABLE_SURFACE = "official_raw"
DERIVED_SURFACES = (
    "flat",
    "official_gap_newline",
    "official_raw_kiwi_sentence_newline_v1",
)


def surface_class(row: dict[str, Any]) -> str:
    """입력 표면 등급. `leaked`는 정답 문단 경계를 본 재현 불가 점수다."""

    # 감사 뒤 사람이 확정한 폴더 이름이 config보다 우선한다. `base_name`은 experiment.json에
    # 기록된 옛 이름이라 rename을 모르므로 실제 경로를 본다.
    location = f"{row.get('relative_run_dir') or ''} {row.get('run_dir') or ''}"
    # validation을 학습에 넣은 진단 run은 표에서 가장 위험한 행이다. 공식 RMSE가 `.3968`까지
    # 나오지만 평가 라벨을 본 값이라 아무 의미가 없다. 표면 문제와 별도로 표시한다.
    if str(row.get("config.primary_data_profile") or "") == "validation_leaked" or (
        "leak_validation" in str(row.get("case_name") or "")
    ):
        return "label_leak"
    if "NOT_DEPLOYABLE_canonical__" in location:
        return "leaked"
    train = str(row.get("config.essay_surface") or "?")
    infer = str(row.get("inference_essay_surface") or train)
    if train == DEPLOYABLE_SURFACE and infer == DEPLOYABLE_SURFACE:
        return "contract"
    if "canonical" in (train, infer):
        return "leaked"
    if infer in DERIVED_SURFACES or train in DERIVED_SURFACES:
        return "derived"
    return "unknown"


# 2026-08-06 정수 반올림 규칙이 적용된 뒤의 공개 리더보드다. 제출 뒤 우리 값이 어느
# 변형과 가장 가까웠는지 대조하려고 여기 고정해 둔다. 규칙 변경 전 값도 함께 남긴다.
LEADERBOARD_AFTER_ROUNDING = (
    ("망상감상대상연맹1557중대", 0.4545, 0.7109, 4.6125, None, None),
    ("털파카팔까알파카 (baseline)", 0.4608, 0.6951, 4.1819, None, None),
    ("LION", 0.4688, 0.7102, 4.3204, 0.4284, 0.7538),
    ("말평뉴비", 0.4613, 0.6909, 3.8035, 0.4282, 0.7470),
)


def _leaderboard_section(rows: list[dict[str, Any]]) -> list[str]:
    """현재 제출 표면 기준으로 리더보드와 우리 상위 run을 나란히 놓는다."""

    lines = [
        "## 공개 리더보드 대조",
        "",
        "공개 리더보드 (2026-08-06 정수 반올림 규칙 적용 후). 규칙 변경 전 값이 있는 팀은",
        "함께 적었다. 우리 값은 현재 단일 제출 표면인 average_matched 기준으로 정렬한다.",
        "",
        "| 팀 | RMSE | ρ | LLM Judge | 변경 전 RMSE | 변경 전 ρ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, rmse, rho, judge, old_rmse, old_rho in LEADERBOARD_AFTER_ROUNDING:
        before = f"{old_rmse:.4f}" if old_rmse is not None else "-"
        before_rho = f"{old_rho:.4f}" if old_rho is not None else "-"
        lines.append(
            f"| {name} | {rmse:.4f} | {rho:.4f} | {judge:.4f} | {before} | {before_rho} |"
        )

    ranked = sorted(
        (
            row
            for row in rows
            if row.get("status") == "complete"
            and finite(row.get("submitted_average_rmse")) is not None
            and surface_class(row) == "contract"
        ),
        key=lambda row: float(row["submitted_average_rmse"]),
    )[:5]
    lines += [
        "",
        "우리 상위 5개 (제출 평균 RMSE 순). **제출 평균**은 average_matched 정수 C/O/E의",
        "삼중 평균이고, **Raw 평균**은 모델의 연속 C/O/E 평균이다. 둘 다 공식",
        "`score.average` gold와 비교한다.",
        "",
        "| Case | Base | 제출 평균 RMSE | 제출 평균 ρ | Raw 평균 RMSE | Raw 평균 ρ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            "| {case} | {base} | {sr} | {ss} | {rr} | {rs} |".format(
                case=row.get("case_name"),
                base=row.get("base_name"),
                sr=fmt(row.get("submitted_average_rmse")),
                ss=fmt(row.get("submitted_average_spearman")),
                rr=fmt(row.get("raw_average_rmse")),
                rs=fmt(row.get("raw_average_spearman")),
            )
        )
    lines.append("")
    return lines


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    csv_path: Path,
    results_root: Path,
) -> None:
    complete = [row for row in rows if row.get("status") == "complete"]
    lines = [
        "# 실험 결과",
        "",
        f"- 생성 시각: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- 결과 root: `{results_root}`",
        f"- 완료: {len(complete)}/{len(rows)}",
        f"- 전체 설정 및 trait별 best/last 값: `{csv_path.name}`",
        "- ρ는 대회 공식 Spearman이다. RMSE와 ρ는 같은 예측/gold 표면에서 계산한다.",
        "- **Raw 평균 RMSE/ρ**는 모델의 연속 C/O/E 평균과 공식 `score.average` gold를",
        "  essay별로 짝지어 전체 샘플에 대해 1회 계산한다.",
        "- **제출 평균 RMSE/ρ**는 현재 제출 규칙인 average_matched 정수 C/O/E의 삼중 평균을",
        "  같은 공식 `score.average` gold와 비교한 값이다.",
        "- **C/O/E RMSE**는 각 trait의 연속 예측과 해당 trait gold를 비교한 값이다.",
        "  trait Spearman과 과거 checkpoint/reader 호환용 overall·alias는 CSV/JSON에만",
        "  남기고 이 전면 표에는 노출하지 않는다.",
        "- 공식 지표가 확정되기 전 artifact는 저장된 `score_predictions.jsonl`에서 **다시 계산**했다.",
        "- **표면** 열: `contract`만 제출 성능이다. `leaked`는 원천 문단 경계를 본 재현 불가 점수,",
        "  `derived`는 실제 입력에서 파생 가능하지만 제출 계약과 다른 표면이고,",
        "  `label_leak`은 validation을 학습에 넣은 진단 run이라 점수에 의미가 없다.",
        "",
        *_leaderboard_section(rows),
        "## 전체 표",
        "",
        "| 순위 | 표면 | Suite | Base | Case | Config | 모델 | 학습 | 상태 | 제출 평균 RMSE | 제출 평균 ρ | Raw 평균 RMSE | Raw 평균 ρ | C RMSE | O RMSE | E RMSE | 데이터 | 행 | 시간(h) | row/s | Peak GiB |",
        "|---:|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        runtime = finite(row.get("train_runtime_seconds"))
        values: Iterable[Any] = (
            row.get("rank"),
            surface_class(row),
            row.get("suite"),
            row.get("base_name"),
            row.get("case_name"),
            row.get("config_id"),
            row.get("model_slug"),
            row.get("training_mode"),
            row.get("status"),
            fmt(row.get("submitted_average_rmse")),
            fmt(row.get("submitted_average_spearman")),
            fmt(row.get("raw_average_rmse")),
            fmt(row.get("raw_average_spearman")),
            trait_rmse(row, "content"),
            trait_rmse(row, "organization"),
            trait_rmse(row, "expression"),
            short_data(row),
            row.get("training_rows"),
            fmt(runtime / 3600 if runtime is not None else None, 2),
            fmt(row.get("train_samples_per_second"), 2),
            fmt(row.get("peak_gpu_memory_reserved_gib"), 2),
        )
        lines.append("| " + " | ".join(md_cell(value) for value in values) + " |")
    if not rows:
        lines.append(
            "| - | - | - | - | - | 결과 없음 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |"
        )

    warnings = [row for row in rows if row.get("warnings")]
    if warnings:
        lines.extend(
            [
                "",
                "## 산출물 경고",
                "",
                "| Case | 모델 | 상태 | 경고 |",
                "|---|---|---|---|",
            ]
        )
        for row in warnings:
            lines.append(
                "| "
                + " | ".join(
                    md_cell(row.get(key))
                    for key in ("case_name", "model_slug", "status", "warnings")
                )
                + " |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    if args.suite and args.base:
        # 라운드별 표. 확정 단계마다 baseline이 다르므로 그 폴더 안에서만 비교한다.
        default_dir = results_root / args.suite / args.base
        default_csv = default_dir / "results.csv"
        default_markdown = default_dir / "results.md"
    elif args.suite:
        default_dir = results_root / args.suite
        default_csv = default_dir / "results.csv"
        default_markdown = default_dir / "results.md"
    else:
        default_csv = results_root / "all_results.csv"
        default_markdown = results_root / "all_results.md"
    csv_path = Path(args.csv).resolve() if args.csv else default_csv
    markdown_path = Path(args.markdown).resolve() if args.markdown else default_markdown

    rows = collect(results_root, suite=args.suite)
    if args.base:
        rows = [row for row in rows if str(row.get("base_name", "")) == args.base]
    write_csv(csv_path, rows)
    write_markdown(markdown_path, rows, csv_path, results_root)
    print(f"rows={len(rows)}")
    print(f"csv={csv_path}")
    print(f"markdown={markdown_path}")


if __name__ == "__main__":
    main()
