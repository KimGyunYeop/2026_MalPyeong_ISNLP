"""Recompute every score experiment under ``main_code/results``.

This report intentionally lives next to, rather than replacing, the historical
``results.md``/``results.csv`` reports.  It reads the saved raw predictions and
recomputes a complete metric matrix for three prediction surfaces:

* ``average_matched``: integer C/O/E with the total matched to the raw total;
* ``independent_half_up``: independent half-up rounding of each trait;
* ``raw_continuous``: the saved continuous head outputs.

For every surface we report trait-level metrics, the macro average of the three
trait metrics, essay-mean-first metrics, and flattened 3N-cell pooled metrics.
Gold C/O/E and the stored gold ``score.average`` are never rounded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .aggregate_results import discover_run_dirs, summarize_run, surface_class
from .official_metrics import official_metrics
from .utils import average_matched_integer_scores, per_trait_integer_scores


TRAITS = ("content", "organization", "expression")
SURFACES = (
    "average_matched",
    "independent_half_up",
    "raw_continuous",
)
SURFACE_TITLES = {
    "average_matched": "Average-matched 정수",
    "independent_half_up": "일반 사사오입 정수",
    "raw_continuous": "Raw continuous",
}
SURFACE_DESCRIPTIONS = {
    "average_matched": (
        "raw C/O/E 합을 사사오입한 총점에 맞춘 뒤, 그 합 안에서 raw와 가장 "
        "가까운 1~5 정수 삼중"
    ),
    "independent_half_up": "각 raw trait에 floor(x + 0.5)를 독립 적용하고 1~5로 clip",
    "raw_continuous": "저장된 분류 head의 연속 기대값 C/O/E",
}

PROTECTED_REPORT_NAMES = {
    # Some older/manual rounds used the singular spelling.  Protect both
    # spellings so an explicit --markdown/--csv path cannot destroy either
    # generation of historical report.
    "result.md",
    "result.csv",
    "results.md",
    "results.csv",
    "all_results.md",
    "all_results.csv",
    "remeasured.md",
}
SURFACE_CLASS_ORDER = {
    "contract": 0,
    "derived": 1,
    "unknown": 2,
    "leaked": 3,
    "label_leak": 4,
}

# Config fields that can change the learned method or its evaluation contract.
# Administrative paths, large registries, and schema bookkeeping are omitted;
# every other resolved-config key is retained automatically, so a new method
# option starts appearing in the report as soon as one experiment changes it.
METHOD_CONFIG_EXCLUDED = {
    "schema_version",
    "model_cache_dir",
    "model_slug",
    "model_source_run",
    "trust_remote_code",
    "prompt_registry",
    "detail_rater_registry",
    "dataset_root",
    "extended_data_dir",
}

# Showing every fixed field makes the report unreadable.  These fixed fields
# define the core model/data/training contract; all varying method fields are
# shown regardless of whether they occur in this curated list.
CORE_FIXED_METHOD_KEYS = (
    "model_id",
    "model_revision",
    "backbone_type",
    "training_mode",
    "use_qlora",
    "primary_data_profile",
    "dataset_schedule",
    "extended_datasets",
    "validation_holdout_size",
    "input_format",
    "essay_surface",
    "pooling",
    "layer_aggregation",
    "head_type",
    "score_head",
    "detail_head_mode",
    "detail_final_source",
    "batch_size",
    "gradient_accumulation",
    "max_length",
    "max_train_steps",
    "lora_r",
    "lora_alpha",
    "lora_targets",
    "lora_include_mlp",
    "mse_loss_weight",
    "score_postprocess",
    "best_checkpoint_metric",
    "seed",
)

METRIC_FIELDS = (
    # 공식 지표를 맨 앞에 둔다. 표를 읽을 때 진단 지표와 헷갈리면 안 된다.
    "official_rmse",
    "official_spearman",
    "content_rmse",
    "content_spearman",
    "organization_rmse",
    "organization_spearman",
    "expression_rmse",
    "expression_spearman",
    "trait_macro_rmse",
    "trait_macro_spearman",
    "mean_first_rmse",
    "mean_first_spearman",
    "pooled_rmse",
    "pooled_spearman",
)

CSV_FIELDS = (
    "record_type",
    "rank",
    "rank_group",
    "surface",
    "surface_title",
    "suite",
    "physical_base_name",
    "logical_base_name",
    "case_name",
    "model_slug",
    "config_id",
    "run_status",
    "analysis_status",
    "surface_class",
    "evaluation_name",
    "checkpoint",
    "same_raw_as_primary",
    "count",
    "input_path",
    "input_sha256",
    "prediction_path",
    "prediction_sha256",
    "raw_stream_sha256",
    "description",
    "method_config_json",
    "explicit_overrides_json",
    *METRIC_FIELDS,
    "relative_run_dir",
    "relative_eval_dir",
    "warnings",
)


class ArtifactError(RuntimeError):
    """A saved evaluation cannot be measured without silently dropping rows."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent / "results"
    parser = argparse.ArgumentParser(
        description=(
            "저장된 score_predictions를 다시 읽어 세 예측 표면과 세 집계 방식의 "
            "result_all.md/result_all.csv를 생성"
        )
    )
    parser.add_argument("--results-root", default=str(default_root))
    parser.add_argument("--suite", help="physical suite 디렉터리 하나만 집계")
    parser.add_argument("--base", help="physical base 디렉터리 하나만 집계")
    parser.add_argument("--markdown", "--markdown-output", dest="markdown")
    parser.add_argument("--csv", "--csv-output", dest="csv")
    parser.add_argument(
        "--write-existing-scopes",
        action="store_true",
        help=(
            "root와 기존 results.md가 있는 모든 디렉터리에 sibling "
            "result_all.md/result_all.csv를 생성"
        ),
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="eval_best_* 진단 부록을 생략하고 primary eval만 읽음",
    )
    return parser.parse_args(argv)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactError(f"{label}가 숫자가 아님")
    number = float(value)
    if not math.isfinite(number):
        raise ArtifactError(f"{label}가 finite가 아님")
    return number


def _score_vector(value: Any, label: str) -> np.ndarray:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{label}가 object가 아님")
    return np.asarray(
        [_finite_number(value.get(trait), f"{label}.{trait}") for trait in TRAITS],
        dtype=np.float64,
    )


def _json_safe(value: Any) -> Any:
    """Normalize tuples/paths and uncommon scalars for stable report JSON."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _stable_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _method_config_from_summary(base_row: Mapping[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for field, value in base_row.items():
        if not field.startswith("config."):
            continue
        name = field.removeprefix("config.")
        if name in METHOD_CONFIG_EXCLUDED:
            continue
        config[name] = _json_safe(value)
    return dict(sorted(config.items()))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.size == 0:
        raise ValueError("RMSE arrays must have the same non-empty shape")
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def _spearman(prediction: np.ndarray, truth: np.ndarray) -> float | None:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    if prediction.shape != truth.shape or prediction.size < 2:
        return None
    if np.unique(prediction).size < 2 or np.unique(truth).size < 2:
        return None
    value = float(np.corrcoef(_average_ranks(prediction), _average_ranks(truth))[0, 1])
    return None if not math.isfinite(value) else value


def metrics_for_surface(
    prediction: np.ndarray,
    gold_traits: np.ndarray,
    gold_average: np.ndarray,
) -> dict[str, float | None]:
    """Compute all 12 requested metrics for one prediction surface."""

    prediction = np.asarray(prediction, dtype=np.float64)
    gold_traits = np.asarray(gold_traits, dtype=np.float64)
    gold_average = np.asarray(gold_average, dtype=np.float64)
    if prediction.shape != gold_traits.shape or prediction.ndim != 2:
        raise ValueError("prediction/gold_traits must have the same [N,3] shape")
    if prediction.shape[1] != len(TRAITS) or gold_average.shape != (prediction.shape[0],):
        raise ValueError("expected three traits and one stored gold average per row")
    if prediction.shape[0] < 2:
        raise ValueError("at least two rows are required")
    if not (
        np.isfinite(prediction).all()
        and np.isfinite(gold_traits).all()
        and np.isfinite(gold_average).all()
    ):
        raise ValueError("metric inputs must be finite")

    result: dict[str, float | None] = {}
    trait_rmses: list[float] = []
    trait_spearmans: list[float] = []
    for index, trait in enumerate(TRAITS):
        rmse = _rmse(prediction[:, index], gold_traits[:, index])
        rho = _spearman(prediction[:, index], gold_traits[:, index])
        result[f"{trait}_rmse"] = rmse
        result[f"{trait}_spearman"] = rho
        trait_rmses.append(rmse)
        if rho is not None:
            trait_spearmans.append(rho)

    result["trait_macro_rmse"] = float(np.mean(trait_rmses))
    result["trait_macro_spearman"] = (
        float(np.mean(trait_spearmans)) if trait_spearmans else None
    )
    prediction_average = prediction.mean(axis=1)
    result["mean_first_rmse"] = _rmse(prediction_average, gold_average)
    result["mean_first_spearman"] = _spearman(prediction_average, gold_average)
    result["pooled_rmse"] = _rmse(prediction.reshape(-1), gold_traits.reshape(-1))
    result["pooled_spearman"] = _spearman(
        prediction.reshape(-1), gold_traits.reshape(-1)
    )
    # 운영측 공지 코드(2026-08-19) 그대로. 위 12개는 진단용이고 리더보드가 읽는 숫자는
    # 이것 하나다. 평가자는 우리가 **실제로 내보낸 값**에 영역별 round_half_up을 적용한
    # 뒤 평균하므로, surface마다 따로 계산해야 "이걸 제출했다면 받을 점수"가 된다.
    official = official_metrics(prediction, gold_average)
    result["official_rmse"] = official["rmse"]
    result["official_spearman"] = official["spearman"]
    return result


def compute_metric_matrix(
    raw_prediction: np.ndarray,
    gold_traits: np.ndarray,
    gold_average: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float | None]]]:
    """Build three prediction surfaces and compute every requested metric."""

    raw_prediction = np.asarray(raw_prediction, dtype=np.float64)
    surfaces = {
        "average_matched": average_matched_integer_scores(raw_prediction),
        "independent_half_up": per_trait_integer_scores(raw_prediction),
        "raw_continuous": raw_prediction.copy(),
    }
    matrix = {
        name: metrics_for_surface(values, gold_traits, gold_average)
        for name, values in surfaces.items()
    }
    return surfaces, matrix


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_raw_stream(ids: Sequence[str], raw: np.ndarray) -> str:
    digest = hashlib.sha256()
    for essay_id, values in zip(ids, raw, strict=True):
        digest.update(essay_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(values, dtype="<f8").tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"{path}: JSON 읽기 실패: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"{path}: JSON root가 object가 아님")
    return value


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Read only LF/CRLF record boundaries; U+2028 inside essays is data."""

    try:
        stream = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ArtifactError(f"{path}: 열기 실패: {exc}") from exc
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactError(
                    f"{path}:{line_number}: JSONL 파싱 실패: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ArtifactError(f"{path}:{line_number}: JSONL row가 object가 아님")
            yield line_number, value


def _resolve_manifest_input(raw_path: Any, eval_dir: Path) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ArtifactError(f"{eval_dir}/inference_manifest.json에 input이 없음")
    value = Path(raw_path)
    candidates = [value]
    if not value.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        candidates.extend((project_root / value, eval_dir / value))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ArtifactError(f"manifest input 파일이 없음: {raw_path}")


def _load_gold(
    path: Path,
    cache: dict[Path, tuple[list[str], dict[str, tuple[np.ndarray, float]], str]],
) -> tuple[list[str], dict[str, tuple[np.ndarray, float]], str]:
    cached = cache.get(path)
    if cached is not None:
        return cached
    order: list[str] = []
    records: dict[str, tuple[np.ndarray, float]] = {}
    for line_number, row in _iter_jsonl(path):
        raw_id = row.get("essay_id", row.get("id"))
        if not isinstance(raw_id, str) or not raw_id:
            raise ArtifactError(f"{path}:{line_number}: id/essay_id가 없음")
        if raw_id in records:
            raise ArtifactError(f"{path}:{line_number}: duplicate gold id {raw_id}")
        score = row.get("score")
        traits = _score_vector(score, f"gold[{raw_id}].score")
        if not isinstance(score, Mapping):
            raise ArtifactError(f"gold[{raw_id}].score가 object가 아님")
        average = _finite_number(score.get("average"), f"gold[{raw_id}].score.average")
        order.append(raw_id)
        records[raw_id] = (traits, average)
    if len(records) < 2:
        raise ArtifactError(f"{path}: gold row가 2개 미만")
    result = (order, records, _sha256_file(path))
    cache[path] = result
    return result


def _extract_raw(row: Mapping[str, Any], essay_id: str) -> np.ndarray:
    surfaces = row.get("score_surfaces")
    canonical: Any = None
    if isinstance(surfaces, Mapping):
        raw_block = surfaces.get("raw_continuous")
        if isinstance(raw_block, Mapping):
            canonical = raw_block.get("scores")
    legacy = row.get("scores")
    source = canonical if canonical is not None else legacy
    raw = _score_vector(source, f"prediction[{essay_id}].raw")
    if canonical is not None and legacy is not None:
        legacy_vector = _score_vector(legacy, f"prediction[{essay_id}].scores")
        if not np.allclose(raw, legacy_vector, rtol=0.0, atol=1e-9):
            raise ArtifactError(f"{essay_id}: canonical raw와 scores가 불일치")
    return raw


def _validate_saved_surfaces(
    row: Mapping[str, Any], essay_id: str, raw: np.ndarray
) -> None:
    expected = average_matched_integer_scores(raw.reshape(1, 3))[0]
    candidates: list[tuple[str, Any]] = []
    surfaces = row.get("score_surfaces")
    if isinstance(surfaces, Mapping):
        matched = surfaces.get("average_matched")
        if isinstance(matched, Mapping) and matched.get("scores") is not None:
            candidates.append(("score_surfaces.average_matched", matched.get("scores")))
    if row.get("submitted_scores") is not None:
        candidates.append(("submitted_scores", row.get("submitted_scores")))
    for label, value in candidates:
        saved = _score_vector(value, f"prediction[{essay_id}].{label}")
        if not np.array_equal(saved, expected):
            raise ArtifactError(f"{essay_id}: 저장 {label}와 raw 재계산이 불일치")


def analyze_evaluation(
    eval_dir: Path,
    base_row: Mapping[str, Any],
    results_root: Path,
    gold_cache: dict[Path, tuple[list[str], dict[str, tuple[np.ndarray, float]], str]],
) -> dict[str, Any]:
    """Strictly join one saved prediction artifact to its own manifest input."""

    prediction_path = eval_dir / "score_predictions.jsonl"
    manifest_path = eval_dir / "inference_manifest.json"
    manifest = _read_json_object(manifest_path)
    input_path = _resolve_manifest_input(manifest.get("input"), eval_dir)
    gold_order, gold_records, input_sha = _load_gold(input_path, gold_cache)

    ids: list[str] = []
    raw_by_id: dict[str, np.ndarray] = {}
    seen: set[str] = set()
    for line_number, record in _iter_jsonl(prediction_path):
        raw_id = record.get("essay_id", record.get("id"))
        if not isinstance(raw_id, str) or not raw_id:
            raise ArtifactError(f"{prediction_path}:{line_number}: essay_id가 없음")
        if raw_id in seen:
            raise ArtifactError(f"{prediction_path}:{line_number}: duplicate essay_id {raw_id}")
        if raw_id not in gold_records:
            raise ArtifactError(f"{prediction_path}:{line_number}: unknown essay_id {raw_id}")
        raw = _extract_raw(record, raw_id)
        _validate_saved_surfaces(record, raw_id, raw)
        embedded_labels = record.get("labels")
        if embedded_labels is not None:
            embedded = _score_vector(
                embedded_labels, f"prediction[{raw_id}].labels"
            )
            if not np.allclose(
                embedded, gold_records[raw_id][0], rtol=0.0, atol=1e-8
            ):
                raise ArtifactError(f"{raw_id}: prediction labels와 manifest gold 불일치")
        seen.add(raw_id)
        ids.append(raw_id)
        raw_by_id[raw_id] = raw

    missing = [essay_id for essay_id in gold_order if essay_id not in seen]
    extra = [essay_id for essay_id in ids if essay_id not in gold_records]
    if missing or extra:
        raise ArtifactError(
            f"prediction/gold ID set 불일치: missing={len(missing)}, extra={len(extra)}"
        )

    expected_count = manifest.get("count", manifest.get("label_count"))
    if isinstance(expected_count, int) and expected_count != len(ids):
        raise ArtifactError(
            f"manifest count={expected_count}, prediction count={len(ids)} 불일치"
        )

    # Metrics are an ID join, not a positional join.  Canonicalizing to the
    # manifest input order makes a legitimately shuffled prediction JSONL
    # produce the same metrics and fingerprint, while missing/duplicate IDs
    # still fail closed above.
    ids = gold_order
    raw_array = np.vstack([raw_by_id[essay_id] for essay_id in ids]).astype(
        np.float64, copy=False
    )
    gold_traits = np.vstack([gold_records[essay_id][0] for essay_id in ids])
    gold_average = np.asarray(
        [gold_records[essay_id][1] for essay_id in ids], dtype=np.float64
    )
    _, metric_matrix = compute_metric_matrix(raw_array, gold_traits, gold_average)

    run_dir = eval_dir.parent
    relative_run = run_dir.relative_to(results_root)
    relative_eval = eval_dir.relative_to(results_root)
    parts = relative_run.parts
    physical_suite = parts[0] if parts else ""
    physical_base = parts[1] if len(parts) >= 2 else ""
    checkpoint = manifest.get("checkpoint")
    checkpoint_label = Path(checkpoint).name if isinstance(checkpoint, str) else ""
    warnings = str(base_row.get("warnings") or "")
    method_config = _method_config_from_summary(base_row)
    explicit_overrides = _json_safe(base_row.get("override_args") or [])
    return {
        "record_type": "primary" if eval_dir.name == "eval" else "variant",
        "suite": physical_suite or str(base_row.get("suite") or ""),
        "physical_base_name": physical_base,
        "logical_base_name": str(base_row.get("base_name") or ""),
        "case_name": str(base_row.get("case_name") or run_dir.name),
        "model_slug": str(base_row.get("model_slug") or ""),
        "config_id": str(base_row.get("config_id") or ""),
        "run_status": str(base_row.get("status") or ""),
        "analysis_status": "complete",
        "surface_class": surface_class(dict(base_row)),
        "evaluation_name": eval_dir.name,
        "checkpoint": checkpoint_label,
        "same_raw_as_primary": None,
        "count": len(ids),
        "input_path": str(input_path),
        "input_name": input_path.name,
        "input_sha256": input_sha,
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": _sha256_file(prediction_path),
        "raw_stream_sha256": _sha256_raw_stream(ids, raw_array),
        "method_config": method_config,
        "method_config_json": _stable_json(method_config),
        "explicit_overrides": explicit_overrides,
        "explicit_overrides_json": _stable_json(explicit_overrides),
        "description": str(base_row.get("description") or ""),
        "relative_run_dir": relative_run.as_posix(),
        "relative_eval_dir": relative_eval.as_posix(),
        "run_dir": str(run_dir.resolve()),
        "eval_dir": str(eval_dir.resolve()),
        "warnings": warnings,
        "metrics": metric_matrix,
        "ranks": {},
    }


def _invalid_evaluation_row(
    eval_dir: Path,
    base_row: Mapping[str, Any],
    results_root: Path,
    error: Exception,
) -> dict[str, Any]:
    run_dir = eval_dir.parent
    relative = run_dir.relative_to(results_root)
    parts = relative.parts
    existing_warning = str(base_row.get("warnings") or "")
    warning = f"{existing_warning}; {error}" if existing_warning else str(error)
    method_config = _method_config_from_summary(base_row)
    explicit_overrides = _json_safe(base_row.get("override_args") or [])
    return {
        "record_type": "primary" if eval_dir.name == "eval" else "variant",
        "suite": parts[0] if parts else str(base_row.get("suite") or ""),
        "physical_base_name": parts[1] if len(parts) >= 2 else "",
        "logical_base_name": str(base_row.get("base_name") or ""),
        "case_name": str(base_row.get("case_name") or run_dir.name),
        "model_slug": str(base_row.get("model_slug") or ""),
        "config_id": str(base_row.get("config_id") or ""),
        "run_status": str(base_row.get("status") or ""),
        "analysis_status": "invalid",
        "surface_class": surface_class(dict(base_row)),
        "evaluation_name": eval_dir.name,
        "checkpoint": "",
        "same_raw_as_primary": None,
        "count": None,
        "input_path": "",
        "input_name": "",
        "input_sha256": "",
        "prediction_path": str((eval_dir / "score_predictions.jsonl").resolve()),
        "prediction_sha256": "",
        "raw_stream_sha256": "",
        "method_config": method_config,
        "method_config_json": _stable_json(method_config),
        "explicit_overrides": explicit_overrides,
        "explicit_overrides_json": _stable_json(explicit_overrides),
        "description": str(base_row.get("description") or ""),
        "relative_run_dir": relative.as_posix(),
        "relative_eval_dir": eval_dir.relative_to(results_root).as_posix(),
        "run_dir": str(run_dir.resolve()),
        "eval_dir": str(eval_dir.resolve()),
        "warnings": warning,
        "metrics": {},
        "ranks": {},
    }


def _unmeasured_row(
    run_dir: Path, base_row: Mapping[str, Any], results_root: Path
) -> dict[str, Any]:
    relative = run_dir.relative_to(results_root)
    parts = relative.parts
    method_config = _method_config_from_summary(base_row)
    explicit_overrides = _json_safe(base_row.get("override_args") or [])
    return {
        "record_type": "unmeasured",
        "suite": parts[0] if parts else str(base_row.get("suite") or ""),
        "physical_base_name": parts[1] if len(parts) >= 2 else "",
        "logical_base_name": str(base_row.get("base_name") or ""),
        "case_name": str(base_row.get("case_name") or run_dir.name),
        "model_slug": str(base_row.get("model_slug") or ""),
        "config_id": str(base_row.get("config_id") or ""),
        "run_status": str(base_row.get("status") or ""),
        "analysis_status": "missing_primary_predictions",
        "surface_class": surface_class(dict(base_row)),
        "evaluation_name": "eval",
        "checkpoint": "",
        "same_raw_as_primary": None,
        "count": None,
        "input_path": "",
        "input_name": "",
        "input_sha256": "",
        "prediction_path": "",
        "prediction_sha256": "",
        "raw_stream_sha256": "",
        "method_config": method_config,
        "method_config_json": _stable_json(method_config),
        "explicit_overrides": explicit_overrides,
        "explicit_overrides_json": _stable_json(explicit_overrides),
        "description": str(base_row.get("description") or ""),
        "relative_run_dir": relative.as_posix(),
        "relative_eval_dir": "",
        "run_dir": str(run_dir.resolve()),
        "eval_dir": "",
        "warnings": str(base_row.get("warnings") or ""),
        "metrics": {},
        "ranks": {},
    }


def collect_report(
    results_root: Path, *, include_variants: bool = True
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect all scoring runs once; paragraph-boundary runs are not regressors."""

    results_root = results_root.resolve()
    evaluations: list[dict[str, Any]] = []
    unmeasured: list[dict[str, Any]] = []
    gold_cache: dict[
        Path, tuple[list[str], dict[str, tuple[np.ndarray, float]], str]
    ] = {}

    for run_dir in discover_run_dirs(results_root):
        try:
            relative = run_dir.relative_to(results_root)
        except ValueError:
            continue
        if relative.parts and relative.parts[0] == "paragraph_boundary":
            continue
        base_row = summarize_run(run_dir, results_root, None)
        primary_path = run_dir / "eval" / "score_predictions.jsonl"
        if not primary_path.is_file():
            unmeasured.append(_unmeasured_row(run_dir, base_row, results_root))
        eval_paths = sorted(run_dir.glob("eval*/score_predictions.jsonl"))
        if not include_variants:
            eval_paths = [path for path in eval_paths if path.parent.name == "eval"]
        run_rows: list[dict[str, Any]] = []
        for prediction_path in eval_paths:
            eval_dir = prediction_path.parent
            try:
                row = analyze_evaluation(
                    eval_dir, base_row, results_root, gold_cache
                )
            except (ArtifactError, OSError, ValueError, KeyError) as exc:
                row = _invalid_evaluation_row(
                    eval_dir, base_row, results_root, exc
                )
            run_rows.append(row)
            evaluations.append(row)
        primary = next(
            (row for row in run_rows if row["record_type"] == "primary"), None
        )
        if primary is not None and primary.get("raw_stream_sha256"):
            for row in run_rows:
                if row["record_type"] == "variant":
                    row["same_raw_as_primary"] = (
                        row.get("raw_stream_sha256")
                        == primary.get("raw_stream_sha256")
                    )

    assign_ranks(evaluations)
    evaluations.sort(key=_row_display_key)
    unmeasured.sort(
        key=lambda row: (
            row.get("suite", ""),
            row.get("physical_base_name", ""),
            row.get("case_name", ""),
            row.get("model_slug", ""),
            row.get("relative_run_dir", ""),
        )
    )
    return evaluations, unmeasured


def _rank_group(row: Mapping[str, Any]) -> str:
    input_sha = str(row.get("input_sha256") or "")
    count = row.get("count")
    classification = str(row.get("surface_class") or "unknown")
    return f"N={count}|input={input_sha[:12]}|surface_class={classification}"


def _metric_value(row: Mapping[str, Any], surface: str, field: str) -> float | None:
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    surface_metrics = metrics.get(surface)
    if not isinstance(surface_metrics, Mapping):
        return None
    value = surface_metrics.get(field)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _metric_sort_key(row: Mapping[str, Any], surface: str) -> tuple[Any, ...]:
    # 2026-08-19 운영측 답변으로 산출식이 확정됐다. 이전 기본 정렬은 "세 영역 RMSE의
    # 산술평균"이었는데 그건 리더보드가 쓰는 값이 아니다. 공식 지표로 정렬한다.
    rmse = _metric_value(row, surface, "official_rmse")
    rho = _metric_value(row, surface, "official_spearman")
    if rmse is None:
        # 공식 지표가 없는 과거 산출물은 예전 기준으로 뒤에 붙인다.
        rmse = _metric_value(row, surface, "trait_macro_rmse")
        rho = _metric_value(row, surface, "mean_first_spearman")
    return (
        math.inf if rmse is None else rmse,
        math.inf if rho is None else -rho,
        str(row.get("relative_run_dir") or ""),
        str(row.get("evaluation_name") or ""),
    )


def assign_ranks(rows: list[dict[str, Any]]) -> None:
    """Rank primary evaluations within comparable input/surface-class cohorts."""

    for row in rows:
        row["rank_group"] = _rank_group(row) if row.get("count") else ""
        row["ranks"] = {}
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if (
            row.get("record_type") != "primary"
            or row.get("analysis_status") != "complete"
            or row.get("surface_class") == "label_leak"
        ):
            continue
        groups.setdefault(str(row["rank_group"]), []).append(row)
    for group_rows in groups.values():
        for surface in SURFACES:
            ranked = sorted(group_rows, key=lambda row: _metric_sort_key(row, surface))
            for rank, row in enumerate(ranked, start=1):
                row["ranks"][surface] = rank


def _row_display_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        0 if row.get("record_type") == "primary" else 1,
        SURFACE_CLASS_ORDER.get(str(row.get("surface_class")), 9),
        str(row.get("input_sha256") or ""),
        _metric_sort_key(row, "average_matched"),
    )


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def filter_scope(
    rows: Sequence[dict[str, Any]], scope_dir: Path
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("run_dir") and _is_within(Path(str(row["run_dir"])), scope_dir)
    ]


def _fmt(value: Any, digits: int = 7) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def _md(value: Any) -> str:
    text = "-" if value in (None, "") else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _experiment_label(row: Mapping[str, Any]) -> str:
    suite = str(row.get("suite") or "")
    base = str(row.get("physical_base_name") or "")
    return "/".join(part for part in (suite, base) if part) or "-"


def _group_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        SURFACE_CLASS_ORDER.get(str(row.get("surface_class")), 9),
        str(row.get("input_sha256") or ""),
        int(row.get("count") or 0),
    )


def _group_title(group_rows: Sequence[Mapping[str, Any]]) -> str:
    row = group_rows[0]
    return (
        f"{row.get('input_name') or 'unknown input'} · N={row.get('count') or '-'} · "
        f"{row.get('surface_class') or 'unknown'} · input SHA "
        f"{str(row.get('input_sha256') or '-')[:12]}"
    )


def _summary_table(
    rows: Sequence[dict[str, Any]], surface: str, *, primary: bool
) -> list[str]:
    lines = [
        "| 순위 | Suite/Base | Case | 모델 | Config | Eval | N | 영역 RMSE 평균 | 영역 ρ 평균 | 평균 후 RMSE | 평균 후 ρ | 3N pooled RMSE | 3N pooled ρ |",
        "|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: _metric_sort_key(item, surface)):
        metrics = row["metrics"][surface]
        rank = row.get("ranks", {}).get(surface, "") if primary else "-"
        values = (
            rank,
            _experiment_label(row),
            row.get("case_name"),
            row.get("model_slug"),
            row.get("config_id"),
            row.get("evaluation_name"),
            row.get("count"),
            _fmt(metrics.get("trait_macro_rmse")),
            _fmt(metrics.get("trait_macro_spearman")),
            _fmt(metrics.get("mean_first_rmse")),
            _fmt(metrics.get("mean_first_spearman")),
            _fmt(metrics.get("pooled_rmse")),
            _fmt(metrics.get("pooled_spearman")),
        )
        lines.append("| " + " | ".join(_md(value) for value in values) + " |")
    return lines


def _trait_detail_table(
    rows: Sequence[dict[str, Any]], surface: str
) -> list[str]:
    lines = [
        "| Suite/Base | Case | 모델 | Config | Eval | C RMSE | C ρ | O RMSE | O ρ | E RMSE | E ρ |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: _metric_sort_key(item, surface)):
        metrics = row["metrics"][surface]
        values = (
            _experiment_label(row),
            row.get("case_name"),
            row.get("model_slug"),
            row.get("config_id"),
            row.get("evaluation_name"),
            _fmt(metrics.get("content_rmse")),
            _fmt(metrics.get("content_spearman")),
            _fmt(metrics.get("organization_rmse")),
            _fmt(metrics.get("organization_spearman")),
            _fmt(metrics.get("expression_rmse")),
            _fmt(metrics.get("expression_spearman")),
        )
        lines.append("| " + " | ".join(_md(value) for value in values) + " |")
    return lines


def _metric_sections(
    rows: Sequence[dict[str, Any]], *, primary: bool, heading_level: int
) -> list[str]:
    valid = [row for row in rows if row.get("analysis_status") == "complete"]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in valid:
        grouped.setdefault(_group_key(row), []).append(row)
    lines: list[str] = []
    for surface in SURFACES:
        prefix = "#" * heading_level
        lines.extend(
            [
                f"{prefix} {SURFACE_TITLES[surface]}",
                "",
                SURFACE_DESCRIPTIONS[surface] + ".",
                "",
            ]
        )
        if not grouped:
            lines.extend(["측정 가능한 결과가 없습니다.", ""])
            continue
        for _, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
            lines.extend(
                [
                    f"{prefix}# {_group_title(group_rows)}",
                    "",
                    *_summary_table(group_rows, surface, primary=primary),
                    "",
                    "<details>",
                    "<summary>C/O/E 영역별 상세</summary>",
                    "",
                    *_trait_detail_table(group_rows, surface),
                    "",
                    "</details>",
                    "",
                ]
            )
    return lines


_MISSING_CONFIG = object()


def _config_value_text(value: Any, *, limit: int = 180) -> str:
    if value is _MISSING_CONFIG:
        return "<missing>"
    text = _stable_json(value)
    if isinstance(value, str):
        text = value
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _method_reference(
    entries: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> dict[str, Any]:
    """Return the deterministic modal *recorded* value for every field.

    Old resolved-config schemas can omit a field.  Missing is provenance, not
    a method value, and therefore must never become the modal reference.
    """

    reference: dict[str, Any] = {}
    for key in keys:
        counts: dict[str, tuple[int, Any]] = {}
        for row in entries:
            config = row.get("method_config")
            value = (
                config.get(key, _MISSING_CONFIG)
                if isinstance(config, Mapping)
                else _MISSING_CONFIG
            )
            if value is _MISSING_CONFIG:
                continue
            encoded = _stable_json(value)
            count, _ = counts.get(encoded, (0, value))
            counts[encoded] = (count + 1, value)
        # Most common is the group reference.  A lexical tie-break makes the
        # report byte-semantics deterministic without pretending it is a
        # hand-selected control experiment.
        if not counts:
            selected = _MISSING_CONFIG
        else:
            _, (_, selected) = min(
                counts.items(), key=lambda item: (-item[1][0], item[0])
            )
        reference[key] = selected
    return reference


def _method_group_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("suite") or ""),
        str(row.get("physical_base_name") or ""),
    )


def _method_configuration_sections(
    rows: Sequence[dict[str, Any]], unmeasured: Sequence[dict[str, Any]]
) -> list[str]:
    """Explain fixed settings and effective changes once per physical group."""

    entries = [row for row in rows if row.get("record_type") == "primary"]
    entries.extend(unmeasured)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in entries:
        # A run with no readable resolved config still belongs in the
        # unmeasured table, but it cannot support a methodology comparison.
        if not isinstance(row.get("method_config"), Mapping) or not row["method_config"]:
            continue
        grouped.setdefault(_method_group_key(row), []).append(row)

    lines = [
        "## 그룹별 방법론 설정",
        "",
        "각 physical `Suite/Base`를 독립 실험 그룹으로 본다. **고정 설정**은 읽기 쉬운",
        "핵심 방법론 항목만 표시한다. 반면 **변경 설정**은 resolved config의 방법론 필드 중",
        "그룹에서 한 번이라도 값이 달라진 항목을 자동으로 전부 추출한다.",
        "실험별 변화는 사람이 임의로 고른 control이 아니라 각 필드의 **기록값 중 그룹 최빈값",
        "(동률이면 JSON lexical order)** 을 reference로 한 effective-config 차이다. 따라서",
        "CLI override가 기본값과 같아 effective 값이 안 바뀐 경우도 구분할 수 있다.",
        "",
    ]
    if not grouped:
        lines.extend(["읽을 수 있는 resolved config가 없습니다.", ""])
        return lines

    for (suite, base), group_rows in sorted(grouped.items()):
        title = "/".join(part for part in (suite, base) if part) or "unknown"
        all_keys = sorted(
            {
                key
                for row in group_rows
                for key in row.get("method_config", {}).keys()
            }
        )
        reference = _method_reference(group_rows, all_keys)
        varied_keys: list[str] = []
        fixed_keys: list[str] = []
        schema_drift_keys: list[str] = []
        for key in all_keys:
            recorded = {
                _stable_json(row["method_config"][key])
                for row in group_rows
                if key in row["method_config"]
            }
            missing_count = sum(key not in row["method_config"] for row in group_rows)
            if len(recorded) > 1:
                varied_keys.append(key)
            elif missing_count:
                schema_drift_keys.append(key)
            else:
                fixed_keys.append(key)

        core_fixed_keys = [key for key in fixed_keys if key in CORE_FIXED_METHOD_KEYS]

        lines.extend(
            [
                f"### {title}",
                "",
                f"- config가 있는 run: {len(group_rows)}개",
                f"- 고정 방법론 설정: {len(fixed_keys)}개 (핵심 {len(core_fixed_keys)}개)",
                f"- 그룹 내 변경 설정: {len(varied_keys)}개",
                f"- schema drift(값은 같고 일부 run에 미기록): {len(schema_drift_keys)}개",
                "",
                "#### 핵심 고정 설정",
                "",
            ]
        )
        if core_fixed_keys:
            lines.extend(["| 설정 | 고정값 |", "|---|---|"])
            for key in core_fixed_keys:
                lines.append(
                    f"| `{_md(key)}` | `{_md(_config_value_text(reference[key]))}` |"
                )
        else:
            lines.append("표시 대상 핵심 고정 설정이 없습니다.")
        lines.extend(
            [
                "",
                "<details>",
                f"<summary>고정 방법론 설정 전체 {len(fixed_keys)}개 펼치기</summary>",
                "",
            ]
        )
        if fixed_keys:
            lines.extend(["| 설정 | 고정값 |", "|---|---|"])
            for key in fixed_keys:
                lines.append(
                    f"| `{_md(key)}` | `{_md(_config_value_text(reference[key]))}` |"
                )
        else:
            lines.append("표시 대상 핵심 고정 설정이 없습니다.")
        lines.extend(["", "</details>", "", "#### 그룹 내 변경 설정", ""])

        if varied_keys:
            lines.extend(
                [
                    "| 설정 | 관측값 × run 수 | 미기록 | modal reference |",
                    "|---|---|---:|---|",
                ]
            )
            for key in varied_keys:
                counts: dict[str, int] = {}
                values: dict[str, Any] = {}
                for row in group_rows:
                    value = row["method_config"].get(key, _MISSING_CONFIG)
                    encoded = (
                        "<missing>" if value is _MISSING_CONFIG else _stable_json(value)
                    )
                    counts[encoded] = counts.get(encoded, 0) + 1
                    values[encoded] = value
                observed = "<br>".join(
                    f"`{_md(_config_value_text(values[encoded], limit=90))}` × {count}"
                    for encoded, count in sorted(
                        counts.items(), key=lambda item: (-item[1], item[0])
                    )
                )
                lines.append(
                    f"| `{_md(key)}` | {observed} | "
                    f"{sum(key not in row['method_config'] for row in group_rows)} | "
                    f"`{_md(_config_value_text(reference[key], limit=90))}` |"
                )
        else:
            lines.append("이 그룹의 resolved method config는 모든 run에서 같습니다.")

        lines.extend(["", "#### Schema drift", ""])
        if schema_drift_keys:
            lines.extend(["| 설정 | 기록값 | 기록 run | 미기록 run |", "|---|---|---:|---:|"])
            for key in schema_drift_keys:
                present = [row for row in group_rows if key in row["method_config"]]
                value = present[0]["method_config"][key]
                lines.append(
                    f"| `{_md(key)}` | `{_md(_config_value_text(value, limit=100))}` | "
                    f"{len(present)} | {len(group_rows) - len(present)} |"
                )
        else:
            lines.append("Schema 차이로 일부 run에만 기록된 설정이 없습니다.")

        lines.extend(
            [
                "",
                "#### 실험별 effective config 변화",
                "",
                "| Case | 설명 | 모델 | Config | Avg-matched 영역 RMSE 평균 | modal 대비 effective 변화 | 명시 CLI override(raw) |",
                "|---|---|---|---|---:|---|---|",
            ]
        )
        ordered = sorted(
            group_rows,
            key=lambda row: (
                _metric_sort_key(row, "average_matched")
                if row.get("record_type") == "primary"
                else (math.inf, math.inf, str(row.get("relative_run_dir")), "")
            ),
        )
        for row in ordered:
            config = row["method_config"]
            differences: list[str] = []
            for key in varied_keys:
                actual = config.get(key, _MISSING_CONFIG)
                if actual is _MISSING_CONFIG:
                    differences.append(f"`{_md(key)}`: `<미기록>`")
                    continue
                if (
                    _stable_json(actual)
                ) == (
                    _stable_json(reference[key])
                ):
                    continue
                differences.append(
                    f"`{_md(key)}`: `{_md(_config_value_text(reference[key], limit=60))}` "
                    f"→ `{_md(_config_value_text(actual, limit=60))}`"
                )
            changes = "<br>".join(differences) if differences else "modal과 동일"
            overrides = row.get("explicit_overrides")
            if isinstance(overrides, list) and overrides:
                override_text = shlex.join(str(value) for value in overrides)
            else:
                override_text = "-"
            values = (
                row.get("case_name"),
                row.get("description"),
                row.get("model_slug"),
                row.get("config_id"),
                _fmt(_metric_value(row, "average_matched", "trait_macro_rmse")),
                changes,
                f"`{_md(override_text)}`" if override_text != "-" else "-",
            )
            lines.append("| " + " | ".join(_md(value) for value in values) + " |")
        lines.append("")
    return lines


def render_markdown(
    rows: Sequence[dict[str, Any]],
    unmeasured: Sequence[dict[str, Any]],
    *,
    results_root: Path,
    scope_dir: Path,
    csv_name: str,
) -> str:
    primary = [row for row in rows if row.get("record_type") == "primary"]
    variants = [row for row in rows if row.get("record_type") == "variant"]
    invalid = [row for row in rows if row.get("analysis_status") != "complete"]
    lines = [
        "# 전체 예측 재측정 결과",
        "",
        f"- 생성 시각: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- 결과 root: `{results_root}`",
        f"- 이 보고서 scope: `{scope_dir}`",
        f"- primary `eval`: {len(primary)}개",
        f"- checkpoint/diagnostic variant: {len(variants)}개",
        f"- primary prediction 없음: {len(unmeasured)}개",
        f"- 재측정 실패: {len(invalid)}개",
        f"- 전체 정밀도 CSV: `{csv_name}`",
        "- **Gold C/O/E와 저장된 `score.average`는 반올림하지 않는다.**",
        "- 세 표면 모두 저장된 raw prediction에서 다시 구성한다. 기존 `metrics.json` 값은",
        "  이 표의 36개 지표 계산에 사용하지 않는다.",
        "- **기본 정렬은 각 표면에서 C/O/E 영역별 RMSE를 각각 구한 뒤 세 값을",
        "  산술평균한 `영역 RMSE 평균` 오름차순**이다.",
        "- `평균 후`는 essay마다 C/O/E 예측 평균을 만든 뒤 저장된 gold",
        "  `score.average`와 전체 RMSE/Spearman을 한 번 계산한다.",
        "- `3N pooled`는 N×3 cell을 펼쳐 한 번 계산한다. 영역 RMSE 평균과 pooled",
        "  RMSE는 서로 다른 수식이다.",
        "- Spearman은 tie에 average rank를 사용한다. 상수 배열은 `-`로 표시한다.",
        "- 서로 다른 input SHA/N/입력 표면 등급은 같은 순위로 섞지 않는다.",
        "  `label_leak`은 수치만 보존하고 순위를 부여하지 않는다.",
        "- 아래 primary 표만 run 순위다. `eval_best_*`, batch1, partial 평가는 같은",
        "  run의 checkpoint/진단 산출물이므로 별도 부록에 둔다.",
        "",
        "## 수식",
        "",
        "`P,Y ∈ R^(N×3)`, 저장된 gold 평균을 `a ∈ R^N`이라 할 때:",
        "",
        "- trait RMSE: `RMSE_t = sqrt(mean_i((P_it - Y_it)^2))`",
        "- 영역 RMSE 평균: `(RMSE_C + RMSE_O + RMSE_E) / 3`",
        "- 평균 후 RMSE: `sqrt(mean_i((mean_t P_it - a_i)^2))`",
        "- pooled RMSE: `sqrt(mean_(i,t)((P_it - Y_it)^2))`",
        "- Spearman도 각각 같은 표면/축에서 tie-aware rank correlation으로 계산",
        "",
        *_method_configuration_sections(primary, unmeasured),
        "## Primary eval 순위",
        "",
        *_metric_sections(primary, primary=True, heading_level=3),
        "## Checkpoint/diagnostic variant 부록",
        "",
        "아래 행은 전수조사에는 포함하지만 primary run 순위에는 포함하지 않는다.",
        "`동일 raw` 여부는 CSV의 `same_raw_as_primary`에서 확인할 수 있다.",
        "",
        "<details>",
        f"<summary>{len(variants)}개 alternate evaluation 전체 지표 펼치기</summary>",
        "",
        *_metric_sections(variants, primary=False, heading_level=3),
        "</details>",
        "",
    ]

    if unmeasured:
        lines.extend(
            [
                "## Primary prediction 미측정 run",
                "",
                "| Suite/Base | Case | 설명 | 모델 | Config | run 상태 | 분석 상태 | 경고 |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for row in unmeasured:
            values = (
                _experiment_label(row),
                row.get("case_name"),
                row.get("description"),
                row.get("model_slug"),
                row.get("config_id"),
                row.get("run_status"),
                row.get("analysis_status"),
                row.get("warnings"),
            )
            lines.append("| " + " | ".join(_md(value) for value in values) + " |")
        lines.append("")

    if invalid:
        lines.extend(
            [
                "## 재측정 오류",
                "",
                "부분 ID만 조용히 계산하지 않고 artifact 전체를 fail-closed 처리했다.",
                "",
                "| Eval | 상태 | 오류 |",
                "|---|---|---|",
            ]
        )
        for row in invalid:
            lines.append(
                "| "
                + " | ".join(
                    _md(value)
                    for value in (
                        row.get("relative_eval_dir"),
                        row.get("analysis_status"),
                        row.get("warnings"),
                    )
                )
                + " |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _csv_rows(
    rows: Sequence[dict[str, Any]], unmeasured: Sequence[dict[str, Any]]
) -> Iterable[dict[str, Any]]:
    for row in rows:
        if row.get("analysis_status") != "complete":
            base = {field: row.get(field, "") for field in CSV_FIELDS}
            base["rank_group"] = row.get("rank_group", "")
            yield base
            continue
        for surface in SURFACES:
            metrics = row["metrics"][surface]
            output = {field: row.get(field, "") for field in CSV_FIELDS}
            output.update(metrics)
            output["surface"] = surface
            output["surface_title"] = SURFACE_TITLES[surface]
            output["rank"] = (
                row.get("ranks", {}).get(surface, "")
                if row.get("record_type") == "primary"
                else ""
            )
            output["rank_group"] = row.get("rank_group", "")
            yield output
    for row in unmeasured:
        output = {field: row.get(field, "") for field in CSV_FIELDS}
        output["rank_group"] = ""
        yield output


def render_csv(
    rows: Sequence[dict[str, Any]], unmeasured: Sequence[dict[str, Any]]
) -> str:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in _csv_rows(rows, unmeasured):
        writer.writerow(row)
    return stream.getvalue()


def _guard_output(path: Path) -> None:
    if path.name in PROTECTED_REPORT_NAMES:
        raise ValueError(
            f"기존 보고서 덮어쓰기 금지: {path}. result_all.md/result_all.csv를 사용하세요."
        )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_scope_report(
    scope_dir: Path,
    all_rows: Sequence[dict[str, Any]],
    all_unmeasured: Sequence[dict[str, Any]],
    *,
    results_root: Path,
    markdown_path: Path | None = None,
    csv_path: Path | None = None,
) -> tuple[Path, Path, int, int]:
    markdown_path = markdown_path or scope_dir / "result_all.md"
    csv_path = csv_path or scope_dir / "result_all.csv"
    markdown_path = markdown_path.resolve()
    csv_path = csv_path.resolve()
    _guard_output(markdown_path)
    _guard_output(csv_path)
    if markdown_path == csv_path:
        raise ValueError("markdown과 csv 출력 경로가 같음")
    # Rank is local to the report scope.  Reuse metric payloads, but copy the
    # outer row so a base-level report cannot overwrite the root report ranks.
    rows = [dict(row) for row in filter_scope(all_rows, scope_dir)]
    assign_ranks(rows)
    rows.sort(key=_row_display_key)
    unmeasured = filter_scope(all_unmeasured, scope_dir)
    markdown = render_markdown(
        rows,
        unmeasured,
        results_root=results_root,
        scope_dir=scope_dir,
        csv_name=csv_path.name,
    )
    csv_text = render_csv(rows, unmeasured)
    _atomic_write(csv_path, csv_text)
    _atomic_write(markdown_path, markdown)
    return markdown_path, csv_path, len(rows), len(unmeasured)


def _explicit_scope(results_root: Path, suite: str | None, base: str | None) -> Path:
    if base and not suite:
        raise ValueError("--base를 쓰려면 --suite도 필요합니다")
    scope = results_root
    if suite:
        scope = scope / suite
    if base:
        scope = scope / base
    if not scope.is_dir():
        raise ValueError(f"scope 디렉터리가 없음: {scope}")
    return scope.resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results_root = Path(args.results_root).resolve()
    if not results_root.is_dir():
        raise SystemExit(f"results root가 없음: {results_root}")
    if args.write_existing_scopes and any(
        (args.suite, args.base, args.markdown, args.csv)
    ):
        raise SystemExit(
            "--write-existing-scopes는 --suite/--base/--markdown/--csv와 같이 쓸 수 없습니다"
        )

    rows, unmeasured = collect_report(
        results_root, include_variants=not args.primary_only
    )
    if args.write_existing_scopes:
        scopes = {results_root}
        scopes.update(path.parent.resolve() for path in results_root.rglob("results.md"))
    else:
        scopes = {_explicit_scope(results_root, args.suite, args.base)}

    outputs: list[tuple[Path, Path, int, int]] = []
    for scope_dir in sorted(scopes):
        if len(scopes) == 1:
            markdown_path = Path(args.markdown).resolve() if args.markdown else None
            csv_path = Path(args.csv).resolve() if args.csv else None
        else:
            markdown_path = None
            csv_path = None
        outputs.append(
            write_scope_report(
                scope_dir,
                rows,
                unmeasured,
                results_root=results_root,
                markdown_path=markdown_path,
                csv_path=csv_path,
            )
        )

    primary_count = sum(row.get("record_type") == "primary" for row in rows)
    variant_count = sum(row.get("record_type") == "variant" for row in rows)
    invalid_count = sum(row.get("analysis_status") != "complete" for row in rows)
    print(
        f"primary={primary_count} variants={variant_count} "
        f"unmeasured={len(unmeasured)} invalid={invalid_count}"
    )
    for markdown_path, csv_path, scoped_rows, scoped_unmeasured in outputs:
        print(
            f"rows={scoped_rows} unmeasured={scoped_unmeasured} "
            f"markdown={markdown_path} csv={csv_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
