"""Fail-closed artifact barrier and base selector for the quantized stage.

The queue deliberately names every mixed-stage config id.  It never discovers a
"best-looking" directory by globbing, because a stale/partial rerun must not be
allowed to become the next experiment's baseline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


MIXED_GROUP = "c02_mixed_traitrmse_avgrho_s43_v1"
MIXED_CONFIG_IDS: OrderedDict[str, str] = OrderedDict(
    (
        ("m00_c02_control", "fae350b35709"),
        ("m01_seed42", "98d18c816947"),
        ("m02_seed44", "f9b75737b76b"),
        ("m03_avg_mse020", "f146a98102de"),
        ("m04_avg_mse030", "7c18b52838f2"),
        ("m05_avg025_list000", "8d1634e422c2"),
        ("m06_avg025_aux010", "41b76ac84c23"),
        ("m07_avg025_list000_aux010", "0fa3ced6b82b"),
        ("m08_avg025_avg_rank020", "c01993403b6f"),
        ("m09_avg025_avg_rank010", "44c9295f5871"),
        ("m11_avg025_avg_pair005", "f05a2bf8aa1c"),
        ("m12_avg025_avg_pair010", "8e42bd9415ae"),
    )
)
MIXED_GPU0_CASES = (
    "m00_c02_control",
    "m02_seed44",
    "m04_avg_mse030",
    "m06_avg025_aux010",
    "m08_avg025_avg_rank020",
    "m11_avg025_avg_pair005",
)
MIXED_GPU1_CASES = (
    "m01_seed42",
    "m03_avg_mse020",
    "m05_avg025_list000",
    "m07_avg025_list000_aux010",
    "m09_avg025_avg_rank010",
    "m12_avg025_avg_pair010",
)
SEED43_CASES = tuple(
    case for case in MIXED_CONFIG_IDS if case not in {"m01_seed42", "m02_seed44"}
)
EXPECTED_ROWS = 400
SECONDARY_EVAL = "eval_best_official_matched_spearman"
RHO_TOLERANCE = 0.002


class BarrierWaiting(RuntimeError):
    """The named run has not produced its complete contract yet."""


class ContractError(RuntimeError):
    """An artifact exists but violates the immutable queue contract."""


def mixed_group_dir(results_root: Path) -> Path:
    return results_root.resolve() / "new_proposed" / MIXED_GROUP


def expected_run_dir(results_root: Path, case: str) -> Path:
    return (
        mixed_group_dir(results_root)
        / case
        / "ax4_light"
        / MIXED_CONFIG_IDS[case]
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BarrierWaiting(f"artifact 없음: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"JSON 손상: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON object가 아님: {path}")
    return value


def _read_token(path: Path, *, waiting_tokens: Iterable[str] = ()) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise BarrierWaiting(f"artifact 없음: {path}") from exc
    if value in set(waiting_tokens):
        raise BarrierWaiting(f"아직 완료 전: {path}={value}")
    return value


def _read_jsonl_ids(path: Path, expected_rows: int) -> tuple[str, ...]:
    try:
        # ``str.splitlines``는 JSON string 안에 합법적으로 들어갈 수 있는 U+2028도
        # record 경계로 취급한다. JSONL 계약의 물리 LF만 소비하도록 file iterator를 쓴다.
        with path.open("r", encoding="utf-8", newline="\n") as stream:
            lines = [line.removesuffix("\n").removesuffix("\r") for line in stream]
    except FileNotFoundError as exc:
        raise BarrierWaiting(f"artifact 없음: {path}") from exc
    except OSError as exc:
        raise ContractError(f"artifact 읽기 실패: {path}: {exc}") from exc
    if len(lines) != expected_rows or any(not line.strip() for line in lines):
        raise ContractError(
            f"JSONL row 수 불일치: {path}: {len(lines)} != {expected_rows}"
        )
    identities: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"JSONL 손상: {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ContractError(f"JSONL object가 아님: {path}:{line_number}")
        identity = value.get("essay_id")
        if not isinstance(identity, str) or not identity:
            raise ContractError(f"essay_id 누락: {path}:{line_number}")
        identities.append(identity)
    if len(set(identities)) != expected_rows:
        raise ContractError(f"essay_id 중복: {path}")
    return tuple(identities)


def _validate_eval(eval_dir: Path, expected_rows: int) -> tuple[str, ...]:
    metrics = _read_json(eval_dir / "metrics.json")
    if metrics.get("count") != expected_rows:
        raise ContractError(
            f"metrics count 불일치: {eval_dir}: {metrics.get('count')!r}"
        )
    official = metrics.get("official")
    if not isinstance(official, dict) or official.get("gold_source") != "score_average":
        raise ContractError(f"official score.average 계약 불일치: {eval_dir}")
    score_ids = _read_jsonl_ids(eval_dir / "score_predictions.jsonl", expected_rows)
    prediction_ids = _read_jsonl_ids(eval_dir / "predictions.jsonl", expected_rows)
    if set(score_ids) != set(prediction_ids):
        raise ContractError(f"두 prediction stream의 essay_id 집합 불일치: {eval_dir}")
    return score_ids


def validate_case(
    results_root: Path, case: str, *, expected_rows: int = EXPECTED_ROWS
) -> Path:
    if case not in MIXED_CONFIG_IDS:
        raise ContractError(f"알 수 없는 mixed case: {case}")
    run_dir = expected_run_dir(results_root, case)
    model_dir = run_dir.parent
    sibling_leaf_names: list[str] = []
    if model_dir.is_dir():
        try:
            sibling_leaf_names = sorted(
                item.name
                for item in model_dir.iterdir()
                if item.is_dir() and not item.name.startswith(".")
            )
        except OSError as exc:
            raise ContractError(
                f"config leaf 목록 읽기 실패: {model_dir}: {exc}"
            ) from exc
    if sibling_leaf_names and sibling_leaf_names != [MIXED_CONFIG_IDS[case]]:
        raise ContractError(
            f"exact config 외 sibling leaf 존재: {model_dir}: {sibling_leaf_names}"
        )
    if not run_dir.is_dir():
        raise BarrierWaiting(f"exact config leaf 없음: {run_dir}")

    experiment = _read_json(run_dir / "experiment.json")
    expected_experiment = {
        "suite": "new_proposed",
        "base_name": MIXED_GROUP,
        "case_name": case,
        "config_id": MIXED_CONFIG_IDS[case],
        "model_argument": "ax4_light",
    }
    mismatched = {
        key: (experiment.get(key), expected)
        for key, expected in expected_experiment.items()
        if experiment.get(key) != expected
    }
    if mismatched:
        raise ContractError(f"experiment identity 불일치: {run_dir}: {mismatched}")

    status = _read_token(
        run_dir / "status.txt", waiting_tokens=("running", "trained")
    )
    if status != "complete":
        raise ContractError(f"실패/미지원 status: {run_dir}: {status!r}")

    # A producer has declared this leaf complete.  From this point onward a
    # missing member is corruption, not something the coordinator may wait for.
    required_after_complete = (
        "train_exit_code.txt",
        "infer_exit_code.txt",
        "secondary_infer_exit_code.txt",
        "status_context.json",
        "resolved_config.json",
        "eval/metrics.json",
        "eval/score_predictions.jsonl",
        "eval/predictions.jsonl",
        f"{SECONDARY_EVAL}/metrics.json",
        f"{SECONDARY_EVAL}/score_predictions.jsonl",
        f"{SECONDARY_EVAL}/predictions.jsonl",
    )
    missing_after_complete = [
        relative for relative in required_after_complete if not (run_dir / relative).is_file()
    ]
    if missing_after_complete:
        raise ContractError(
            f"complete leaf의 필수 artifact 누락: {run_dir}: {missing_after_complete}"
        )
    for name in (
        "train_exit_code.txt",
        "infer_exit_code.txt",
        "secondary_infer_exit_code.txt",
    ):
        value = _read_token(run_dir / name)
        if value != "0":
            raise ContractError(f"nonzero exit code: {run_dir / name}={value!r}")

    context = _read_json(run_dir / "status_context.json")
    if (
        context.get("status") != "complete"
        or context.get("phase") != "complete"
        or context.get("exit_code") != 0
    ):
        raise ContractError(f"status_context 완료 계약 불일치: {run_dir}")

    config = _read_json(run_dir / "resolved_config.json")
    expected_seed = 42 if case == "m01_seed42" else 44 if case == "m02_seed44" else 43
    required_config = {
        "model_slug": "ax4_light",
        "model_id": "skt/A.X-4.0-Light",
        "training_mode": "lora_only",
        "primary_data_profile": "full",
        "dataset_schedule": "competition_only",
        "extended_datasets": "",
        "essay_surface": "official_raw",
        "score_postprocess": "average_matched",
        "best_checkpoint_metric": "submitted_trait_macro_rmse",
        "seed": expected_seed,
    }
    config_mismatch = {
        key: (config.get(key), expected)
        for key, expected in required_config.items()
        if config.get(key) != expected
    }
    if config_mismatch:
        raise ContractError(f"resolved config 계약 불일치: {run_dir}: {config_mismatch}")

    primary_ids = _validate_eval(run_dir / "eval", expected_rows)
    secondary_ids = _validate_eval(run_dir / SECONDARY_EVAL, expected_rows)
    if set(primary_ids) != set(secondary_ids):
        raise ContractError(f"primary/secondary essay_id 집합 불일치: {run_dir}")
    return run_dir


def cases_for_lane(lane: str) -> tuple[str, ...]:
    if lane == "all":
        return tuple(MIXED_CONFIG_IDS)
    if lane == "gpu0":
        return MIXED_GPU0_CASES
    if lane == "gpu1":
        return MIXED_GPU1_CASES
    raise ContractError(f"지원하지 않는 lane: {lane}")


def validate_barrier(
    results_root: Path, *, lane: str = "all", expected_rows: int = EXPECTED_ROWS
) -> tuple[list[Path], list[str]]:
    complete: list[Path] = []
    waiting: list[str] = []
    for case in cases_for_lane(lane):
        try:
            complete.append(
                validate_case(results_root, case, expected_rows=expected_rows)
            )
        except BarrierWaiting as exc:
            waiting.append(f"{case}: {exc}")
    return complete, waiting


def _finite(value: str, field: str, case: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{case}: {field}가 숫자가 아님: {value!r}") from exc
    if not math.isfinite(number):
        raise ContractError(f"{case}: {field}가 finite가 아님")
    return number


def select_seed43_base(
    results_root: Path,
    csv_path: Path,
    *,
    rho_tolerance: float = RHO_TOLERANCE,
) -> dict[str, Any]:
    """Select minimum matched trait-RMSE within best-rho minus tolerance."""

    if rho_tolerance < 0 or not math.isfinite(rho_tolerance):
        raise ContractError("rho tolerance는 finite non-negative여야 합니다")
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            report_rows = list(csv.DictReader(stream))
    except FileNotFoundError as exc:
        raise ContractError(f"result_all.csv 없음: {csv_path}") from exc
    except OSError as exc:
        raise ContractError(f"result_all.csv 읽기 실패: {csv_path}: {exc}") from exc

    selected_rows: dict[str, dict[str, str]] = {}
    for row in report_rows:
        case = row.get("case_name", "")
        if case not in MIXED_CONFIG_IDS:
            continue
        if (
            row.get("record_type") == "primary"
            and row.get("surface") == "average_matched"
            and row.get("analysis_status") == "complete"
            and row.get("surface_class") == "contract"
            and row.get("evaluation_name") == "eval"
            and row.get("config_id") == MIXED_CONFIG_IDS[case]
        ):
            if case in selected_rows:
                raise ContractError(f"result_all.csv exact primary row 중복: {case}")
            selected_rows[case] = row

    missing = sorted(set(MIXED_CONFIG_IDS) - set(selected_rows))
    if missing:
        raise ContractError(f"result_all.csv exact primary row 누락: {missing}")
    for case, row in selected_rows.items():
        if row.get("count") != str(EXPECTED_ROWS):
            raise ContractError(f"{case}: result_all count={row.get('count')!r}")

    candidates: list[dict[str, Any]] = []
    for case in SEED43_CASES:
        row = selected_rows[case]
        run_dir = validate_case(results_root, case)
        candidates.append(
            {
                "case_name": case,
                "config_id": MIXED_CONFIG_IDS[case],
                "rmse": _finite(row.get("trait_macro_rmse", ""), "trait_macro_rmse", case),
                "rho": _finite(row.get("mean_first_spearman", ""), "mean_first_spearman", case),
                "run_dir": str(run_dir.resolve()),
                "resolved_config": str((run_dir / "resolved_config.json").resolve()),
            }
        )
    best_rho = max(item["rho"] for item in candidates)
    rho_floor = best_rho - rho_tolerance
    eligible = [item for item in candidates if item["rho"] >= rho_floor]
    winner = min(eligible, key=lambda item: (item["rmse"], -item["rho"], item["case_name"]))
    return {
        "schema_version": 1,
        "policy": {
            "seed": 43,
            "prediction_surface": "average_matched",
            "primary_metric": "trait_macro_rmse",
            "primary_direction": "min",
            "constraint_metric": "mean_first_spearman",
            "constraint": "rho >= best_seed43_rho - tolerance",
            "rho_tolerance": rho_tolerance,
            "best_seed43_rho": best_rho,
            "rho_floor": rho_floor,
        },
        "selected": winner,
        "eligible": sorted(eligible, key=lambda item: (item["rmse"], -item["rho"])),
        "all_seed43_candidates": candidates,
        "source_report": str(csv_path.resolve()),
        "source_report_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def verify_selected_base(results_root: Path, config_path: Path) -> Path:
    resolved = config_path.resolve(strict=True)
    allowed = {
        (expected_run_dir(results_root, case) / "resolved_config.json").resolve()
        for case in SEED43_CASES
    }
    if resolved not in allowed:
        raise ContractError(f"선택 config가 exact seed43 mixed leaf가 아님: {resolved}")
    case = next(
        case
        for case in SEED43_CASES
        if (expected_run_dir(results_root, case) / "resolved_config.json").resolve()
        == resolved
    )
    validate_case(results_root, case)
    return resolved


def verify_selection_manifest(
    results_root: Path, manifest_path: Path, config_path: Path
) -> dict[str, Any]:
    """Recompute the recorded policy and reject a stale/tampered selection."""

    manifest = _read_json(manifest_path.resolve())
    policy = manifest.get("policy")
    selected = manifest.get("selected")
    if not isinstance(policy, dict) or not isinstance(selected, dict):
        raise ContractError(f"selection manifest schema 불일치: {manifest_path}")
    tolerance = policy.get("rho_tolerance")
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        raise ContractError("selection manifest rho_tolerance가 숫자가 아님")
    if not math.isclose(float(tolerance), RHO_TOLERANCE, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError(
            f"selection policy tolerance 불일치: {tolerance!r} != {RHO_TOLERANCE}"
        )
    report_value = manifest.get("source_report")
    if not isinstance(report_value, str) or not report_value:
        raise ContractError("selection manifest source_report 누락")
    report = Path(report_value).resolve(strict=True)
    recorded_digest = manifest.get("source_report_sha256")
    actual_digest = hashlib.sha256(report.read_bytes()).hexdigest()
    if recorded_digest != actual_digest:
        raise ContractError("selection source report가 manifest 생성 뒤 변경됨")
    recomputed = select_seed43_base(
        results_root, report, rho_tolerance=float(tolerance)
    )
    recorded_key = (
        selected.get("case_name"),
        selected.get("config_id"),
        selected.get("resolved_config"),
        selected.get("rmse"),
        selected.get("rho"),
    )
    recomputed_selected = recomputed["selected"]
    recomputed_key = (
        recomputed_selected.get("case_name"),
        recomputed_selected.get("config_id"),
        recomputed_selected.get("resolved_config"),
        recomputed_selected.get("rmse"),
        recomputed_selected.get("rho"),
    )
    if recorded_key != recomputed_key:
        raise ContractError(
            f"selection manifest winner가 재계산 결과와 다름: "
            f"recorded={recorded_key}, recomputed={recomputed_key}"
        )
    resolved = verify_selected_base(results_root, config_path)
    if Path(str(selected.get("resolved_config"))).resolve() != resolved:
        raise ContractError("selection manifest와 QUANT_BASE_CONFIG가 다름")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("barrier", "select", "verify-base", "verify-selection")
    )
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--lane", choices=("all", "gpu0", "gpu1"), default="all")
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    parser.add_argument("--csv")
    parser.add_argument("--output")
    parser.add_argument("--config")
    parser.add_argument("--manifest")
    parser.add_argument("--rho-tolerance", type=float, default=RHO_TOLERANCE)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results_root = Path(args.results_root)
    try:
        if args.mode == "barrier":
            complete, waiting = validate_barrier(
                results_root, lane=args.lane, expected_rows=args.expected_rows
            )
            if not args.quiet:
                print(
                    f"BARRIER lane={args.lane} complete={len(complete)}/"
                    f"{len(cases_for_lane(args.lane))} waiting={len(waiting)}"
                )
                for item in waiting:
                    print(f"WAITING {item}")
            return 3 if waiting else 0
        if args.mode == "verify-base":
            if not args.config:
                raise ContractError("verify-base에는 --config가 필요합니다")
            selected = verify_selected_base(results_root, Path(args.config))
            print(selected)
            return 0
        if args.mode == "verify-selection":
            if not args.config or not args.manifest:
                raise ContractError(
                    "verify-selection에는 --config와 --manifest가 필요합니다"
                )
            manifest = verify_selection_manifest(
                results_root, Path(args.manifest), Path(args.config)
            )
            print(manifest["selected"]["resolved_config"])
            return 0
        if not args.csv or not args.output:
            raise ContractError("select에는 --csv와 --output이 필요합니다")
        complete, waiting = validate_barrier(results_root)
        if waiting:
            raise BarrierWaiting(f"mixed barrier incomplete: {len(complete)}/12")
        selection = select_seed43_base(
            results_root,
            Path(args.csv),
            rho_tolerance=args.rho_tolerance,
        )
        atomic_write_json(Path(args.output), selection)
        print(selection["selected"]["resolved_config"])
        return 0
    except BarrierWaiting as exc:
        if not args.quiet:
            print(f"WAITING: {exc}")
        return 3
    except (ContractError, OSError, ValueError) as exc:
        print(f"CONTRACT ERROR: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
