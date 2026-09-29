"""Small deterministic utilities for the Qwen-vs-Gemma rationale teacher A/B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from . import TRAIN_SOURCE_SPLITS, TRAITS
from .artifacts import read_rows, sha256_text, write_json, write_jsonl
from .schema import essay_id, official_raw_essay, prompt_text, validate_scores


def _unique_by_id(
    rows: list[dict[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = essay_id(row)
        if key in result:
            raise ValueError(f"{label}: 중복 essay_id {key}")
        result[key] = row
    return result


def _require_accepted_training_row(row: Mapping[str, Any], *, label: str) -> None:
    meta = row.get("pseudo_meta")
    qc = meta.get("qc") if isinstance(meta, Mapping) else None
    proxy = meta.get("proxy_judge") if isinstance(meta, Mapping) else None
    proxy_filter = proxy.get("filter") if isinstance(proxy, Mapping) else None
    if not (
        isinstance(meta, Mapping)
        and meta.get("status") == "accepted"
        and meta.get("parse_ok") is True
        and meta.get("teacher_score_copy_exact") is True
        and meta.get("input_source_split") in TRAIN_SOURCE_SPLITS
        and isinstance(qc, Mapping)
        and qc.get("pass") is True
        and isinstance(proxy_filter, Mapping)
        and proxy_filter.get("accepted") is True
    ):
        raise ValueError(
            f"{label}: proxy Judge까지 통과한 accepted train row가 아닙니다: "
            f"{essay_id(row)}"
        )


def align_training_rows(
    *,
    source_path: str | Path,
    qwen_path: str | Path,
    gemma_path: str | Path,
    qwen_output: str | Path,
    gemma_output: str | Path,
    manifest_path: str | Path,
    limit: int | None = None,
) -> dict[str, Any]:
    """Write paired training files on the exact accepted-ID intersection."""

    source = read_rows(source_path)
    qwen = _unique_by_id(read_rows(qwen_path), label="qwen")
    gemma = _unique_by_id(read_rows(gemma_path), label="gemma")
    source_order = [essay_id(row) for row in source]
    common = [key for key in source_order if key in qwen and key in gemma]
    if limit is not None:
        if limit < 1:
            raise ValueError("align limit은 양수여야 합니다")
        common = common[:limit]
    if not common:
        raise ValueError("Qwen/Gemma accepted ID 교집합이 비었습니다")

    qwen_rows = [qwen[key] for key in common]
    gemma_rows = [gemma[key] for key in common]
    for q_row, g_row in zip(qwen_rows, gemma_rows, strict=True):
        _require_accepted_training_row(q_row, label="qwen")
        _require_accepted_training_row(g_row, label="gemma")
        q_meta = q_row.get("pseudo_meta")
        g_meta = g_row.get("pseudo_meta")
        if not isinstance(q_meta, Mapping) or not isinstance(g_meta, Mapping):
            raise ValueError("paired pseudo row에 pseudo_meta가 없습니다")
        q_hash = q_meta.get("rationale_prompt_sha256")
        g_hash = g_meta.get("rationale_prompt_sha256")
        if not isinstance(q_hash, str) or q_hash != g_hash:
            raise ValueError(
                f"paired prompt hash 불일치: {essay_id(q_row)} {q_hash} != {g_hash}"
            )
        q_score_contract = (
            q_meta.get("score_source"),
            q_meta.get("conditioning_score_postprocess"),
        )
        g_score_contract = (
            g_meta.get("score_source"),
            g_meta.get("conditioning_score_postprocess"),
        )
        if q_score_contract != g_score_contract or not all(q_score_contract):
            raise ValueError(
                "paired conditioning score 계약 불일치: "
                f"{essay_id(q_row)} {q_score_contract} != {g_score_contract}"
            )
        q_score_values = (
            validate_scores(q_row.get("conditioning_scores", {})),
            validate_scores(q_meta.get("canonical_fixed_scores", {})),
            validate_scores(q_meta.get("score_source_values", {})),
        )
        g_score_values = (
            validate_scores(g_row.get("conditioning_scores", {})),
            validate_scores(g_meta.get("canonical_fixed_scores", {})),
            validate_scores(g_meta.get("score_source_values", {})),
        )
        if q_score_values != g_score_values:
            raise ValueError(
                "paired conditioning score 값 불일치: "
                f"{essay_id(q_row)} {q_score_values} != {g_score_values}"
            )
        if q_score_values[0] != q_score_values[1]:
            raise ValueError(
                "conditioning_scores와 canonical_fixed_scores가 다릅니다: "
                f"{essay_id(q_row)}"
            )

    q_target = Path(qwen_output)
    g_target = Path(gemma_output)
    write_jsonl(q_target, qwen_rows)
    write_jsonl(g_target, gemma_rows)
    manifest = {
        "schema_version": 1,
        "stage": "rationale_teacher_ab_align",
        "source": str(Path(source_path).resolve()),
        "qwen_input": str(Path(qwen_path).resolve()),
        "gemma_input": str(Path(gemma_path).resolve()),
        "qwen_accepted": len(qwen),
        "gemma_accepted": len(gemma),
        "intersection_count": len(common),
        "essay_ids_sha256": sha256_text("\n".join(common)),
        "rationale_prompt_sha256": qwen_rows[0]["pseudo_meta"][
            "rationale_prompt_sha256"
        ],
        "score_source": qwen_rows[0]["pseudo_meta"]["score_source"],
        "conditioning_score_postprocess": qwen_rows[0]["pseudo_meta"][
            "conditioning_score_postprocess"
        ],
        "qwen_output": str(q_target.resolve()),
        "gemma_output": str(g_target.resolve()),
    }
    write_json(manifest_path, manifest)
    return manifest


def make_submitted_score_file(
    *, source_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Convert score-model JSONL to Docker-equivalent integer conditioning scores."""

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in read_rows(source_path):
        key = essay_id(row)
        if key in seen:
            raise ValueError(f"중복 score prediction ID: {key}")
        seen.add(key)
        scores = validate_scores(row.get("submitted_scores", {}))
        if any(float(scores[trait]).is_integer() is False for trait in TRAITS):
            raise ValueError(f"submitted score가 정수가 아닙니다: {key}")
        output.append(
            {
                "essay_id": key,
                "scores": {trait: int(scores[trait]) for trait in TRAITS},
            }
        )
    if not output:
        raise ValueError("score prediction이 비었습니다")
    target = Path(output_path)
    write_jsonl(target, output)
    return {
        "count": len(output),
        "source": str(Path(source_path).resolve()),
        "output": str(target.resolve()),
        "essay_ids_sha256": sha256_text("\n".join(row["essay_id"] for row in output)),
    }


def build_evaluation_rows(
    *,
    validation_path: str | Path,
    inference_path: str | Path,
    output_path: str | Path,
    arm: str,
) -> dict[str, Any]:
    """Join student predictions to source text for evaluation-only proxy Judge input."""

    validation_rows = read_rows(validation_path)
    validation = _unique_by_id(validation_rows, label="validation")
    predictions = _unique_by_id(read_rows(inference_path), label="inference")
    validation_order = [essay_id(row) for row in validation_rows]
    if set(predictions) != set(validation):
        missing = sorted(set(validation) - set(predictions))[:5]
        extra = sorted(set(predictions) - set(validation))[:5]
        raise ValueError(
            f"validation/inference ID 불일치: missing={missing} extra={extra}"
        )
    output: list[dict[str, Any]] = []
    for key in validation_order:
        source = validation[key]
        predicted = predictions[key]
        judge = predicted.get("judge")
        if not isinstance(judge, Mapping):
            raise ValueError(f"student inference judge가 없습니다: {key}")
        scores = validate_scores(
            {
                trait: judge.get(trait, {}).get("score")
                if isinstance(judge.get(trait), Mapping)
                else None
                for trait in TRAITS
            }
        )
        output.append(
            {
                "essay_id": key,
                "prompt_text": prompt_text(source),
                "essay_text": official_raw_essay(source),
                "source_split": "evaluation",
                "conditioning_scores": scores,
                "judge": dict(judge),
                "pseudo_meta": {
                    "status": "evaluation",
                    "input_source_split": "official_validation",
                    "student_arm": arm,
                },
            }
        )
    target = Path(output_path)
    write_jsonl(target, output)
    return {"arm": arm, "count": len(output), "output": str(target.resolve())}


def compare_proxy_judges(
    *,
    qwen_path: str | Path,
    gemma_path: str | Path,
    output_path: str | Path,
    baseline_path: str | Path | None = None,
) -> dict[str, Any]:
    qwen = _unique_by_id(read_rows(qwen_path), label="qwen judge")
    gemma = _unique_by_id(read_rows(gemma_path), label="gemma judge")
    if set(qwen) != set(gemma):
        raise ValueError("paired proxy Judge ID set이 다릅니다")
    ids = sorted(qwen)
    q_values = [float(qwen[key]["statistics"]["overall_mean"]) for key in ids]
    g_values = [float(gemma[key]["statistics"]["overall_mean"]) for key in ids]
    deltas = [g - q for q, g in zip(q_values, g_values, strict=True)]
    result = {
        "interpretation": "local paired proxy; official LLM Judge score가 아님",
        "count": len(ids),
        "qwen_mean": mean(q_values),
        "gemma_mean": mean(g_values),
        "gemma_minus_qwen_mean": mean(deltas),
        "gemma_win_tie_loss": {
            "win": sum(delta > 0 for delta in deltas),
            "tie": sum(delta == 0 for delta in deltas),
            "loss": sum(delta < 0 for delta in deltas),
        },
        "essay_ids_sha256": sha256_text("\n".join(ids)),
    }
    if baseline_path is not None:
        baseline = _unique_by_id(read_rows(baseline_path), label="baseline judge")
        if set(baseline) != set(qwen):
            raise ValueError("baseline/new proxy Judge ID set이 다릅니다")
        baseline_values = [
            float(baseline[key]["statistics"]["overall_mean"]) for key in ids
        ]
        result["existing_baseline_mean"] = mean(baseline_values)
        result["qwen_v1_minus_existing_baseline"] = mean(
            qwen_value - baseline_value
            for qwen_value, baseline_value in zip(
                q_values, baseline_values, strict=True
            )
        )
        result["gemma_v1_minus_existing_baseline"] = mean(
            gemma_value - baseline_value
            for gemma_value, baseline_value in zip(
                g_values, baseline_values, strict=True
            )
        )
    write_json(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rationale teacher A/B utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    align = sub.add_parser("align")
    align.add_argument("--source", required=True)
    align.add_argument("--qwen", required=True)
    align.add_argument("--gemma", required=True)
    align.add_argument("--qwen-output", required=True)
    align.add_argument("--gemma-output", required=True)
    align.add_argument("--manifest", required=True)
    align.add_argument("--limit", type=int)

    scores = sub.add_parser("make-score-file")
    scores.add_argument("--source", required=True)
    scores.add_argument("--output", required=True)

    evaluation = sub.add_parser("build-eval")
    evaluation.add_argument("--validation", required=True)
    evaluation.add_argument("--inference", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--arm", required=True)

    compare = sub.add_parser("compare")
    compare.add_argument("--qwen", required=True)
    compare.add_argument("--gemma", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "align":
        result = align_training_rows(
            source_path=args.source,
            qwen_path=args.qwen,
            gemma_path=args.gemma,
            qwen_output=args.qwen_output,
            gemma_output=args.gemma_output,
            manifest_path=args.manifest,
            limit=args.limit,
        )
    elif args.command == "make-score-file":
        result = make_submitted_score_file(
            source_path=args.source, output_path=args.output
        )
    elif args.command == "build-eval":
        result = build_evaluation_rows(
            validation_path=args.validation,
            inference_path=args.inference,
            output_path=args.output,
            arm=args.arm,
        )
    else:
        result = compare_proxy_judges(
            qwen_path=args.qwen,
            gemma_path=args.gemma,
            output_path=args.output,
            baseline_path=args.baseline,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
