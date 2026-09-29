from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping

from . import TRAITS
from .artifacts import read_rows, sha256_text, write_json, write_jsonl
from .config import bind_adapter_prompt, load_config, with_prompt_file
from .metrics import regression_metrics
from .modeling import GeneratorRuntime
from .prompts import build_messages
from .schema import (
    canonicalize_with_fixed_scores,
    conditioning_scores,
    essay_id,
    human_scores,
    official_raw_essay,
    parse_generated_judge,
    prompt_text,
    rationale_audit,
    score_prediction_map,
    validate_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="사전학습 또는 LoRA 모델의 채점 JSON 생성"
    )
    # 2026-08-25 최종 제출본 레시피. 인자를 안 주면 제출본을 그대로 재현한다.
    # 프롬프트 기본값(dataclass)은 baseline으로 두어야 프롬프트를 선언하지 않는
    # legacy recipe가 조용히 v4로 바뀌지 않는다. 그래서 제출본 고정은 여기서 한다.
    parser.add_argument(
        "--recipe",
        default="main_code_relonation/recipes/r17_qwen35_lora_fixed_prompt_v4.json",
        help="기본값은 최종 제출본 근거모델 레시피(v4 prompt, Qwen3.5-9B LoRA r32)",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-predictions")
    parser.add_argument("--adapter-path")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--rationale-prompt-file")
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="recipe의 base 양자화 설정을 명시적으로 재정의합니다",
    )
    return parser.parse_args()


def optional_labels(row: Mapping[str, Any]) -> dict[str, float] | None:
    try:
        return human_scores(row)
    except ValueError:
        return None


def build_submission_record(
    row: Mapping[str, Any],
    raw: str,
    *,
    fixed_scores: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = parse_generated_judge(raw)
    generated_scores = validate_scores(
        {trait: parsed[trait]["score"] for trait in TRAITS}
    )
    judge = (
        canonicalize_with_fixed_scores(parsed, fixed_scores)
        if fixed_scores is not None
        else canonicalize_with_fixed_scores(parsed, generated_scores)
    )
    return {"essay_id": essay_id(row), "judge": judge}, generated_scores


def main() -> None:
    args = parse_args()
    config = load_config(args.recipe)
    prompt_was_explicit = config.rationale_prompt_source != "baseline_fallback"
    if args.rationale_prompt_file:
        config = with_prompt_file(config, args.rationale_prompt_file)
        prompt_was_explicit = True
    if args.load_in_4bit is not None:
        config = config.with_updates(load_in_4bit=args.load_in_4bit)
    if args.adapter_path:
        config = config.with_updates(
            adapter_path=str(Path(args.adapter_path).resolve())
        )
    if config.adapter_path:
        config = bind_adapter_prompt(
            config,
            config.adapter_path,
            prompt_was_explicit=prompt_was_explicit,
        )

    input_path = Path(args.input).resolve()
    rows = read_rows(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("추론 row가 비었습니다")

    fixed: dict[str, dict[str, float]] | None = None
    score_path: Path | None = None
    if config.score_mode == "fixed":
        if not args.score_predictions:
            raise ValueError("fixed score_mode에는 --score-predictions가 필요합니다")
        score_path = Path(args.score_predictions).resolve()
        # 학습 때와 같은 양자화를 거쳐야 프롬프트 표기와 복사 검증이 일치한다.
        fixed = {
            key: conditioning_scores(value)
            for key, value in score_prediction_map(read_rows(score_path)).items()
        }
        missing = [essay_id(row) for row in rows if essay_id(row) not in fixed]
        if missing:
            raise ValueError(f"고정 점수에 없는 essay_id: {missing[:5]}")
    elif args.score_predictions:
        raise ValueError("joint score_mode에는 --score-predictions를 넣지 마세요")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runtime = GeneratorRuntime(config)
    predictions: list[dict[str, Any]] = []
    submission: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    score_mismatch_count = 0
    started_all = time.perf_counter()

    for index, row in enumerate(rows):
        key = essay_id(row)
        fixed_scores = fixed[key] if fixed is not None else None
        messages = build_messages(
            prompt_text(row),
            official_raw_essay(row),
            scores=fixed_scores,
            prompt_template=(
                config.rationale_prompt_text if fixed_scores is not None else None
            ),
            skeleton_hint=config.rationale_skeleton_hint,
        )
        raw_generation = ""
        try:
            generated = runtime.generate(messages, max_new_tokens=args.max_new_tokens)
            raw_generation = generated.raw
            record, generated_scores = build_submission_record(
                row, raw_generation, fixed_scores=fixed_scores
            )
            final_scores = {trait: record["judge"][trait]["score"] for trait in TRAITS}
            if fixed_scores is not None and generated_scores != fixed_scores:
                score_mismatch_count += 1
            audit = rationale_audit(record["judge"], official_raw_essay(row))
            detail = {
                **record,
                "raw_generation": generated.raw,
                "conditioning_scores": fixed_scores,
                "generated_scores": generated_scores,
                "final_scores": final_scores,
                "generated_score_mismatch": (
                    generated_scores != fixed_scores
                    if fixed_scores is not None
                    else False
                ),
                "audit": audit,
                "runtime": {
                    "row_index": index,
                    "prompt_tokens": generated.prompt_tokens,
                    "completion_tokens": generated.completion_tokens,
                    "latency_seconds": generated.latency_seconds,
                },
            }
            predictions.append(detail)
            submission.append(record)
        except Exception as exc:
            failures.append(
                {
                    "essay_id": key,
                    "row_index": index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "raw_generation": raw_generation,
                    "prompt_text_sha256": sha256_text(prompt_text(row)),
                    "essay_sha256": sha256_text(official_raw_essay(row)),
                }
            )

    write_jsonl(output / "predictions.jsonl", predictions)
    write_jsonl(output / "failures.jsonl", failures)
    write_json(output / "submission.json", submission)

    labels: list[list[float]] = []
    predicted: list[list[float]] = []
    row_lookup = {essay_id(row): row for row in rows}
    for record in submission:
        label = optional_labels(row_lookup[record["essay_id"]])
        if label is not None:
            labels.append([label[trait] for trait in TRAITS])
            predicted.append([record["judge"][trait]["score"] for trait in TRAITS])
    metrics = regression_metrics(labels, predicted) if labels else None
    if metrics is not None:
        write_json(output / "metrics.json", metrics)

    fixed_score_parity = True
    if fixed is not None:
        fixed_score_parity = all(
            all(
                record["judge"][trait]["score"] == fixed[record["essay_id"]][trait]
                for trait in TRAITS
            )
            for record in submission
        )
    manifest = {
        "pipeline": "rationale_generation",
        "status": "complete" if not failures else "failed_rows",
        "recipe": str(Path(args.recipe).resolve()),
        "config": config.to_dict(),
        "config_id": config.fingerprint(),
        "chat_template_sha256": sha256_text(str(runtime.tokenizer.chat_template)),
        "input": str(input_path),
        "input_sha256": sha256_text(input_path.read_text(encoding="utf-8")),
        "score_predictions": str(score_path) if score_path else None,
        "score_predictions_sha256": (
            sha256_text(score_path.read_text(encoding="utf-8")) if score_path else None
        ),
        "requested_count": len(rows),
        "success_count": len(submission),
        "failure_count": len(failures),
        "fixed_score_parity": fixed_score_parity,
        "generated_score_mismatch_count": score_mismatch_count,
        "elapsed_seconds": time.perf_counter() - started_all,
        "metrics": metrics,
    }
    write_json(output / "inference_manifest.json", manifest)
    print(
        f"submission={output / 'submission.json'} success={len(submission)} "
        f"failed={len(failures)} score_parity={fixed_score_parity}"
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)}개 생성/파싱 실패. failures.jsonl을 확인하세요"
        )


if __name__ == "__main__":
    main()
