from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from . import TRAIN_SOURCE_SPLITS, TRAITS
from .api import (
    NONTHINKING_TEMPLATE_KWARGS,
    chat_completion,
    completion_parts,
    local_base_url,
    resolve_model_id,
    response_audit,
    sampling_for_attempt,
)
from .artifacts import (
    append_jsonl,
    read_jsonl,
    sha256_text,
    write_json,
    write_jsonl,
)
from .schema import (
    _extract_first_json_official,
    essay_id,
    official_raw_essay,
    prompt_text,
)


JUDGE_DIMENSIONS = (
    "domain_match",
    "score_rationale_consistency",
    "specificity",
    "groundedness",
)

PROXY_JUDGE_VERSION = "task_spec_260708_3trait_x_4_proxy_v1"
OFFICIAL_JUDGE_REFERENCE = "Qwen3.6-35B-A3B Q4_K_M GGUF"

JUDGE_INSTRUCTION = """[역할]
당신은 한국어 논증적 글 채점 결과의 타당성을 검토하는 매우 엄격하고 일관된 심사자이다.
점수의 정답 여부를 새로 채점하지 않는다. predicted_score를 전제로 rationale이 해당 영역에 맞고,
점수와 정합적이며, 구체적이고, essay_text에 충실한지를 평가한다. 애매하거나 근거가 부족하면 감점하라.

[중요 원칙]
- 1~5 전 구간을 사용하고 기본 점수는 3점으로 둔다. 명확한 강점이 입증될 때만 4~5점을 준다.
- generic한 총평과 템플릿형 설명은 낮게 평가한다.
- essay_text에 없는 내용을 만들면 groundedness는 1~2점이다.
- 다른 영역 기준을 섞으면 domain_match는 1~2점이다.
- 높은 predicted_score에 비해 근거가 약하거나, 낮은 점수인데 실제 결함을 입증하지 못하면
  score_rationale_consistency를 낮게 평가한다.
- rationale 길이가 아니라 실제 문장, 표현, 논지, 전개 또는 오류 양상을 짚는지를 본다.

[평가 항목]
1. domain_match: rationale이 해당 영역의 평가 기준에 맞는가
2. score_rationale_consistency: predicted_score와 rationale이 부합하는가
3. specificity: 실제 글의 특정 문장, 표현, 논지, 문단 전개, 오류 양상을 구체적으로 짚는가
4. groundedness: rationale이 실제 essay_text에 근거하고 없는 내용을 만들지 않는가

[영역별 기준]
- content: 문제 대응, 근거의 충분성·구체성, 주장과 근거의 논리적 연결
- organization: 서론·본론·결론 구조, 문단 간 연결, 논리 전개 순서
- expression: 문장의 자연스러움, 어휘, 맞춤법·띄어쓰기·문법·주술 호응

[공통 점수 기준]
5: 매우 구체적이고 정확하며 영역 기준과 essay_text에 긴밀히 연결되어 점수를 설득력 있게 정당화함
4: 전반적으로 타당하고 충분히 구체적이며 essay_text와 연결이 분명함
3: 기본 타당성은 있으나 구체성, 명확성, 근거 연결 중 하나 이상이 부족함
2: 영역 혼동, 근거 부족, 일반론 또는 essay_text 연결 부족이 뚜렷함
1: 명백한 영역 혼동, 환각, 근거 부재 또는 점수-설명 모순이 있음

[항목별 세부 기준]
domain_match: 5=해당 영역만 정확히 사용, 4=대부분 정확, 3=일부 혼합, 2=상당히 혼합, 1=거의 다른 영역
score_rationale_consistency: 5=매우 잘 부합, 4=대체로 부합, 3=다소 애매, 2=근거가 약함, 1=명확히 모순
specificity: 5=특정 요소를 분명히 지목, 4=비교적 구체적, 3=최소 근거, 2=상당히 추상적, 1=템플릿 총평
groundedness: 5=본문에 명확히 근거, 4=대체로 근거, 3=일부 추정, 2=확인 어려움, 1=없는 내용을 말함

[출력 규칙]
- JSON 객체 하나만 출력하고 코드블록 마크다운을 사용하지 마라.
- content, organization, expression을 모두 포함하라.
- 각 영역에 domain_match, score_rationale_consistency, specificity, groundedness를 포함하라.
- 각 항목은 {\"evidence\":\"판정 근거\",\"score\":1~5 정수} 형식이다.
- **각 evidence는 60자 이내 한 문장이다.** 12개 항목을 모두 써야 하므로, 길게 쓰면 응답이
  중간에서 잘려 통째로 폐기된다. 판정의 결정적 이유 하나만 짧게 적어라.

[출력 형식]
{"content":{"domain_match":{"evidence":"","score":1},"score_rationale_consistency":{"evidence":"","score":1},"specificity":{"evidence":"","score":1},"groundedness":{"evidence":"","score":1}},"organization":{"domain_match":{"evidence":"","score":1},"score_rationale_consistency":{"evidence":"","score":1},"specificity":{"evidence":"","score":1},"groundedness":{"evidence":"","score":1}},"expression":{"domain_match":{"evidence":"","score":1},"score_rationale_consistency":{"evidence":"","score":1},"specificity":{"evidence":"","score":1},"groundedness":{"evidence":"","score":1}}}

이제 아래 정보를 바탕으로 세 영역 모두를 평가하라."""


def judge_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    payload = {
        "prompt_text": prompt_text(row),
        "essay_text": official_raw_essay(row),
        "prediction": row["judge"],
    }
    return [
        {
            "role": "user",
            "content": (
                f"{JUDGE_INSTRUCTION}\n\n"
                + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            ),
        }
    ]


def parse_proxy_judge(raw: str) -> dict[str, dict[str, dict[str, Any]]]:
    cleaned = re.sub(r"```(?:json)?", "", raw.strip()).replace("```", "").strip()
    extracted = _extract_first_json_official(cleaned)
    if extracted is None:
        raise ValueError("첫 balanced JSON 객체가 없습니다")
    try:
        value = json.loads(extracted)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge JSON을 파싱할 수 없습니다: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Judge 최상위 값은 객체여야 합니다")

    result: dict[str, dict[str, dict[str, Any]]] = {}
    for trait in TRAITS:
        source = value.get(trait)
        if not isinstance(source, dict):
            raise ValueError(f"Judge 결과에 {trait}이 없습니다")
        result[trait] = {}
        for dimension in JUDGE_DIMENSIONS:
            item = source.get(dimension)
            if not isinstance(item, dict):
                raise ValueError(f"Judge 결과에 {trait}.{dimension}이 없습니다")
            score = item.get("score")
            if (
                isinstance(score, bool)
                or not isinstance(score, int)
                or not 1 <= score <= 5
            ):
                raise ValueError(f"{trait}.{dimension}.score는 1~5 정수여야 합니다")
            evidence = item.get("evidence")
            if not isinstance(evidence, str) or not evidence.strip():
                raise ValueError(f"{trait}.{dimension}.evidence가 비었습니다")
            result[trait][dimension] = {
                "evidence": evidence.strip(),
                "score": score,
            }
    return result


def judge_statistics(
    result: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    by_trait = {
        trait: mean(
            float(result[trait][dimension]["score"]) for dimension in JUDGE_DIMENSIONS
        )
        for trait in TRAITS
    }
    by_dimension = {
        dimension: mean(float(result[trait][dimension]["score"]) for trait in TRAITS)
        for dimension in JUDGE_DIMENSIONS
    }
    all_scores = [
        float(result[trait][dimension]["score"])
        for trait in TRAITS
        for dimension in JUDGE_DIMENSIONS
    ]
    return {
        "overall_mean": mean(all_scores),
        "minimum": min(all_scores),
        "by_trait": by_trait,
        "by_dimension": by_dimension,
    }


def passes_filter(
    result: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    minimum_mean: float,
    minimum_scores: Mapping[str, int],
) -> tuple[bool, list[str]]:
    statistics = judge_statistics(result)
    reasons: list[str] = []
    if statistics["overall_mean"] < minimum_mean:
        reasons.append(
            f"overall_mean<{minimum_mean:g}: {statistics['overall_mean']:.4f}"
        )
    for dimension in JUDGE_DIMENSIONS:
        threshold = minimum_scores[dimension]
        lowest = min(int(result[trait][dimension]["score"]) for trait in TRAITS)
        if lowest < threshold:
            reasons.append(f"{dimension}_minimum<{threshold}: {lowest}")
    return not reasons, reasons


def _input_source_split(row: Mapping[str, Any]) -> str:
    meta = row.get("pseudo_meta")
    if isinstance(meta, Mapping) and meta.get("input_source_split") is not None:
        return str(meta["input_source_split"])
    return str(row.get("source_split", ""))


def select_pseudo_rows(
    rows: Iterable[dict[str, Any]],
    *,
    source_splits: set[str] | None,
    limit: int | None,
    evaluation_mode: bool = False,
) -> list[dict[str, Any]]:
    if limit is not None and limit < 0:
        raise ValueError("limit은 0 이상이어야 합니다")
    if limit == 0:
        return []
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        item_id = essay_id(row)
        train_split = str(row.get("source_split", "")).lower()
        input_split = _input_source_split(row)
        is_evaluation = (
            "validation" in input_split.lower() or "test" in input_split.lower()
        )
        if evaluation_mode:
            if train_split != "evaluation" or not is_evaluation:
                raise ValueError(
                    f"evaluation_mode에는 명시적 validation/test row만 허용됩니다: {item_id}"
                )
        elif train_split != "train" or is_evaluation:
            raise ValueError(f"train pseudo row만 Judge할 수 있습니다: {item_id}")
        if source_splits and input_split not in source_splits:
            continue
        if item_id in seen:
            raise ValueError(f"중복 pseudo essay ID: {item_id}")
        seen.add(item_id)
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _request_hash(
    model: str,
    model_revision: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "model_revision": model_revision,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "chat_template_kwargs": NONTHINKING_TEMPLATE_KWARGS,
    }
    return sha256_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _existing_successes(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    result: set[tuple[str, str]] = set()
    for row in read_jsonl(path):
        request_hash = row.get("request_sha256")
        if row.get("parse_ok") and isinstance(request_hash, str):
            result.add((essay_id(row), request_hash))
    return result


def _log(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=False, allow_nan=False), file=sys.stderr)


def evaluate_teacher_rows(
    *,
    pseudo_path: str | Path,
    output_path: str | Path,
    base_url: str,
    model: str | None = None,
    model_revision: str = "local_server_checkpoint",
    source_splits: set[str] | None = None,
    limit: int | None = None,
    # 12개 evidence 블록을 모두 담아야 한다. 1200으로는 `finish_reason=length`에 걸려 balanced
    # JSON이 안 나오고 전량 파싱 실패한다(2026-08-06 실측). 프롬프트에서 evidence를 60자로
    # 묶었고, 여기에 여유를 둔다.
    max_tokens: int = 2048,
    timeout: float = 900.0,
    retries: int = 2,
    minimum_mean: float = 3.0,
    minimum_scores: Mapping[str, int] | None = None,
    accepted_output_path: str | Path | None = None,
    concurrency: int = 8,
    evaluation_mode: bool = False,
) -> dict[str, Any]:
    endpoint = local_base_url(base_url)
    if retries < 0:
        raise ValueError("retries는 0 이상이어야 합니다")
    if concurrency < 1:
        raise ValueError("concurrency는 1 이상이어야 합니다")
    thresholds = dict(minimum_scores or {name: 3 for name in JUDGE_DIMENSIONS})
    if set(thresholds) != set(JUDGE_DIMENSIONS):
        raise ValueError("네 Judge dimension의 threshold가 모두 필요합니다")
    if not 1 <= minimum_mean <= 5 or any(
        not isinstance(value, int) or not 1 <= value <= 5
        for value in thresholds.values()
    ):
        raise ValueError("Judge filter threshold는 1~5 범위여야 합니다")

    all_pseudo = read_jsonl(pseudo_path)
    rows = select_pseudo_rows(
        all_pseudo,
        source_splits=source_splits,
        limit=limit,
        evaluation_mode=evaluation_mode,
    )
    if evaluation_mode and accepted_output_path is not None:
        raise ValueError(
            "evaluation_mode에서는 학습 accepted output을 만들 수 없습니다"
        )
    judge_model = model or resolve_model_id(endpoint)
    target = Path(output_path)
    audit_path = target.with_suffix(target.suffix + ".audit.jsonl")
    done = _existing_successes(target)
    done_ids = {item_id for item_id, _request in done}
    prompt_hash = sha256_text(JUDGE_INSTRUCTION)

    evaluated = skipped = failed = accepted_this_run = 0

    # 계획 단계는 순차로 돈다. resume 판정을 여기서 끝내고, 네트워크가 필요한 일만 아래에서
    # 병렬로 돌린다(teacher_generate와 같은 구조).
    pending: list[tuple[dict[str, Any], str, list[dict[str, str]]]] = []
    for row in rows:
        item_id = essay_id(row)
        messages = judge_messages(row)
        request_hash = _request_hash(judge_model, model_revision, messages, max_tokens)
        if (item_id, request_hash) in done:
            skipped += 1
            continue
        if item_id in done_ids:
            raise ValueError(
                f"기존 proxy output에 다른 request의 성공 row가 있습니다: {item_id}. "
                "새 output 경로를 사용하십시오"
            )
        pending.append((row, request_hash, messages))

    def judge_one(messages: list[dict[str, str]]) -> dict[str, Any]:
        """judge 1건. 공유 상태를 건드리지 않아 스레드에서 안전하다."""
        raw = ""
        reasoning: str | None = None
        errors: list[str] = []
        api_audit: dict[str, Any] = {}
        result: dict[str, dict[str, dict[str, Any]]] | None = None
        sampling = sampling_for_attempt(0)
        for attempt_index in range(retries + 1):
            # 재시도는 parse 실패에만 걸린다. greedy로 고정하면 같은 출력이 다시 오므로
            # 표본을 흔든다. 판정 자체를 다시 굴리는 것이 아니라 형식을 다시 받는 것이다.
            sampling = sampling_for_attempt(attempt_index)
            try:
                response = chat_completion(
                    endpoint,
                    judge_model,
                    messages,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    temperature=sampling["temperature"],
                    top_p=sampling["top_p"],
                    seed=sampling["seed"],
                )
                raw, reasoning = completion_parts(response)
                api_audit = response_audit(response)
                result = parse_proxy_judge(raw)
                break
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                result = None
        return {
            "raw": raw,
            "reasoning": reasoning,
            "errors": errors,
            "api_audit": api_audit,
            "result": result,
            "sampling": sampling,
        }

    # 파일 쓰기는 전부 주 스레드에서만 한다. `map`이 제출 순서대로 결과를 돌려주므로 출력
    # 파일 순서는 순차 실행과 같다.
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        outcomes = pool.map(judge_one, (messages for _r, _h, messages in pending))
        for (row, request_hash, _messages), outcome in zip(pending, outcomes):
            item_id = essay_id(row)
            raw = outcome["raw"]
            reasoning = outcome["reasoning"]
            errors = outcome["errors"]
            api_audit = outcome["api_audit"]
            result = outcome["result"]

            retry_count = max(0, len(errors) if result is not None else retries)
            common = {
                "essay_id": item_id,
                "stage": "teacher_judge",
                "proxy_judge_model": judge_model,
                "model_revision": model_revision,
                "proxy_prompt_version": PROXY_JUDGE_VERSION,
                "proxy_prompt_sha256": prompt_hash,
                "official_judge_reference": OFFICIAL_JUDGE_REFERENCE,
                "exact_official_judge": False,
                "interpretation": "local proxy; official LLM Judge score가 아님",
                "input_source_split": _input_source_split(row),
                "teacher_request_sha256": (
                    row.get("pseudo_meta", {}).get("request_sha256")
                    if isinstance(row.get("pseudo_meta"), Mapping)
                    else None
                ),
                "request_sha256": request_hash,
                # 실제로 채택된 시도의 sampling이다. 0번 시도는 항상 greedy/seed 42다.
                "sampling": outcome["sampling"],
                "temperature": outcome["sampling"]["temperature"],
                "seed": outcome["sampling"]["seed"],
                "chat_template_kwargs": NONTHINKING_TEMPLATE_KWARGS,
                "attempt_count": min(retries + 1, len(errors) + (result is not None)),
                "retry_count": retry_count,
                "errors": errors,
                "response": api_audit,
                "raw_output_sha256": sha256_text(raw),
                "raw_output_characters": len(raw),
                "reasoning_sha256": sha256_text(reasoning or ""),
                "reasoning_characters": len(reasoning or ""),
            }
            if result is None:
                failed += 1
                append_jsonl(
                    audit_path,
                    {**common, "parse_ok": False, "raw_output": raw},
                )
                _log(
                    {
                        "stage": "teacher_judge",
                        "essay_id": item_id,
                        "status": "failed",
                        "error": errors[-1] if errors else "unknown",
                    }
                )
                continue

            statistics = judge_statistics(result)
            accepted, rejection_reasons = passes_filter(
                result,
                minimum_mean=minimum_mean,
                minimum_scores=thresholds,
            )
            record = {
                **common,
                "parse_ok": True,
                "result": result,
                "statistics": statistics,
                "filter": {
                    "accepted": accepted,
                    "rejection_reasons": rejection_reasons,
                    "minimum_mean": minimum_mean,
                    "minimum_scores": thresholds,
                },
                "raw_output": raw,
            }
            append_jsonl(target, record)
            append_jsonl(
                audit_path,
                {
                    **common,
                    "parse_ok": True,
                    "accepted": accepted,
                    "statistics": statistics,
                },
            )
            evaluated += 1
            accepted_this_run += int(accepted)
            _log(
                {
                    "stage": "teacher_judge",
                    "essay_id": item_id,
                    "status": "accepted" if accepted else "rejected",
                    "overall_mean": statistics["overall_mean"],
                    "retry_count": retry_count,
                    "latency_seconds": api_audit.get("latency_seconds"),
                }
            )

    latest_evaluations: dict[str, dict[str, Any]] = {}
    if target.exists():
        for record in read_jsonl(target):
            if record.get("proxy_judge_model") == judge_model and record.get(
                "parse_ok"
            ):
                latest_evaluations[essay_id(record)] = record
    selected_by_id = {essay_id(row): row for row in rows}
    accepted_rows: list[dict[str, Any]] = []
    for item_id, evaluation in latest_evaluations.items():
        if item_id not in selected_by_id:
            continue
        accepted, rejection_reasons = passes_filter(
            evaluation["result"],
            minimum_mean=minimum_mean,
            minimum_scores=thresholds,
        )
        current_filter = {
            "accepted": accepted,
            "rejection_reasons": rejection_reasons,
            "minimum_mean": minimum_mean,
            "minimum_scores": thresholds,
        }
        if not accepted:
            continue
        pseudo = dict(selected_by_id[item_id])
        meta = dict(pseudo.get("pseudo_meta") or {})
        meta["proxy_judge"] = {
            "model": judge_model,
            "model_revision": model_revision,
            "prompt_version": PROXY_JUDGE_VERSION,
            "request_sha256": evaluation["request_sha256"],
            "statistics": evaluation["statistics"],
            "filter": current_filter,
            "exact_official_judge": False,
        }
        meta["status"] = "accepted"
        pseudo["pseudo_meta"] = meta
        accepted_rows.append(pseudo)
    if accepted_output_path is not None:
        write_jsonl(accepted_output_path, accepted_rows)

    valid_results = [
        record
        for item_id, record in latest_evaluations.items()
        if item_id in selected_by_id
    ]
    manifest = {
        "schema_version": 1,
        "stage": "teacher_judge",
        "pseudo_path": str(Path(pseudo_path).resolve()),
        "output_path": str(target.resolve()),
        "audit_path": str(audit_path.resolve()),
        "accepted_output_path": (
            str(Path(accepted_output_path).resolve())
            if accepted_output_path is not None
            else None
        ),
        "proxy_judge_model": judge_model,
        "model_revision": model_revision,
        "proxy_prompt_version": PROXY_JUDGE_VERSION,
        "proxy_prompt_sha256": prompt_hash,
        "official_judge_reference": OFFICIAL_JUDGE_REFERENCE,
        "exact_official_judge": False,
        "interpretation": "local proxy; official LLM Judge score가 아님",
        "source_split_filter": sorted(source_splits) if source_splits else None,
        "limit": limit,
        "filter": {"minimum_mean": minimum_mean, "minimum_scores": thresholds},
        "concurrency": concurrency,
        "selected": len(rows),
        "evaluated_this_run": evaluated,
        "resumed_skips": skipped,
        "failed_this_run": failed,
        "accepted_this_run": accepted_this_run,
        "valid_latest": len(valid_results),
        "accepted_latest": len(accepted_rows),
        "evaluation_mode": evaluation_mode,
        "mean_latest": (
            mean(
                float(record["statistics"]["overall_mean"]) for record in valid_results
            )
            if valid_results
            else None
        ),
    }
    write_json(target.with_suffix(target.suffix + ".manifest.json"), manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="기술서 3 trait x 4항목 형식의 local proxy Judge"
    )
    parser.add_argument("--pseudo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--accepted-output")
    parser.add_argument(
        "--base-url", "--api-base", dest="base_url", default="http://127.0.0.1:9000"
    )
    parser.add_argument("--model")
    parser.add_argument("--model-revision", default="local_server_checkpoint")
    parser.add_argument("--source-split", action="append", dest="source_splits")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--minimum-mean", type=float, default=3.0)
    parser.add_argument("--minimum-domain-match", type=int, default=3)
    parser.add_argument("--minimum-consistency", type=int, default=3)
    parser.add_argument("--minimum-specificity", type=int, default=3)
    parser.add_argument("--minimum-groundedness", type=int, default=3)
    parser.add_argument(
        "--evaluation-mode",
        action="store_true",
        help="학습 채택과 분리해 validation/test rationale를 proxy 평가",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = {
        "domain_match": args.minimum_domain_match,
        "score_rationale_consistency": args.minimum_consistency,
        "specificity": args.minimum_specificity,
        "groundedness": args.minimum_groundedness,
    }
    manifest = evaluate_teacher_rows(
        pseudo_path=args.pseudo,
        output_path=args.output,
        accepted_output_path=args.accepted_output,
        base_url=args.base_url,
        model=args.model,
        model_revision=args.model_revision,
        source_splits=(
            None
            if args.evaluation_mode
            else (
                set(args.source_splits)
                if args.source_splits
                else set(TRAIN_SOURCE_SPLITS)
            )
        ),
        limit=args.limit,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        retries=args.retries,
        concurrency=args.concurrency,
        minimum_mean=args.minimum_mean,
        minimum_scores=thresholds,
        evaluation_mode=args.evaluation_mode,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
