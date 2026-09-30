from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Mapping

import torch

from main_code.postprocess import ScorePostprocessor

from . import TRAIN_SOURCE_SPLITS
from .artifacts import append_jsonl, read_rows, sha256_text, write_json
from .config import RationaleConfig, load_config, with_prompt_file
from .data import RationaleCollator, RationaleSFTDataset, tokenization_audit
from .modeling import load_model, load_tokenizer
from .prompts import baseline_prompt_template, prompt_template_sha256
from .schema import conditioning_scores, essay_id, validate_scores

_AVERAGE_MATCHED = ScorePostprocessor("average_matched")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="고정 점수 조건부 JSON 근거 생성 LoRA SFT"
    )
    # 2026-08-25 최종 제출본 레시피. 인자를 안 주면 제출본을 그대로 재현한다.
    # 프롬프트 기본값(dataclass)은 baseline으로 두어야 프롬프트를 선언하지 않는
    # legacy recipe가 조용히 v4로 바뀌지 않는다. 그래서 제출본 고정은 여기서 한다.
    parser.add_argument(
        "--recipe",
        default="main_code_relonation/recipes/r18_report_qwen35.json",
        help="기본값은 최종 제출본 근거모델 레시피(v4 prompt, Qwen3.5-9B LoRA r32)",
    )
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument(
        "--rationale-prompt-file",
        help="recipe prompt를 명시적으로 대체합니다; adapter sidecar에 원문을 저장합니다",
    )
    return parser.parse_args()


def accepted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        meta = row.get("pseudo_meta")
        if not isinstance(meta, dict):
            continue
        status = meta.get("status")
        qc = meta.get("qc")
        if (
            status in {"accepted", "passed"}
            and meta.get("parse_ok") is True
            and meta.get("input_source_split") in TRAIN_SOURCE_SPLITS
            and meta.get("teacher_score_copy_exact") is True
            and isinstance(qc, dict)
            and qc.get("pass") is True
        ):
            result.append(row)
    if not result:
        raise ValueError("학습 가능한 accepted pseudo row가 없습니다")
    return result


def environment_manifest() -> dict[str, Any]:
    import peft
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
    }


def validate_training_prompt_rows(
    rows: list[dict[str, Any]], config: RationaleConfig
) -> None:
    """Prevent baseline/v1 pseudo labels from being silently mixed."""

    legacy_hash = prompt_template_sha256(baseline_prompt_template())
    mismatches: list[tuple[str, str]] = []
    for row in rows:
        meta = row.get("pseudo_meta")
        if not isinstance(meta, dict):
            continue
        declared = meta.get("rationale_prompt_sha256")
        # Rows produced before prompt provenance existed are baseline-only.
        row_hash = str(declared) if declared else legacy_hash
        if row_hash != config.rationale_prompt_sha256:
            mismatches.append((str(row.get("essay_id", row.get("id"))), row_hash))
    if mismatches:
        raise ValueError(
            "학습 pseudo row의 rationale prompt가 recipe와 다릅니다: "
            f"expected={config.rationale_prompt_sha256} examples={mismatches[:3]}"
        )


def validate_training_score_rows(
    rows: list[dict[str, Any]], config: RationaleConfig
) -> dict[str, Any]:
    """Validate and fingerprint the fixed-score surface used by SFT.

    Legacy baseline rows predate explicit postprocess provenance.  New v1 rows must
    declare it, and human_average_matched rows are independently reconstructed with
    the same shared postprocessor used by the submission engine.
    """

    legacy_prompt_hash = prompt_template_sha256(baseline_prompt_template())
    contracts: set[tuple[str, str]] = set()
    score_records: list[dict[str, Any]] = []
    for row in rows:
        meta = row.get("pseudo_meta")
        if not isinstance(meta, Mapping):
            raise ValueError(f"pseudo_meta가 없습니다: {essay_id(row)}")
        source = meta.get("score_source")
        postprocess = meta.get("conditioning_score_postprocess")
        if source == "human" and postprocess is None:
            # rationale_ax_v2 legacy artifact predates this field.
            postprocess = "none"
        if not isinstance(source, str) or not isinstance(postprocess, str):
            if config.rationale_prompt_sha256 != legacy_prompt_hash:
                raise ValueError(
                    "v1 학습 row에 conditioning score provenance가 없습니다: "
                    f"{essay_id(row)}"
                )
            source = str(source or "legacy_unspecified")
            postprocess = str(postprocess or "none")
        contract = (source, postprocess)
        contracts.add(contract)

        fixed = conditioning_scores(validate_scores(row.get("conditioning_scores", {})))
        canonical_value = meta.get("canonical_fixed_scores")
        if isinstance(canonical_value, Mapping):
            canonical = conditioning_scores(validate_scores(canonical_value))
            if canonical != fixed:
                raise ValueError(
                    "conditioning_scores와 canonical_fixed_scores가 다릅니다: "
                    f"{essay_id(row)}"
                )

        if contract == ("human_average_matched", "average_matched"):
            source_value = meta.get("score_source_values")
            if not isinstance(source_value, Mapping):
                raise ValueError(
                    "human_average_matched row에 원래 인간 점수가 없습니다: "
                    f"{essay_id(row)}"
                )
            source_scores = conditioning_scores(validate_scores(source_value))
            expected = conditioning_scores(_AVERAGE_MATCHED.apply_row(source_scores))
            if expected != fixed:
                raise ValueError(
                    "human 점수의 average_matched 재계산값과 학습 점수가 다릅니다: "
                    f"{essay_id(row)} expected={expected} actual={fixed}"
                )
        score_records.append({"essay_id": essay_id(row), "scores": fixed})

    if len(contracts) != 1:
        raise ValueError(
            f"학습 row에 conditioning score 계약이 섞였습니다: {contracts}"
        )
    source, postprocess = next(iter(contracts))
    payload = "\n".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for item in score_records
    )
    return {
        "score_source": source,
        "conditioning_score_postprocess": postprocess,
        "row_count": len(score_records),
        "conditioning_scores_sha256": sha256_text(payload),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.recipe)
    if args.rationale_prompt_file:
        config = with_prompt_file(config, args.rationale_prompt_file)
    if config.adapter_path is not None:
        raise ValueError("SFT recipe에는 adapter_path를 넣지 마세요")

    input_path = Path(args.train_file).resolve()
    rows = accepted_rows(read_rows(input_path))
    if args.limit is not None:
        rows = rows[: args.limit]
    validate_training_prompt_rows(rows, config)
    score_contract = validate_training_score_rows(rows, config)
    dataset = RationaleSFTDataset(rows)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(config)
    collator = RationaleCollator(tokenizer, config)
    audit = tokenization_audit(rows, collator)
    write_json(output / "tokenization_audit.json", audit)
    write_json(output / "resolved_config.json", config.to_dict())
    write_json(
        output / "run_manifest.json",
        {
            "pipeline": "rationale_lora_sft",
            "recipe": str(Path(args.recipe).resolve()),
            "config_id": config.fingerprint(),
            "input": str(input_path),
            "input_sha256": sha256_text(input_path.read_text(encoding="utf-8")),
            "source_row_count": len(read_rows(input_path)),
            "training_row_count": len(rows),
            "score_mode": config.score_mode,
            "rationale_prompt_id": config.rationale_prompt_id,
            "rationale_prompt_sha256": config.rationale_prompt_sha256,
            "rationale_prompt_text": config.rationale_prompt_text,
            "conditioning_score_contract": score_contract,
            "supervision": "assistant JSON only; user/chat-template tokens masked",
            "environment": environment_manifest(),
        },
    )

    model = load_model(config, for_training=True)
    model.print_trainable_parameters()

    from transformers import Trainer, TrainerCallback, TrainingArguments

    log_path = output / "train_log.jsonl"

    class JsonLogCallback(TrainerCallback):
        def on_log(
            self, args: Any, state: Any, control: Any, logs: Any = None, **_: Any
        ) -> None:
            if logs:
                append_jsonl(
                    log_path,
                    {
                        "step": int(state.global_step),
                        "epoch": state.epoch,
                        **dict(logs),
                    },
                )

    training_args = TrainingArguments(
        output_dir=str(output / "trainer_checkpoints"),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=5,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=config.torch_dtype == "bfloat16" and torch.cuda.is_available(),
        fp16=config.torch_dtype == "float16" and torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
        seed=config.seed,
        data_seed=config.seed,
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[JsonLogCallback()],
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    adapter_dir = output / "final_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    (adapter_dir / "rationale_prompt.txt").write_text(
        config.rationale_prompt_text, encoding="utf-8"
    )
    write_json(
        adapter_dir / "rationale_runtime_config.json",
        {
            "schema_version": 1,
            "rationale_prompt_id": config.rationale_prompt_id,
            "rationale_prompt_text": config.rationale_prompt_text,
            "rationale_prompt_sha256": config.rationale_prompt_sha256,
            "score_mode": config.score_mode,
            "training_score_source": score_contract["score_source"],
            "training_conditioning_score_postprocess": score_contract[
                "conditioning_score_postprocess"
            ],
            "training_conditioning_scores_sha256": score_contract[
                "conditioning_scores_sha256"
            ],
            "chat_template_sha256": sha256_text(str(tokenizer.chat_template)),
        },
    )
    write_json(output / "train_metrics.json", dict(result.metrics))
    write_json(
        output / "completed.json",
        {
            "status": "complete",
            "adapter": str(adapter_dir.resolve()),
            "global_step": trainer.state.global_step,
            "train_metrics": dict(result.metrics),
        },
    )
    print(f"adapter={adapter_dir} rows={len(rows)} config={config.fingerprint()}")


if __name__ == "__main__":
    main()
