from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from . import TRAITS
from .artifacts import sha256_text
from .config import RationaleConfig
from .modeling import assistant_supervision_ids
from .prompts import build_messages
from .schema import (
    compact_judge_json,
    essay_id,
    official_raw_essay,
    prompt_text,
    validate_score,
    validate_scores,
)


def judge_from_row(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    value = row.get("judge")
    if not isinstance(value, Mapping):
        raise ValueError(f"pseudo judge가 필요합니다: {essay_id(row)}")
    result: dict[str, dict[str, Any]] = {}
    for trait in TRAITS:
        item = value.get(trait)
        if not isinstance(item, Mapping):
            raise ValueError(f"{trait} pseudo judge가 없습니다: {essay_id(row)}")
        score = validate_score(item.get("score"), trait)
        rationale = item.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"{trait} rationale이 비었습니다: {essay_id(row)}")
        result[trait] = {"score": score, "rationale": rationale.strip()}
    return result


def conditioning_scores(row: Mapping[str, Any]) -> dict[str, float]:
    value = row.get("conditioning_scores")
    if isinstance(value, Mapping):
        return validate_scores(value)
    judge = judge_from_row(row)
    return validate_scores({trait: judge[trait]["score"] for trait in TRAITS})


def reject_validation_training_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    bad: list[str] = []
    for row in rows:
        split_values = [row.get("source_split"), row.get("split")]
        meta = row.get("pseudo_meta")
        if isinstance(meta, Mapping):
            split_values.extend(
                [meta.get("source_split"), meta.get("input_source_split")]
            )
        normalized = [str(value).lower() for value in split_values if value is not None]
        if any("validation" in split or "test" in split for split in normalized):
            bad.append(essay_id(row))
    if bad:
        raise ValueError(
            f"validation/test pseudo row를 학습에 사용할 수 없습니다: {bad[:3]}"
        )


@dataclass(frozen=True)
class RationaleExample:
    row: Mapping[str, Any]


class RationaleSFTDataset(Dataset[RationaleExample]):
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            raise ValueError("SFT 학습 row가 비었습니다")
        reject_validation_training_rows(rows)
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> RationaleExample:
        return RationaleExample(self.rows[index])


class RationaleCollator:
    def __init__(self, tokenizer: Any, config: RationaleConfig) -> None:
        self.tokenizer = tokenizer
        self.config = config

    def encode_row(self, row: Mapping[str, Any]) -> tuple[list[int], list[int], int]:
        judge = judge_from_row(row)
        scores = conditioning_scores(row) if self.config.score_mode == "fixed" else None
        if scores is not None:
            target_scores = validate_scores(
                {trait: judge[trait]["score"] for trait in TRAITS}
            )
            if target_scores != scores:
                raise ValueError(
                    f"conditioning score와 assistant target score가 다릅니다: "
                    f"{essay_id(row)} conditioning={scores} target={target_scores}"
                )
        messages = build_messages(
            prompt_text(row),
            official_raw_essay(row),
            scores=scores,
            prompt_template=(
                self.config.rationale_prompt_text if scores is not None else None
            ),
            skeleton_hint=self.config.rationale_skeleton_hint,
        )
        target = compact_judge_json(judge)
        ids, labels, prompt_length = assistant_supervision_ids(
            self.tokenizer, messages, target, config=self.config
        )
        if len(ids) > self.config.max_length:
            raise ValueError(
                f"SFT input이 max_length를 넘습니다: {essay_id(row)} "
                f"{len(ids)}>{self.config.max_length}"
            )
        return ids, labels, prompt_length

    def __call__(self, examples: Sequence[RationaleExample]) -> dict[str, torch.Tensor]:
        encoded = [self.encode_row(example.row) for example in examples]
        features = [
            {"input_ids": ids, "attention_mask": [1] * len(ids)}
            for ids, _, _ in encoded
        ]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        padded_length = int(batch["input_ids"].shape[1])
        labels = [
            label + [-100] * (padded_length - len(label)) for _, label, _ in encoded
        ]
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def tokenization_audit(
    rows: Sequence[Mapping[str, Any]], collator: RationaleCollator
) -> dict[str, Any]:
    if not rows:
        raise ValueError("tokenization audit row가 비었습니다")
    lengths: list[int] = []
    prompt_lengths: list[int] = []
    target_lengths: list[int] = []
    row_hashes: list[str] = []
    for row in rows:
        ids, labels, prompt_length = collator.encode_row(row)
        lengths.append(len(ids))
        prompt_lengths.append(prompt_length)
        target_lengths.append(sum(label != -100 for label in labels))
        row_hashes.append(
            sha256_text(
                f"{essay_id(row)}\0{prompt_text(row)}\0{official_raw_essay(row)}"
            )
        )

    def summary(values: Sequence[int]) -> dict[str, int]:
        ordered = sorted(values)
        return {
            "min": ordered[0],
            "median": ordered[len(ordered) // 2],
            "p95": ordered[min(len(ordered) - 1, math.floor(len(ordered) * 0.95))],
            "max": ordered[-1],
        }

    return {
        "count": len(rows),
        "total_tokens": summary(lengths),
        "prompt_tokens": summary(prompt_lengths),
        "assistant_target_tokens": summary(target_lengths),
        "row_fingerprint": sha256_text("".join(sorted(row_hashes))),
        "chat_template_hash": sha256_text(str(collator.tokenizer.chat_template)),
        "rationale_prompt_id": collator.config.rationale_prompt_id,
        "rationale_prompt_sha256": collator.config.rationale_prompt_sha256,
        "assistant_mask_contract": "prompt=-100, assistant JSON and template terminal supervised",
        "chat_prefix_verified_for_all_rows": True,
    }
