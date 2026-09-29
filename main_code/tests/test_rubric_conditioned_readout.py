from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from main_code.datasets import (
    INPUT_TEMPLATE,
    RUBRIC_CRITERION_TAGS,
    RegressionCollator,
    deployment_surface_fingerprints,
    format_input,
    format_input_segments,
)
from main_code.infer import score_batches
from main_code.models import RegressionScorer
from main_code.tests.config_helpers import legacy_config


def rc_config(**updates):
    base = legacy_config(
        input_format="rubric_conditioned_v1",
        rubric_profile="short_3trait_v1",
        criterion_readout="shared",
        backbone_type="decoder",
        pooling="mean",
        organization_pooling="shared",
        detail_head_mode="scalar",
        detail_final_source="criterion",
        essay_surface="official_raw",
    )
    return replace(base, **updates).validate()


def row(essay="  원문\t본문\r  "):
    return {
        "essay_id": "e1",
        "prompt": "[C1 판단]: 이 문자열도 문제에 있을 수 있다.",
        "essay": essay,
    }


class CharTokenizer:
    model_max_length = 100_000
    padding_side = "right"
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        if isinstance(text, list):
            raise AssertionError("RC test는 per-row structured tokenization을 사용해야 한다")
        add_special_tokens = kwargs.get("add_special_tokens", True)
        offsets = [(index, index + 1) for index in range(len(text))]
        ids = [2 + ord(char) % 101 for char in text]
        if add_special_tokens:
            offsets = [(0, 0), *offsets]
            ids = [1, *ids]
        if kwargs.get("return_offsets_mapping"):
            return {"input_ids": ids, "offset_mapping": offsets}
        return {"input_ids": ids}

    def pad(self, examples, *, padding, return_tensors):
        assert padding is True and return_tensors == "pt"
        width = max(len(example["input_ids"]) for example in examples)
        ids = []
        masks = []
        for example in examples:
            amount = width - len(example["input_ids"])
            if self.padding_side == "left":
                ids.append([self.pad_token_id] * amount + example["input_ids"])
                masks.append([0] * amount + example["attention_mask"])
            else:
                ids.append(example["input_ids"] + [self.pad_token_id] * amount)
                masks.append(example["attention_mask"] + [0] * amount)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class TinyBackbone(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(128, hidden_size)
        self.calls = 0

    def forward(self, input_ids, attention_mask, **kwargs):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_legacy_defaults_are_neutral_and_invalid_combinations_fail():
    legacy = legacy_config().validate()
    assert legacy.rubric_profile == "none"
    assert legacy.criterion_readout == "shared"
    with pytest.raises(ValueError, match="versioned rubric_profile"):
        replace(rc_config(), rubric_profile="none").validate()
    with pytest.raises(ValueError, match="input_format='rubric_conditioned_v1'"):
        replace(legacy_config(), rubric_profile="full_9criterion_1to5_v1").validate()


def test_baseline_v1_formatted_text_is_unchanged():
    config = legacy_config(input_format="baseline_v1", essay_surface="canonical")
    item = row("원래 입력")
    assert format_input(item, config) == INPUT_TEMPLATE.format(
        prompt=item["prompt"], essay=item["essay"]
    )
    empty = row("")
    assert format_input(empty, config) == INPUT_TEMPLATE.format(
        prompt=empty["prompt"], essay=""
    )
    with pytest.raises(ValueError, match="빈 essay"):
        format_input(empty, rc_config())


def test_formatting_preserves_raw_essay_and_shared_anchor_pairs_are_identical():
    short_shared = rc_config()
    short_anchor = replace(
        short_shared, criterion_readout="textual_anchor_residual"
    ).validate()
    full_shared = replace(
        short_shared, rubric_profile="full_9criterion_1to5_v1"
    ).validate()
    full_anchor = replace(
        full_shared, criterion_readout="textual_anchor_residual"
    ).validate()
    assert format_input(row(), short_shared) == format_input(row(), short_anchor)
    assert format_input(row(), full_shared) == format_input(row(), full_anchor)
    assert format_input(row(), short_shared) != format_input(row(), full_shared)
    assert (
        deployment_surface_fingerprints([row()], short_shared)[
            "formatted_input_utf8_sha256"
        ]
        == deployment_surface_fingerprints([row()], short_anchor)[
            "formatted_input_utf8_sha256"
        ]
    )
    assert (
        deployment_surface_fingerprints([row()], full_shared)[
            "formatted_input_utf8_sha256"
        ]
        == deployment_surface_fingerprints([row()], full_anchor)[
            "formatted_input_utf8_sha256"
        ]
    )
    segments = format_input_segments(row(), full_shared)
    assert segments.text[slice(*segments.essay_span)] == row()["essay"]
    assert len(segments.criterion_anchor_spans) == 9
    for tag, span in zip(RUBRIC_CRITERION_TAGS, segments.criterion_anchor_spans):
        assert segments.text[slice(*span)] == f"[{tag} 판단]:"
        assert span[0] >= segments.scoring_end

    short_segments = format_input_segments(row(), short_shared)
    baseline = legacy_config(
        input_format="baseline_v1", essay_surface="official_raw"
    ).validate()
    assert (
        short_segments.text[: short_segments.scoring_end]
        == format_input(row(), baseline)
    )


@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_collator_masks_suffix_and_preserves_nine_anchor_positions(padding_side):
    tokenizer = CharTokenizer()
    tokenizer.padding_side = padding_side
    collator = RegressionCollator(
        tokenizer, rc_config(), include_labels=False, include_metadata=False
    )
    batch = collator([row("짧은 글"), row("조금 더 긴 글")])
    assert batch["shared_pooling_mask"].shape == batch["attention_mask"].shape
    assert batch["criterion_anchor_positions"].shape == (2, 9)
    for item_index in range(2):
        positions = batch["criterion_anchor_positions"][item_index]
        assert torch.all(positions[1:] > positions[:-1])
        assert torch.all(batch["attention_mask"][item_index, positions] == 1)
        assert torch.all(batch["shared_pooling_mask"][item_index, positions] == 0)
    # RC0의 shared prefix는 suffix를 붙이기 전 base tokenization과 정확히 같다.
    first_segments = format_input_segments(row("짧은 글"), rc_config())
    standalone = tokenizer(
        first_segments.text[: first_segments.scoring_end],
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=True,
    )["input_ids"]
    observed_prefix = batch["input_ids"][0][
        batch["shared_pooling_mask"][0].bool()
    ].tolist()
    assert observed_prefix == standalone
    assert collator.input_length_summary()["overflow_count"] == 0


def test_rc_overflow_is_fail_closed():
    collator = RegressionCollator(
        CharTokenizer(),
        replace(rc_config(), max_length=10),
        include_labels=False,
        include_metadata=False,
    )
    with pytest.raises(ValueError, match="truncation은 허용하지 않습니다"):
        collator([row()])


def test_zero_gate_matches_shared_and_anchor_uses_one_backbone_forward():
    tokenizer = CharTokenizer()
    shared_batch = RegressionCollator(
        tokenizer, rc_config(), include_labels=False, include_metadata=False
    )([row("주장과 근거가 있는 글")])

    torch.manual_seed(7)
    shared_backbone = TinyBackbone()
    shared = RegressionScorer(shared_backbone, 8, rc_config())
    torch.manual_seed(7)
    anchor_backbone = TinyBackbone()
    anchor = RegressionScorer(
        anchor_backbone,
        8,
        rc_config(criterion_readout="textual_anchor_residual"),
    )
    shared.eval()
    anchor.eval()
    shared_result = shared(**shared_batch, return_detail_predictions=True)
    anchor_result = anchor(**shared_batch, return_detail_predictions=True)
    assert torch.equal(shared_result["detail_scores"], anchor_result["detail_scores"])
    assert torch.equal(shared_result["scores"], anchor_result["scores"])
    assert shared_backbone.calls == 1
    assert anchor_backbone.calls == 1
    assert "criterion_anchor_gates" not in shared.scoring_state_dict()
    assert anchor.scoring_state_dict()["criterion_anchor_gates"].shape == (9,)

    anchor.train()
    assert anchor.criterion_anchor_gates is not None
    anchor.criterion_anchor_gates.data[0] = 0.5
    changed = anchor(**shared_batch, return_detail_predictions=True)
    assert not torch.equal(
        changed["detail_scores"][:, 0], shared_result["detail_scores"][:, 0]
    )
    changed["detail_scores"].sum().backward()
    assert anchor.criterion_anchor_gates.grad is not None
    assert torch.isfinite(anchor.criterion_anchor_gates.grad).all()


def test_score_batches_forwards_structured_readout_tensors():
    class CapturingScorer:
        def __init__(self):
            self.seen = None

        def eval(self):
            return self

        def __call__(self, **kwargs):
            self.seen = kwargs
            return {
                "scores": torch.full((1, 3), 3.0),
                "detail_scores": torch.full((1, 9), 3.0),
            }

    scorer = CapturingScorer()
    batch = {
        "input_ids": torch.ones((1, 12), dtype=torch.long),
        "attention_mask": torch.ones((1, 12), dtype=torch.long),
        "shared_pooling_mask": torch.tensor([[1, 1, 1] + [0] * 9]),
        "criterion_anchor_positions": torch.arange(3, 12).unsqueeze(0),
        "essay_ids": ["e1"],
        "rows": [row("글")],
    }
    loaded = SimpleNamespace(
        scorer=scorer,
        config=SimpleNamespace(score_head="regression", detail_head_mode="scalar"),
    )
    score_batches(
        loaded,
        [batch],
        torch.device("cpu"),
        prompt_lookup={},
        prompt_routing_active=False,
    )
    assert scorer.seen is not None
    assert torch.equal(
        scorer.seen["shared_pooling_mask"], batch["shared_pooling_mask"]
    )
    assert torch.equal(
        scorer.seen["criterion_anchor_positions"],
        batch["criterion_anchor_positions"],
    )
