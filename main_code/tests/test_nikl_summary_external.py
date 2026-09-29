from __future__ import annotations

import pytest

from main_code.config import EXTERNAL_DATASETS
from main_code.prepare_data import convert_nikl_summary
from main_code.tests.config_helpers import legacy_config


def _evaluator(name: str, values: tuple[int, int, int, int, int]) -> dict:
    c1, c2, c3, organization, expression = values
    return {
        "id": name,
        "content": {
            "description": c1,
            "claims": c2,
            "arguments": c3,
            "comment": "내용 근거",
        },
        "organization": {"completion": organization, "comment": "조직 근거"},
        "expression": {"accuracy": expression, "comment": "표현 근거"},
    }


def test_nikl_summary_converter_recomputes_three_traits_from_1to7_ratings() -> None:
    document = {
        "id": "NEWS-1",
        "metadata": {"title": "기사 제목", "topic": "사회"},
        "paragraph": [{"form": "첫 문단."}, {"form": "둘째 문단."}],
    }
    candidate = {
        "summary": "평가할 요약문입니다.",
        "evaluation": {
            "evaluators": [
                _evaluator("A", (1, 3, 5, 7, 6)),
                _evaluator("B", (3, 5, 7, 5, 4)),
                _evaluator("C", (5, 7, 1, 3, 2)),
            ],
            # Provider overall is provenance only and can use its imprecise
            # 16.7 multiplier. It must not become the C/O/E target.
            "average_score": 100.2,
        },
    }
    row = convert_nikl_summary(
        {
            "alias": "nikl24_summary",
            "document": document,
            "candidate": candidate,
            "candidate_slot": "SC1",
            "available_variant_count": 2,
            "selected_variant_index": 0,
        }
    )

    # Native content rater means are 3, 5 and 13/3. Their grand mean is 37/9;
    # the exact linear 1--7 -> 1--5 map is applied before the rater mean.
    assert row["score"]["content"] == pytest.approx(1 + (37 / 9 - 1) * 4 / 6)
    assert row["score"]["organization"] == pytest.approx(1 + (5 - 1) * 4 / 6)
    assert row["score"]["expression"] == pytest.approx(1 + (4 - 1) * 4 / 6)
    assert "첫 문단.\n\n둘째 문단." in row["prompt"]
    assert row["essay"] == "평가할 요약문입니다."
    assert row["document_id"] == row["id"]
    details = row["score_details"]
    assert details["stored_average_score_0_100"] == pytest.approx(100.2)
    assert len(details["traits"]["content"]["criteria_order"]) == 3
    assert len(details["traits"]["organization"]["criteria_order"]) == 1
    assert len(details["traits"]["expression"]["criteria_order"]) == 1
    assert details["traits"]["content"]["criteria"]["content_1"][
        "source_rater_scores_1to7"
    ]["A"] == 1


def test_new_summary_aliases_are_valid_external_direct_score_sources() -> None:
    aliases = {
        "nikl24_summary",
        "nikl25_argumentative_summary",
        "nikl25_cooperative_summary",
    }
    assert aliases <= set(EXTERNAL_DATASETS)
    config = legacy_config(
        model_id="tri_7b",
        training_mode="lora_only",
        input_format="source_aware_v1",
        essay_surface="flat",
        detail_head_mode="none",
        detail_final_source="direct",
        detail_expected_loss_weight=0.0,
        inbatch_sampling="random",
        dataset_schedule="pretrain_then_competition",
        extended_datasets="nikl24_summary,nikl25_argumentative_summary",
        max_train_steps=549,
        final_competition_epochs=1,
    )
    config.validate()
