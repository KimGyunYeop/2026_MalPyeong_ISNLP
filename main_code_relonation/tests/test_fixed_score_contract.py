from __future__ import annotations

import json

from main_code_relonation.infer import build_submission_record


def test_direct_scorer_float_values_survive_rationale_pass_exactly() -> None:
    row = {"id": "e1", "prompt": "논제", "essay": "글"}
    fixed = {
        "content": 3.487192153930664,
        "organization": 2.991230010986328,
        "expression": 4.000000476837158,
    }
    raw = json.dumps(
        {
            "content": {"score": 5, "rationale": "주장을 이유와 연결했다."},
            "organization": {"score": 1, "rationale": "도입과 결론의 기능이 보인다."},
            "expression": {"score": 2, "rationale": "문장이 대체로 자연스럽다."},
        },
        ensure_ascii=False,
    )
    record, generated = build_submission_record(row, raw, fixed_scores=fixed)
    assert generated == {"content": 5.0, "organization": 1.0, "expression": 2.0}
    assert {
        trait: record["judge"][trait]["score"] for trait in fixed
    } == fixed
