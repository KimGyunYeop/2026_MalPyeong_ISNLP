from __future__ import annotations

from main_code.train import filter_external_rows_by_purpose


def test_external_purpose_filter_uses_prepared_metadata() -> None:
    rows = [
        {"id": "p", "metadata": {"purpose": "설득"}},
        {"id": "e", "metadata": {"purpose": "설명"}},
        {"id": "missing"},
    ]

    assert filter_external_rows_by_purpose(rows, "설득") == [rows[0]]
    assert filter_external_rows_by_purpose(rows, "") is rows
