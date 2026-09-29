from __future__ import annotations

import unittest

from main_code.config import TRAITS
from main_code.train import align_external_scores_to_competition

POLICY = "quantile_to_competition_v1"


def _row(item_id: str, source: str, content: float) -> dict:
    # organization/expression은 content와 같은 값으로 두어 trait별 사상만 본다.
    return {
        "id": item_id,
        "_dataset_name": source,
        "prompt_num": "Q1",
        "prompt": "논제",
        "essay": f"{item_id} 본문",
        "score": {
            "content": content,
            "organization": content,
            "expression": content,
            "average": content,
        },
    }


def _competition(values: list[float]) -> list[dict]:
    return [
        {
            "id": f"c{index}",
            "dataset_group": "competition",
            "score": {trait: value for trait in TRAITS},
        }
        for index, value in enumerate(values)
    ]


class ExternalScoreAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        # 대회 label 분포는 1~5의 중간 대역에 몰려 있다.
        self.competition = _competition([2.0, 2.5, 3.0, 3.5, 4.0])
        # 외부 corpus는 고득점 편향이 크고 범위도 좁다.
        self.external = [
            _row("e1", "aihub24_essay", 4.6),
            _row("e2", "aihub24_essay", 4.8),
            _row("e3", "aihub24_essay", 4.8),
            _row("e4", "aihub24_essay", 5.0),
        ]

    def test_none_policy_returns_the_same_rows(self) -> None:
        result = align_external_scores_to_competition(
            self.external, self.competition, "none"
        )
        self.assertIs(result, self.external)

    def test_unknown_policy_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "external_score_alignment"):
            align_external_scores_to_competition(
                self.external, self.competition, "quantile_v99"
            )

    def test_scores_land_inside_the_competition_range(self) -> None:
        aligned = align_external_scores_to_competition(
            self.external, self.competition, POLICY
        )
        values = [row["score"]["content"] for row in aligned]
        self.assertEqual(min(values), 2.0)
        self.assertEqual(max(values), 4.0)
        for value in values:
            self.assertGreaterEqual(value, 2.0)
            self.assertLessEqual(value, 4.0)

    def test_order_and_ties_are_preserved(self) -> None:
        aligned = align_external_scores_to_competition(
            self.external, self.competition, POLICY
        )
        by_id = {row["id"]: row["score"]["content"] for row in aligned}
        self.assertLess(by_id["e1"], by_id["e2"])
        # 원래 동점이던 두 행은 같은 분위수로 가야 한다.
        self.assertEqual(by_id["e2"], by_id["e3"])
        self.assertLess(by_id["e3"], by_id["e4"])

    def test_input_rows_are_not_mutated(self) -> None:
        before = [dict(row["score"]) for row in self.external]
        align_external_scores_to_competition(self.external, self.competition, POLICY)
        after = [dict(row["score"]) for row in self.external]
        self.assertEqual(before, after)

    def test_average_follows_the_aligned_traits(self) -> None:
        aligned = align_external_scores_to_competition(
            self.external, self.competition, POLICY
        )
        for row in aligned:
            scores = row["score"]
            expected = sum(scores[trait] for trait in TRAITS) / len(TRAITS)
            self.assertAlmostEqual(scores["average"], expected, places=9)

    def test_each_source_is_aligned_independently(self) -> None:
        # 두 source의 절대 척도가 다르지만 각자 대회 분포로 사상되어야 한다.
        rows = [
            _row("a1", "aihub24_essay", 4.6),
            _row("a2", "aihub24_essay", 5.0),
            _row("b1", "nikl24_summary", 1.0),
            _row("b2", "nikl24_summary", 7.0),
        ]
        aligned = align_external_scores_to_competition(rows, self.competition, POLICY)
        by_id = {row["id"]: row["score"]["content"] for row in aligned}
        self.assertEqual(by_id["a1"], 2.0)
        self.assertEqual(by_id["a2"], 4.0)
        self.assertEqual(by_id["b1"], 2.0)
        self.assertEqual(by_id["b2"], 4.0)

    def test_single_external_row_maps_to_the_distribution_start(self) -> None:
        aligned = align_external_scores_to_competition(
            [_row("only", "aihub24_essay", 4.9)], self.competition, POLICY
        )
        self.assertEqual(aligned[0]["score"]["content"], 2.0)


if __name__ == "__main__":
    unittest.main()
