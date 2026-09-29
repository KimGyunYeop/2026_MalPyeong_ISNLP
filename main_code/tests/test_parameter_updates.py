from __future__ import annotations

import unittest

import torch
from torch import nn

from main_code.train import (
    capture_parameter_update_snapshot,
    summarize_parameter_updates,
)


class FakeScorer(nn.Module):
    """Small scorer with the same score-side/LoRA ownership boundary."""

    def __init__(self, *, with_lora: bool = True) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.base_projection = nn.Linear(2, 2, bias=False)
        if with_lora:
            self.backbone.lora_projection = nn.Linear(2, 2, bias=False)
        self.heads = nn.Linear(2, 3)

    def scoring_parameters(self):
        return self.heads.parameters()


class FakeDetailScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.heads = nn.ModuleDict({"content": nn.Linear(2, 1)})
        self.detail_heads = nn.ModuleDict({"content_1": nn.Linear(2, 1)})
        self.detail_evaluator_severity = nn.Parameter(torch.zeros(2, 1))

    def scoring_parameters(self):
        yield from self.heads.parameters()
        yield from self.detail_heads.parameters()
        yield self.detail_evaluator_severity


class ParameterUpdateDiagnosticsTest(unittest.TestCase):
    def test_reports_changed_score_side_and_unchanged_lora(self) -> None:
        model = FakeScorer()
        snapshot = capture_parameter_update_snapshot(model)

        self.assertTrue(snapshot["score_side"])
        self.assertTrue(snapshot["lora"])
        self.assertTrue(
            all(
                tensor.device.type == "cpu"
                for group in snapshot.values()
                for tensor in group.values()
            )
        )

        with torch.no_grad():
            model.heads.weight.add_(0.25)
            # A frozen backbone change must not accidentally enter either
            # diagnostic group.
            model.backbone.base_projection.weight.add_(10.0)

        summary = summarize_parameter_updates(model, snapshot)
        score_side = summary["score_side"]
        lora = summary["lora"]

        self.assertEqual(summary["snapshot_device"], "cpu")
        self.assertEqual(score_side["tensor_count"], 2)
        self.assertEqual(score_side["parameter_count"], 9)
        self.assertTrue(score_side["changed"])
        self.assertGreater(score_side["delta_l2"], 0.0)
        self.assertAlmostEqual(score_side["max_abs_delta"], 0.25)
        self.assertTrue(summary["direct_heads"]["changed"])
        self.assertEqual(summary["direct_heads"]["tensor_count"], 2)
        self.assertEqual(summary["detail_heads"]["tensor_count"], 0)
        self.assertEqual(summary["detail_rater_severity"]["tensor_count"], 0)

        self.assertEqual(lora["tensor_count"], 1)
        self.assertEqual(lora["parameter_count"], 4)
        self.assertFalse(lora["changed"])
        self.assertEqual(lora["delta_l2"], 0.0)
        self.assertEqual(lora["max_abs_delta"], 0.0)

    def test_reports_changed_lora_and_empty_head_only_group(self) -> None:
        model = FakeScorer()
        snapshot = capture_parameter_update_snapshot(model)
        with torch.no_grad():
            model.backbone.lora_projection.weight.sub_(0.125)
        summary = summarize_parameter_updates(model, snapshot)
        self.assertTrue(summary["lora"]["changed"])
        self.assertGreater(summary["lora"]["delta_l2"], 0.0)
        self.assertFalse(summary["score_side"]["changed"])

        head_only = FakeScorer(with_lora=False)
        head_only_summary = summarize_parameter_updates(
            head_only, capture_parameter_update_snapshot(head_only)
        )
        self.assertEqual(
            head_only_summary["lora"],
            {
                "parameter_count": 0,
                "tensor_count": 0,
                "initial_l2": None,
                "final_l2": None,
                "delta_l2": None,
                "max_abs_delta": None,
                "changed": False,
            },
        )

    def test_separates_dormant_direct_detail_and_rater_updates(self) -> None:
        model = FakeDetailScorer()
        snapshot = capture_parameter_update_snapshot(model)
        with torch.no_grad():
            model.detail_heads["content_1"].weight.add_(0.5)
            model.detail_evaluator_severity.add_(0.25)

        summary = summarize_parameter_updates(model, snapshot)

        self.assertFalse(summary["direct_heads"]["changed"])
        self.assertTrue(summary["detail_heads"]["changed"])
        self.assertTrue(summary["detail_rater_severity"]["changed"])
        self.assertTrue(summary["score_side"]["changed"])


if __name__ == "__main__":
    unittest.main()
