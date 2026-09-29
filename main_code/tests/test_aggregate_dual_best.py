from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from main_code.aggregate_results import (
    ALTERNATE_FIELDS,
    collect,
    summarize_run,
    write_csv,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def metrics(rmse: float, spearman: float, *, count: int = 2) -> dict[str, object]:
    traits = {
        "content": {"rmse": rmse + 0.01, "spearman": spearman - 0.01},
        "organization": {"rmse": rmse + 0.02, "spearman": spearman - 0.02},
        "expression": {"rmse": rmse + 0.03, "spearman": spearman - 0.03},
    }
    prompt_traits = {
        name: {"rmse": block["rmse"] + 0.1, "spearman": block["spearman"] - 0.1}
        for name, block in traits.items()
    }
    return {
        "count": count,
        "overall": {"rmse": rmse, "spearman": spearman},
        "traits": traits,
        "prompt_macro": {
            "overall": {"rmse": rmse + 0.1, "spearman": spearman - 0.1},
            "traits": prompt_traits,
        },
    }


class AggregateDualBestTest(unittest.TestCase):
    def make_run(self, root: Path, name: str, primary_rmse: float) -> Path:
        run_dir = root / "proposed" / "base" / name / "fake" / "config"
        write_json(
            run_dir / "resolved_config.json",
            {
                "model_id": "example/fake",
                "model_slug": "fake",
                "best_checkpoint_metric": "rmse",
            },
        )
        write_json(
            run_dir / "experiment.json",
            {
                "suite": "proposed",
                "base_name": "base",
                "case_name": name,
                "model_slug": "fake",
            },
        )
        write_json(run_dir / "run.json", {"training_rows": 2, "trainer": {}})
        write_json(run_dir / "eval/metrics.json", metrics(primary_rmse, 0.70))
        (run_dir / "eval/predictions.jsonl").write_text(
            '{"essay_id":"a"}\n{"essay_id":"b"}\n', encoding="utf-8"
        )
        write_json(
            run_dir / "eval/inference_manifest.json",
            {
                "checkpoint": str(run_dir / "best_checkpoint"),
                "checkpoint_type": "best_checkpoint",
            },
        )
        return run_dir

    def test_exposes_opposite_metric_eval_without_changing_primary_rank(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            alternate_run = self.make_run(root, "with_alternate", 0.60)
            self.make_run(root, "primary_only", 0.50)

            write_json(alternate_run / "best_checkpoint_spearman/config.json", {})
            write_json(
                alternate_run / "best_checkpoint_spearman/selection.json",
                {
                    "metric": "eval_overall_spearman",
                    "metric_value": 0.81,
                    "epoch": 1.5,
                    "global_step": 42,
                },
            )
            write_json(
                alternate_run / "eval_best_spearman/metrics.json",
                metrics(0.40, 0.81),
            )
            # Deliberate mismatch exercises the secondary-only warning.
            (alternate_run / "eval_best_spearman/predictions.jsonl").write_text(
                '{"essay_id":"a"}\n', encoding="utf-8"
            )
            write_json(
                alternate_run / "eval_best_spearman/inference_manifest.json",
                {
                    "checkpoint": str(alternate_run / "best_checkpoint_spearman"),
                    "checkpoint_type": "custom_checkpoint",
                },
            )

            rows = collect(root)
            by_case = {row["case_name"]: row for row in rows}
            alternate = by_case["with_alternate"]
            primary_only = by_case["primary_only"]

            # Ranking and primary metrics continue to use eval/ only.  The
            # spectacular alternate RMSE must not reorder these rows.
            self.assertEqual(primary_only["rank"], 1)
            self.assertEqual(alternate["rank"], 2)
            self.assertEqual(alternate["overall_rmse"], 0.60)
            self.assertEqual(alternate["alternate_overall_rmse"], 0.40)
            self.assertEqual(alternate["alternate_overall_spearman"], 0.81)
            self.assertAlmostEqual(alternate["alternate_content_rmse"], 0.41)
            self.assertAlmostEqual(
                alternate["alternate_organization_spearman"], 0.79
            )
            self.assertAlmostEqual(alternate["alternate_expression_rmse"], 0.43)
            self.assertAlmostEqual(
                alternate["alternate_prompt_macro_overall_rmse"], 0.50
            )
            self.assertEqual(
                alternate["alternate_checkpoint_metric_name"],
                "eval_overall_spearman",
            )
            self.assertEqual(alternate["alternate_checkpoint_metric_value"], 0.81)
            self.assertEqual(alternate["alternate_checkpoint_epoch"], 1.5)
            self.assertEqual(alternate["alternate_checkpoint_step"], 42)
            self.assertEqual(
                alternate["alternate_checkpoint_type"], "best_checkpoint_spearman"
            )
            self.assertEqual(alternate["alternate_prediction_records"], 1)
            self.assertEqual(alternate["alternate_metric_count"], 2)
            self.assertEqual(alternate["alternate_infer_status"], "complete")
            self.assertIsNone(alternate["alternate_infer_exit_code"])
            self.assertIn("secondary metric count 2 != predictions 1", alternate["warnings"])

            self.assertTrue(
                all(primary_only[field] is None for field in ALTERNATE_FIELDS)
            )
            self.assertEqual(primary_only["status"], "complete")

            csv_path = root / "summary.csv"
            write_csv(csv_path, rows)
            with csv_path.open(encoding="utf-8", newline="") as stream:
                csv_rows = list(csv.DictReader(stream))
            csv_by_case = {row["case_name"]: row for row in csv_rows}
            self.assertEqual(
                csv_by_case["primary_only"]["alternate_overall_rmse"], ""
            )
            self.assertEqual(
                csv_by_case["with_alternate"]["alternate_overall_rmse"],
                "0.4",
            )

    def test_ignores_orphan_secondary_eval_without_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self.make_run(root, "orphan", 0.50)
            write_json(
                run_dir / "eval_best_spearman/metrics.json", metrics(0.40, 0.80)
            )

            row = summarize_run(run_dir, root, None)

            metric_fields = [
                field
                for field in ALTERNATE_FIELDS
                if field not in {"alternate_infer_status", "alternate_infer_exit_code"}
            ]
            self.assertTrue(all(row[field] is None for field in metric_fields))
            self.assertEqual(row["alternate_infer_status"], "incomplete")
            self.assertIn(
                "secondary eval은 있으나 checkpoint 없음: best_checkpoint_spearman",
                row["warnings"],
            )

    def test_secondary_failure_status_does_not_change_primary_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self.make_run(root, "secondary_failed", 0.50)
            write_json(run_dir / "best_checkpoint_spearman/config.json", {})
            (run_dir / "status.txt").write_text("infer_failed\n", encoding="utf-8")
            (run_dir / "secondary_infer_exit_code.txt").write_text(
                "17\n", encoding="utf-8"
            )
            write_json(
                run_dir / "status_context.json",
                {"status": "infer_failed", "phase": "secondary_infer"},
            )

            row = summarize_run(run_dir, root, None)

            self.assertEqual(row["status"], "complete")
            self.assertEqual(row["infer_status"], "complete")
            self.assertEqual(row["alternate_infer_status"], "failed")
            self.assertEqual(row["alternate_infer_exit_code"], 17)
            self.assertIsNone(row["alternate_overall_rmse"])
            self.assertIn("secondary infer exit=17", row["warnings"])


if __name__ == "__main__":
    unittest.main()
