from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from main_code.train import BestCheckpointCallback


class BestCheckpointCallbackTest(unittest.TestCase):
    def test_tracks_both_metrics_and_preserves_selected_legacy_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded = SimpleNamespace(
                config=SimpleNamespace(best_checkpoint_metric="rmse")
            )
            saved: list[Path] = []

            def fake_save_checkpoint(_loaded: object, directory: Path) -> Path:
                target = Path(directory)
                target.mkdir(parents=True, exist_ok=True)
                (target / "config.json").write_text("{}\n", encoding="utf-8")
                saved.append(target)
                return target

            callback = BestCheckpointCallback(loaded, root / "best_checkpoint")
            with patch("main_code.train.save_checkpoint", fake_save_checkpoint):
                callback.on_evaluate(
                    None,
                    SimpleNamespace(epoch=0.25, global_step=10),
                    None,
                    metrics={
                        "eval_overall_rmse": 0.55,
                        "eval_overall_spearman": 0.70,
                    },
                )
                callback.on_evaluate(
                    None,
                    SimpleNamespace(epoch=0.5, global_step=20),
                    None,
                    metrics={
                        "eval_overall_rmse": 0.56,
                        "eval_overall_spearman": 0.73,
                    },
                )
                callback.on_evaluate(
                    None,
                    SimpleNamespace(epoch=0.75, global_step=30),
                    None,
                    metrics={
                        "eval_overall_rmse": 0.50,
                        "eval_overall_spearman": 0.72,
                    },
                )

            self.assertEqual(callback.metric_name, "eval_overall_rmse")
            self.assertEqual(callback.best_metric, 0.50)
            self.assertEqual(callback.best_epoch, 0.75)
            self.assertEqual(callback.best_global_step, 30)

            summary = callback.selection_summary()
            self.assertEqual(summary["rmse"]["metric_value"], 0.50)
            self.assertEqual(summary["rmse"]["global_step"], 30)
            self.assertEqual(summary["spearman"]["metric_value"], 0.73)
            self.assertEqual(summary["spearman"]["global_step"], 20)
            self.assertEqual(
                summary["rmse"]["checkpoint"], str(root / "best_checkpoint_rmse")
            )
            self.assertEqual(
                summary["spearman"]["checkpoint"],
                str(root / "best_checkpoint_spearman"),
            )

            # RMSE improves twice: both the legacy and metric-specific paths move.
            # Spearman improves twice and never rewrites the legacy RMSE path.
            self.assertEqual(saved.count(root / "best_checkpoint"), 2)
            self.assertEqual(saved.count(root / "best_checkpoint_rmse"), 2)
            self.assertEqual(saved.count(root / "best_checkpoint_spearman"), 2)

            legacy = json.loads(
                (root / "best_checkpoint" / "selection.json").read_text(
                    encoding="utf-8"
                )
            )
            rmse = json.loads(
                (root / "best_checkpoint_rmse" / "selection.json").read_text(
                    encoding="utf-8"
                )
            )
            spearman = json.loads(
                (root / "best_checkpoint_spearman" / "selection.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(legacy, rmse)
            self.assertEqual(legacy["global_step"], 30)
            self.assertEqual(spearman["global_step"], 20)

    def test_spearman_selection_controls_legacy_path_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded = SimpleNamespace(
                config=SimpleNamespace(best_checkpoint_metric="spearman")
            )

            def fake_save_checkpoint(_loaded: object, directory: Path) -> Path:
                target = Path(directory)
                target.mkdir(parents=True, exist_ok=True)
                return target

            callback = BestCheckpointCallback(loaded, root / "best_checkpoint")
            with patch("main_code.train.save_checkpoint", fake_save_checkpoint):
                callback.on_evaluate(
                    None,
                    SimpleNamespace(epoch=0.5, global_step=10),
                    None,
                    metrics={
                        "eval_overall_rmse": 0.55,
                        "eval_overall_spearman": 0.75,
                    },
                )
                callback.on_evaluate(
                    None,
                    SimpleNamespace(epoch=1.0, global_step=20),
                    None,
                    metrics={
                        "eval_overall_rmse": 0.50,
                        "eval_overall_spearman": 0.74,
                    },
                )

            self.assertEqual(callback.metric_name, "eval_overall_spearman")
            self.assertEqual(callback.best_metric, 0.75)
            legacy = json.loads(
                (root / "best_checkpoint" / "selection.json").read_text(
                    encoding="utf-8"
                )
            )
            rmse = json.loads(
                (root / "best_checkpoint_rmse" / "selection.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(legacy["metric"], "eval_overall_spearman")
            self.assertEqual(legacy["global_step"], 10)
            self.assertEqual(rmse["global_step"], 20)

    def test_trait_specialist_can_select_organization_rmse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded = SimpleNamespace(
                config=SimpleNamespace(best_checkpoint_metric="organization_rmse")
            )

            def fake_save_checkpoint(_loaded: object, directory: Path) -> Path:
                target = Path(directory)
                target.mkdir(parents=True, exist_ok=True)
                return target

            callback = BestCheckpointCallback(loaded, root / "best_checkpoint")
            with patch("main_code.train.save_checkpoint", fake_save_checkpoint):
                callback.on_evaluate(
                    None,
                    SimpleNamespace(epoch=0.5, global_step=10),
                    None,
                    metrics={
                        "eval_overall_rmse": 1.20,
                        "eval_overall_spearman": 0.10,
                        "eval_organization_rmse": 0.64,
                    },
                )
                callback.on_evaluate(
                    None,
                    SimpleNamespace(epoch=1.0, global_step=20),
                    None,
                    metrics={
                        "eval_overall_rmse": 1.30,
                        "eval_overall_spearman": 0.09,
                        "eval_organization_rmse": 0.61,
                    },
                )

            self.assertEqual(callback.metric_name, "eval_organization_rmse")
            self.assertEqual(callback.best_metric, 0.61)
            self.assertEqual(callback.best_global_step, 20)
            self.assertEqual(
                callback.selection_summary()["organization_rmse"]["global_step"],
                20,
            )

    def test_mixed_metric_pair_tracks_min_rmse_and_max_spearman(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded = SimpleNamespace(
                config=SimpleNamespace(
                    best_checkpoint_metric="submitted_trait_macro_rmse"
                )
            )

            def fake_save_checkpoint(_loaded: object, directory: Path) -> Path:
                target = Path(directory)
                target.mkdir(parents=True, exist_ok=True)
                return target

            callback = BestCheckpointCallback(loaded, root / "best_checkpoint")
            metric_rows = (
                (10, 0.60, 0.70),
                (20, 0.62, 0.75),
                (30, 0.55, 0.74),
            )
            with patch("main_code.train.save_checkpoint", fake_save_checkpoint):
                for step, rmse, rho in metric_rows:
                    callback.on_evaluate(
                        None,
                        SimpleNamespace(epoch=step / 10, global_step=step),
                        None,
                        metrics={
                            "eval_submitted_trait_macro_rmse": rmse,
                            "eval_official_matched_spearman": rho,
                        },
                    )

            summary = callback.selection_summary()
            self.assertEqual(
                summary["submitted_trait_macro_rmse"]["global_step"], 30
            )
            self.assertEqual(
                summary["official_matched_spearman"]["global_step"], 20
            )
            legacy = json.loads(
                (root / "best_checkpoint" / "selection.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(legacy["metric"], "eval_submitted_trait_macro_rmse")
            self.assertEqual(legacy["global_step"], 30)


if __name__ == "__main__":
    unittest.main()
