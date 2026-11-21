"""Sonuçların tamlığı ve testten bağımsız örnek model seçimi."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kt_project.experiment_results import aggregate_runs, select_prototype, audit_main_run


def sample_runs():
    return [{"method": method, "seed": seed, "training_seconds": 10,
             "validation": {"bce": val}, "best_epoch": 1,
             "test": {"auc": auc, "accuracy": .7, "bce": .5}}
            for method in ("baseline", "kpt", "kmpt")
            for seed, val, auc in ((17, .5, .7), (42, .4, .8), (2026, .6, .9))]


class ExperimentResultsTests(unittest.TestCase):
    def test_sample_standard_deviation_uses_three_independent_seeds(self):
        summary = aggregate_runs(sample_runs())
        self.assertAlmostEqual(summary["kpt"]["test"]["auc"]["mean"], .8)
        self.assertAlmostEqual(summary["kpt"]["test"]["auc"]["sample_sd"], .1)

    def test_missing_or_duplicated_run_is_rejected(self):
        runs = sample_runs()
        with self.assertRaises(ValueError):
            aggregate_runs(runs[:-1])
        with self.assertRaises(ValueError):
            aggregate_runs(runs[:-1] + [runs[0]])

    def test_prototype_selection_is_unchanged_when_test_scores_change(self):
        runs = sample_runs()
        policy = {"prototype_method": "kmpt"}
        original = select_prototype(runs, policy)
        changed = copy.deepcopy(runs)
        for row in changed:
            row["test"]["auc"] = 1 if row["seed"] == 17 else 0
        self.assertEqual(original, select_prototype(changed, policy))
        self.assertEqual(original["seed"], 42)
        self.assertFalse(original["test_metrics_used_for_selection"])


class SavedExperimentAuditTests(unittest.TestCase):
    """Yalnızca geçici dizinde üretilen küçük test kayıtlarıyla hata yakalama."""

    def make_fixture(self, root):
        import torch
        from kt_project.training import state_digest
        run = root / "run"
        run.mkdir()
        processed = root / "data/processed"
        processed.mkdir(parents=True)
        students = np.repeat(np.arange(6), 2)
        splits = np.repeat([0, 0, 1, 2, 2, 2], 2)
        labels = np.array([0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1], dtype=np.int8)
        questions = np.full(12, 3)
        questions[11] = 1
        np.savez(processed / "interactions.npz", student=students, split=splits,
                 correct=labels, question=questions)
        (processed / "metadata.json").write_text(json.dumps({"source_sha256": "fixture"}))
        paths = ["src/kt_project/models.py", "src/kt_project/sequences.py",
                 "src/kt_project/training.py", "src/kt_project/prepare_data.py"]
        hashes = {}
        for name in paths:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# temporary test fixture\n")
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        config = {"seeds": [17, 42, 2026], "evaluate_test": True,
                  "train_student_limit": None, "validation_student_limit": None,
                  "train_target_limit": None, "validation_target_limit": None}
        (run / "config.json").write_text(json.dumps(config))
        (run / "status.json").write_text(json.dumps({"status": "complete"}))
        probabilities = np.array([.2, .8, .7], dtype=np.float32)
        y = labels[[7, 9, 11]]
        bce = float(-(y*np.log(probabilities)+(1-y)*np.log1p(-probabilities)).mean())
        results = {"source_sha256": "fixture", "test_evaluated": True,
                   "model_config": {"hidden_size": 2}, "implementation_sha256": hashes, "runs": []}
        for seed in config["seeds"]:
            folder = run / f"seed_{seed}"
            folder.mkdir()
            state = {"weights": torch.tensor([float(seed)])}
            digest = state_digest(state)
            torch.save({"model_state": state}, folder / "pretrained.pt")
            for method in ("baseline", "kpt", "kmpt"):
                row = {"method": method, "seed": seed, "best_epoch": 2, "training_seconds": 1,
                       "training_target_count": 2, "validation_target_count": 1,
                       "initial_lkm_sha256": digest, "validation": {"bce": .5},
                       "test": {"samples": 3, "auc": 1., "accuracy": 1., "bce": bce,
                                "known_question_subset": {"samples": 2, "auc": 1.}}}
                results["runs"].append(row)
                np.savez(folder / f"{method}_test_predictions.npz", label=y,
                         probability=probabilities, row_index=np.array([7, 9, 11]))
                (folder / f"{method}_history.json").write_text(json.dumps([
                    {"validation": {"bce": .6}}, {"validation": {"bce": .5}}]))
                torch.save({"method": method, "seed": seed, "best_epoch": 2,
                            "initial_lkm_sha256": digest, "config": {"hidden_size": 2},
                            "model_state": {"weight": torch.ones(2)}}, folder / f"{method}_best.pt")
        (run / "results.json").write_text(json.dumps(results))
        return run

    def test_complete_fixture_is_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.make_fixture(root)
            _, _, integrity = audit_main_run(root, run)
            self.assertTrue(integrity["all_checks_passed"])
            self.assertEqual(integrity["complete_method_seed_pairs"], 9)
            self.assertEqual(integrity["test_targets"], 3)

    def test_wrong_ground_truth_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.make_fixture(root)
            path = run / "seed_17/kpt_test_predictions.npz"
            np.savez(path, label=np.array([1, 1, 1]), probability=np.array([.2, .8, .7]),
                     row_index=np.array([7, 9, 11]))
            with self.assertRaises(AssertionError):
                audit_main_run(root, run)

    def test_nonfinite_saved_weights_are_rejected(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.make_fixture(root)
            path = run / "seed_17/baseline_best.pt"
            saved = torch.load(path, weights_only=True)
            saved["model_state"]["weight"][0] = float("inf")
            torch.save(saved, path)
            with self.assertRaises(ValueError):
                audit_main_run(root, run)


if __name__ == "__main__":
    unittest.main()
