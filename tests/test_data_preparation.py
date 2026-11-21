"""Gelecek cevaplar, kopyalar ve öğrenci ayrımı için anlamlı veri kontrolleri."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kt_project.prepare_data import load_clean_events, prepare_dataset


class PreparationTests(unittest.TestCase):
    @staticmethod
    def write_csv(path, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["user_id", "problem_id", "order_id",
                "skill_id", "correct", "ms_first_response", "hint_count"])
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def sample_rows():
        return [{"user_id": student, "problem_id": step if step < 4 else 1000+student,
                 "order_id": student*10+step, "skill_id": "10_11" if step == 0 else "10",
                 "correct": (student+step) % 2, "ms_first_response": 1000+step,
                 "hint_count": 0} for student in range(20) for step in range(5)]

    def test_test_answers_do_not_change_training_difficulty_or_references(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            rows = self.sample_rows()
            self.write_csv(base / "a.csv", rows)
            prepare_dataset(base / "a.csv", base / "a")
            splits = json.loads((base / "a" / "student_splits.json").read_text())
            self.assertFalse(set(splits["train"]) & set(splits["test"]))
            self.assertFalse(set(splits["train"]) & set(splits["validation"]))
            self.assertFalse(set(splits["validation"]) & set(splits["test"]))
            heldout = set(splits["test"]) | set(splits["validation"])
            for row in rows:
                if row["user_id"] in heldout:
                    row["correct"] = 1-int(row["correct"])
            self.write_csv(base / "b.csv", rows)
            prepare_dataset(base / "b.csv", base / "b")
            with np.load(base / "a" / "interactions.npz") as a, np.load(base / "b" / "interactions.npz") as b:
                np.testing.assert_array_equal(a["difficulty"], b["difficulty"])
                np.testing.assert_array_equal(a["split"], b["split"])
                self.assertTrue((a["question"][a["split"] == 2] == 1).any())
            self.assertEqual((base / "a" / "references.json").read_bytes(),
                             (base / "b" / "references.json").read_bytes())

    def test_multiple_skills_are_one_event_and_conflicts_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            rows = self.sample_rows()
            duplicate = dict(rows[0], skill_id="12")
            conflict = dict(rows[1], correct=1-int(rows[1]["correct"]))
            self.write_csv(path, rows + [duplicate, conflict])
            events = load_clean_events(path, min_interactions=1)
            selected = [event for event in events if event.student == 0]
            self.assertEqual(len(selected), 4)
            self.assertEqual(selected[0].skills, (10, 11, 12))
            self.assertFalse(any(event.order == 1 for event in selected))


if __name__ == "__main__":
    unittest.main()
