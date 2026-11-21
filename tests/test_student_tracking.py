"""Öğrenci göstergelerinde zaman sınırı ve öneri kurallarının kontrolleri."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kt_project.models import KnowledgeTracer, LargeKnowledgeModel, ModelConfig
from kt_project.patterns import behavior_vector, fit_observation_limits, training_snapshots
from kt_project.recommendations import challenge_probability, recommend_next, training_question_catalog
from kt_project.replay import observed_skill_rates, choose_replay_student
from kt_project.sequences import InteractionStore
from kt_project.tracking import FrozenTracer, ReferenceBank, snapshot_for


def toy_store(path):
    np.savez(path, student=np.repeat(np.arange(3), 8), question=np.tile(np.arange(3, 11), 3),
             skills=np.tile([2, 3], (24, 1)), correct=np.arange(24) % 2,
             difficulty=np.full(24, .4, dtype=np.float32), split=np.repeat([0, 0, 1], 8),
             duration_ms=np.arange(24, dtype=np.float32)*1000+1000,
             hints=np.zeros(24, dtype=np.float32))
    return InteractionStore(path)


def toy_tracer(store):
    instance = object.__new__(FrozenTracer)
    instance.config = ModelConfig(30, 10, hidden_size=16, layers=1, attention_heads=4,
                                  feedforward_size=32, max_history=8, prompt_length=2, dropout=0)
    instance.model = KnowledgeTracer(LargeKnowledgeModel(instance.config), "kmpt").eval()
    instance.model.requires_grad_(False)
    instance.device, instance.precision, instance.skill_width = torch.device("cpu"), "float32", 2
    instance.store = store
    instance.arrays = {key: torch.as_tensor(store.arrays[source], dtype=dtype).clone()
                       for key, source, dtype in [("question", "question", torch.long),
                           ("skills", "skills", torch.long), ("response", "correct", torch.long),
                           ("difficulty", "difficulty", torch.float32)]}
    return instance


class StudentTrackingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.store = toy_store(self.path / "data.npz")
        self.config = json.loads((ROOT / "configs/tracking.json").read_text())

    def tearDown(self):
        self.temp.cleanup()

    def test_future_answers_do_not_change_reference_prediction(self):
        tracer = toy_tracer(self.store)
        snapshot = snapshot_for(self.store, 0, 5)
        question = {"question": 7, "skills": [2, 3], "difficulty": .4}
        before = tracer.predict_requests([(snapshot, question)])
        tracer.arrays["response"][5:] = 1-tracer.arrays["response"][5:]
        after = tracer.predict_requests([(snapshot, question)])
        np.testing.assert_array_equal(before, after)
        self.assertNotIn("label", tracer.query_batch([(snapshot, question)]))

    def test_reference_bank_is_fixed_and_score_is_mean_of_same_questions(self):
        path = self.path / "references.json"
        path.write_text(json.dumps({"2": [{"question": q, "skills": [2], "difficulty": .4}
                                          for q in [3, 5, 7]]}))
        bank = ReferenceBank(path)
        changed = bank.for_skill(2)
        changed[0]["question"] = 29
        self.assertEqual(bank.for_skill(2)[0]["question"], 3)
        tracer = toy_tracer(self.store)
        snapshot = snapshot_for(self.store, 0, 5)
        expected = tracer.predict_requests([(snapshot, item) for item in bank.for_skill(2)]).mean()
        actual = tracer.reference_scores([snapshot], [[2]], bank)[0][2]
        self.assertAlmostEqual(actual, float(expected), places=7)

    def test_observation_features_and_limits_exclude_future_and_validation(self):
        snapshot = snapshot_for(self.store, 0, 5)
        limits = fit_observation_limits(self.store)
        before = behavior_vector(self.store, snapshot, {2: .6}, limits)
        self.store.arrays["correct"][5:] = 1-self.store.arrays["correct"][5:]
        self.store.arrays["duration_ms"][16:] = 1e12
        self.store.arrays["hints"][16:] = 999
        np.testing.assert_array_equal(before, behavior_vector(self.store, snapshot, {2: .6}, limits))
        self.assertEqual(limits, fit_observation_limits(self.store))

    def test_snapshots_and_catalog_only_use_training(self):
        self.store.arrays["question"][16:] = 29
        snapshots = training_snapshots(self.store, 5, 4)
        self.assertEqual({snapshot.student for snapshot in snapshots}, {0, 1})
        self.assertTrue(all(snapshot.stop <= self.store.spans[snapshot.student][1] for snapshot in snapshots))
        self.assertNotIn(29, training_question_catalog(self.store))

    def test_recommendation_avoids_seen_questions_and_uses_success_rule(self):
        self.store.arrays["correct"][:5] = 1
        snapshot = snapshot_for(self.store, 0, 5)
        class ConstantTracer:
            def predict_requests(self, requests, batch_size):
                return np.array([.55 for _ in requests])
        result = recommend_next(self.store, snapshot, {2: .4, 3: .7}, ConstantTracer(),
                                training_question_catalog(self.store), self.config)
        self.assertTrue(result["available"])
        self.assertEqual(result["skill"], 2)
        self.assertNotIn(result["question"], self.store.arrays["question"][:5])
        self.assertEqual(result["target_probability"], .55)
        self.assertFalse(result["rule_is_learned_policy"])
        self.assertEqual(challenge_probability([0, 0, 0], self.config)[0], .8)

    def test_invalid_history_boundaries_are_rejected(self):
        with self.assertRaises(ValueError):
            snapshot_for(self.store, 0, 9)

    def test_displayed_observed_ratio_uses_only_skill_answers_in_history(self):
        snapshot = snapshot_for(self.store, 0, 5)
        rates = observed_skill_rates(self.store, snapshot, [2, 9], 20)
        self.assertEqual(rates[2]["answers"], 5)
        self.assertAlmostEqual(rates[2]["ratio"], .4)
        self.assertEqual(rates[9]["answers"], 0)
        self.assertIsNone(rates[9]["ratio"])

    def test_demo_student_selection_is_independent_of_response_outcomes(self):
        config = {**self.config, "replay_steps": 2}
        before = choose_replay_student(self.store, config)
        self.store.arrays["correct"][:] = 1-self.store.arrays["correct"]
        self.assertEqual(before, choose_replay_student(self.store, config))


if __name__ == "__main__":
    unittest.main()
