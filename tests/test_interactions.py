"""Yeni giriş yollarında veri ayrımı, ses dönüşümü ve cevap geçmişinin korunması."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kt_project.intent import LABELS, UNKNOWN, normalize_text
from kt_project.interactions import InteractionService
from kt_project.media import load_audio, word_error_rate
from kt_project.tracking import Snapshot


class InteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(ROOT / "outputs/interactions/intent_model.joblib")
        cls.rows = json.loads((ROOT / "data/interactions/intent_corpus.json").read_text(encoding="utf-8"))

    def test_vocabulary_and_idf_fit_training_only_with_disjoint_demo(self):
        training = [row["text"] for row in self.rows if row["split"] == "train"]
        fresh = clone(self.model.pipeline.named_steps["tfidf"]).fit(training)
        self.assertEqual(fresh.vocabulary_, self.model.pipeline.named_steps["tfidf"].vocabulary_)
        np.testing.assert_allclose(fresh.idf_, self.model.pipeline.named_steps["tfidf"].idf_)
        normalized = [normalize_text(row["text"]) for row in self.rows]
        self.assertEqual(len(set(normalized)), len(normalized))
        manifest = json.loads((ROOT / "data/interactions/manifest.json").read_text(encoding="utf-8"))
        for sample in manifest["samples"]:
            self.assertNotIn(normalize_text(sample["expected_text"]), normalized)

    def test_acceptance_threshold_is_selected_by_validation_only(self):
        validation = [row for row in self.rows if row["split"] == "validation"]
        options = []
        # Bağımsız kopyada dene; kayıtlı model veya diğer testler değişmesin.
        import copy
        model = copy.deepcopy(self.model)
        for row in self.model.metadata["threshold_candidates"]:
            model.threshold = row["threshold"]
            score = f1_score([r["label"] for r in validation],
                             [model.predict(r["text"])["label"] for r in validation],
                             labels=LABELS, average="macro", zero_division=0)
            options.append((score, model.threshold))
        self.assertEqual(max(options)[1], self.model.threshold)

    def test_unclear_multiple_or_unsupported_requests_are_not_forced_into_skill(self):
        for text in ["", "Matematik çalışmak istiyorum.", "Kesirlerde çarpma çalışalım.",
                     "Kesirler ve tam sayılarda toplama çalışalım.",
                     "Negatif ondalık sayılarda çıkarma çalışalım."]:
            result = self.model.predict(text)
            self.assertFalse(result["accepted"], text)
            self.assertEqual(result["label"], UNKNOWN)
            self.assertIsNone(result["skill"])

    def service_stub(self):
        instance = object.__new__(InteractionService)
        instance.snapshot = Snapshot(0, 0, 5)
        instance.store = SimpleNamespace(arrays={
            "question": np.arange(3, 8), "skills": np.full((5, 1), 81),
            "correct": np.array([1, 0, 1, 0, 1]), "order": np.arange(5),
        })
        instance.intent = self.model
        class Tracer:
            def reference_scores(self, snapshots, skills, bank):
                return [{skill: .72 for skill in row} for row in skills]
            def predict_requests(self, requests, batch_size):
                return np.full(len(requests), .65)
        instance.tracer = Tracer()
        instance.bank = None
        instance.catalog = {i: {"question": i, "skills": [skill], "difficulty": .3}
                            for i, skill in [(100, 81), (200, 82), (300, 84)]}
        instance.config = json.loads((ROOT / "configs/tracking.json").read_text())
        instance.behavior = {"cluster": 0}
        instance.inverse_questions = {100: 1000, 200: 2000, 300: 3000}
        return instance

    def test_accepted_request_selects_requested_existing_skill_without_adding_answer(self):
        service = self.service_stub()
        before = service.store.arrays["correct"].copy()
        result = service.respond("Paydaları farklı kesirlerde toplama ve çıkarma yaparken zorlanıyorum.")
        self.assertTrue(result["intent"]["accepted"])
        self.assertEqual(result["recommendation"]["skill"], 84)
        self.assertEqual(result["recommendation"]["original_problem_id"], 3000)
        self.assertTrue(result["history_unchanged"])
        self.assertFalse(result["response_is_assessed_answer"])
        np.testing.assert_array_equal(before, service.store.arrays["correct"])
        self.assertEqual(service.snapshot.stop, 5)

    def test_rejected_request_does_not_call_recommender(self):
        with patch("kt_project.interactions.recommend_next") as recommender:
            result = self.service_stub().respond("Kesirlerde çarpma yapmak istiyorum.")
        recommender.assert_not_called()
        self.assertIsNone(result["recommendation"])
        self.assertTrue(result["history_unchanged"])

    def test_stereo_audio_is_mixed_and_resampled_to_model_rate(self):
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "audio.wav"
            signal = (.3*np.sin(2*np.pi*440*np.arange(24000)/24000)*32767).astype("<i2")
            with wave.open(str(path), "wb") as stream:
                stream.setparams((2, 2, 24000, 0, "NONE", "not compressed"))
                stream.writeframes(np.stack([signal, signal], axis=1).tobytes())
            samples, meta = load_audio(path)
        self.assertEqual(samples.shape, (16000,))
        self.assertEqual(meta["source_channels"], 2)
        self.assertEqual(meta["source_sample_rate"], 24000)
        self.assertEqual(meta["model_sample_rate"], 16000)
        self.assertAlmostEqual(float(np.sqrt(np.mean(samples**2))), .3/np.sqrt(2), places=3)

    def test_silent_or_short_audio_is_rejected_before_asr(self):
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "empty.wav"
            for frames in (16000, 1000):
                with wave.open(str(path), "wb") as stream:
                    stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    stream.writeframes(np.zeros(frames, dtype="<i2").tobytes())
                with self.assertRaises(ValueError):
                    load_audio(path)

    def test_word_error_metric_counts_substitutions_deletions_and_insertions(self):
        self.assertEqual(word_error_rate("bir iki üç", "bir dört üç"), 1/3)
        self.assertEqual(word_error_rate("bir iki üç", "bir üç"), 1/3)
        self.assertEqual(word_error_rate("bir iki üç", "bir dört iki üç"), 1/3)
        self.assertEqual(word_error_rate("Çıkarma, işlemi!", "cikarma islemi"), 0)
        with self.assertRaises(ValueError):
            word_error_rate("", "bir")


if __name__ == "__main__":
    unittest.main()
