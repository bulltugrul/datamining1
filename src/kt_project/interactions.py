"""Beceri talebini mevcut öğrenci geçmişiyle gerekçeli öneriye bağla."""

import hashlib
import json
from pathlib import Path

import joblib

from .patterns import behavior_vector
from .recommendations import recommend_next, training_question_catalog
from .replay import observed_skill_rates
from .sequences import InteractionStore
from .tracking import active_skills, FrozenTracer, ReferenceBank, snapshot_for


def history_digest(store, snapshot):
    digest = hashlib.sha256()
    for key in ("question", "skills", "correct", "order"):
        digest.update(store.arrays[key][snapshot.start:snapshot.stop].tobytes())
    return digest.hexdigest()


class InteractionService:
    def __init__(self, root):
        self.root = Path(root)
        self.store = InteractionStore(self.root / "data/processed/interactions.npz")
        self.tracer = FrozenTracer(self.root, self.store)
        self.bank = ReferenceBank(self.root / "data/processed/references.json")
        self.intent = joblib.load(self.root / "outputs/interactions/intent_model.joblib")
        interaction_config = json.loads((self.root / "configs/interactions.json").read_text(encoding="utf-8"))
        if self.intent.metadata["configuration"] != interaction_config:
            raise ValueError("İfade sınıflandırıcısının ayarları eğitimden sonra değişmiş.")
        corpus_digest = hashlib.sha256((self.root / "data/interactions/intent_corpus.json").read_bytes()).hexdigest()
        if self.intent.metadata["corpus_sha256"] != corpus_digest:
            raise ValueError("İfade derlemi sınıflandırıcı eğitiminden sonra değişmiş.")
        self.patterns = joblib.load(self.root / "outputs/patterns/behavior_model.joblib")
        self.config = json.loads((self.root / "configs/tracking.json").read_text(encoding="utf-8"))
        session = json.loads((self.root / "outputs/replay/session.json").read_text(encoding="utf-8"))
        student = session["encoded_student"]
        start, _ = self.store.spans[student]
        self.snapshot = snapshot_for(self.store, student, start+session["events"][-1]["after"]["history_count"])
        self.catalog = training_question_catalog(self.store)
        self.vocabulary = json.loads((self.root / "data/processed/vocabulary.json").read_text(encoding="utf-8"))
        self.inverse_questions = {int(v): int(k) for k, v in self.vocabulary["question"].items()}
        if self.patterns.metadata["checkpoint_sha256"] != self.tracer.provenance["checkpoint_sha256"]:
            raise ValueError("Davranış modeli ile bilgi takibi modeli farklı.")
        if self.patterns.metadata.get("inference_policy", "legacy_causal_lkm_bidirectional_prompt") != self.tracer.provenance["inference_policy"]:
            raise ValueError("Davranış modelinin dikkat politikası farklı.")
        if self.patterns.metadata["reference_bank_sha256"] != self.bank.source_sha256:
            raise ValueError("Referans soruları davranış modelinden sonra değişmiş.")
        for key in ("window_size", "minimum_history", "reference_skills_per_snapshot"):
            if self.patterns.metadata["configuration"][key] != self.config[key]:
                raise ValueError("Davranış modelinin eğitim ayarı değişmiş: " + key)
        active = active_skills(self.store, self.snapshot, self.config["reference_skills_per_snapshot"],
                               self.config["window_size"], self.bank)
        scores = self.tracer.reference_scores([self.snapshot], [active], self.bank)[0]
        features = behavior_vector(self.store, self.snapshot, scores, self.patterns.limits, self.config["window_size"])
        self.behavior = self.patterns.predict(features)

    def respond(self, text):
        before = history_digest(self.store, self.snapshot)
        route = self.intent.predict(text)
        output = {"recognized_text": text, "intent": route, "response_is_assessed_answer": False,
                  "student_history_answers": self.snapshot.stop-self.snapshot.start,
                  "recommendation": None}
        if route["accepted"]:
            skill = route["skill"]["encoded_skill"]
            name = route["skill"]["name"]
            score = self.tracer.reference_scores([self.snapshot], [[skill]], self.bank)[0]
            recommendation = recommend_next(self.store, self.snapshot, score, self.tracer,
                                             self.catalog, self.config, self.behavior, preferred_skill=skill)
            if recommendation["available"]:
                recommendation["original_problem_id"] = self.inverse_questions[recommendation["question"]]
                recommendation["skill_name"] = name
                recommendation["reason"] = recommendation["reason"].replace(f"Beceri {skill}:", f"{name}:", 1)
                output["reply"] = (f"İsteğini {name.lower()} olarak algıladım. "
                                   f"Mevcut sorulardan {recommendation['original_problem_id']} kimlikli soruyu öneriyorum. "
                                   f"Bu soru için tahmini doğru cevap olasılığı %{100*recommendation['predicted_probability']:.1f}.")
            else:
                output["reply"] = recommendation["reason"]
            output["reference_success_percent"] = 100*score[skill]
            output["observed_skill"] = observed_skill_rates(self.store, self.snapshot, [skill], self.config["window_size"])[skill]
            output["recommendation"] = recommendation
        else:
            output["reply"] = route["reason"]
        output["history_unchanged"] = before == history_digest(self.store, self.snapshot)
        if not output["history_unchanged"]:
            raise AssertionError("Bir çalışma isteği öğrenci cevap geçmişini değiştirdi.")
        return output
