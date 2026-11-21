"""Yeni cevap olaylarını gerçek alım zamanlarıyla, eğitim verisine yazmadan izle."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import uuid

import joblib
import numpy as np
import torch

from .patterns import FEATURE_NAMES, behavior_vector
from .recommendations import recommend_next, training_question_catalog
from .replay import load_display_metadata, observed_skill_rates, pattern_state_digest
from .sequences import InteractionStore
from .tracking import FrozenTracer, ReferenceBank, active_skills, snapshot_for
from .training import cpu_state, state_digest


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def store_hash(store):
    digest = hashlib.sha256()
    for key, value in sorted(store.arrays.items()):
        digest.update(key.encode())
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def parse_time(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Saat dilimi olmayan zaman kaydı kabul edilmez.")
    return result.astimezone(timezone.utc)


@dataclass
class LiveResources:
    root: Path
    tracer: object
    bank: object
    patterns: object
    catalog: dict
    config: dict
    names: dict
    inverse_questions: dict
    default_skills: list

    @classmethod
    def load(cls, root):
        root = Path(root)
        store = InteractionStore(root / "data/processed/interactions.npz")
        tracer = FrozenTracer(root, store)
        bank = ReferenceBank(root / "data/processed/references.json")
        patterns = joblib.load(root / "outputs/patterns/behavior_model.joblib")
        if patterns.metadata["checkpoint_sha256"] != tracer.provenance["checkpoint_sha256"]:
            raise ValueError("Davranış grupları ile etkin ağırlıklar uyuşmuyor.")
        if patterns.metadata["inference_policy"] != tracer.provenance["inference_policy"]:
            raise ValueError("Davranış gruplarının dikkat politikası farklı.")
        if patterns.metadata["reference_bank_sha256"] != bank.source_sha256:
            raise ValueError("Sabit referans kümesi davranış modelinden sonra değişmiş.")
        config = json.loads((root / "configs/tracking.json").read_text(encoding="utf-8"))
        for key in ("window_size", "minimum_history", "reference_skills_per_snapshot"):
            if config[key] != patterns.metadata["configuration"][key]:
                raise ValueError("Davranış modeli ayarı uyuşmuyor: " + key)
        names, inverse = load_display_metadata(root)
        modal = json.loads((root / "configs/interactions.json").read_text(encoding="utf-8"))
        vocab = json.loads((root / "data/processed/vocabulary.json").read_text(encoding="utf-8"))
        skills = [int(vocab["skill"][entry["raw_skill_id"]]) for entry in modal["skills"].values()]
        for encoded, entry in zip(skills, modal["skills"].values()):
            names[encoded] = entry["name"]
        return cls(root, tracer, bank, patterns, training_question_catalog(store), config, names, inverse, skills)

    def signature(self):
        return {"model_selection": self.tracer.selection, "reference_bank_sha256": self.bank.source_sha256,
                "pattern_state_sha256": pattern_state_digest(self.patterns),
                "tracking_config_sha256": content_hash(self.config)}


class LiveSession:
    """Tek bekleyen soru, cevaptan önce tahmin ve kalıcı oturum kaydı.

    Cevap doğruluğu dışarıda değerlendirilip girilir; soru metni üretilmez veya
    model tarafından cevap değerlendirmesi yapılmaz. Geçmiş başlangıçta boştur.
    Otomatik örnekler insan cevabı olarak etiketlenemez.
    """

    def __init__(self, resources, alias="Katılımcı", *, origin="manual_assessed_response",
                 directory=None, _clock=None, _monotonic=None, _saved=None):
        self.resources = resources
        self.directory = Path(directory or resources.root / "outputs/live/sessions")
        self.directory.mkdir(parents=True, exist_ok=True)
        self._clock = _clock or (lambda: datetime.now(timezone.utc))
        self._monotonic = _monotonic or time.monotonic
        self._lock = threading.RLock()
        self._pending_tick = None
        self.tracer = copy.copy(resources.tracer)
        self._model_state_before = state_digest(cpu_state(resources.tracer.model))
        self._source_store_before = store_hash(resources.tracer.store)
        self._bank_content_before = resources.bank.content_sha256()
        self._disk_record_hash = None
        if _saved is None:
            if origin not in {"manual_assessed_response", "automated_smoke_test"}:
                raise ValueError("Geçersiz cevap kaynağı.")
            if not isinstance(alias, str) or not 1 <= len(alias.strip()) <= 80:
                raise ValueError("Katılımcı etiketi 1 ile 80 karakter arasında olmalı.")
            self.data = {"format": "live_session_v1", "session_id": uuid.uuid4().hex,
                "participant_alias": alias.strip(), "response_origin": origin,
                "human_learning_study": False,
                "created_at": self._now(), "closed_at": None, "status": "open",
                "initial_history_count": 0, "historical_calendar_times_fabricated": False,
                "tracked_skills": list(resources.default_skills), "resources": resources.signature(),
                "pending": None, "events": [],
                "note": "Yeni cevapların sunucu alım saatleri kaydedilir. Tahminler cevaplar arasında son değerinde tutulur; saat modeli veya sürekli öğrenme ölçümü değildir."}
            self._sync([])
            self.data["initial_state"] = self._state()
            self.data["initial_state_recorded_at"] = self._now()
            self._persist()
        else:
            self.data = copy.deepcopy(_saved)
            self._validate_saved()
            if self.data["resources"] != resources.signature():
                raise ValueError("Oturumun ağırlıkları, referansları veya grup ayarları değişmiş.")
            self._sync(self.data["events"])

    @classmethod
    def resume(cls, resources, session_id, *, directory=None, **testing):
        if not isinstance(session_id, str) or len(session_id) != 32 or any(c not in "0123456789abcdef" for c in session_id):
            raise ValueError("Geçersiz oturum kimliği.")
        folder = Path(directory or resources.root / "outputs/live/sessions")
        saved = json.loads((folder / f"{session_id}.json").read_text(encoding="utf-8"))
        if saved["session_id"] != session_id:
            raise ValueError("Dosya ile oturum kimliği farklı.")
        return cls(resources, directory=folder, _saved=saved, **testing)

    @property
    def path(self):
        return self.directory / f"{self.data['session_id']}.json"

    def _now(self):
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("Saat dilimi gerekli.")
        return value.astimezone(timezone.utc).isoformat(timespec="microseconds")

    def _ordered_now(self):
        now = self._now()
        previous = self.data["events"][-1]["state_updated_at"] if self.data["events"] else self.data["initial_state_recorded_at"]
        if self.data["pending"]:
            previous = self.data["pending"]["presented_at"]
        if parse_time(now) < parse_time(previous):
            raise ValueError("Sistem saati geriye gitmiş; zaman kaydı düzeltilmeden yeni olay kabul edilmez.")
        return now

    def _persist(self):
        if self.path.exists():
            current = json.loads(self.path.read_text(encoding="utf-8"))
            if current.get("record_sha256") != self._disk_record_hash:
                raise ValueError("Oturum başka bir görünümde değişmiş; kaydı yeniden açın.")
        payload = {**self.data, "record_sha256": content_hash(self.data)}
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)
        self._disk_record_hash = payload["record_sha256"]

    def _validate_saved(self):
        expected = self.data.pop("record_sha256", None)
        if expected != content_hash(self.data) or self.data["format"] != "live_session_v1":
            raise ValueError("Oturum kaydının içerik denetimi başarısız.")
        self._disk_record_hash = expected
        previous = parse_time(self.data["initial_state_recorded_at"])
        chain = "initial"
        for index, event in enumerate(self.data["events"], 1):
            payload = {key: value for key, value in event.items() if key != "event_sha256"}
            if event["event_sha256"] != content_hash(payload) or event["previous_event_sha256"] != chain:
                raise ValueError("Cevap kayıt zinciri değişmiş.")
            chain = event["event_sha256"]
            presented, received, updated = map(parse_time, (event["presented_at"], event["answer_received_at"], event["state_updated_at"]))
            if not previous <= presented <= received <= updated or event["sequence"] != index:
                raise ValueError("Oturum olay sırası tutarsız.")
            previous = updated

    def _sync(self, events):
        # Bir dolgu satırı, sıfır geçmişte boş tensör indekslemeyi önler.
        # Snapshot.stop bu satırı geçmişe almaz; tamamlanmamış soru da eklenmez.
        width, size = self.tracer.skill_width, max(1, len(events))
        arrays = {"question": np.zeros(size, dtype=np.int64), "skills": np.zeros((size, width), dtype=np.int64),
                  "correct": np.zeros(size, dtype=np.int64), "difficulty": np.zeros(size, dtype=np.float32),
                  "duration_ms": np.full(size, np.nan), "hints": np.full(size, np.nan)}
        for index, event in enumerate(events):
            question = event["question"]
            arrays["question"][index] = question["question"]
            arrays["skills"][index, :len(question["skills"])] = question["skills"]
            arrays["correct"][index] = event["correct"]
            arrays["difficulty"][index] = question["difficulty"]
            arrays["duration_ms"][index] = 1000*event["response_duration_seconds"]
            arrays["hints"][index] = event["hint_count"]
        self.store = SimpleNamespace(arrays=arrays, spans={0: (0, len(events))})
        self.tracer.store = self.store
        self.tracer.arrays = {key: torch.as_tensor(arrays[source], dtype=dtype, device=self.tracer.device)
            for key, source, dtype in (("question", "question", torch.long), ("skills", "skills", torch.long),
                ("response", "correct", torch.long), ("difficulty", "difficulty", torch.float32))}

    def _state(self):
        resource, count = self.resources, self.store.spans[0][1]
        snapshot = snapshot_for(self.store, 0, count)
        tracked = self.data["tracked_skills"]
        active = active_skills(self.store, snapshot, resource.config["reference_skills_per_snapshot"],
                               resource.config["window_size"], resource.bank)
        scores = self.tracer.reference_scores([snapshot], [sorted(set(tracked+active))], resource.bank,
                                              resource.config["prediction_batch_size"])[0]
        observed = observed_skill_rates(self.store, snapshot, tracked, resource.config["window_size"])
        features = behavior_vector(self.store, snapshot, {s: scores[s] for s in active if s in scores},
            resource.patterns.limits, resource.config["window_size"]) if count else np.full(4, np.nan)
        group = resource.patterns.predict(features) if count >= resource.patterns.minimum_history else None
        recent = self.store.arrays["correct"][max(0,count-resource.config["window_size"]):count]
        recommendation = recommend_next(self.store, snapshot, scores, self.tracer, resource.catalog, resource.config, behavior=group)
        if recommendation["available"]:
            recommendation["original_problem_id"] = resource.inverse_questions[recommendation["question"]]
        return {"history_count": count, "recent_window_count": len(recent),
            "observed_recent_correct_ratio": float(recent.mean()) if len(recent) else None,
            "skill_scores": [{"skill": skill, "name": resource.names.get(skill, f"Beceri {skill}"),
                "predicted_success_percent": 100*scores[skill],
                "observed_correct_percent": 100*observed[skill]["ratio"] if observed[skill]["ratio"] is not None else None,
                "recent_skill_answers": observed[skill]["answers"],
                "reference_question_count": len(resource.bank.for_skill(skill))} for skill in tracked if skill in scores],
            "behavior_features": {name: float(value) if np.isfinite(value) else None for name, value in zip(FEATURE_NAMES, features)},
            "behavior": group, "behavior_minimum_answers": resource.patterns.minimum_history,
            "recommendation": recommendation}

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(self.data)

    def prepare(self, original_problem_id):
        with self._lock:
            if self.data["status"] != "open" or self.data["pending"] is not None:
                raise ValueError("Oturum kapalı veya tamamlanmamış bir soru var.")
            if isinstance(original_problem_id, bool) or not isinstance(original_problem_id, (int, np.integer)):
                raise ValueError("Soru kimliği tam sayı olmalı.")
            inverse = {value: key for key, value in self.resources.inverse_questions.items()}
            encoded = inverse.get(int(original_problem_id))
            if encoded not in self.resources.catalog:
                raise ValueError("Eğitimden bilinen bir soru seçilmeli.")
            question = {key: self.resources.catalog[encoded][key] for key in ("question", "skills", "difficulty")}
            started = self._ordered_now()
            snap = snapshot_for(self.store, 0, len(self.data["events"]))
            probability = float(self.tracer.predict_requests([(snap, question)], self.resources.config["prediction_batch_size"])[0])
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Geçersiz model olasılığı.")
            now = self._ordered_now()
            if parse_time(now) < parse_time(started):
                raise ValueError("Tahmin sırasında sistem saati geriye gitti.")
            pending = {"question_token": uuid.uuid4().hex, "question": copy.deepcopy(question),
                "original_problem_id": int(original_problem_id), "history_count_before": len(self.data["events"]),
                "prediction_started_at": started, "predicted_at": now, "presented_at": now,
                "predicted_correct_probability": probability}
            self.data["pending"] = pending
            self._pending_tick = self._monotonic()
            try:
                self._persist()
            except Exception:
                self.data["pending"] = None
                self._pending_tick = None
                raise
            return copy.deepcopy(pending)

    def answer(self, question_token, correct, hint_count=0):
        with self._lock:
            if isinstance(correct, bool) or not isinstance(correct, (int, np.integer)) or correct not in (0,1):
                raise ValueError("Değerlendirilmiş cevap 0 veya 1 olmalı.")
            if isinstance(hint_count, bool) or not isinstance(hint_count, (int,np.integer)) or not 0 <= hint_count <= 100:
                raise ValueError("İpucu sayısı 0 ile 100 arasında tam sayı olmalı.")
            for event in self.data["events"]:
                if event["question_token"] == question_token:
                    if event["correct"] != correct or event["hint_count"] != hint_count:
                        raise ValueError("Aynı cevap kaydı farklı sonuçla yeniden gönderilemez.")
                    return copy.deepcopy(event)
            pending = self.data["pending"]
            if self.data["status"] != "open" or pending is None or pending["question_token"] != question_token:
                raise ValueError("Cevap, açık bekleyen soruya ait değil.")
            received = self._ordered_now()
            seconds = (self._monotonic()-self._pending_tick) if self._pending_tick is not None else (
                parse_time(received)-parse_time(pending["presented_at"])).total_seconds()
            if not np.isfinite(seconds) or seconds < 0:
                raise ValueError("Geçersiz cevap süresi.")
            event = {**copy.deepcopy(pending), "sequence": len(self.data["events"])+1,
                "answer_received_at": received, "correct": int(correct), "hint_count": int(hint_count),
                "response_duration_seconds": float(seconds),
                "duration_clock": "monotonic" if self._pending_tick is not None else "UTC_after_resume",
                "response_origin": self.data["response_origin"],
                "previous_event_sha256": self.data["events"][-1]["event_sha256"] if self.data["events"] else "initial"}
            candidate = self.data["events"]+[event]
            old_skills = list(self.data["tracked_skills"])
            try:
                for skill in event["question"]["skills"]:
                    if self.resources.bank.has_skill(skill) and skill not in self.data["tracked_skills"]:
                        self.data["tracked_skills"].append(skill)
                self._sync(candidate)
                event["after"] = self._state()
                event["state_updated_at"] = self._now()
                if parse_time(event["state_updated_at"]) < parse_time(received):
                    raise ValueError("Güncelleme sırasında sistem saati geriye gitti.")
                event["event_sha256"] = content_hash(event)
                self.data["events"].append(event)
                self.data["pending"] = None
                self._persist()
            except Exception:
                self.data["tracked_skills"] = old_skills
                if self.data["events"] and self.data["events"][-1] is event:
                    self.data["events"].pop()
                self.data["pending"] = pending
                self._sync(self.data["events"])
                raise
            self._pending_tick = None
            return copy.deepcopy(event)

    def close(self):
        with self._lock:
            if self.data["pending"]:
                raise ValueError("Önce açık sorunun sonucunu kaydedin.")
            if self.data["status"] == "open":
                self.data["closed_at"] = self._ordered_now()
                self.data["status"] = "closed"
                try:
                    self._persist()
                except Exception:
                    self.data["closed_at"] = None
                    self.data["status"] = "open"
                    raise
            return self.snapshot()

    def verify_frozen_resources(self):
        return {"model_weights_unchanged": state_digest(cpu_state(self.tracer.model)) == self._model_state_before,
                "source_store_unmodified": store_hash(self.resources.tracer.store) == self._source_store_before,
                "references_and_pattern_state_unchanged": self.resources.signature() == self.data["resources"]
                    and self.resources.bank.content_sha256() == self._bank_content_before}
