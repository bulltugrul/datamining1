"""Sabit referans sorularında, yalnızca gözlenmiş geçmişle bilgi takibi."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .models import KnowledgeTracer, LargeKnowledgeModel, ModelConfig
from .sequences import InteractionStore
from .training import autocast_context
from .data_audit import file_digest


@dataclass(frozen=True)
class Snapshot:
    student: int
    start: int
    stop: int


class ReferenceBank:
    def __init__(self, path: str | Path):
        raw = Path(path).read_bytes()
        self.source_sha256 = hashlib.sha256(raw).hexdigest()
        self._items = json.loads(raw)

    def for_skill(self, skill):
        # Kopyalar döndürülür; çağıran oturum boyunca referansları değiştiremez.
        return json.loads(json.dumps(self._items.get(str(int(skill)), [])))

    def has_skill(self, skill):
        return str(int(skill)) in self._items

    def content_sha256(self):
        return hashlib.sha256(json.dumps(self._items, sort_keys=True).encode()).hexdigest()


def snapshot_for(store: InteractionStore, student: int, stop: int):
    left, right = store.spans[int(student)]
    if not left <= stop <= right:
        raise ValueError("Geçmiş sınırı öğrencinin kayıt aralığında değil.")
    return Snapshot(int(student), left, int(stop))


def active_skills(store, snapshot, limit=3, window_size=20, bank=None):
    left = max(snapshot.start, snapshot.stop-window_size)
    counts = Counter(int(skill) for row in store.arrays["skills"][left:snapshot.stop]
                     for skill in row if skill > 1 and (bank is None or bank.has_skill(skill)))
    return [skill for skill, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


class FrozenTracer:
    def __init__(self, root: Path, store: InteractionStore, device=None):
        root = Path(root)
        active_path = root / "configs/active_model.json"
        active = json.loads(active_path.read_text(encoding="utf-8")) if active_path.exists() else {}
        selection_file = active.get("selection_file", "outputs/main/three_seed_v1/analysis/prototype_selection.json")
        selection = json.loads((root / selection_file).read_text(encoding="utf-8"))
        self.checkpoint_path = root / selection["checkpoint"]
        saved = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        self.config = ModelConfig(**saved["config"])
        if saved.get("format") == "plkt_method_v2":
            if selection.get("inference_policy") == "same_target_step_block":
                from .plkt_temporal import build_temporal_tracer
                implementation = root / "src/kt_project/plkt_temporal.py"
                if file_digest(implementation) != selection["inference_implementation_sha256"]:
                    raise ValueError("Etkin dikkat uygulaması seçim kaydıyla uyuşmuyor.")
                self.model = build_temporal_tracer(self.config, saved["method"])
            else:
                from .plkt_training import build_tracer
                self.model = build_tracer(self.config, saved["variant"])
        else:
            self.model = KnowledgeTracer(LargeKnowledgeModel(self.config), saved["method"])
        self.model.load_state_dict(saved["model_state"])
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.store = store
        self.selection = selection
        prepared = json.loads((root / "data/processed/metadata.json").read_text(encoding="utf-8"))
        self.provenance = {"source_sha256": prepared["source_sha256"],
                           "checkpoint_sha256": file_digest(self.checkpoint_path),
                           "inference_policy": selection.get("inference_policy",
                               "strict_token" if saved.get("format") == "plkt_method_v2"
                               else "legacy_causal_lkm_bidirectional_prompt")}
        expected_sha = selection.get("checkpoint_sha256")
        if expected_sha is not None and expected_sha != self.provenance["checkpoint_sha256"]:
            raise ValueError("Etkin model ağırlığının içerik özeti seçim kaydıyla uyuşmuyor.")
        self.precision = "bfloat16" if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else "float32"
        self.arrays = {key: torch.as_tensor(store.arrays[source], dtype=dtype, device=self.device)
                       for key, source, dtype in [
                           ("question", "question", torch.long), ("skills", "skills", torch.long),
                           ("response", "correct", torch.long), ("difficulty", "difficulty", torch.float32)]}
        self.skill_width = store.arrays["skills"].shape[1]

    def query_batch(self, requests):
        """İstek (Snapshot, soru özellikleri); hedef cevabı kabul edilmez."""
        starts, stops, questions, concepts, difficulties = [], [], [], [], []
        for snapshot, question in requests:
            snapshot_for(self.store, snapshot.student, snapshot.stop)
            if snapshot.start != self.store.spans[snapshot.student][0]:
                raise ValueError("Başlangıç sınırı farklı bir öğrenciye ait.")
            starts.append(max(snapshot.start, snapshot.stop-self.config.max_history))
            stops.append(snapshot.stop)
            q = int(question["question"])
            if not 1 <= q < self.config.question_vocabulary_size or q == 2:
                raise ValueError("Geçersiz hedef soru kimliği.")
            questions.append(q)
            ids = [int(value) for value in question["skills"] if int(value) > 0]
            if len(ids) > self.skill_width or any(value >= self.config.skill_vocabulary_size for value in ids):
                raise ValueError("Geçersiz beceri bilgisi.")
            concepts.append(ids + [0]*(self.skill_width-len(ids)))
            difficulty = float(question["difficulty"])
            if not np.isfinite(difficulty) or not 0 <= difficulty <= 1:
                raise ValueError("Soru zorluğu sıfır ile bir arasında olmalı.")
            difficulties.append(difficulty)
        start = torch.as_tensor(starts, dtype=torch.long, device=self.device)
        stop = torch.as_tensor(stops, dtype=torch.long, device=self.device)
        offsets = torch.arange(self.config.max_history, device=self.device)[None, :]
        valid = offsets < (stop-start)[:, None]
        indices = (start[:, None]+offsets).clamp_max(len(self.arrays["question"])-1)
        result = {}
        for key, pad in (("question", 0), ("skills", 0), ("response", 3), ("difficulty", 0)):
            values = self.arrays[key][indices]
            mask = valid.unsqueeze(-1) if key == "skills" else valid
            result["history_"+key] = torch.where(mask, values, pad)
        result.update({
            "history_padding": ~valid,
            "target_question": torch.as_tensor(questions, dtype=torch.long, device=self.device),
            "target_skills": torch.as_tensor(concepts, dtype=torch.long, device=self.device),
            "target_difficulty": torch.as_tensor(difficulties, dtype=torch.float32, device=self.device),
        })
        return result

    @torch.inference_mode()
    def predict_requests(self, requests, batch_size=512):
        if not requests:
            return np.array([], dtype=np.float32)
        values = []
        for left in range(0, len(requests), batch_size):
            batch = self.query_batch(requests[left:left+batch_size])
            with autocast_context(self.device, self.precision):
                logits = self.model(batch)
            probabilities = torch.sigmoid(logits.float())
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("Öğrenci takibi olasılığı sonlu değil.")
            values.append(probabilities.cpu().numpy())
        return np.concatenate(values)

    def reference_scores(self, snapshots, skill_lists, bank, batch_size=512, progress=None):
        if len(snapshots) != len(skill_lists):
            raise ValueError("Geçmiş ve beceri listesi sayıları farklı.")
        scores = [{} for _ in snapshots]
        # İş yükünü sınırla; bütün sorgular için dev geçmiş dizileri oluşturma.
        for first in range(0, len(snapshots), 128):
            requests, groups = [], []
            for index in range(first, min(first+128, len(snapshots))):
                for skill in skill_lists[index]:
                    entries = bank.for_skill(skill)
                    left = len(requests)
                    requests.extend((snapshots[index], question) for question in entries)
                    if entries:
                        groups.append((index, int(skill), left, len(requests)))
            probabilities = self.predict_requests(requests, batch_size)
            for index, skill, left, right in groups:
                scores[index][skill] = float(probabilities[left:right].mean())
            if progress:
                progress(min(first+128, len(snapshots)), len(snapshots))
        return scores
