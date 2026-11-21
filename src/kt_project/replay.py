"""Mevcut kayıtlarla, cevap öncesi tahmin ve cevap sonrası durum güncellemesi."""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .data_audit import detect_encoding, integer_id
from .patterns import FEATURE_NAMES, behavior_vector
from .recommendations import recommend_next
from .tracking import active_skills, snapshot_for
from .training import cpu_state, state_digest


def load_display_metadata(root):
    root = Path(root)
    vocabulary = json.loads((root / "data/processed/vocabulary.json").read_text(encoding="utf-8"))
    inverse_questions = {int(encoded): int(original) for original, encoded in vocabulary["question"].items()}
    names = {int(encoded): f"Beceri {original}" for original, encoded in vocabulary["skill"].items()}
    raw = root / "data/raw/skill_builder_data_corrected_collapsed.csv"
    # Çok becerili satırın tek bir adını bütün becerilere yayma.
    with raw.open(encoding=detect_encoding(raw), newline="") as stream:
        for row in csv.DictReader(stream):
            original = integer_id(row.get("skill_id"))
            name = (row.get("skill_name") or "").strip()
            encoded = vocabulary["skill"].get(str(original)) if original is not None else None
            if encoded is not None and name and name.lower() != "nan":
                names[int(encoded)] = name
    return names, inverse_questions


def choose_replay_student(store, config):
    minimum = config["replay_initial_history"] + config["replay_steps"]
    eligible = [student for student in store.select_students(config["replay_split"])
                if store.spans[student][1]-store.spans[student][0] >= minimum]
    if not eligible:
        raise ValueError("Örnek oturum uzunluğunu sağlayan öğrenci yok.")
    # Sadece kayıt uzunluğu ve sabit başlangıç; sonuç eğrisine göre seçim yok.
    return int(np.random.default_rng(config["random_seed"]).choice(eligible))


def observed_skill_rates(store, snapshot, skills, window_size):
    left = max(snapshot.start, snapshot.stop-window_size)
    concepts = store.arrays["skills"][left:snapshot.stop]
    labels = store.arrays["correct"][left:snapshot.stop]
    output = {}
    for skill in skills:
        selected = (concepts == skill).any(axis=1)
        output[skill] = {"answers": int(selected.sum()),
                         "ratio": float(labels[selected].mean()) if selected.any() else None}
    return output


def pattern_state_digest(model):
    digest = hashlib.sha256()
    for array in (model.clusterer.cluster_centers_, model.scaler.mean_, model.scaler.scale_,
                  model.imputer.statistics_):
        digest.update(np.asarray(array).tobytes())
    return digest.hexdigest()


def current_state(store, snapshot, tracked_skills, tracer, bank, pattern_model,
                  catalog, config, names, inverse_questions):
    active = active_skills(store, snapshot, config["reference_skills_per_snapshot"], config["window_size"], bank)
    query_skills = sorted(set(tracked_skills).union(active))
    scores = tracer.reference_scores([snapshot], [query_skills], bank,
                                     config["prediction_batch_size"])[0]
    pattern_scores = {skill: scores[skill] for skill in active if skill in scores}
    features = behavior_vector(store, snapshot, pattern_scores, pattern_model.limits, config["window_size"])
    group = pattern_model.predict(features) if snapshot.stop-snapshot.start >= pattern_model.minimum_history else None
    recommendation = recommend_next(store, snapshot, pattern_scores, tracer, catalog, config, behavior=group)
    if recommendation["available"]:
        skill = recommendation["skill"]
        recommendation["skill_name"] = names.get(skill, f"Beceri {skill}")
        recommendation["original_problem_id"] = inverse_questions.get(recommendation["question"])
        recommendation["reason"] = recommendation["reason"].replace(
            f"Beceri {skill}:", f"{recommendation['skill_name']}:", 1)
    observed = observed_skill_rates(store, snapshot, tracked_skills, config["window_size"])
    left = max(snapshot.start, snapshot.stop-config["window_size"])
    rows = []
    for skill in tracked_skills:
        if skill not in scores:
            continue
        ratio = observed[skill]["ratio"]
        rows.append({"skill": skill, "name": names.get(skill, f"Beceri {skill}"),
                     "predicted_success_percent": 100*scores[skill],
                     "observed_correct_percent": 100*ratio if ratio is not None else None,
                     "recent_skill_answers": observed[skill]["answers"],
                     "reference_question_count": len(bank.for_skill(skill))})
    return {
        "history_count": snapshot.stop-snapshot.start,
        "recent_window_count": snapshot.stop-left,
        "observed_recent_correct_ratio": float(store.arrays["correct"][left:snapshot.stop].mean()),
        "skill_scores": rows, "active_pattern_skills": active,
        "behavior_features": {name: float(value) if np.isfinite(value) else None
                              for name, value in zip(FEATURE_NAMES, features)},
        "behavior": group, "recommendation": recommendation,
    }


def run_replay(root, store, tracer, bank, pattern_model, catalog, config, progress=None):
    names, inverse_questions = load_display_metadata(root)
    student = choose_replay_student(store, config)
    start, _ = store.spans[student]
    stop = start+config["replay_initial_history"]
    snapshot = snapshot_for(store, student, stop)
    tracked = active_skills(store, snapshot, 6, config["window_size"], bank)
    model_before = state_digest(cpu_state(tracer.model))
    pattern_before, bank_before = pattern_state_digest(pattern_model), bank.content_sha256()
    initial = current_state(store, snapshot, tracked, tracer, bank, pattern_model,
                            catalog, config, names, inverse_questions)
    started = datetime.now(ZoneInfo("Europe/Istanbul")).isoformat(timespec="milliseconds")
    events = []
    for step in range(1, config["replay_steps"]+1):
        target = snapshot.stop
        question = {"question": int(store.arrays["question"][target]),
                    "skills": store.arrays["skills"][target][store.arrays["skills"][target] > 0].tolist(),
                    "difficulty": float(store.arrays["difficulty"][target])}
        # Gerçek cevap burada henüz okunmaz; istek hiçbir cevap etiketi içermez.
        probability = float(tracer.predict_requests([(snapshot, question)], config["prediction_batch_size"])[0])
        actual = int(store.arrays["correct"][target])
        history_stop_before = snapshot.stop
        snapshot = snapshot_for(store, student, target+1)
        for skill in question["skills"]:
            if skill > 1 and bank.has_skill(skill) and skill not in tracked and len(tracked) < 6:
                tracked.append(skill)
        state = current_state(store, snapshot, tracked, tracer, bank, pattern_model,
                              catalog, config, names, inverse_questions)
        timestamp = datetime.now(ZoneInfo("Europe/Istanbul")).isoformat(timespec="milliseconds")
        event = {"step": step, "target_row_index": target, "history_stop_before": history_stop_before,
                 "source_order_id": int(store.arrays["order"][target]), "source_timestamp": None,
                 "display_timestamp": timestamp, "question": question["question"],
                 "original_problem_id": inverse_questions.get(question["question"]),
                 "predicted_before_response": probability, "observed_correct": actual, "after": state}
        events.append(event)
        if progress:
            progress(step, config["replay_steps"])
    integrity = {
        "model_weights_unchanged": model_before == state_digest(cpu_state(tracer.model)),
        "behavior_model_unchanged": pattern_before == pattern_state_digest(pattern_model),
        "reference_bank_unchanged": bank_before == bank.content_sha256(),
        "each_prediction_precedes_target_response": all(row["history_stop_before"] == row["target_row_index"] for row in events),
        "source_calendar_times_fabricated": False,
    }
    if not all(value for key, value in integrity.items() if key != "source_calendar_times_fabricated"):
        raise AssertionError("Oturum güvenilirlik kontrolü başarısız.")
    return {"student_alias": "Örnek öğrenci", "encoded_student": student,
            "source_split": "validation", "selection_uses_response_outcomes": False,
            "selection_rule": "fixed random seed among validation histories of sufficient length",
            "display_started_at": started, "source_calendar_timestamp_available": False,
            "initial": initial, "events": events, "integrity": integrity,
            "reference_bank_sha256": bank.source_sha256, "model_selection": tracer.selection,
            "note": "Mevcut geçmişin yeniden oynatımıdır; gerçek zamana yayılmış öğrenme kazanımı veya öneri etkisi ölçülmez."}
