"""Geçmişe ve sabit model skorlarına dayanan açıklanabilir seçim kuralları."""

from __future__ import annotations

from collections import Counter

import numpy as np


def training_question_catalog(store):
    catalog = {}
    arrays = store.arrays
    for index in np.flatnonzero(arrays["split"] == 0):
        question = int(arrays["question"][index])
        if question < 3:
            continue
        entry = catalog.setdefault(question, {"question": question, "skills": set(),
                                              "difficulty": float(arrays["difficulty"][index]),
                                              "training_support": 0})
        entry["skills"].update(int(skill) for skill in arrays["skills"][index] if skill > 1)
        entry["training_support"] += 1
    return {question: {**entry, "skills": sorted(entry["skills"])} for question, entry in catalog.items()}


def challenge_probability(recent_correct, config):
    recent = np.asarray(recent_correct)
    if len(recent) >= 3 and np.all(recent[-3:] == 0):
        return config["probability_after_repeated_errors"], "son üç cevap yanlış; daha erişilebilir başarı hedefi"
    if len(recent) >= 3 and np.all(recent[-3:] == 1):
        return config["probability_after_repeated_successes"], "son üç cevap doğru; daha zorlayıcı başarı hedefi"
    return config["recommended_probability"], "orta düzeyde zorlayıcı başarı hedefi"


def recommend_next(store, snapshot, reference_scores, tracer, catalog, config,
                   behavior=None, preferred_skill=None):
    if not reference_scores:
        return {"available": False, "reason": "Öneri için bilinen bir beceri ve referans tahmini gerekli."}
    if preferred_skill is not None and int(preferred_skill) in reference_scores:
        skill = int(preferred_skill)
        skill_reason = "öğrencinin ifade ettiği beceri önceliklendirildi"
    else:
        skill = min(reference_scores, key=lambda value: (reference_scores[value], value))
        skill_reason = "izlenen beceriler arasında referans başarı tahmini en düşük olan beceri"
    seen = set(store.arrays["question"][snapshot.start:snapshot.stop].tolist())
    candidates = [entry for entry in catalog.values() if skill in entry["skills"]]
    fresh = [entry for entry in candidates if entry["question"] not in seen]
    repeated = not bool(fresh)
    candidates = fresh if fresh else candidates
    if not candidates:
        return {"available": False, "skill": skill, "reason": "Bu beceride eğitimden bilinen soru yok."}
    candidates.sort(key=lambda entry: (entry["difficulty"], entry["question"]))
    # Çok büyük kataloglarda adayları zorluk aralığından sabit ve eşit aralıklarla al.
    if len(candidates) > 256:
        positions = np.linspace(0, len(candidates)-1, 256, dtype=int)
        candidates = [candidates[int(index)] for index in positions]
    left = max(snapshot.start, snapshot.stop-config["window_size"])
    recent = store.arrays["correct"][left:snapshot.stop]
    target, challenge_reason = challenge_probability(recent, config)
    probabilities = tracer.predict_requests([(snapshot, entry) for entry in candidates],
                                             config["prediction_batch_size"])
    best = min(range(len(candidates)), key=lambda index:
               (abs(float(probabilities[index])-target), candidates[index]["question"]))
    entry, probability = candidates[best], float(probabilities[best])
    action = "daha erişilebilir soru ile pekiştirme" if target >= .8 else (
        "daha zorlayıcı soru ile ilerleme" if target <= .55 else "mevcut düzeyde alıştırma")
    reason = (f"Beceri {skill}: {skill_reason}; sabit referans başarı tahmini "
              f"%{100*reference_scores[skill]:.1f}. {challenge_reason}. "
              f"Seçilen soruda tahmin %{100*probability:.1f}, hedef %{100*target:.0f}.")
    if repeated:
        reason += " Bu beceride bütün bilinen adaylar görüldüğü için tekrar sorusu seçildi."
    return {"available": True, "question": entry["question"], "skill": int(skill),
            "difficulty": entry["difficulty"], "predicted_probability": probability,
            "target_probability": float(target), "reference_probability": reference_scores[skill],
            "repeat_question": repeated, "action": action, "reason": reason,
            "behavior_group": behavior["cluster"] if behavior else None,
            "rule_is_learned_policy": False,
            "note": "Seçim kuralıdır; önerinin öğrenme kazanımı sağladığı ölçülmemiştir."}
