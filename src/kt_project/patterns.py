"""Eğitim geçmişlerinden öğrenilen, değişebilir öğrenci davranış grupları."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .tracking import Snapshot, snapshot_for, active_skills

FEATURE_NAMES = ["recent_correct_ratio", "active_skill_reference_probability",
                 "log_median_response_seconds", "mean_hint_count"]


def training_snapshots(store, minimum_history=5, per_student=4):
    snapshots = []
    for student in store.select_students(0):
        start, stop = store.spans[student]
        length = stop-start
        count = min(per_student, length-minimum_history+1)
        if count <= 0:
            continue
        for offset in np.linspace(minimum_history, length, count, dtype=int):
            snapshots.append(snapshot_for(store, student, start+int(offset)))
    return snapshots


def fit_observation_limits(store):
    training = store.arrays["split"] == 0
    seconds = store.arrays["duration_ms"][training].astype(float)/1000
    seconds = seconds[np.isfinite(seconds) & (seconds > 0)]
    hints = store.arrays["hints"][training].astype(float)
    hints = hints[np.isfinite(hints) & (hints >= 0)]
    return {
        "duration_lower_seconds": float(np.quantile(seconds, .01)) if len(seconds) else 0.,
        "duration_upper_seconds": float(np.quantile(seconds, .99)) if len(seconds) else 1.,
        "hint_upper": max(1., float(np.quantile(hints, .99))) if len(hints) else 1.,
        "fitted_on": "training students only",
    }


def behavior_vector(store, snapshot, reference_scores, limits, window_size=20):
    if snapshot.stop <= snapshot.start:
        raise ValueError("Davranış özelliği için gözlenmiş geçmiş gerekli.")
    left = max(snapshot.start, snapshot.stop-window_size)
    values = store.arrays
    correct = values["correct"][left:snapshot.stop]
    seconds = values["duration_ms"][left:snapshot.stop].astype(float)/1000
    seconds = seconds[np.isfinite(seconds) & (seconds > 0)]
    if len(seconds):
        seconds = np.clip(seconds, limits["duration_lower_seconds"], limits["duration_upper_seconds"])
    hints = values["hints"][left:snapshot.stop].astype(float)
    hints = hints[np.isfinite(hints) & (hints >= 0)]
    if len(hints):
        hints = np.clip(hints, 0, limits["hint_upper"])
    probabilities = list(reference_scores.values())
    return np.array([
        float(correct.mean()), float(np.mean(probabilities)) if probabilities else np.nan,
        float(np.log1p(np.median(seconds))) if len(seconds) else np.nan,
        float(hints.mean()) if len(hints) else np.nan,
    ], dtype=float)


@dataclass
class BehaviorModel:
    imputer: SimpleImputer
    scaler: StandardScaler
    clusterer: KMeans
    limits: dict
    profiles: list[dict]
    window_size: int
    minimum_history: int
    metadata: dict

    def predict(self, feature):
        feature = np.asarray(feature, dtype=float).reshape(1, -1)
        if feature.shape[1] != len(FEATURE_NAMES):
            raise ValueError("Davranış özelliklerinin sayısı farklı.")
        with threadpool_limits(limits=2):
            z = self.scaler.transform(self.imputer.transform(feature))
            cluster = int(self.clusterer.predict(z)[0])
            distances = self.clusterer.transform(z)[0]
        return {"cluster": cluster, "description": self.profiles[cluster]["description"],
                "distance_to_center": float(distances[cluster]),
                "note": "En yakın davranış grubudur; güven olasılığı veya kalıcı öğrenme tarzı değildir."}


def _finite(value):
    return float(value) if np.isfinite(value) else None


def fit_behavior_model(store, tracer, bank, config, output: Path, progress=None):
    snapshots = training_snapshots(store, config["minimum_history"],
                                   config["training_snapshots_per_student"])
    if not snapshots:
        raise ValueError("Davranış gruplama için eğitim geçmişi bulunamadı.")
    assert all(store.arrays["split"][snapshot.start] == 0 for snapshot in snapshots)
    limits = fit_observation_limits(store)
    skills = [active_skills(store, snapshot, config["reference_skills_per_snapshot"],
                           config["window_size"], bank) for snapshot in snapshots]
    scores = tracer.reference_scores(snapshots, skills, bank, config["prediction_batch_size"], progress)
    features = np.stack([behavior_vector(store, snapshot, values, limits, config["window_size"])
                         for snapshot, values in zip(snapshots, scores)])
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    scaler = StandardScaler()
    z = scaler.fit_transform(imputer.fit_transform(features))
    candidates, best = [], None
    with threadpool_limits(limits=2):
        for count in config["cluster_candidates"]:
            if count >= len(z):
                continue
            clusterer = KMeans(n_clusters=count, init="k-means++", n_init=10,
                              max_iter=300, random_state=config["random_seed"])
            labels = clusterer.fit_predict(z)
            sizes = np.bincount(labels, minlength=count)
            if len(np.unique(labels)) < 2:
                continue
            score = float(silhouette_score(z, labels, sample_size=min(config["silhouette_sample_size"], len(z)),
                                            random_state=config["random_seed"]))
            accepted = bool(sizes.min()/len(z) >= config["minimum_cluster_fraction"])
            row = {"clusters": count, "silhouette": score, "min_cluster_fraction": float(sizes.min()/len(z)),
                   "cluster_sizes": sizes.tolist(), "size_constraint_passed": accepted}
            candidates.append(row)
            if accepted and (best is None or score > best[0]):
                best = score, clusterer, labels
    if best is None:
        raise ValueError("Grup büyüklüğü koşulunu sağlayan kümeleme bulunamadı; sessizce farklı kural uygulanmadı.")
    score, clusterer, labels = best
    profiles = []
    global_accuracy = float(np.mean(features[:, 0]))
    global_times = features[:, 2][np.isfinite(features[:, 2])]
    global_time = float(np.median(global_times)) if len(global_times) else np.nan
    for cluster in range(clusterer.n_clusters):
        chosen = labels == cluster
        group = features[chosen]
        accuracy = float(group[:, 0].mean())
        valid_times = group[:, 2][np.isfinite(group[:, 2])]
        time_value = float(np.median(valid_times)) if len(valid_times) else np.nan
        time_description = (f"yanıt süresi görece {'uzun' if time_value >= global_time else 'kısa'}"
                            if np.isfinite(time_value) else "yanıt süresi bilgisi eksik")
        description = (f"Doğru cevap oranı görece {'yüksek' if accuracy >= global_accuracy else 'düşük'}; "
                       f"{time_description}")
        profiles.append({"cluster": cluster, "snapshots": int(chosen.sum()),
                         "students_with_snapshot_in_group": len({snapshots[i].student for i in np.flatnonzero(chosen)}),
                         "mean_observed_accuracy": accuracy,
                         "mean_reference_probability": _finite(np.nanmean(group[:, 1])),
                         "median_response_seconds": _finite(np.expm1(time_value)),
                         "mean_hints": _finite(np.nanmean(group[:, 3])), "description": description})
    metadata = {"fitted_on": "training students only", "training_students": len(store.select_students(0)),
                "training_snapshots": len(snapshots), "selected_clusters": clusterer.n_clusters,
                "selected_silhouette": score, "feature_names": FEATURE_NAMES,
                "expanded_feature_names": imputer.get_feature_names_out(FEATURE_NAMES).tolist(),
                "reference_bank_sha256": bank.source_sha256,
                "model_method": tracer.selection["method"], "model_seed": tracer.selection["seed"],
                "configuration": config, "limits": limits, "candidates": candidates,
                **getattr(tracer, "provenance", {}),
                "profiles": profiles,
                "interpretation": "Gruplar yakın dönem pencerelerini açıklar; öğrencinin değişmez niteliği değildir."}
    model = BehaviorModel(imputer, scaler, clusterer, limits, profiles,
                           config["window_size"], config["minimum_history"], metadata)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "behavior_model.joblib")
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    width = config["reference_skills_per_snapshot"]
    skill_matrix = np.zeros((len(snapshots), width), dtype=np.int32)
    probability_matrix = np.full((len(snapshots), width), np.nan, dtype=np.float32)
    for index, values in enumerate(scores):
        for column, (skill, probability) in enumerate(values.items()):
            skill_matrix[index, column], probability_matrix[index, column] = skill, probability
    np.savez_compressed(output / "training_snapshots.npz", features=features, cluster=labels,
                        student=np.array([item.student for item in snapshots], dtype=np.int32),
                        start=np.array([item.start for item in snapshots], dtype=np.int64),
                        stop=np.array([item.stop for item in snapshots], dtype=np.int64),
                        reference_skill=skill_matrix, reference_probability=probability_matrix)
    return model
