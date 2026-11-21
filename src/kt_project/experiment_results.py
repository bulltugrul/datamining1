"""Ana deneyin tamlığını, ortak örneklerini ve istatistiklerini denetle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

METHODS = ("baseline", "kpt", "kmpt")
SEEDS = (17, 42, 2026)


def aggregate_runs(runs):
    expected = {(method, seed) for method in METHODS for seed in SEEDS}
    observed = [(row["method"], row["seed"]) for row in runs]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Ana karşılaştırma her yöntem ve başlangıç için tam bir sonuç gerektirir.")
    summaries = {}
    baseline = {row["seed"]: row for row in runs if row["method"] == "baseline"}
    for method in METHODS:
        selected = [row for row in runs if row["method"] == method]
        if any(row.get("test") is None for row in selected):
            raise ValueError("Ana karşılaştırmada test sonucu eksik.")
        scores = {}
        for metric in ("auc", "accuracy", "bce"):
            values = np.array([row["test"][metric] for row in selected], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Ana ölçüler sonlu olmalı.")
            scores[metric] = {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1))}
        differences = np.array([row["test"]["auc"] - baseline[row["seed"]]["test"]["auc"]
                                for row in selected])
        summaries[method] = {
            "seeds": len(selected), "test": scores,
            "auc_difference_from_baseline": {"mean": float(differences.mean()),
                                               "sample_sd": float(differences.std(ddof=1))},
            "mean_training_seconds": float(np.mean([row["training_seconds"] for row in selected])),
        }
    return summaries


def select_prototype(runs, policy):
    candidates = [row for row in runs if row["method"] == policy["prototype_method"]]
    if len(candidates) != 3:
        raise ValueError("Örnek oturum seçimi için üç KMPT doğrulama sonucu gerekli.")
    chosen = min(candidates, key=lambda row: (row["validation"]["bce"], row["seed"]))
    return {"method": chosen["method"], "seed": chosen["seed"],
            "criterion": "minimum validation BCE within the preselected KMPT method",
            "validation_bce": chosen["validation"]["bce"],
            "best_epoch": chosen["best_epoch"],
            "test_metrics_used_for_selection": False}


def audit_main_run(root: Path, run_dir: Path):
    root, run_dir = Path(root), Path(run_dir)
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "complete":
        raise ValueError("Ana deney henüz tamamlanmadı.")
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "data/processed/metadata.json").read_text(encoding="utf-8"))
    if results["source_sha256"] != metadata["source_sha256"]:
        raise ValueError("Veri kaynağının içerik özeti uyuşmuyor.")
    if not results["test_evaluated"] or config["seeds"] != list(SEEDS):
        raise ValueError("Ana deney ayarları uyuşmuyor.")
    if any(config.get(key) is not None for key in
           ("train_student_limit", "validation_student_limit", "train_target_limit", "validation_target_limit")):
        raise ValueError("Ana deneyde örnek sınırlaması olmamalı.")
    summary = aggregate_runs(results["runs"])
    source_hashes = results["implementation_sha256"]
    required_sources = {"src/kt_project/models.py", "src/kt_project/sequences.py",
                        "src/kt_project/training.py", "src/kt_project/prepare_data.py"}
    if set(source_hashes) != required_sources:
        raise ValueError("Uygulama kaynaklarının içerik özeti eksik.")
    for name, expected in source_hashes.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Eğitimden sonra uygulama kaynağı değişmiş: {name}")
    for seed in SEEDS:
        values = {row["initial_lkm_sha256"] for row in results["runs"] if row["seed"] == seed}
        if len(values) != 1:
            raise ValueError("Bir başlangıç içindeki LKM ağırlıkları farklı.")
    if len({row["initial_lkm_sha256"] for row in results["runs"]}) != 3:
        raise ValueError("Üç farklı başlangıcın ön eğitim ağırlıkları farklı olmalı.")
    import torch
    from .training import state_digest
    for seed in SEEDS:
        checkpoint = torch.load(run_dir / f"seed_{seed}/pretrained.pt", weights_only=True, map_location="cpu")
        expected_digest = next(row["initial_lkm_sha256"] for row in results["runs"] if row["seed"] == seed)
        if state_digest(checkpoint["model_state"]) != expected_digest:
            raise ValueError("Ön eğitim ağırlıklarının içerik özeti farklı.")
    with np.load(root / "data/processed/interactions.npz", allow_pickle=False) as packed:
        actual_labels = packed["correct"].copy()
        splits = packed["split"].copy()
        questions = packed["question"].copy()
        students = packed["student"].copy()
    student_groups = [set(students[splits == code].tolist()) for code in range(3)]
    if any(student_groups[left] & student_groups[right]
           for left in range(3) for right in range(left+1, 3)):
        raise ValueError("Aynı öğrenci farklı gruplarda bulunuyor.")
    has_history = np.r_[False, students[1:] == students[:-1]]
    expected_rows = np.flatnonzero((splits == 2) & has_history)
    train_count = int(((splits == 0) & has_history).sum())
    validation_count = int(((splits == 1) & has_history).sum())
    checks = []
    for row in results["runs"]:
        method, seed = row["method"], row["seed"]
        if row["training_target_count"] != train_count or row["validation_target_count"] != validation_count:
            raise ValueError("Tam eğitim veya doğrulama hedefleri kullanılmamış.")
        folder = run_dir / f"seed_{seed}"
        history = json.loads((folder / f"{method}_history.json").read_text(encoding="utf-8"))
        selected = history[row["best_epoch"]-1]
        best_bce = min(item["validation"]["bce"] for item in history)
        if abs(selected["validation"]["bce"]-best_bce) > 1e-6:
            raise ValueError("Seçilen dönem doğrulama BCE kuralına uymuyor.")
        if not np.isclose(selected["validation"]["bce"], row["validation"]["bce"], atol=1e-6):
            raise ValueError("Kayıtlı seçilmiş doğrulama sonucu uyuşmuyor.")
        with np.load(folder / f"{method}_test_predictions.npz", allow_pickle=False) as packed:
            y, p, indices = packed["label"], packed["probability"], packed["row_index"]
        np.testing.assert_array_equal(indices, expected_rows)
        np.testing.assert_array_equal(y, actual_labels[indices])
        if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
            raise ValueError("Geçersiz olasılık bulundu.")
        auc = float(roc_auc_score(y, p))
        accuracy = float(accuracy_score(y, p >= .5))
        clipped = np.clip(p.astype(np.float64), np.finfo(np.float64).eps,
                          1-np.finfo(np.float64).eps)
        bce = float(-(y*np.log(clipped)+(1-y)*np.log1p(-clipped)).mean())
        if not np.isclose(auc, row["test"]["auc"], rtol=0, atol=1e-10):
            raise ValueError("Kaydedilen tahminlerin AUC değeri farklı.")
        if not np.isclose(accuracy, row["test"]["accuracy"], rtol=0, atol=1e-10):
            raise ValueError("Kaydedilen tahminlerin doğruluğu farklı.")
        if not np.isclose(bce, row["test"]["bce"], rtol=0, atol=5e-5):
            raise ValueError("Kaydedilen tahminlerin BCE değeri sayısal toleransı aşıyor.")
        known = questions[indices] != 1
        if int(known.sum()) != row["test"]["known_question_subset"]["samples"]:
            raise ValueError("Bilinen soru alt grubunun sayısı uyuşmuyor.")
        if not np.isclose(roc_auc_score(y[known], p[known]),
                          row["test"]["known_question_subset"]["auc"], atol=1e-10, rtol=0):
            raise ValueError("Bilinen soru alt grubunun AUC değeri uyuşmuyor.")
        if not (folder / f"{method}_best.pt").is_file():
            raise ValueError("Model ağırlığı eksik.")
        saved = torch.load(folder / f"{method}_best.pt", weights_only=True, map_location="cpu")
        if (saved["method"] != method or saved["seed"] != seed
                or saved["best_epoch"] != row["best_epoch"]
                or saved["initial_lkm_sha256"] != row["initial_lkm_sha256"]
                or saved["config"] != results["model_config"]):
            raise ValueError("Kayıtlı modelin deney bilgileri farklı.")
        if not all(bool(torch.isfinite(value).all()) for value in saved["model_state"].values()):
            raise ValueError("Kayıtlı ağırlıklarda sonlu olmayan değer var.")
        checks.append({"method": method, "seed": seed, "samples": len(y),
                       "same_test_targets": True, "ground_truth_verified": True,
                       "auc_recomputed": auc, "accuracy_recomputed": accuracy,
                       "bce_recomputed_from_probabilities": bce,
                       "validation_selection_verified": True})
    integrity = {
        "complete_method_seed_pairs": len(checks), "all_checks_passed": True,
        "same_test_targets_for_all_runs": True, "test_targets": len(expected_rows),
        "training_targets": train_count, "validation_targets": validation_count,
        "test_students": int(len(np.unique(students[expected_rows]))),
        "unseen_question_test_targets": int((questions[expected_rows] == 1).sum()),
        "implementation_hashes_verified": True, "checks": checks,
        "saved_checkpoint_metadata_and_finiteness_verified": True,
        "variability_interpretation": "sample SD across initialization seeds on one fixed student split",
    }
    return results, summary, integrity
