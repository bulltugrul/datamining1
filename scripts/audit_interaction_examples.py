"""Örnek dosyalar, yönlendirme ölçüleri ve mevcut sorularla bağlantıyı doğrula."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from kt_project.intent import LABELS, normalize_text
from kt_project.interactions import history_digest
from kt_project.media import load_audio, word_error_rate
from kt_project.recommendations import training_question_catalog
from kt_project.sequences import InteractionStore
from kt_project.tracking import snapshot_for


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main():
    result = read("outputs/interactions/examples.json")
    model = joblib.load(ROOT / "outputs/interactions/intent_model.joblib")
    corpus = read("data/interactions/intent_corpus.json")
    manifest = read("data/interactions/manifest.json")
    metrics = read("outputs/interactions/intent_metrics.json")
    assert hashlib.sha256((ROOT / "data/interactions/intent_corpus.json").read_bytes()).hexdigest() == metrics["corpus_sha256"]
    test_rows = [row for row in corpus if row["split"] == "test"]
    actual = [row["label"] for row in test_rows]
    predicted = [model.predict(row["text"])["label"] for row in test_rows]
    assert accuracy_score(actual, predicted) == metrics["test_accuracy"]
    assert f1_score(actual, predicted, labels=LABELS, average="macro", zero_division=0) == metrics["test_macro_f1"]
    provenance = read("outputs/interactions/asr_model_provenance.json")
    config = read("configs/interactions.json")
    assert provenance["revision"] == config["asr_revision"]
    for filename, digest in provenance["file_sha256"].items():
        assert hashlib.sha256((ROOT / "data/models/whisper-base" / filename).read_bytes()).hexdigest() == digest
    store = InteractionStore(ROOT / "data/processed/interactions.npz")
    session = read("outputs/replay/session.json")
    assert result["model_selection"]["checkpoint_sha256"] == session["model_selection"]["checkpoint_sha256"]
    assert result["model_selection"].get("inference_policy") == session["model_selection"].get("inference_policy")
    start, _ = store.spans[session["encoded_student"]]
    snapshot = snapshot_for(store, session["encoded_student"], start+session["events"][-1]["after"]["history_count"])
    assert result["state_sha256_before"]["history"] == history_digest(store, snapshot)
    catalog = training_question_catalog(store)
    inverse = {int(value): int(key) for key, value in read("data/processed/vocabulary.json")["question"].items()}
    normalized = {normalize_text(row["text"]) for row in corpus}
    entries = {row["id"]: row for row in manifest["samples"]}
    assert len(entries) == len(result["results"]) == 9
    for row in result["results"]:
        sample = entries[row["sample_id"]]
        assert normalize_text(sample["expected_text"]) not in normalized
        assert hashlib.sha256((ROOT / sample["path"]).read_bytes()).hexdigest() == row["input_sha256"] == sample["sha256"]
        if row["modality"] == "audio":
            _, audio_metadata = load_audio(ROOT / sample["path"])
            assert audio_metadata["model_sample_rate"] == row["recognition"]["model_sample_rate"] == 16000
            assert row["recognition"]["revision"] == config["asr_revision"]
        if row["modality"] == "text":
            assert (ROOT / sample["path"]).read_text(encoding="utf-8") == row["recognized_text"]
        assert row["recognized_text"] == row["recognition"]["text"]
        routing = model.predict(row["recognized_text"])
        assert routing["label"] == row["intent"]["label"]
        assert row["evaluation"]["label_correct"] == (routing["label"] == sample["expected_label"])
        assert row["evaluation"]["normalized_word_error_rate"] == word_error_rate(sample["expected_text"], row["recognized_text"])
        assert row["evaluation"]["reference_word_count"] == len(normalize_text(sample["expected_text"]).split())
        assert row["history_unchanged"] and not row["response_is_assessed_answer"]
        assert row["student_history_answers"] == snapshot.stop-snapshot.start
        recommendation = row["recommendation"]
        if recommendation is not None and recommendation["available"]:
            question = recommendation["question"]
            assert question in catalog
            assert routing["skill"]["encoded_skill"] in catalog[question]["skills"]
            assert recommendation["original_problem_id"] == inverse[question]
            if not recommendation["repeat_question"]:
                assert question not in store.arrays["question"][snapshot.start:snapshot.stop]
    for summary in result["modality_summary"]:
        subset = [row for row in result["results"] if row["modality"] == summary["modality"]]
        assert len(subset) == summary["examples"]
        assert sum(row["evaluation"]["label_correct"] for row in subset) == summary["correct_skill_routes"]
        errors = sum(row["evaluation"]["reference_word_count"]*row["evaluation"]["normalized_word_error_rate"] for row in subset)
        total = sum(row["evaluation"]["reference_word_count"] for row in subset)
        assert np.isclose(summary["pooled_normalized_word_error_rate"], errors/total)
    assert result["state_sha256_before"] == result["state_sha256_after"]
    assert all(result["integrity"].values())
    core = read("outputs/main/three_seed_v1/results.json")
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
               for name, digest in core["implementation_sha256"].items())
    report = {"input_hashes_verified": True, "asr_weights_and_revision_verified": True,
              "corpus_and_demo_expressions_disjoint": True, "test_metrics_recomputed": True,
              "recognition_error_metrics_recomputed": True, "routing_from_actual_recognized_text_verified": True,
              "recommendations_are_known_training_questions": True, "requests_do_not_append_answers": True,
              "unchanged_state_hashes_verified": True, "main_experiment_sources_unchanged": True,
              "examples_checked": 9}
    (ROOT / "outputs/validation/interaction_integrity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
