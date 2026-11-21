"""Davranış grupları ve yeniden oynatım çıktılarının gerçek veriyle denetimi."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from kt_project.patterns import behavior_vector, fit_observation_limits
from kt_project.recommendations import training_question_catalog, challenge_probability
from kt_project.replay import observed_skill_rates, choose_replay_student
from kt_project.sequences import InteractionStore
from kt_project.tracking import ReferenceBank, snapshot_for


def main():
    store = InteractionStore(ROOT / "data/processed/interactions.npz")
    metadata = json.loads((ROOT / "outputs/patterns/metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(ROOT / "outputs/patterns/behavior_model.joblib")
    bank = ReferenceBank(ROOT / "data/processed/references.json")
    config = json.loads((ROOT / "configs/tracking.json").read_text(encoding="utf-8"))
    assert metadata["fitted_on"] == "training students only"
    assert metadata["reference_bank_sha256"] == bank.source_sha256
    assert metadata["limits"] == fit_observation_limits(store)
    with np.load(ROOT / "outputs/patterns/training_snapshots.npz", allow_pickle=False) as packed:
        features, students, starts, stops, labels = (packed[key] for key in
                                                    ("features", "student", "start", "stop", "cluster"))
        ref_skills, ref_probs = packed["reference_skill"], packed["reference_probability"]
        for index, (student, start, stop) in enumerate(zip(students, starts, stops)):
            assert store.arrays["split"][start] == 0
            assert store.spans[int(student)][0] == start
            assert start+config["minimum_history"] <= stop <= store.spans[int(student)][1]
            scores = {int(skill): float(probability) for skill, probability in zip(ref_skills[index], ref_probs[index]) if skill > 0}
            actual = behavior_vector(store, snapshot_for(store, int(student), int(stop)), scores,
                                      model.limits, config["window_size"])
            np.testing.assert_allclose(actual, features[index], atol=1e-7, rtol=0, equal_nan=True)
        imputed = model.imputer.transform(features)
        np.testing.assert_allclose(model.scaler.mean_, imputed.mean(axis=0), atol=1e-12)
        with threadpool_limits(limits=2):
            np.testing.assert_array_equal(model.clusterer.predict(model.scaler.transform(imputed)), labels)
    accepted = [row for row in metadata["candidates"] if row["size_constraint_passed"]]
    selected = max(accepted, key=lambda row: row["silhouette"])
    assert selected["clusters"] == metadata["selected_clusters"]
    session = json.loads((ROOT / "outputs/replay/session.json").read_text(encoding="utf-8"))
    assert metadata.get("inference_policy", "legacy_causal_lkm_bidirectional_prompt") == session["model_selection"].get("inference_policy", "legacy_causal_lkm_bidirectional_prompt")
    assert session["encoded_student"] == choose_replay_student(store, config)
    assert not session["selection_uses_response_outcomes"]
    assert session["source_split"] == "validation"
    student = session["encoded_student"]
    start, stop = store.spans[student]
    assert store.arrays["split"][start] == 1
    catalog = training_question_catalog(store)
    times = [datetime.fromisoformat(session["display_started_at"])]
    for event in session["events"]:
        target = event["target_row_index"]
        assert start <= target < stop
        assert event["history_stop_before"] == target
        assert event["observed_correct"] == int(store.arrays["correct"][target])
        assert event["source_order_id"] == int(store.arrays["order"][target])
        assert event["source_timestamp"] is None
        assert event["after"]["history_count"] == target+1-start
        snapshot = snapshot_for(store, student, target+1)
        displayed_skills = [row["skill"] for row in event["after"]["skill_scores"]]
        observed = observed_skill_rates(store, snapshot, displayed_skills, config["window_size"])
        for row in event["after"]["skill_scores"]:
            assert row["reference_question_count"] == len(bank.for_skill(row["skill"]))
            assert row["recent_skill_answers"] == observed[row["skill"]]["answers"]
            expected = observed[row["skill"]]["ratio"]
            if expected is None:
                assert row["observed_correct_percent"] is None
            else:
                assert np.isclose(row["observed_correct_percent"], 100*expected)
        recommendation = event["after"]["recommendation"]
        if recommendation["available"]:
            assert recommendation["question"] in catalog
            if not recommendation["repeat_question"]:
                assert recommendation["question"] not in store.arrays["question"][start:target+1]
            left = max(start, target+1-config["window_size"])
            expected_target, _ = challenge_probability(store.arrays["correct"][left:target+1], config)
            assert expected_target == recommendation["target_probability"]
        times.append(datetime.fromisoformat(event["display_timestamp"]))
    assert all(left <= right for left, right in zip(times, times[1:]))
    assert all(value for key, value in session["integrity"].items() if key != "source_calendar_times_fabricated")
    main_results = json.loads((ROOT / "outputs/main/three_seed_v1/results.json").read_text())
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
               for name, digest in main_results["implementation_sha256"].items())
    report = {"training_only_pattern_snapshots_verified": True, "snapshots_checked": len(features),
              "selected_clusters": metadata["selected_clusters"],
              "feature_values_recomputed": True, "scaler_training_statistics_verified": True,
              "group_selection_verified": True, "real_replay_outcomes_verified": True,
              "replay_steps": len(session["events"]), "recommendation_rules_verified": True,
              "model_and_reference_immutability_verified": True,
              "source_calendar_timestamps_not_fabricated": True,
              "main_experiment_source_hashes_unchanged": True}
    output = ROOT / "outputs/validation/student_tracking_integrity.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
