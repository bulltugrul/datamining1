"""Metin, ses ve görsel dosyalarını gerçekten işle; aynı öneri akışını kullan."""

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".runtime/huggingface"))
os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / ".runtime/cuda"))
import torch
from kt_project.intent import train_intent_model, normalize_text
from kt_project.interactions import InteractionService, history_digest
from kt_project.media import SpeechReader, read_image, word_error_rate
from kt_project.replay import pattern_state_digest
from kt_project.training import cpu_state, state_digest


def main():
    torch.set_num_threads(4)
    directory = ROOT / "outputs/interactions"
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "intent_model.joblib").exists():
        model = train_intent_model(ROOT)
        print(json.dumps({"phase": "intent_training", "test_accuracy": model.metadata["test_accuracy"],
                          "threshold": model.threshold}), flush=True)
    service = InteractionService(ROOT)
    manifest = json.loads((ROOT / "data/interactions/manifest.json").read_text(encoding="utf-8"))
    before = {"knowledge_model": state_digest(cpu_state(service.tracer.model)),
              "patterns": pattern_state_digest(service.patterns),
              "references": service.bank.content_sha256(),
              "history": history_digest(service.store, service.snapshot)}
    asr, results = None, []
    for sample in manifest["samples"]:
        path = ROOT / sample["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != sample["sha256"]:
            raise ValueError("Örnek dosya özeti değişti: " + sample["id"])
        if sample["modality"] == "text":
            recognition = {"text": path.read_text(encoding="utf-8"), "engine": "UTF-8 text input"}
        elif sample["modality"] == "audio":
            if asr is None:
                asr = SpeechReader(ROOT)
            recognition = asr.read(path)
        else:
            recognition = read_image(ROOT, path)
        # Beklenen metin ve etiket yalnızca işleme tamamlandıktan sonra değerlendirmede kullanılır.
        result = service.respond(recognition["text"])
        results.append({"sample_id": sample["id"], "modality": sample["modality"], "input_path": sample["path"],
                        "input_sha256": sample["sha256"], "recognition": recognition, **result,
                        "evaluation": {"expected_text": sample["expected_text"],
                                       "expected_label": sample["expected_label"],
                                       "label_correct": result["intent"]["label"] == sample["expected_label"],
                                       "exact_text_match": recognition["text"] == sample["expected_text"],
                                       "reference_word_count": len(normalize_text(sample["expected_text"]).split()),
                                       "normalized_word_error_rate": word_error_rate(sample["expected_text"], recognition["text"])}})
        print(json.dumps({"phase": "example", "sample": sample["id"], "text": recognition["text"],
                          "accepted": result["intent"]["accepted"], "label": result["intent"]["label"]}, ensure_ascii=False), flush=True)
    rejection_examples = [service.respond(text) for text in [
        "Matematik çalışmak istiyorum.", "Kesirlerde çarpma yapmak istiyorum.",
        "Kesirler ve tam sayılarda toplama çalışalım.",
    ]]
    after = {"knowledge_model": state_digest(cpu_state(service.tracer.model)),
             "patterns": pattern_state_digest(service.patterns), "references": service.bank.content_sha256(),
             "history": history_digest(service.store, service.snapshot)}
    integrity = {key+"_unchanged": before[key] == after[key] for key in before}
    core = json.loads((ROOT / "outputs/main/three_seed_v1/results.json").read_text())
    integrity["main_experiment_source_hashes_unchanged"] = all(
        hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
        for name, digest in core["implementation_sha256"].items())
    integrity["rejected_requests_have_no_recommendation"] = all(
        not row["intent"]["accepted"] and row["recommendation"] is None for row in rejection_examples)
    if not all(integrity.values()):
        raise AssertionError(integrity)
    output = {"purpose": "small working modality examples, not multimodal KT training or natural speech benchmark",
              "student_context": "existing replay final history; no response or correctness appended",
              "model_selection": service.tracer.selection,
              "results": results, "rejection_examples": rejection_examples,
              "integrity": integrity, "state_sha256_before": before, "state_sha256_after": after,
              "modality_summary": [{"modality": mode, "examples": len(rows),
                                    "correct_skill_routes": sum(r["evaluation"]["label_correct"] for r in rows),
                                    "mean_normalized_word_error_rate": sum(r["evaluation"]["normalized_word_error_rate"] for r in rows)/len(rows),
                                    "pooled_normalized_word_error_rate": sum(r["evaluation"]["normalized_word_error_rate"]*r["evaluation"]["reference_word_count"] for r in rows)/sum(r["evaluation"]["reference_word_count"] for r in rows)}
                                   for mode in ["text", "audio", "image"]
                                   if (rows := [r for r in results if r["modality"] == mode])]}
    (directory / "examples.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"phase": "complete", "examples": len(results), "integrity": integrity}), flush=True)


if __name__ == "__main__":
    main()
