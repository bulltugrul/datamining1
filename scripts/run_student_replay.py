"""Eğitimden öğrenilmiş gruplar ve sabit KMPT ile gerçek kayıtları yeniden oynat."""

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / ".runtime/cuda"))
sys.path.insert(0, str(ROOT / "src"))

import joblib
import torch
from kt_project.recommendations import training_question_catalog
from kt_project.replay import run_replay
from kt_project.sequences import InteractionStore
from kt_project.tracking import FrozenTracer, ReferenceBank


def main():
    torch.set_num_threads(4)
    config = json.loads((ROOT / "configs/tracking.json").read_text(encoding="utf-8"))
    store = InteractionStore(ROOT / "data/processed/interactions.npz")
    tracer = FrozenTracer(ROOT, store)
    bank = ReferenceBank(ROOT / "data/processed/references.json")
    patterns = joblib.load(ROOT / "outputs/patterns/behavior_model.joblib")
    if patterns.metadata["reference_bank_sha256"] != bank.source_sha256:
        raise ValueError("Davranış modeli farklı referans sorularıyla öğrenilmiş.")
    if patterns.metadata["checkpoint_sha256"] != tracer.provenance["checkpoint_sha256"]:
        raise ValueError("Davranış modeli farklı bilgi takibi ağırlığıyla öğrenilmiş.")
    if patterns.metadata.get("inference_policy", "legacy_causal_lkm_bidirectional_prompt") != tracer.provenance["inference_policy"]:
        raise ValueError("Davranış modelinin dikkat politikası farklı.")
    for key in ("window_size", "minimum_history", "reference_skills_per_snapshot"):
        if patterns.metadata["configuration"][key] != config[key]:
            raise ValueError(f"Davranış modelinin özellik ayarı değişmiş: {key}")
    catalog = training_question_catalog(store)
    def progress(done, total):
        if done % 5 == 0 or done == total:
            print(json.dumps({"phase": "replay", "step": done, "total": total}), flush=True)
    result = run_replay(ROOT, store, tracer, bank, patterns, catalog, config, progress)
    output = ROOT / "outputs/replay"
    output.mkdir(parents=True, exist_ok=True)
    (output / "session.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"phase": "complete", "steps": len(result["events"]),
                      "integrity": result["integrity"]}), flush=True)


if __name__ == "__main__":
    main()
