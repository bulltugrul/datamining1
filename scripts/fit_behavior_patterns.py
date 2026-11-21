"""Eğitim geçmişlerinde sabit KMPT göstergeleriyle davranış gruplarını öğren."""

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / ".runtime/cuda"))
sys.path.insert(0, str(ROOT / "src"))

import torch
from kt_project.patterns import fit_behavior_model
from kt_project.sequences import InteractionStore
from kt_project.tracking import FrozenTracer, ReferenceBank


def main():
    torch.set_num_threads(4)
    config = json.loads((ROOT / "configs/tracking.json").read_text(encoding="utf-8"))
    output = ROOT / "outputs/patterns"
    if (output / "behavior_model.joblib").exists():
        raise SystemExit("Davranış modeli var; otomatik yeniden eğitim yapılmadı.")
    store = InteractionStore(ROOT / "data/processed/interactions.npz")
    tracer = FrozenTracer(ROOT, store)
    bank = ReferenceBank(ROOT / "data/processed/references.json")
    def progress(done, total):
        if done % 512 == 0 or done == total:
            print(json.dumps({"phase": "reference_scores", "snapshots": done, "total": total}), flush=True)
    model = fit_behavior_model(store, tracer, bank, config, output, progress)
    print(json.dumps({"phase": "complete", "snapshots": model.metadata["training_snapshots"],
                      "clusters": model.metadata["selected_clusters"],
                      "silhouette": model.metadata["selected_silhouette"]}), flush=True)


if __name__ == "__main__":
    main()
