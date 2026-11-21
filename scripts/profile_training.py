"""Ana deneyden önce yalnızca eğitim girdileriyle hız ve bellek ölç."""

import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / ".runtime" / "cuda"))
sys.path.insert(0, str(ROOT / "src"))

import torch
from kt_project.models import LargeKnowledgeModel, KnowledgeTracer, ModelConfig
from kt_project.sequences import InteractionStore, PrefixDataset, DevicePrefixBatches
from kt_project.training import set_seed, autocast_context


def main():
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise SystemExit("Bu ölçüm CUDA gerektirir.")
    metadata = json.loads((ROOT / "data/processed/metadata.json").read_text())
    config = ModelConfig(metadata["question_vocabulary_size"], metadata["skill_vocabulary_size"])
    store = InteractionStore(ROOT / "data/processed/interactions.npz")
    dataset = PrefixDataset(store, 0, config.max_history)
    reports = []
    for size in (128, 512, 1024):
        set_seed(17)
        model = KnowledgeTracer(LargeKnowledgeModel(config), "kpt").cuda().train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0005)
        iterator = iter(DevicePrefixBatches(dataset, size, "cuda", seed=17, shuffle=True))
        torch.cuda.reset_peak_memory_stats()
        try:
            # İlk iki adım kütüphane başlatma ve ısınma içindir.
            for step in range(10):
                batch = next(iterator)
                if step == 2:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(torch.device("cuda"), "bfloat16"):
                    loss = model.loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Profil kaybı sonlu değil.")
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            seconds = time.perf_counter()-started
            report = {"batch_size": size, "measured_steps": 8,
                      "seconds_per_step": seconds/8, "targets_per_second": 8*size/seconds,
                      "peak_allocated_gpu_mb": torch.cuda.max_memory_allocated()/1024**2}
        except torch.OutOfMemoryError:
            report = {"batch_size": size, "out_of_memory": True}
        reports.append(report)
        print(json.dumps(report), flush=True)
        del iterator, optimizer, model, loss, batch
        gc.collect()
        torch.cuda.empty_cache()
    output = ROOT / "outputs/pilot/training_profile.json"
    output.write_text(json.dumps({"purpose": "training_data_only_performance_probe", "runs": reports}, indent=2))


if __name__ == "__main__":
    main()
