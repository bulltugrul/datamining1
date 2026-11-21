"""Yalnızca eğitim örneklerinde yeni modelin toplu işlem süresi ve belleği."""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from kt_project.models import ModelConfig
from kt_project.plkt_models import PaperLKM, PaperKnowledgeTracer
from kt_project.sequences import InteractionStore, PrefixDataset, DevicePrefixBatches
from kt_project.training import autocast_context, set_seed


def main():
    torch.set_num_threads(4)
    device = torch.device("cuda")
    config = json.loads((ROOT / "configs/paper_alignment.json").read_text())
    meta = json.loads((ROOT / "data/processed/metadata.json").read_text())
    model_config = ModelConfig(meta["question_vocabulary_size"], meta["skill_vocabulary_size"], **config["model"])
    store = InteractionStore(ROOT / "data/processed/interactions.npz")
    data = PrefixDataset(store, 0, model_config.max_history, target_limit=10000)
    rows = []
    for size in (64, 256, 512):
        set_seed(17)
        model = PaperKnowledgeTracer(PaperLKM(model_config), "kmpt").to(device).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
        iterator = iter(DevicePrefixBatches(data, size, device, shuffle=True))
        torch.cuda.reset_peak_memory_stats()
        times = []
        for i in range(10):
            batch = next(iterator)
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, "bfloat16"):
                loss = model.loss(batch, auxiliary=i % 5 == 0)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            if i >= 3:
                times.append(time.perf_counter()-started)
        rows.append({"batch_size": size, "seconds_per_step": sum(times)/len(times),
                     "targets_per_second": size/(sum(times)/len(times)),
                     "peak_gpu_mb": torch.cuda.max_memory_allocated()/1024**2})
        del model, optimizer
        print(json.dumps(rows[-1]), flush=True)
    output = ROOT / "outputs/method_audit"
    output.mkdir(parents=True, exist_ok=True)
    (output / "throughput_profile.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
