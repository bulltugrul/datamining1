"""Kayıtlı pilot ağırlıklarını yeniden yükle ve gerçek tahminlerle denetle."""

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / ".runtime" / "cuda"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from kt_project.models import LargeKnowledgeModel, KnowledgeTracer, ModelConfig
from kt_project.sequences import InteractionStore, PrefixDataset, DevicePrefixBatches
from kt_project.training import evaluate, state_digest, autocast_context


def main():
    run = ROOT / "outputs/pilot/lkm_128_device_batches"
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert not results["test_evaluated"]
    assert all(item["test"] is None for item in results["runs"])
    assert len({item["initial_lkm_sha256"] for item in results["runs"]}) == 1
    device = torch.device(results["device"])
    torch.set_num_threads(4)
    store = InteractionStore(ROOT / "data/processed/interactions.npz")
    dataset = PrefixDataset(store, 1, results["model_config"]["max_history"],
                            config["validation_student_limit"], config["validation_target_limit"], 17)
    pretrained = torch.load(run / "seed_17/pretrained.pt", weights_only=True, map_location="cpu")
    assert state_digest(pretrained["model_state"]) == results["runs"][0]["initial_lkm_sha256"]
    reports = []
    for item in results["runs"]:
        method = item["method"]
        saved = torch.load(run / f"seed_17/{method}_best.pt", weights_only=True, map_location="cpu")
        model_config = ModelConfig(**saved["config"])
        model = KnowledgeTracer(LargeKnowledgeModel(model_config), method).to(device)
        model.load_state_dict(saved["model_state"])
        batches = DevicePrefixBatches(dataset, config["evaluation_batch_size"], device)
        metrics, predictions = evaluate(model, batches, device, results["precision"])
        for key in ("bce", "accuracy", "auc"):
            assert np.isclose(metrics[key], item["validation"][key], rtol=1e-5, atol=1e-6)
        assert (store.arrays["split"][predictions["row_index"]] == 1).all()
        np.testing.assert_array_equal(predictions["label"], store.arrays["correct"][predictions["row_index"]])
        batch = next(iter(batches))
        changed = {**batch, "label": 1-batch["label"]}
        model.eval()
        with torch.no_grad(), autocast_context(device, results["precision"]):
            a, b = model(batch), model(changed)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        gradient = None
        if method != "baseline":
            model.train()
            model.zero_grad(set_to_none=True)
            with autocast_context(device, results["precision"]):
                loss = model.loss(batch)
            loss.backward()
            gradient = float(model.prompt_encoder.soft_prompt.grad.float().norm())
            assert np.isfinite(gradient) and gradient > 0
        reports.append({"method": method, "checkpoint_reloaded": True,
                        "validation_metrics_reproduced": True,
                        "target_label_invariance": True, "prompt_gradient_norm": gradient})
        del model
    report = {"device": str(device), "common_pretrained_state_verified": True,
              "test_not_evaluated": True, "checks": reports}
    output = ROOT / "outputs/validation/pilot_integrity.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
