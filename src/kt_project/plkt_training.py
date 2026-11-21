"""Sürüm2: ortak yeni ön eğitim, doğrulamada seçim ve en sonda kilitli test."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .models import ModelConfig
from .plkt_models import PaperLKM, PaperKnowledgeTracer
from .plkt_sequences import PaperPretrainingDataset, PaperPretrainingCollator
from .sequences import InteractionStore, PrefixDataset, to_device
from .training import (autocast_context, cpu_state, evaluate, loader, log_event,
                       optimizer_step, prefix_batches, save_json, set_seed, state_digest)


def variant_settings(variant):
    return ("kmpt", "prefix_control") if variant == "kmpt_prefix" else (variant, "paper_tail")


def build_tracer(config, variant, state=None):
    lkm = PaperLKM(config)
    if state is not None:
        lkm.load_state_dict(state)
    method, layout = variant_settings(variant)
    return PaperKnowledgeTracer(lkm, method, layout)


def pretrain(store, model_config, config, seed, device, precision, directory, progress):
    path = directory / "pretrained.pt"
    if path.exists():
        saved = torch.load(path, weights_only=True, map_location="cpu")
        return saved["model_state"], saved["state_sha256"]
    set_seed(seed)
    model = PaperLKM(model_config).to(device)
    dataset = PaperPretrainingDataset(store, model_config.max_history, seed=seed)
    collator = PaperPretrainingCollator(seed+11)
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config["pretraining_lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "float16")
    history, stale, best, best_state, best_epoch = [], 0, float("inf"), None, None
    for epoch in range(1, config["pretraining_epochs"]+1):
        dataset.set_epoch(epoch-1)
        collator.set_epoch(epoch-1)
        model.train()
        sums, total = {}, 0
        started = time.perf_counter()
        for cpu_batch in loader(dataset, config["pretraining_batch_size"], collator, seed+epoch, True):
            batch = to_device(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                loss, parts = model.pretraining_loss(batch)
            optimizer_step(loss, optimizer, scaler, parameters)
            size = len(batch["fbo_label"])
            for key, value in {"loss": loss.detach(), **parts}.items():
                sums[key] = sums.get(key, 0.)+float(value)*size
            total += size
        row = {"epoch": epoch, **{key: value/total for key, value in sums.items()},
               "samples": total, "seconds": time.perf_counter()-started}
        history.append(row)
        progress(phase="pretraining", seed=seed, epoch=epoch, loss=row["loss"], seconds=row["seconds"])
        significant = row["loss"] < best*(1-config["pretraining_min_relative_improvement"])
        stale = 0 if significant else stale+1
        if row["loss"] < best:
            best, best_state, best_epoch = row["loss"], cpu_state(model), epoch
        if stale >= config["pretraining_patience"]:
            break
    digest = state_digest(best_state)
    torch.save({"model_state": best_state, "state_sha256": digest, "config": model_config.to_dict(),
                "seed": seed, "best_epoch": best_epoch, "format": "plkt_method_v2"}, path)
    save_json(directory / "pretraining_history.json", history)
    save_json(directory / "pretraining_stopping.json", {"epochs": len(history), "best_epoch": best_epoch,
        "criterion": "training objective relative improvement, not heldout/test performance",
        "reason": "relative_plateau" if stale >= config["pretraining_patience"] else "epoch_budget_reached"})
    return best_state, digest


def tune(store, model_config, initial, digest, config, seed, variant, device, precision, directory, progress):
    metrics_path = directory / f"{variant}_metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    set_seed(seed+100)
    model = build_tracer(model_config, variant, initial).to(device)
    assert state_digest(cpu_state(model.lkm)) == digest
    train = PrefixDataset(store, 0, model_config.max_history)
    validation = PrefixDataset(store, 1, model_config.max_history)
    val_batches = prefix_batches(validation, config["evaluation_batch_size"], device, seed, config)
    initial_validation, _ = evaluate(model, val_batches, device, precision)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=config["finetuning_lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "float16")
    rng = np.random.default_rng(seed+1979)
    best, best_epoch, best_state, stale = float("inf"), None, None, 0
    history, total_seconds = [], 0.
    for epoch in range(1, config["finetuning_epochs"]+1):
        model.train()
        total, response_n, auxiliary_n = 0, 0, 0
        loss_sum, response_sum, auxiliary_sum = 0., 0., 0.
        started = time.perf_counter()
        batches = prefix_batches(train, config["batch_size"], device, seed+epoch, config, True)
        for cpu_batch in batches:
            batch = to_device(cpu_batch, device)
            size = len(batch["label"])
            auxiliary = rng.random(size) < .2 if variant.startswith("kmpt") else np.zeros(size, dtype=bool)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device)
            with autocast_context(device, precision):
                for is_aux, selected in ((False, ~auxiliary), (True, auxiliary)):
                    count = int(selected.sum())
                    if not count:
                        continue
                    mask = torch.as_tensor(selected, device=device)
                    part = model.loss({key: value[mask] for key, value in batch.items()}, auxiliary=is_aux)
                    loss = loss+part*count/size
                    if is_aux:
                        auxiliary_sum += float(part.detach())*count
                        auxiliary_n += count
                    else:
                        response_sum += float(part.detach())*count
                        response_n += count
            optimizer_step(loss, optimizer, scaler, parameters)
            loss_sum += float(loss.detach())*size
            total += size
        metrics, _ = evaluate(model, val_batches, device, precision)
        seconds = time.perf_counter()-started
        total_seconds += seconds
        row = {"epoch": epoch, "validation": metrics, "seconds": seconds,
               "train_mixed_objective": loss_sum/total,
               "train_response_bce": response_sum/response_n if response_n else None,
               "train_auxiliary_question_ce": auxiliary_sum/auxiliary_n if auxiliary_n else None,
               "response_samples": response_n, "auxiliary_samples": auxiliary_n}
        history.append(row)
        progress(phase="tuning", seed=seed, variant=variant, epoch=epoch,
                 validation_bce=metrics["bce"], validation_auc=metrics["auc"], seconds=seconds)
        if metrics["bce"] < best-1e-6:
            best, best_epoch, best_state, stale = metrics["bce"], epoch, cpu_state(model), 0
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    metrics, predictions = evaluate(model, val_batches, device, precision)
    np.savez_compressed(directory / f"{variant}_validation_predictions.npz", **predictions)
    saved = {"model_state": best_state, "config": model_config.to_dict(), "format": "plkt_method_v2",
             "variant": variant, "method": model.method, "layout": model.layout, "seed": seed,
             "best_epoch": best_epoch, "initial_lkm_sha256": digest}
    torch.save(saved, directory / f"{variant}_best.pt")
    save_json(directory / f"{variant}_history.json", history)
    result = {key: saved[key] for key in ("variant", "method", "layout", "seed", "best_epoch", "initial_lkm_sha256")}
    result.update(initial_validation=initial_validation, validation=metrics, test=None,
                  training_target_count=len(train), validation_target_count=len(validation),
                  parameters=model.parameter_counts(), training_seconds=total_seconds)
    save_json(metrics_path, result)
    return result


def run(root, config, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Deney ayarları değişti; farklı çıktı klasörü gerekli.")
    if (output / "status.json").exists():
        status = json.loads((output / "status.json").read_text())
        if status["status"] == "complete":
            raise FileExistsError("Bu deney tamamlanmış; üzerine yazılmaz.")
    save_json(config_path, config)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = "bfloat16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "float32"
    metadata = json.loads((root / "data/processed/metadata.json").read_text())
    model_config = ModelConfig(metadata["question_vocabulary_size"], metadata["skill_vocabulary_size"], **config["model"])
    store = InteractionStore(root / "data/processed/interactions.npz")
    source_files = ["src/kt_project/plkt_models.py", "src/kt_project/plkt_sequences.py", "src/kt_project/plkt_training.py",
                    "src/kt_project/models.py", "src/kt_project/sequences.py", "src/kt_project/training.py",
                    "src/kt_project/prepare_data.py", "configs/paper_alignment.json"]
    provenance = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_files}
    if (output / "source_hashes.json").exists():
        if json.loads((output / "source_hashes.json").read_text()) != provenance:
            raise ValueError("Eğitim başladıktan sonra kaynak değişmiş; eski deneye devam edilmez.")
    save_json(output / "source_hashes.json", provenance)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    result = {"purpose": config["purpose"], "source_sha256": metadata["source_sha256"],
              "model_config": model_config.to_dict(), "implementation_sha256": provenance,
              "device": str(device), "precision": precision, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
              "runs": [], "test_evaluated": False,
              "test_scope": "same holdout previously inspected in v1; methodological re-evaluation, not a new untouched confirmation set"}
    def progress(**values):
        save_json(output / "status.json", {"status": "running", **values})
        log_event(**values)
    try:
        for seed in config["seeds"]:
            directory = output / f"seed_{seed}"
            directory.mkdir(exist_ok=True)
            initial, digest = pretrain(store, model_config, config, seed, device, precision, directory, progress)
            for variant in config["variants"]:
                row = tune(store, model_config, initial, digest, config, seed, variant, device, precision, directory, progress)
                result["runs"].append(row)
                save_json(output / "results.json", result)
        # All training/epoch/variant decisions are fixed before this point.
        locked = {"config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  "test_not_used_for_epoch_or_variant_selection": True,
                  "checkpoints": {f"seed_{row['seed']}/{row['variant']}_best.pt": hashlib.sha256(
                      (output / f"seed_{row['seed']}/{row['variant']}_best.pt").read_bytes()).hexdigest()
                                  for row in result["runs"]}}
        save_json(output / "test_evaluation_lock.json", locked)
        test = PrefixDataset(store, 2, model_config.max_history)
        test_batches = prefix_batches(test, config["evaluation_batch_size"], device, 17, config)
        for row in result["runs"]:
            directory = output / f"seed_{row['seed']}"
            saved = torch.load(directory / f"{row['variant']}_best.pt", weights_only=True, map_location="cpu")
            model = build_tracer(model_config, row["variant"]).to(device)
            model.load_state_dict(saved["model_state"])
            metrics, predictions = evaluate(model, test_batches, device, precision)
            np.savez_compressed(directory / f"{row['variant']}_test_predictions.npz", **predictions)
            row["test"] = metrics
            progress(phase="locked_test", seed=row["seed"], variant=row["variant"], auc=metrics["auc"])
        result["test_evaluated"] = True
        result["total_seconds"] = time.perf_counter()-started
        result["peak_allocated_gpu_mb"] = torch.cuda.max_memory_allocated()/1024**2 if device.type == "cuda" else None
        save_json(output / "results.json", result)
        save_json(output / "status.json", {"status": "complete", "total_seconds": result["total_seconds"]})
        log_event(phase="complete", seconds=result["total_seconds"])
        return result
    except Exception as error:
        save_json(output / "status.json", {"status": "failed", "error": str(error)})
        raise
