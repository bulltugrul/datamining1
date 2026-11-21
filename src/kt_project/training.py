"""Ortak ön eğitimden başlayan deneyler ve bağımsız değerlendirme."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .models import LargeKnowledgeModel, KnowledgeTracer, ModelConfig
from .sequences import (InteractionStore, PrefixDataset, PretrainingDataset,
                        PretrainingCollator, DevicePrefixBatches, collate_prefix, to_device)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def state_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def log_event(**values):
    print(json.dumps(values, ensure_ascii=True), flush=True)


def loader(dataset, batch_size, collate_fn, seed, shuffle=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn, generator=generator, num_workers=0,
                      pin_memory=torch.cuda.is_available())


def autocast_context(device, precision):
    if device.type != "cuda" or precision == "float32":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bfloat16" else torch.float16)


def prefix_batches(dataset, size, device, seed, config, shuffle=False):
    if config.get("device_batches", False):
        return DevicePrefixBatches(dataset, size, device, seed, shuffle)
    return loader(dataset, size, collate_prefix, seed, shuffle)


def optimizer_step(loss, optimizer, scaler, parameters):
    if not torch.isfinite(loss):
        raise FloatingPointError("Eğitim kaybı sonlu değil.")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    return float(norm)


@torch.inference_mode()
def evaluate(model, batches, device, precision):
    model.eval()
    labels, probabilities, row_indices = [], [], []
    loss_sum, count = 0.0, 0
    for cpu_batch in batches:
        batch = to_device(cpu_batch, device)
        with autocast_context(device, precision):
            logits = model(batch)
        logits = logits.float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Değerlendirme logits sonlu değil.")
        loss_sum += F.binary_cross_entropy_with_logits(logits, batch["label"], reduction="sum").item()
        count += len(logits)
        labels.append(batch["label"].cpu().numpy())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        row_indices.append(cpu_batch["row_index"].cpu().numpy())
    y, p, rows = np.concatenate(labels), np.concatenate(probabilities), np.concatenate(row_indices)
    metrics = {"samples": count, "bce": loss_sum/count,
               "accuracy": float(accuracy_score(y, p >= .5)),
               "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None}
    return metrics, {"label": y, "probability": p, "row_index": rows}


def pretrain(lkm, store, config, seed, device, precision, output):
    dataset = PretrainingDataset(store, lkm.config.max_history,
                                 config.get("train_student_limit"), seed)
    collator = PretrainingCollator(seed)
    optimizer = torch.optim.AdamW(lkm.parameters(), lr=config["pretraining_lr"],
                                  weight_decay=config["weight_decay"])
    parameters = list(lkm.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "float16")
    history = []
    for epoch in range(config["pretraining_epochs"]):
        dataset.set_epoch(epoch)
        collator.set_epoch(epoch)
        batches = loader(dataset, config["pretraining_batch_size"], collator, seed+epoch, True)
        lkm.train()
        sums, count = {"loss": 0., "question_loss": 0., "response_loss": 0., "order_loss": 0.}, 0
        started = time.perf_counter()
        for cpu_batch in batches:
            batch = to_device(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                loss, parts = lkm.pretraining_loss(batch)
            optimizer_step(loss, optimizer, scaler, parameters)
            size = len(batch["fbo_label"])
            count += size
            sums["loss"] += loss.item()*size
            for key, value in parts.items():
                sums[key] += value.item()*size
        row = {"epoch": epoch+1, "samples": count,
               **{key: value/count for key, value in sums.items()},
               "seconds": time.perf_counter()-started}
        history.append(row)
        log_event(phase="pretraining", seed=seed, **row)
    state = cpu_state(lkm)
    digest = state_digest(state)
    torch.save({"config": lkm.config.to_dict(), "model_state": state,
                "state_sha256": digest, "seed": seed}, output / "pretrained.pt")
    save_json(output / "pretraining_history.json", history)
    return state, digest, history


def finetune(method, initial_state, initial_digest, model_config, store, config,
             seed, device, precision, output):
    # KPT ve KMPT yeni bileşenleri de aynı başlangıçla oluşturulur.
    set_seed(seed+100)
    lkm = LargeKnowledgeModel(model_config)
    lkm.load_state_dict(initial_state)
    if state_digest(cpu_state(lkm)) != initial_digest:
        raise AssertionError("Ortak LKM başlangıcı farklı.")
    model = KnowledgeTracer(lkm, method).to(device)
    train_data = PrefixDataset(store, 0, model_config.max_history,
                               config.get("train_student_limit"), config.get("train_target_limit"), seed)
    val_data = PrefixDataset(store, 1, model_config.max_history,
                             config.get("validation_student_limit"), config.get("validation_target_limit"), seed)
    val_loader = prefix_batches(val_data, config["evaluation_batch_size"], device, seed, config)
    initial_validation, _ = evaluate(model, val_loader, device, precision)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["finetuning_lr"], weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "float16")
    rng = np.random.default_rng(seed+1979)
    history, best_state, best_epoch, no_improvement = [], None, None, 0
    best_bce, total_seconds = float("inf"), 0.
    for epoch in range(config["finetuning_epochs"]):
        model.train()
        batches = prefix_batches(train_data, config["batch_size"], device, seed+epoch, config, True)
        mixed_sum = response_sum = auxiliary_sum = 0.
        total_count = response_count = auxiliary_count = 0
        started = time.perf_counter()
        for cpu_batch in batches:
            batch = to_device(cpu_batch, device)
            size = len(batch["label"])
            auxiliary = rng.random(size) < .2 if method == "kmpt" else np.zeros(size, dtype=bool)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device)
            with autocast_context(device, precision):
                for is_aux, chosen in ((False, ~auxiliary), (True, auxiliary)):
                    n = int(chosen.sum())
                    if not n:
                        continue
                    selected = torch.as_tensor(chosen, device=device)
                    part_batch = {key: value[selected] for key, value in batch.items()}
                    part_loss = model.loss(part_batch, auxiliary=is_aux)
                    loss = loss + part_loss*(n/size)
                    if is_aux:
                        auxiliary_sum += part_loss.detach().float().item()*n
                        auxiliary_count += n
                    else:
                        response_sum += part_loss.detach().float().item()*n
                        response_count += n
            optimizer_step(loss, optimizer, scaler, parameters)
            mixed_sum += loss.item()*size
            total_count += size
        validation, _ = evaluate(model, val_loader, device, precision)
        seconds = time.perf_counter()-started
        total_seconds += seconds
        row = {"epoch": epoch+1, "train_mixed_objective": mixed_sum/total_count,
               "train_response_bce": response_sum/response_count if response_count else None,
               "train_auxiliary_question_ce": auxiliary_sum/auxiliary_count if auxiliary_count else None,
               "response_samples": response_count, "auxiliary_samples": auxiliary_count,
               "validation": validation, "seconds": seconds}
        history.append(row)
        log_event(phase="finetuning", method=method, seed=seed, epoch=epoch+1,
                  validation_bce=validation["bce"], validation_auc=validation["auc"], seconds=seconds)
        if validation["bce"] < best_bce-1e-6:
            best_bce, best_epoch, best_state, no_improvement = validation["bce"], epoch+1, cpu_state(model), 0
        else:
            no_improvement += 1
        if no_improvement >= config["patience"]:
            break
    model.load_state_dict(best_state)
    validation, predictions = evaluate(model, val_loader, device, precision)
    np.savez_compressed(output / f"{method}_validation_predictions.npz", **predictions)
    test_metrics = None
    if config["evaluate_test"]:
        test_data = PrefixDataset(store, 2, model_config.max_history)
        test_loader = prefix_batches(test_data, config["evaluation_batch_size"], device, seed, config)
        test_metrics, test_predictions = evaluate(model, test_loader, device, precision)
        np.savez_compressed(output / f"{method}_test_predictions.npz", **test_predictions)
        known = store.arrays["question"][test_predictions["row_index"]] != 1
        if known.any():
            y, p = test_predictions["label"][known], test_predictions["probability"][known]
            test_metrics["known_question_subset"] = {
                "samples": int(known.sum()), "accuracy": float(accuracy_score(y, p >= .5)),
                "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None}
    torch.save({"config": model_config.to_dict(), "method": method, "model_state": best_state,
                "initial_lkm_sha256": initial_digest, "seed": seed, "best_epoch": best_epoch},
               output / f"{method}_best.pt")
    save_json(output / f"{method}_history.json", history)
    return {"method": method, "seed": seed, "parameters": model.parameter_counts(),
            "initial_lkm_sha256": initial_digest, "best_epoch": best_epoch,
            "initial_validation": initial_validation, "validation": validation,
            "test": test_metrics, "training_seconds": total_seconds,
            "training_target_count": len(train_data), "validation_target_count": len(val_data)}


def run_experiment(root: Path, config: dict, output: Path):
    output = Path(output)
    if (output / "status.json").exists():
        raise FileExistsError("Bu deney klasöründe kayıt var. Yeni bir çıktı klasörü seçin.")
    if config["evaluate_test"]:
        limits = ("train_student_limit", "validation_student_limit", "train_target_limit", "validation_target_limit")
        if any(config.get(key) is not None for key in limits) or len(config["seeds"]) != 3:
            raise ValueError("Ana test karşılaştırması tam veri ve üç başlangıç gerektirir.")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "status.json", {"status": "running", "purpose": config["purpose"]})
    save_json(output / "config.json", config)
    try:
        torch.set_num_threads(4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        precision = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") if (
            device.type == "cuda" and config["mixed_precision"]) else "float32"
        metadata = json.loads((root / "data" / "processed" / "metadata.json").read_text(encoding="utf-8"))
        model_config = ModelConfig(metadata["question_vocabulary_size"], metadata["skill_vocabulary_size"], **config["model"])
        store = InteractionStore(root / "data" / "processed" / "interactions.npz")
        results = {"purpose": config["purpose"], "device": str(device), "precision": precision,
                   "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                   "source_sha256": metadata["source_sha256"], "model_config": model_config.to_dict(),
                   "test_evaluated": config["evaluate_test"], "runs": []}
        results["implementation_sha256"] = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("src/kt_project/models.py", "src/kt_project/sequences.py",
                         "src/kt_project/training.py", "src/kt_project/prepare_data.py")}
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for seed in config["seeds"]:
            set_seed(seed)
            seed_output = output / f"seed_{seed}"
            seed_output.mkdir()
            lkm = LargeKnowledgeModel(model_config).to(device)
            state, digest, _ = pretrain(lkm, store, config, seed, device, precision, seed_output)
            del lkm
            for method in ("baseline", "kpt", "kmpt"):
                result = finetune(method, state, digest, model_config, store, config,
                                  seed, device, precision, seed_output)
                results["runs"].append(result)
                save_json(output / "results.json", results)
        results["total_seconds"] = time.perf_counter()-started
        results["peak_allocated_gpu_mb"] = (torch.cuda.max_memory_allocated()/1024**2
                                              if device.type == "cuda" else None)
        save_json(output / "results.json", results)
        save_json(output / "status.json", {"status": "complete", "purpose": config["purpose"],
                                           "total_seconds": results["total_seconds"]})
        log_event(phase="complete", seconds=results["total_seconds"],
                  peak_gpu_mb=results["peak_allocated_gpu_mb"])
        return results
    except Exception as error:
        save_json(output / "status.json", {"status": "failed", "error": str(error)})
        raise
