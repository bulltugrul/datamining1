"""Eski gösterim çıktılarını arşivle ve düzeltilmiş ağırlıklarla yeniden üret."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / ".runtime/cuda"))
import torch
from kt_project.patterns import fit_behavior_model
from kt_project.sequences import InteractionStore
from kt_project.tracking import FrozenTracer, ReferenceBank


def script(name):
    subprocess.run([sys.executable, str(ROOT / "scripts" / name)], cwd=ROOT, check=True)


def main():
    status = json.loads((ROOT / "outputs/main/paper_alignment_v2/status.json").read_text())
    if status["status"] != "complete":
        raise RuntimeError("Yeni deney bitmeden öğrenci gösterimi değiştirilemez.")
    selection_file = "outputs/main/paper_alignment_v2/interpretations/prototype_selection.json"
    selection = json.loads((ROOT / selection_file).read_text(encoding="utf-8"))
    integrity = json.loads((ROOT / "outputs/main/paper_alignment_v2/analysis/integrity.json").read_text())
    if not integrity["all_checks_passed"]:
        raise RuntimeError("Yeni deney denetimi geçmedi.")
    archive = ROOT / "outputs/archive/pre_alignment_v1"
    if not archive.exists():
        archive.mkdir(parents=True)
        for name in ("patterns", "replay", "interactions", "figures", "validation"):
            shutil.copytree(ROOT / "outputs" / name, archive / name)
        for name in ("05_Ogrenci_Gostergeleri_ve_Davranis_Gruplari.ipynb",
                     "06_Ornek_Oturum_ve_Calisma_Onerileri.ipynb", "07_Metin_Ses_ve_Gorsel_Etkilesimler.ipynb"):
            target = archive / "notebooks"
            target.mkdir(exist_ok=True)
            shutil.copy2(ROOT / "notebooks" / name, target / name)
        shutil.copytree(ROOT / "docs", archive / "docs")
        (archive / "archive_complete.json").write_text(json.dumps({"purpose": "legacy v1 display outputs before model correction"}))
    if not (archive / "archive_complete.json").exists():
        raise RuntimeError("Eski çıktı arşivi eksik; üzerine yazmadan kontrol edilmeli.")
    (ROOT / "configs/active_model.json").write_text(json.dumps({
        "version": "paper_alignment_v2", "selection_file": selection_file,
        "scope": "scaled method with explicit same-target-step block mask interpretation; strict-token reference retained",
        "inference_policy": selection["inference_policy"],
    }, indent=2), encoding="utf-8")
    torch.set_num_threads(4)
    config = json.loads((ROOT / "configs/tracking.json").read_text())
    store = InteractionStore(ROOT / "data/processed/interactions.npz")
    tracer = FrozenTracer(ROOT, store)
    bank = ReferenceBank(ROOT / "data/processed/references.json")
    if tracer.provenance["checkpoint_sha256"] != selection["checkpoint_sha256"]:
        raise AssertionError("Etkin model seçimi uyuşmuyor.")
    print(json.dumps({"phase": "refresh_patterns", "seed": selection["seed"]}), flush=True)
    def progress(done, total):
        if done % 512 == 0 or done == total:
            print(json.dumps({"phase": "reference_scores", "done": done, "total": total}), flush=True)
    fit_behavior_model(store, tracer, bank, config, ROOT / "outputs/patterns", progress)
    # Aynı gerçek öğrenci seçimi ve aynı kurallar; yalnızca yeni ağırlıkların çıktıları.
    del tracer
    script("run_student_replay.py")
    script("audit_student_tracking.py")
    script("build_tracking_figures.py")
    script("write_tracking_report.py")
    script("run_interaction_examples.py")
    script("audit_interaction_examples.py")
    script("write_interaction_report.py")
    print(json.dumps({"phase": "corrected_tracking_complete"}), flush=True)


if __name__ == "__main__":
    main()
