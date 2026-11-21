"""Devam eden eğitim tamamlanınca denetim, yeni gösterimler ve defter kontrolü."""

import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "outputs/method_audit/finalization_status.json"


def record(**values):
    STATE.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(values, ensure_ascii=True), flush=True)


def main():
    record(status="waiting_training")
    path = ROOT / "outputs/main/paper_alignment_v2/status.json"
    while True:
        try:
            status = json.loads(path.read_text())
        except json.JSONDecodeError:
            time.sleep(1)
            continue
        if status["status"] == "failed":
            raise RuntimeError("Ana eğitim başarısız: " + status.get("error", ""))
        if status["status"] == "complete":
            break
        time.sleep(15)
    scripts = ["record_method_audit.py", "analyze_paper_alignment.py", "draw_prompt_visibility.py",
               "refresh_corrected_tracking.py", "build_tracking_notebooks.py", "build_interaction_notebook.py",
               "build_method_audit_notebook.py", "finalize_method_revision.py"]
    for script in scripts:
        record(status="running", step=script)
        subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)
    record(status="running", step="execute_corrected_notebooks")
    subprocess.run([sys.executable, str(ROOT / "scripts/execute_notebooks.py"),
        "notebooks/05_Ogrenci_Gostergeleri_ve_Davranis_Gruplari.ipynb",
        "notebooks/06_Ornek_Oturum_ve_Calisma_Onerileri.ipynb",
        "notebooks/07_Metin_Ses_ve_Gorsel_Etkilesimler.ipynb",
        "notebooks/08_Makale_Yontem_Denetimi_ve_Duzeltilmis_Deneyler.ipynb"], cwd=ROOT, check=True)
    record(status="complete", steps=len(scripts)+1)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        record(status="failed", error=str(error))
        raise
