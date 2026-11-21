from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kt_project.data_audit import audit_assist2009, file_digest


def main() -> None:
    parser = argparse.ArgumentParser(description="Ham dosyayı koru ve veri denetimini üret.")
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    if source.name != "skill_builder_data_corrected_collapsed.csv":
        raise SystemExit("Bu aşama için düzeltilmiş collapsed Skill Builder dosyası bekleniyor.")
    destination = ROOT / "data" / "raw" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_digest(source) != file_digest(destination):
            raise SystemExit("Ham klasörde farklı bir dosya var; otomatik üzerine yazılmadı.")
    else:
        shutil.copy2(source, destination)
    report = audit_assist2009(destination)
    output = ROOT / "outputs" / "audit" / "assist2009_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    provenance = {
        "dataset": report["dataset"],
        "original_filename": source.name,
        "sha256": report["sha256"],
        "copied_at": datetime.now(ZoneInfo("Europe/Istanbul")).isoformat(),
        "source_page": "https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010",
        "previous_notebooks_or_weights_imported": False,
    }
    (destination.parent / "manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"raw": report["raw"], "eligibility_preview": report["eligibility_preview"],
                      "fields": report["fields"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
