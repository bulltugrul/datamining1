from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kt_project.prepare_data import prepare_dataset


if __name__ == "__main__":
    result = prepare_dataset(
        ROOT / "data" / "raw" / "skill_builder_data_corrected_collapsed.csv",
        ROOT / "data" / "processed")
    print(json.dumps(result, ensure_ascii=True, indent=2))
