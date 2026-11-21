"""Yeni yöntem deneyi; önceki sürümün kaynaklarını ve sonuçlarını korur."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / ".runtime/cuda"))
from kt_project.plkt_training import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/main/paper_alignment_v2", help="Proje içinde yeni deney klasörü.")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT):
        parser.error("Deney çıktısı proje klasörü içinde olmalı.")
    config = json.loads((ROOT / "configs/paper_alignment.json").read_text())
    run(ROOT, config, output)
