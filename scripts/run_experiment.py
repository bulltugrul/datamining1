from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / ".runtime" / "cuda"))
sys.path.insert(0, str(ROOT / "src"))
from kt_project.training import run_experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/pilot/lkm_128_v1"))
    args = parser.parse_args()
    config_path = ROOT / args.config
    output = ROOT / args.output
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_experiment(ROOT, config, output)
