from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kt_project.environment import check_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    report = check_environment()
    output = ROOT / "outputs" / "audit" / "environment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    safe_view = {key: value for key, value in report.items() if key != "python_executable"}
    print(json.dumps(safe_view, ensure_ascii=True, indent=2))
    if args.require_cuda and not report["cuda_calculation_passed"]:
        raise SystemExit("CUDA hesaplama kontrolü geçmedi.")


if __name__ == "__main__":
    main()
