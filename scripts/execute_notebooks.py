"""Defterleri gerçek proje çekirdeğinde baştan sona çalıştır ve doğrula."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ["JUPYTER_CONFIG_DIR"] = str(ROOT / ".runtime" / "jupyter" / "config")
os.environ["JUPYTER_DATA_DIR"] = str(ROOT / ".runtime" / "jupyter" / "data")
os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / ".runtime" / "jupyter" / "runtime")
os.environ["JUPYTER_PATH"] = str(ROOT / ".venv" / "share" / "jupyter")
os.environ["IPYTHONDIR"] = str(ROOT / ".runtime" / "ipython")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".runtime" / "matplotlib")
os.environ["CUDA_CACHE_PATH"] = str(ROOT / ".runtime" / "cuda")

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="*", help="Yalnızca bu defterleri çalıştır; boşsa hepsi.")
    args = parser.parse_args()
    results = []
    paths = [ROOT / name for name in args.notebooks] if args.notebooks else sorted((ROOT / "notebooks").glob("*.ipynb"))
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        NotebookClient(nb, timeout=240, kernel_name="kt-project",
                       resources={"metadata": {"path": str(ROOT)}},
                       allow_errors=False).execute()
        nbformat.validate(nb)
        code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
        if not all(cell.execution_count is not None for cell in code_cells):
            raise RuntimeError(f"Çalışmamış hücre var: {path.name}")
        errors = [output for cell in code_cells for output in cell.get("outputs", [])
                  if output.output_type == "error"]
        if errors:
            raise RuntimeError(f"Defterde hata çıktısı var: {path.name}")
        nbformat.write(nb, path)
        result = {"notebook": path.name, "executed_code_cells": len(code_cells), "errors": 0}
        results.append(result)
        print(json.dumps(result), flush=True)
    output = ROOT / "outputs" / "validation"
    output.mkdir(parents=True, exist_ok=True)
    record_path = output / "notebook_execution.json"
    previous = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else []
    records = {row["notebook"]: row for row in previous}
    records.update({row["notebook"]: row for row in results})
    record_path.write_text(json.dumps(list(records.values()), indent=2), encoding="utf-8")
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze", "--disable-pip-version-check"],
                             check=True, capture_output=True, text=True)
    (ROOT / "outputs" / "audit" / "package_versions.txt").write_text(frozen.stdout, encoding="utf-8")


if __name__ == "__main__":
    main()
