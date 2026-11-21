"""JupyterLab'i gercek Windows kullanicisina ait calisma klasorleriyle ac."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def current_principal() -> str:
    """Profil ortam degiskeni yerine sureci calistiran hesabin kimligini kullan."""
    if os.name != "nt":
        return f"uid:{os.getuid()}"
    whoami = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "whoami.exe"
    result = subprocess.run(
        [str(whoami), "/user", "/fo", "csv", "/nh"],
        check=True, capture_output=True, text=True, timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    rows = list(csv.reader(io.StringIO(result.stdout.strip())))
    if not rows or len(rows[0]) < 2 or not re.fullmatch(r"S-\d+(?:-\d+)+", rows[0][1]):
        raise RuntimeError("Windows kullanici kimligi belirlenemedi.")
    return rows[0][1]


def runtime_environment(project_root: Path, environ=None, principal=None) -> dict[str, str]:
    env = dict(os.environ if environ is None else environ)
    project_root = project_root.resolve()
    project_key = hashlib.sha256(str(project_root).casefold().encode("utf-8")).hexdigest()[:12]
    identity = current_principal() if principal is None else principal
    user_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    local_data = Path(env.get("LOCALAPPDATA") or tempfile.gettempdir())
    base = local_data / "KnowledgeTracing" / project_key / user_key
    for name, relative in {
        "JUPYTER_CONFIG_DIR": "jupyter/config",
        "JUPYTER_DATA_DIR": "jupyter/data",
        "JUPYTER_RUNTIME_DIR": "jupyter/runtime",
        "IPYTHONDIR": "ipython",
        "MPLCONFIGDIR": "matplotlib",
        "CUDA_CACHE_PATH": "cuda",
    }.items():
        env[name] = str(base / relative)
    kernel_path = str(project_root / ".venv" / "share" / "jupyter")
    existing_path = env.get("JUPYTER_PATH", "")
    env["JUPYTER_PATH"] = kernel_path + (os.pathsep + existing_path if existing_path else "")
    return env


def prepare_runtime(env: dict[str, str]) -> None:
    for name in ("JUPYTER_CONFIG_DIR", "JUPYTER_DATA_DIR", "JUPYTER_RUNTIME_DIR",
                 "IPYTHONDIR", "MPLCONFIGDIR", "CUDA_CACHE_PATH"):
        Path(env[name]).mkdir(parents=True, exist_ok=True)
    # Klasorlerin kullanicinin gercek oturumunda yazilabilir oldugunu denetle.
    with tempfile.TemporaryFile(dir=env["JUPYTER_RUNTIME_DIR"]):
        pass


def jupyter_command(project_root: Path, port: int = 8888, no_browser: bool = False) -> list[str]:
    executable = project_root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    command = [str(executable), "-m", "jupyterlab",
               "notebooks/01_Gereksinimler_ve_Veri_Kontrolu.ipynb",
               "--ServerApp.ip=127.0.0.1", f"--ServerApp.port={port}",
               f"--ServerApp.root_dir={project_root}"]
    if no_browser:
        command.append("--no-browser")
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Calisma klasorlerini kontrol et ve cik.")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
    try:
        env = runtime_environment(ROOT)
        prepare_runtime(env)
        if args.check:
            print("Jupyter kullanici klasorleri hazir ve yazilabilir.")
            return 0
        return subprocess.call(jupyter_command(ROOT, args.port, args.no_browser), cwd=ROOT, env=env)
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"JupyterLab baslatilamadi: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
