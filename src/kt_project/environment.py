"""Çalışma ortamını ve gerçek bir CUDA işlemini denetle."""

from __future__ import annotations

from datetime import datetime
from importlib import metadata
import platform
import sys
from zoneinfo import ZoneInfo


def check_environment() -> dict:
    import torch

    packages = ["torch", "numpy", "pandas", "scikit-learn", "matplotlib",
                "jupyterlab", "ipykernel", "nbformat", "nbclient"]
    report = {
        "checked_at": datetime.now(ZoneInfo("Europe/Istanbul")).isoformat(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "packages": {name: metadata.version(name) for name in packages},
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "gpu_memory_gib": None,
        "cuda_calculation_passed": False,
    }
    if report["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        report["gpu_name"] = props.name
        report["gpu_memory_gib"] = round(props.total_memory / 1024**3, 2)
        # Aygıtın görünmesi yetmez: ileri hesap ve geri yayılım gerçekten çalışmalı.
        x = torch.randn(64, 64, device="cuda", requires_grad=True)
        loss = (x @ x.T).square().mean()
        loss.backward()
        torch.cuda.synchronize()
        passed = bool(torch.isfinite(loss).item() and torch.isfinite(x.grad).all().item())
        report["cuda_calculation_passed"] = passed
        report["cuda_calculation_device"] = str(x.device)
        if not passed:
            raise RuntimeError("CUDA ileri hesap veya gradyan kontrolü başarısız.")
    return report
