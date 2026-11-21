"""Sabit sürümlü açık ASR ağırlığını proje içine bir kez indir."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".runtime/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from huggingface_hub import snapshot_download


def main():
    config = json.loads((ROOT / "configs/interactions.json").read_text(encoding="utf-8"))
    target = ROOT / "data/models/whisper-base"
    snapshot_download(config["asr_model"], revision=config["asr_revision"], local_dir=target,
                      allow_patterns=["*.json", "merges.txt", "model.safetensors", "README.md"], max_workers=2)
    files = {path.relative_to(target).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in target.iterdir() if path.is_file()}
    output = ROOT / "outputs/interactions"
    output.mkdir(parents=True, exist_ok=True)
    (output / "asr_model_provenance.json").write_text(json.dumps({
        "model": config["asr_model"], "revision": config["asr_revision"],
        "license": "apache-2.0", "file_sha256": files,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"downloaded": str(target), "files": len(files)}))


if __name__ == "__main__":
    main()
