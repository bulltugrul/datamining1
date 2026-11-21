"""Dosya içeriğinden gerçek ASR ve OCR; beklenen metin bu işlevlere verilmez."""

import json
import math
from pathlib import Path
import subprocess
import tempfile
import wave

import numpy as np
from scipy.signal import resample_poly


def load_audio(path, target_rate=16000):
    with wave.open(str(path), "rb") as stream:
        channels, width, source_rate, frames = (stream.getnchannels(), stream.getsampwidth(),
                                               stream.getframerate(), stream.getnframes())
        if channels not in (1, 2) or width != 2:
            raise ValueError("Örnek ses 16 bit PCM, tek veya iki kanallı WAV olmalı.")
        duration = frames/source_rate
        if not .2 <= duration <= 30:
            raise ValueError("Ses süresi 0,2–30 saniye arasında olmalı.")
        samples = np.frombuffer(stream.readframes(frames), dtype="<i2").astype(np.float32)/32768
    samples = samples.reshape(-1, channels).mean(axis=1)
    if float(np.sqrt(np.mean(samples**2))) < .001:
        raise ValueError("Ses çok sessiz; lütfen konuşma içeren bir dosya kullan.")
    if source_rate != target_rate:
        divisor = math.gcd(source_rate, target_rate)
        samples = resample_poly(samples, target_rate//divisor, source_rate//divisor).astype(np.float32)
    return samples, {"source_sample_rate": source_rate, "model_sample_rate": target_rate,
                     "source_channels": channels, "duration_seconds": duration}


class SpeechReader:
    def __init__(self, root):
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self.config = json.loads((Path(root) / "configs/interactions.json").read_text(encoding="utf-8"))
        directory = Path(root) / "data/models/whisper-base"
        self.processor = WhisperProcessor.from_pretrained(directory, local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(directory, local_files_only=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval().requires_grad_(False)

    def read(self, path):
        import torch
        samples, metadata = load_audio(path, self.config["sample_rate"])
        features = self.processor(samples, sampling_rate=self.config["sample_rate"],
                                  return_tensors="pt", return_attention_mask=True)
        with torch.inference_mode():
            ids = self.model.generate(input_features=features.input_features.to(self.device),
                                      attention_mask=features.attention_mask.to(self.device),
                                      language=self.config["asr_language"], task="transcribe",
                                      do_sample=False, num_beams=1, max_new_tokens=96)
        return {"text": self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip(),
                "engine": self.config["asr_model"], "revision": self.config["asr_revision"],
                "language": self.config["asr_language"], **metadata}


def read_image(root, path, language="tr"):
    root = Path(root)
    temporary = root / "tmp"
    temporary.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary) as work:
        output = Path(work) / "recognized.json"
        result = subprocess.run([
            "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
            str(root / "scripts/windows_ocr.ps1"), "-ImagePath", str(Path(path).resolve()),
            "-OutputPath", str(output), "-Language", language,
        ], capture_output=True, timeout=45)
        if result.returncode:
            raise RuntimeError("Windows OCR çalışamadı: " + result.stderr.decode("utf-8", errors="replace"))
        return json.loads(output.read_text(encoding="utf-8"))


def word_error_rate(expected, recognized):
    from .intent import normalize_text
    actual, predicted = normalize_text(expected).split(), normalize_text(recognized).split()
    if not actual:
        raise ValueError("Karşılaştırma metni boş olamaz.")
    previous = list(range(len(predicted)+1))
    for i, first in enumerate(actual, 1):
        current = [i]
        for j, second in enumerate(predicted, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(first != second)))
        previous = current
    return previous[-1]/len(actual)
