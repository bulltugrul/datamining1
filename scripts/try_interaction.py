"""Yeni kısa metin, WAV veya PNG girdisini aynı örnek öğrenci bağlamında dene."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from kt_project.interactions import InteractionService
from kt_project.media import SpeechReader, read_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", choices=["text", "audio", "image"], default="text")
    parser.add_argument("--text")
    parser.add_argument("--path", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.type == "text":
        if args.text is None:
            parser.error("Metin için --text gerekli.")
        recognition = {"text": args.text}
    else:
        if args.path is None:
            parser.error("Dosya için --path gerekli.")
        recognition = SpeechReader(ROOT).read(args.path) if args.type == "audio" else read_image(ROOT, args.path)
    output = {"recognition": recognition, **InteractionService(ROOT).respond(recognition["text"])}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
