"""Açıkça yazılmış küçük ifade derlemi ve sentetik gösterim girdileri."""

import hashlib
import json
from pathlib import Path
import subprocess
import textwrap

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]

# Bölümler elle ve önceden ayrılmıştır. Bunlar gerçek öğrenci beyanları değildir.
CORPUS = {
    "integers": {
        "train": [
            "Tam sayılarda toplama yapmak istiyorum.", "Tam sayılarda çıkarma konusunda yardım lazım.",
            "Negatif tam sayıları toplarken zorlanıyorum.", "Pozitif ve negatif tam sayılarla çıkarma çalışalım.",
            "Tam sayıların toplamını bulmayı öğrenmek istiyorum.", "İşaretli tam sayılarda çıkarma alıştırması arıyorum.",
            "Eksi işaretli tam sayıları toplama konusunu anlat.", "Tam sayılarda toplama ve çıkarma için soru öner.",
            "İki negatif tam sayıyı nasıl toplarım?", "Bir tam sayıdan başka bir tam sayıyı çıkarma.",
            "Tam sayılarla toplama işleminde işaretleri karıştırıyorum.", "Negatif bir tam sayıdan pozitif tam sayı çıkarma.",
            "Tam sayılarda toplama egzersizi yapabilir miyiz?", "Tam sayı çıkarma işlemine yeniden bakalım.",
            "Artı ve eksi tam sayılarla toplama pratiği istiyorum.", "Tam sayılarda çıkarma soruları çözmek istiyorum.",
        ],
        "validation": [
            "Tam sayılarda toplama becerimi geliştirelim.", "Negatif tam sayılarla çıkarma örneği göster.",
            "İşaretli tam sayıların toplamında hata yapıyorum.", "Tam sayılarla toplama ve çıkarma tekrarına ihtiyacım var.",
        ],
        "test": [
            "Eksi ve artı tam sayılarda toplama pratiği yapalım.", "Tam sayıların çıkarma işlemi için alıştırma seç.",
            "Negatif tam sayılarda toplama konusunu pekiştirelim.", "Tam sayılarla çıkarma işlemini tekrar etmek isterim.",
        ],
    },
    "decimals": {
        "train": [
            "Ondalık sayılarda toplama yapmak istiyorum.", "Ondalık sayılarda çıkarma konusunu anlat.",
            "Pozitif ondalık sayıları toplamakta zorlanıyorum.", "Virgüllü sayılarda çıkarma çalışalım.",
            "Ondalık sayıların toplamını bulmayı öğrenelim.", "Ondalık sayılarda toplama ve çıkarma için soru öner.",
            "Virgüllü sayıların toplama işleminde yardım lazım.", "Pozitif ondalık sayılarda çıkarma alıştırması istiyorum.",
            "İki ondalık sayıyı toplarken virgülü nereye koyarım?", "Ondalık bir sayıdan başka bir ondalık sayı çıkarma.",
            "Ondalık sayılarda toplama işleminde basamakları hizalayamıyorum.", "Virgüllü sayıların farkını çıkarma ile bulalım.",
            "Pozitif ondalıklarla toplama pratiği yapalım.", "Ondalık çıkarma işleminde hata yapıyorum.",
            "Virgülden sonraki basamaklarla toplama çalışması istiyorum.", "Ondalık sayılarla çıkarma soruları çözmek istiyorum.",
        ],
        "validation": [
            "Pozitif ondalık sayılarla toplama örneği arıyorum.", "Virgüllü sayılarda çıkarma pratiğine ihtiyacım var.",
            "Ondalık sayıların toplamında virgülleri karıştırıyorum.", "Ondalık toplama ve çıkarma becerimi geliştirelim.",
        ],
        "test": [
            "Virgüllü sayıların toplama işlemini pekiştirelim.", "Ondalık sayılarla çıkarma alıştırması seçer misin?",
            "Pozitif ondalıklarla toplama konusunu yeniden çalışalım.", "Ondalık sayı çıkarma işlemi için yardım istiyorum.",
        ],
    },
    "fractions": {
        "train": [
            "Kesirlerde toplama yapmak istiyorum.", "Kesirlerde çıkarma konusunda yardım lazım.",
            "Paydası farklı kesirleri toplarken zorlanıyorum.", "Kesirlerle toplama ve çıkarma için soru öner.",
            "Paydaları eşit kesirlerde çıkarma çalışalım.", "Kesirlerin toplamını bulmayı öğrenmek istiyorum.",
            "Payda eşitleyerek kesirleri toplama konusunu anlat.", "Bir kesirden başka bir kesri çıkarma alıştırması istiyorum.",
            "Basit kesirlerde toplama egzersizi yapalım.", "Kesir çıkarma işleminde paydaları karıştırıyorum.",
            "İki kesri nasıl toplarım?", "Paydaları eşitleyip kesirlerde çıkarma yapalım.",
            "Kesirlerde toplama becerimi tekrar etmek isterim.", "Farklı paydalı kesirlerde çıkarma sorusu çözmek istiyorum.",
            "Kesir toplarken payı ve paydayı nasıl kullanırım?", "Kesirlerin farkını çıkarma işlemiyle bulalım.",
        ],
        "validation": [
            "Kesirlerle toplama pratiğine ihtiyacım var.", "Eşit paydalı kesirlerde çıkarma örneği göster.",
            "Kesirlerin toplamında payda eşitlemeyi unutuyorum.", "Kesir toplama ve çıkarma alıştırması seçer misin?",
        ],
        "test": [
            "Farklı paydalı kesirlerle toplama örneği çözelim.", "Kesirlerin çıkarma işlemini pekiştirelim.",
            "Kesirlerde toplama için yeni bir alıştırma seç.", "Kesir çıkarma yaparken paydaları eşitlemede zorlanıyorum.",
        ],
    },
    "out_of_scope": {
        "train": [],
        "validation": [
            "Üçgenlerin alanını hesaplamak istiyorum.", "Kesirlerde çarpma yapmak istiyorum.",
            "Bugün hava nasıl?", "Matematik çalışmak istiyorum.",
            "Denklem çözmeyi öğrenmek istiyorum.", "Kesirler ve tam sayılarda toplama çalışalım.",
            "Ondalık sayılarla bölme işlemi yapalım.", "Bir konu seçemedim, yardım eder misin?",
        ],
        "test": [
            "Daire çevresi hesaplama alıştırması öner.", "Kesirlerle bölme pratiği yapalım.",
            "Tarih sınavına hazırlanmak istiyorum.", "Hangi konuyu anlamadığımı bilmiyorum.",
            "Tam sayılarda çarpma konusunu anlat.", "Kesirlerle ve ondalık sayılarla toplama yapmak isterim.",
            "Negatif ondalık sayılarda çıkarma çalışmak istiyorum.", "Bana bir müzik önerir misin?",
        ],
    },
}

DEMO_TEXTS = {
    "integers": "Negatif ve pozitif tam sayılarda toplama ve çıkarma alıştırması yapalım.",
    "decimals": "Ondalık sayılarda toplama ve çıkarma çalışmak istiyorum.",
    "fractions": "Paydaları farklı kesirlerde toplama ve çıkarma yaparken zorlanıyorum.",
}


def main():
    directory = ROOT / "data/interactions"
    directory.mkdir(parents=True, exist_ok=True)
    rows = [{"id": f"{label}_{split}_{i:02}", "split": split, "label": label, "text": text,
             "origin": "authored demonstration expression, not a real student statement"}
            for label, partitions in CORPUS.items() for split, texts in partitions.items()
            for i, text in enumerate(texts, 1)]
    assert len({row["text"] for row in rows}) == len(rows)
    assert not set(DEMO_TEXTS.values()).intersection(row["text"] for row in rows)
    (directory / "intent_corpus.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    samples = []
    for modality, extension in [("text", "txt"), ("audio", "wav"), ("image", "png")]:
        for label, text in DEMO_TEXTS.items():
            path = directory / f"{modality}_{label}.{extension}"
            if modality == "text":
                path.write_text(text, encoding="utf-8")
            if modality == "image":
                # Bu bir metin OCR girdisidir; görsel matematik çözümleme görevi değildir.
                image = Image.new("RGB", (1500, 360), "white")
                draw = ImageDraw.Draw(image)
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 48)
                draw.multiline_text((65, 65), textwrap.fill(text, width=48), font=font, fill="#152538", spacing=25)
                image.save(path)
            samples.append({"id": f"{modality}_{label}", "modality": modality,
                            "path": path.relative_to(ROOT).as_posix(), "expected_text": text,
                            "expected_label": label, "synthetic": True, "absolute_path": str(path.resolve())})
    manifest = {"purpose": "controlled examples of topic requests, not new assessment questions",
                "audio_origin": "Microsoft Tolga TTS; synthetic, not recorded student speech",
                "image_origin": "Pillow rendered printed Turkish text; not handwriting or diagram recognition",
                "samples": samples}
    temporary = ROOT / "tmp/interaction_synthesis.json"
    temporary.parent.mkdir(exist_ok=True)
    temporary.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(ROOT / "scripts/synthesize_examples.ps1"), "-ManifestPath", str(temporary)], check=True)
    for sample in samples:
        sample.pop("absolute_path")
        sample["sha256"] = hashlib.sha256((ROOT / sample["path"]).read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"corpus_records": len(rows), "demo_inputs": len(samples)}))


if __name__ == "__main__":
    main()
