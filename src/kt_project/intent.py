"""Küçük, açıkça etiketlenmiş derlemden öğrenilen beceri isteği sınıflandırıcısı."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

UNKNOWN = "out_of_scope"
LABELS = ["integers", "decimals", "fractions", UNKNOWN]


def normalize_text(text):
    value = unicodedata.normalize("NFKD", text.lower().replace("ı", "i"))
    value = "".join(c for c in value if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def scope_issue(text):
    value = normalize_text(text)
    if len(text) > 500:
        return "İstek çok uzun; lütfen tek beceriyi kısa bir cümleyle belirt."
    if not value:
        return "İstek okunamadı; lütfen çalışmak istediğin beceriyi belirt."
    topics = [bool(re.search(pattern, value)) for pattern in
              [r"tam say|negatif|pozitif ve negatif", r"ondalik|virgul", r"kesir|kesri"]]
    # Bu sınırlar öğrenilmiş sınıflandırıcının anlamadığı niyetleri zorla etiketlemesini önler.
    if topics[1] and "negatif" in value:
        return "Bu küçük örnek pozitif ondalık sayıları kapsıyor; isteğini netleştirelim."
    if sum(topics) > 1:
        return "Birden fazla beceri belirttin; önce hangisini çalışmak istediğini seçelim."
    if re.search(r"carp|bolme|bolmek|denklem|geometri", value):
        return "Bu örnek yalnızca üç beceride toplama ve çıkarma isteklerini kapsıyor."
    if not re.search(r"topla|cikar", value):
        return "Lütfen beceriyi ve toplama veya çıkarma işlemini açıkça belirt."
    return None


@dataclass
class IntentModel:
    pipeline: Pipeline
    threshold: float
    minimum_similarity: float
    training_vectors: object
    skills: dict
    metadata: dict

    def predict(self, text):
        if not isinstance(text, str):
            raise TypeError("Metin bir karakter dizisi olmalı.")
        issue = scope_issue(text)
        vector = self.pipeline.named_steps["tfidf"].transform([text])
        probabilities = self.pipeline.named_steps["classifier"].predict_proba(vector)[0]
        index = int(np.argmax(probabilities))
        label = str(self.pipeline.classes_[index])
        score = float(probabilities[index])
        similarity = float((vector @ self.training_vectors.T).max())
        accepted = issue is None and score >= self.threshold and similarity >= self.minimum_similarity
        return {"accepted": accepted, "label": label if accepted else UNKNOWN,
                "classifier_score": score, "nearest_training_similarity": similarity,
                "skill": dict(self.skills[label]) if accepted else None,
                "reason": "Bilinen beceri isteği tanındı." if accepted else (
                    issue or "İstek yeterince belirgin değil; lütfen tek beceriyi yeniden ifade et."),
                "score_note": "Sınıflandırma skoru kalibre edilmiş güven veya öğrenci başarı yüzdesi değildir."}


def train_intent_model(root):
    root = Path(root)
    source = root / "data/interactions/intent_corpus.json"
    rows = json.loads(source.read_text(encoding="utf-8"))
    config = json.loads((root / "configs/interactions.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((root / "data/processed/vocabulary.json").read_text(encoding="utf-8"))
    normalized = [normalize_text(row["text"]) for row in rows]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Derlemde normalleştirme sonrası yinelenen ifade var.")
    training = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    testing = [row for row in rows if row["split"] == "test"]
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                  preprocessor=normalize_text, lowercase=False, sublinear_tf=True)),
        ("classifier", LogisticRegression(C=config["classifier_C"], max_iter=1000,
                                          random_state=config["random_seed"])),
    ])
    pipeline.fit([r["text"] for r in training], [r["label"] for r in training])
    vectors = pipeline.named_steps["tfidf"].transform([row["text"] for row in training])
    skills = {label: {**entry, "encoded_skill": vocabulary["skill"][entry["raw_skill_id"]]}
              for label, entry in config["skills"].items()}
    model = IntentModel(pipeline, 0., config["minimum_text_similarity"], vectors, skills, {})
    candidates = []
    actual = [r["label"] for r in validation]
    for threshold in config["threshold_candidates"]:
        model.threshold = threshold
        predicted = [model.predict(row["text"])["label"] for row in validation]
        candidates.append({"threshold": threshold,
                           "validation_macro_f1": float(f1_score(actual, predicted, labels=LABELS, average="macro", zero_division=0))})
    # Eşit F1 durumunda daha yüksek eşik: belirsiz isteğe zorla etiket atamama.
    selected = max(candidates, key=lambda row: (row["validation_macro_f1"], row["threshold"]))
    model.threshold = selected["threshold"]
    predicted = [model.predict(row["text"])["label"] for row in testing]
    actual = [row["label"] for row in testing]
    model.metadata = {
        "method": "character TF-IDF (3..5) + logistic regression with explicit scope guards",
        "origin": "authored small example corpus, not real student statements",
        "training_records": len(training), "validation_records": len(validation), "test_records": len(testing),
        "tfidf_fitted_on": "train only", "threshold_selected_on": "validation only",
        "threshold": model.threshold, "threshold_candidates": candidates, "labels": LABELS,
        "test_accuracy": float(accuracy_score(actual, predicted)),
        "test_macro_f1": float(f1_score(actual, predicted, labels=LABELS, average="macro", zero_division=0)),
        "test_confusion_matrix": confusion_matrix(actual, predicted, labels=LABELS).tolist(),
        "test_predictions": [{"id": row["id"], "text": row["text"], "actual": row["label"], "predicted": p}
                             for row, p in zip(testing, predicted)],
        "corpus_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "configuration": config,
        "interpretation": "Küçük yazılmış derlemde işlev kontrolüdür; gerçek öğrencilerde genelleme veya ASR/OCR başarısı değildir.",
    }
    output = root / "outputs/interactions"
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "intent_model.joblib")
    (output / "intent_metrics.json").write_text(json.dumps(model.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return model
