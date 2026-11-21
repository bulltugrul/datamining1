"""Ham ASSIST2009 dosyasını öğrenci kayıtlarını yayımlamadan incele."""

from __future__ import annotations

from collections import Counter
import codecs
import csv
import hashlib
from pathlib import Path
import re
import statistics


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detect_encoding(path: Path) -> str:
    decoder = codecs.getincrementaldecoder("utf-8-sig")()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                decoder.decode(block)
            decoder.decode(b"", final=True)
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "latin-1"


def integer_id(value: str | None) -> int | None:
    text = (value or "").strip()
    if not re.fullmatch(r"\d+(?:\.0+)?", text):
        return None
    return int(text.split(".")[0])


def parse_skills(value: str | None) -> tuple[int, ...]:
    text = (value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ()
    pieces = re.split(r"[_~,;|\s]+", text)
    parsed = [integer_id(piece) for piece in pieces]
    if any(piece is None for piece in parsed):
        return ()
    return tuple(sorted(set(parsed)))


def audit_assist2009(path: Path, min_interactions: int = 5) -> dict:
    path = Path(path)
    required = {"user_id", "problem_id", "order_id", "skill_id", "correct"}
    counts = Counter()
    students, questions, skills = set(), set(), set()
    events = {}
    skill_names = set()
    encoding = detect_encoding(path)
    with path.open(encoding=encoding, newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        missing = required.difference(columns)
        if missing:
            raise ValueError(f"Zorunlu sütunlar eksik: {sorted(missing)}")
        for row in reader:
            counts["raw_rows"] += 1
            student = integer_id(row["user_id"])
            question = integer_id(row["problem_id"])
            order = integer_id(row["order_id"])
            concepts = parse_skills(row["skill_id"])
            label = integer_id(row["correct"])
            if student is not None:
                students.add(student)
            if question is not None:
                questions.add(question)
            skills.update(concepts)
            if row.get("skill_name", "").strip():
                skill_names.add(row["skill_name"].strip())
            if not concepts:
                counts["missing_or_invalid_skill_rows"] += 1
            if len(concepts) > 1:
                counts["multi_skill_rows"] += 1
            if label not in (0, 1):
                counts["invalid_binary_label_rows"] += 1
            if None in (student, question, order):
                counts["invalid_identifier_rows"] += 1
            for field in ("ms_first_response", "hint_count"):
                try:
                    value = float(row.get(field, ""))
                    if value >= 0 and value < float("inf"):
                        counts[f"valid_{field}_rows"] += 1
                except (TypeError, ValueError):
                    pass
            if not concepts or label not in (0, 1) or None in (student, question, order):
                continue
            key = (student, order, question)
            if key in events:
                counts["repeated_event_rows"] += 1
                previous = events[key]
                previous["skills"].update(concepts)
                previous["labels"].add(label)
            else:
                events[key] = {"skills": set(concepts), "labels": {label}}

    valid = {key: value for key, value in events.items() if len(value["labels"]) == 1}
    counts["conflicting_events"] = len(events) - len(valid)
    per_student = Counter(key[0] for key in valid)
    retained_students = {student for student, count in per_student.items()
                         if count >= min_interactions}
    retained_questions, retained_skills = set(), set()
    retained_count = 0
    for (student, _, question), event in valid.items():
        if student in retained_students:
            retained_count += 1
            retained_questions.add(question)
            retained_skills.update(event["skills"])
    sizes = [per_student[student] for student in retained_students]
    return {
        "dataset": "ASSISTments 2009-2010 corrected collapsed Skill Builder",
        "file_name": path.name,
        "file_bytes": path.stat().st_size,
        "sha256": file_digest(path),
        "text_encoding": encoding,
        "columns": columns,
        "raw": {
            **dict(counts),
            "students": len(students),
            "questions": len(questions),
            "skills": len(skills),
        },
        "eligibility_preview": {
            "min_interactions": min_interactions,
            "events_before_minimum": len(valid),
            "retained_events": retained_count,
            "retained_students": len(retained_students),
            "retained_questions": len(retained_questions),
            "retained_skills": len(retained_skills),
            "min_events_per_student": min(sizes) if sizes else None,
            "median_events_per_student": statistics.median(sizes) if sizes else None,
            "max_events_per_student": max(sizes) if sizes else None,
        },
        "fields": {
            "ordering": [name for name in ("order_id", "position") if name in columns],
            "documented_calendar_timestamp_available": False,
            "duration_columns_are_not_timestamps": [name for name in
                ("ms_first_response", "overlap_time") if name in columns],
            "explicit_difficulty_column_available": False,
            "full_question_content_available": False,
        },
        "skill_name_examples": sorted(skill_names)[:12],
        "note": "Bu çıktı ham veri denetimidir; ayrılmış eğitim veya test sonucu değildir.",
    }
