"""Öğrenci ayrımını ve yalnızca eğitimden türetilen özellikleri hazırla."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import json
from pathlib import Path

import numpy as np

from .data_audit import detect_encoding, file_digest, integer_id, parse_skills

PAD, UNK, MASK = 0, 1, 2


@dataclass(slots=True)
class Interaction:
    student: int
    order: int
    question: int
    skills: tuple[int, ...]
    correct: int
    duration_ms: float
    hints: float


def _nonnegative(value: str | None) -> float:
    try:
        number = float(value or "nan")
        return number if np.isfinite(number) and number >= 0 else float("nan")
    except ValueError:
        return float("nan")


def load_clean_events(source: Path, min_interactions: int = 5) -> list[Interaction]:
    events: dict[tuple[int, int, int], Interaction] = {}
    conflicts = set()
    with source.open(encoding=detect_encoding(source), newline="") as stream:
        for row in csv.DictReader(stream):
            student, question, order = (integer_id(row[name]) for name in
                                        ("user_id", "problem_id", "order_id"))
            skills = parse_skills(row["skill_id"])
            correct = integer_id(row["correct"])
            if None in (student, question, order) or not skills or correct not in (0, 1):
                continue
            key = student, order, question
            if key in events:
                previous = events[key]
                if previous.correct != correct:
                    conflicts.add(key)
                previous.skills = tuple(sorted(set(previous.skills).union(skills)))
            else:
                events[key] = Interaction(student, order, question, skills, correct,
                    _nonnegative(row.get("ms_first_response")),
                    _nonnegative(row.get("hint_count")))
    per_student = Counter(event.student for key, event in events.items() if key not in conflicts)
    result = [event for key, event in events.items()
              if key not in conflicts and per_student[event.student] >= min_interactions]
    return sorted(result, key=lambda event: (event.student, event.order, event.question))


def prepare_dataset(source: Path, output: Path, split_seed: int = 2026,
                    min_interactions: int = 5, reference_size: int = 32) -> dict:
    events = load_clean_events(Path(source), min_interactions)
    students = sorted({event.student for event in events})
    if len(students) < 10:
        raise ValueError("Üç öğrenci grubu oluşturmak için en az 10 uygun öğrenci gerekli.")
    shuffled = np.random.default_rng(split_seed).permutation(students).tolist()
    n_train, n_val = int(len(students) * .70), int(len(students) * .15)
    members = [set(shuffled[:n_train]), set(shuffled[n_train:n_train+n_val]),
               set(shuffled[n_train+n_val:])]
    if any(members[i].intersection(members[j]) for i in range(3) for j in range(i+1, 3)):
        raise AssertionError("Öğrenci kümeleri çakışıyor.")
    split_for = {student: group for group, group_members in enumerate(members)
                 for student in group_members}
    train = [event for event in events if split_for[event.student] == 0]
    question_map = {question: index + 3 for index, question in
                    enumerate(sorted({event.question for event in train}))}
    skill_map = {skill: index + 2 for index, skill in
                 enumerate(sorted({skill for event in train for skill in event.skills}))}
    student_map = {student: index for index, student in enumerate(students)}
    question_count = Counter(event.question for event in train)
    question_errors = Counter()
    question_skills = defaultdict(set)
    for event in train:
        question_errors[event.question] += 1 - event.correct
        question_skills[event.question].update(event.skills)
    global_difficulty = sum(1-event.correct for event in train) / len(train)
    difficulty = {question: question_errors[question] / count
                  for question, count in question_count.items()}
    max_skills = max(len(event.skills) for event in events)
    encoded_skills = np.zeros((len(events), max_skills), dtype=np.int32)
    for index, event in enumerate(events):
        encoded_skills[index, :len(event.skills)] = [skill_map.get(skill, UNK) for skill in event.skills]
    arrays = {
        "student": np.array([student_map[e.student] for e in events], dtype=np.int32),
        "order": np.array([e.order for e in events], dtype=np.int64),
        "question": np.array([question_map.get(e.question, UNK) for e in events], dtype=np.int32),
        "skills": encoded_skills,
        "correct": np.array([e.correct for e in events], dtype=np.int8),
        "difficulty": np.array([difficulty.get(e.question, global_difficulty) for e in events], dtype=np.float32),
        "duration_ms": np.array([e.duration_ms for e in events], dtype=np.float32),
        "hints": np.array([e.hints for e in events], dtype=np.float32),
        "split": np.array([split_for[e.student] for e in events], dtype=np.int8),
    }
    # Sabit referans soruları yalnızca eğitimdeki sorulardan seç.
    by_skill = defaultdict(list)
    for question, concepts in question_skills.items():
        for skill in concepts:
            by_skill[skill].append(question)
    references = {}
    for skill, candidates in sorted(by_skill.items()):
        supported = [q for q in candidates if question_count[q] >= 3]
        pool = supported if supported else candidates
        pool.sort(key=lambda q: (difficulty[q], q))
        positions = np.linspace(0, len(pool)-1, min(reference_size, len(pool)), dtype=int)
        selected = [pool[int(position)] for position in positions]
        references[str(skill_map[skill])] = [
            {"question": question_map[q],
             "skills": sorted(skill_map[s] for s in question_skills[q]),
             "difficulty": difficulty[q], "training_support": question_count[q]}
            for q in selected]
    summary = {}
    for code, name in enumerate(("train", "validation", "test")):
        selected = arrays["split"] == code
        skill_present = arrays["skills"][selected]
        summary[name] = {
            "students": len(members[code]), "interactions": int(selected.sum()),
            "unseen_question_interactions": int((arrays["question"][selected] == UNK).sum()),
            "interactions_with_unseen_skill": int((skill_present == UNK).any(axis=1).sum()),
        }
    metadata = {
        "source_sha256": file_digest(Path(source)), "split_seed": split_seed,
        "student_splits_disjoint": True, "minimum_interactions": min_interactions,
        "multi_skill_policy": "one event, set of concepts; mean concept embedding",
        "question_vocabulary_size": len(question_map) + 3,
        "skill_vocabulary_size": len(skill_map) + 2,
        "max_skills_per_event": max_skills,
        "global_training_difficulty": global_difficulty,
        "difficulty_fitted_on": "train students only",
        "unknown_question_policy": "UNK question id, known event skills, global training difficulty",
        "reference_policy": "training questions, fixed difficulty-spanning subset, uniform weights",
        "splits": summary,
        "calendar_timestamps_available": False,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "interactions.npz", **arrays)
    for filename, value in {
        "metadata.json": metadata,
        "references.json": references,
        "vocabulary.json": {"question": question_map, "skill": skill_map, "student": student_map},
        "student_splits.json": {name: sorted(members[i])
                                 for i, name in enumerate(("train", "validation", "test"))},
    }.items():
        (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata
