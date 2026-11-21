"""Öğrenci geçmişleri, nedensel tahmin örnekleri ve RTM/FBO verileri."""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import torch
from torch.utils.data import Dataset

QUESTION_PAD, QUESTION_UNK, QUESTION_MASK = 0, 1, 2
RESPONSE_MASK, RESPONSE_PAD = 2, 3


class InteractionStore:
    def __init__(self, path: str | Path):
        with np.load(path, allow_pickle=False) as packed:
            self.arrays = {key: packed[key] for key in packed.files}
        self.spans = {}
        students = self.arrays["student"]
        starts = np.r_[0, np.flatnonzero(students[1:] != students[:-1]) + 1]
        ends = np.r_[starts[1:], len(students)]
        for start, end in zip(starts, ends):
            self.spans[int(students[start])] = (int(start), int(end))

    def select_students(self, split: int, limit: int | None = None, seed: int = 17):
        students = [student for student, (start, _) in self.spans.items()
                    if self.arrays["split"][start] == split]
        if limit is not None and limit < len(students):
            students = np.random.default_rng(seed).choice(students, limit, replace=False).tolist()
        return sorted(students)


class PrefixDataset(Dataset):
    """Her hedef cevabı geçmişten ayır; ilk etkileşim bağlam olarak kullanılır."""

    def __init__(self, store: InteractionStore, split: int, max_history: int,
                 student_limit: int | None = None, target_limit: int | None = None,
                 seed: int = 17):
        self.store, self.max_history = store, max_history
        examples = [(max(start, target-max_history), target)
                    for student in store.select_students(split, student_limit, seed)
                    for start, end in [store.spans[student]]
                    for target in range(start+1, end)]
        if target_limit is not None and target_limit < len(examples):
            chosen = np.random.default_rng(seed).choice(len(examples), target_limit, replace=False)
            examples = [examples[int(index)] for index in sorted(chosen)]
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        start, target = self.examples[index]
        arrays = self.store.arrays
        return {
            "history_question": arrays["question"][start:target],
            "history_skills": arrays["skills"][start:target],
            "history_response": arrays["correct"][start:target],
            "history_difficulty": arrays["difficulty"][start:target],
            "target_question": arrays["question"][target],
            "target_skills": arrays["skills"][target],
            "target_difficulty": arrays["difficulty"][target],
            "label": arrays["correct"][target],
            "row_index": target,
        }


def collate_prefix(items):
    batch, length = len(items), max(len(item["history_question"]) for item in items)
    skill_width = len(items[0]["target_skills"])
    result = {
        "history_question": np.zeros((batch, length), dtype=np.int64),
        "history_skills": np.zeros((batch, length, skill_width), dtype=np.int64),
        "history_response": np.full((batch, length), RESPONSE_PAD, dtype=np.int64),
        "history_difficulty": np.zeros((batch, length), dtype=np.float32),
        "history_padding": np.ones((batch, length), dtype=bool),
    }
    for row, item in enumerate(items):
        size = len(item["history_question"])
        for key in ("history_question", "history_skills", "history_response", "history_difficulty"):
            result[key][row, :size] = item[key]
        result["history_padding"][row, :size] = False
    for key in ("target_question", "target_skills", "row_index"):
        result[key] = np.asarray([item[key] for item in items], dtype=np.int64)
    for key in ("target_difficulty", "label"):
        result[key] = np.asarray([item[key] for item in items], dtype=np.float32)
    return {key: torch.from_numpy(value) for key, value in result.items()}


class PretrainingDataset(Dataset):
    """Eğitim öğrencilerinden iki ardışık parça ve 2:1 FBO örnekleri."""

    def __init__(self, store: InteractionStore, max_events: int, student_limit=None, seed=17):
        self.store, self.max_events, self.seed, self.epoch = store, max_events, seed, 0
        self.windows = [(student, left, min(left+max_events, end))
                        for student in store.select_students(0, student_limit, seed)
                        for start, end in [store.spans[student]]
                        for left in range(start, end, max_events)
                        if min(left+max_events, end)-left >= 2]
        if len({window[0] for window in self.windows}) < 2:
            raise ValueError("FBO için en az iki eğitim öğrencisi gerekli.")

    def __len__(self):
        return len(self.windows) + math.ceil(len(self.windows)/2)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        positive = index < len(self.windows)
        student, start, end = self.windows[index % len(self.windows)]
        middle = start + (end-start)//2
        first = np.arange(start, middle)
        if positive:
            second = np.arange(middle, end)
        else:
            rng = np.random.default_rng(self.seed + 1_000_003*self.epoch + index)
            candidate = int(rng.integers(len(self.windows)))
            while self.windows[candidate][0] == student:
                candidate = int(rng.integers(len(self.windows)))
            _, other_start, other_end = self.windows[candidate]
            other_middle = other_start + (other_end-other_start)//2
            second = np.arange(other_middle, min(other_end, other_middle+self.max_events-len(first)))
        arrays = self.store.arrays
        indices = np.r_[first, second]
        result = {key: arrays[source][indices] for key, source in {
            "question": "question", "skills": "skills", "response": "correct",
            "difficulty": "difficulty"}.items()}
        result.update({"first_length": len(first), "fbo_label": float(positive)})
        return result


class PretrainingCollator:
    def __init__(self, seed: int):
        self.seed = seed
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.rng = np.random.default_rng(self.seed + 104729*epoch)

    def __call__(self, items):
        batch = len(items)
        width = max(len(item["question"])+2 for item in items)
        skill_width = items[0]["skills"].shape[-1]
        output = {
            "question": np.zeros((batch, width), dtype=np.int64),
            "skills": np.zeros((batch, width, skill_width), dtype=np.int64),
            "response": np.full((batch, width), RESPONSE_PAD, dtype=np.int64),
            "difficulty": np.zeros((batch, width), dtype=np.float32),
            "special": np.zeros((batch, width), dtype=np.int64),
            "padding": np.ones((batch, width), dtype=bool),
            "question_labels": np.full((batch, width), -100, dtype=np.int64),
            "response_labels": np.full((batch, width), -100, dtype=np.int64),
            "fbo_label": np.asarray([item["fbo_label"] for item in items], dtype=np.float32),
        }
        for row, item in enumerate(items):
            size, first = len(item["question"]), item["first_length"]
            # [CLS], segment A, [SEP], segment B
            real_positions = np.r_[np.arange(1, first+1), np.arange(first+2, size+2)]
            for key in ("question", "skills", "response", "difficulty"):
                output[key][row, real_positions] = item[key]
            output["special"][row, 0] = 1
            output["special"][row, first+1] = 2
            output["padding"][row, :size+2] = False
            masked = self.rng.choice(real_positions, max(1, round(.15*size)), replace=False)
            for key, label_key, mask_value in (("question", "question_labels", QUESTION_MASK),
                                                ("response", "response_labels", RESPONSE_MASK)):
                output[label_key][row, masked] = output[key][row, masked]
                for position in masked:
                    chance = self.rng.random()
                    if chance < .8:
                        output[key][row, position] = mask_value
                    elif chance < .9:
                        alternatives = real_positions[real_positions != position]
                        replacement = int(self.rng.choice(alternatives))
                        output[key][row, position] = item[key][int(np.flatnonzero(real_positions == replacement)[0])]
        return {key: torch.from_numpy(value) for key, value in output.items()}


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


class DevicePrefixBatches:
    """Küçük temel dizilerden geçmişleri aygıt üzerinde toplu oluştur.

    Her hedef için geçmişi ayrı Python döngüsüyle kopyalamaz. Geçmiş dışındaki
    bütün konumlar, model girdisi üretilmeden önce sıfırlanır.
    """

    def __init__(self, dataset: PrefixDataset, batch_size: int, device, seed=17, shuffle=False):
        self.dataset, self.batch_size = dataset, batch_size
        self.device, self.seed, self.shuffle = torch.device(device), seed, shuffle
        store = dataset.store
        cache_name = str(self.device)
        if not hasattr(store, "_device_cache"):
            store._device_cache = {}
        if cache_name not in store._device_cache:
            store._device_cache[cache_name] = {
                key: torch.as_tensor(store.arrays[source], dtype=dtype, device=self.device)
                for key, source, dtype in [
                    ("question", "question", torch.long), ("skills", "skills", torch.long),
                    ("response", "correct", torch.long), ("difficulty", "difficulty", torch.float32)]}
        self.arrays = store._device_cache[cache_name]
        self.examples = torch.as_tensor(np.asarray(dataset.examples, dtype=np.int64), device=self.device)

    def __len__(self):
        return math.ceil(len(self.dataset)/self.batch_size)

    def __iter__(self):
        order = (torch.randperm(len(self.dataset), generator=torch.Generator().manual_seed(self.seed))
                 if self.shuffle else torch.arange(len(self.dataset)))
        offsets = torch.arange(self.dataset.max_history, device=self.device)[None, :]
        for left in range(0, len(order), self.batch_size):
            selected = order[left:left+self.batch_size].to(self.device)
            start, target = self.examples[selected].unbind(dim=1)
            valid = offsets < (target-start)[:, None]
            indices = (start[:, None]+offsets).clamp_max(len(self.arrays["question"])-1)
            history = {}
            for key, pad in (("question", 0), ("skills", 0), ("response", RESPONSE_PAD), ("difficulty", 0)):
                values = self.arrays[key][indices]
                mask = valid.unsqueeze(-1) if key == "skills" else valid
                history["history_"+key] = torch.where(mask, values, pad)
            yield {**history, "history_padding": ~valid,
                   "target_question": self.arrays["question"][target],
                   "target_skills": self.arrays["skills"][target],
                   "target_difficulty": self.arrays["difficulty"][target],
                   "label": self.arrays["response"][target].float(), "row_index": target}
