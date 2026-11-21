"""Makale Bölüm3.2: bütün yarım parçalar havuzundan FBO ve konum bazlı RTM."""

import numpy as np
import torch

from .sequences import PretrainingDataset, PretrainingCollator, QUESTION_MASK, RESPONSE_MASK


class PaperPretrainingDataset(PretrainingDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.segments = [(student, left, right) for student, start, stop in self.windows
                         for left, right in [(start, start+(stop-start)//2), (start+(stop-start)//2, stop)]]

    def __getitem__(self, index):
        positive = index < len(self.windows)
        if positive:
            first_id, second_id = 2*index, 2*index+1
        else:
            rng = np.random.default_rng(self.seed+1_000_003*self.epoch+index)
            while True:
                first_id, second_id = rng.integers(len(self.segments), size=2).tolist()
                if not (first_id % 2 == 0 and second_id == first_id+1):
                    break
        _, first_start, first_stop = self.segments[first_id]
        _, second_start, second_stop = self.segments[second_id]
        first, second = np.arange(first_start, first_stop), np.arange(second_start, second_stop)
        indices = np.r_[first, second]
        arrays = self.store.arrays
        result = {key: arrays[source][indices] for key, source in {
            "question": "question", "skills": "skills", "response": "correct", "difficulty": "difficulty"}.items()}
        result.update(first_length=len(first), fbo_label=float(positive),
                      segment_ids=(first_id, second_id), source_indices=indices)
        return result


class PaperPretrainingCollator(PretrainingCollator):
    def __call__(self, items):
        # Reuse padding/CLS/SEP allocation, then reconstruct the uncorrupted values.
        output = super().__call__(items)
        output["question_labels"].fill_(-100)
        output["response_labels"].fill_(-100)
        for row, item in enumerate(items):
            size, first = len(item["question"]), item["first_length"]
            positions = np.r_[np.arange(1, first+1), np.arange(first+2, size+2)]
            for key in ("question", "response"):
                output[key][row, positions] = torch.from_numpy(item[key]).to(output[key].dtype)
            chosen = self.rng.choice(positions, max(1, round(.15*size)), replace=False)
            for position in chosen:
                for key in ("question", "response"):
                    output[key+"_labels"][row, position] = output[key][row, position]
                chance = self.rng.random()
                if chance < .8:
                    output["question"][row, position], output["response"][row, position] = QUESTION_MASK, RESPONSE_MASK
                elif chance < .9:
                    # One source interaction for the selected position's q/r pair.
                    source = int(self.rng.choice(np.flatnonzero(positions != position)))
                    output["question"][row, position] = int(item["question"][source])
                    output["response"][row, position] = int(item["response"][source])
        return output
