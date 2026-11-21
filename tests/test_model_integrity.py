"""Hedef cevabın gizlenmesi, nedensellik ve promptların etkisini denetle."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kt_project.models import LargeKnowledgeModel, KnowledgeTracer, ModelConfig
from kt_project.sequences import (InteractionStore, PretrainingDataset, PretrainingCollator,
                                  PrefixDataset, DevicePrefixBatches, collate_prefix)


def configuration():
    return ModelConfig(30, 12, hidden_size=16, layers=2, attention_heads=4,
                       feedforward_size=32, max_history=8, prompt_length=2,
                       prompt_layers=1, dropout=0)


def example(length=3, label=1):
    return {"history_question": np.arange(3, 3+length),
            "history_skills": np.tile([2, 3], (length, 1)),
            "history_response": np.arange(length) % 2,
            "history_difficulty": np.full(length, .4),
            "target_question": 7, "target_skills": np.array([2, 3]),
            "target_difficulty": .4, "label": label, "row_index": length}


class ModelIntegrityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def test_target_label_never_changes_prediction(self):
        batch = collate_prefix([example()])
        changed = {**batch, "label": 1-batch["label"]}
        for method in ("baseline", "kpt", "kmpt"):
            model = KnowledgeTracer(LargeKnowledgeModel(configuration()), method).eval()
            with torch.no_grad():
                torch.testing.assert_close(model(batch), model(changed), rtol=0, atol=0)

    def test_future_interaction_cannot_change_earlier_hidden_state(self):
        lkm = LargeKnowledgeModel(configuration()).eval()
        inputs = torch.randn(1, 5, 16)
        changed = inputs.clone()
        changed[:, 4] += torch.randn(16)*100
        padding = torch.zeros(1, 5, dtype=torch.bool)
        with torch.no_grad():
            a = lkm.encode(inputs, padding, causal=True)
            b = lkm.encode(changed, padding, causal=True)
        torch.testing.assert_close(a[:, :4], b[:, :4], rtol=0, atol=1e-6)

    def test_padding_does_not_change_same_students_prediction(self):
        short = collate_prefix([example(2)])
        together = collate_prefix([example(2), example(7)])
        for method in ("baseline", "kpt", "kmpt"):
            model = KnowledgeTracer(LargeKnowledgeModel(configuration()), method).eval()
            with torch.no_grad():
                torch.testing.assert_close(model(short)[0], model(together)[0], atol=1e-6, rtol=1e-5)

    def test_prompt_receives_gradient_and_changes_in_optimizer_step(self):
        batch = collate_prefix([example(2, 0), example(5, 1)])
        for method in ("kpt", "kmpt"):
            model = KnowledgeTracer(LargeKnowledgeModel(configuration()), method).train()
            prompt = model.prompt_encoder.soft_prompt
            before = prompt.detach().clone()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
            model.loss(batch).backward()
            self.assertIsNotNone(prompt.grad)
            self.assertTrue(torch.isfinite(prompt.grad).all())
            self.assertGreater(prompt.grad.norm().item(), 0)
            optimizer.step()
            self.assertFalse(torch.equal(prompt, before))

    def test_auxiliary_answer_is_only_allowed_during_kmpt_training(self):
        batch = collate_prefix([example(2, 0), example(5, 1)])
        model = KnowledgeTracer(LargeKnowledgeModel(configuration()), "kmpt").train()
        loss = model.loss(batch, auxiliary=True)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.lkm.question_head.weight.grad.norm().item(), 0)
        model.eval()
        with self.assertRaises(ValueError):
            model.loss(batch, auxiliary=True)

    def test_encoder_layers_start_with_distinct_attention_weights(self):
        lkm = LargeKnowledgeModel(configuration())
        self.assertFalse(torch.equal(lkm.encoder.layers[0].self_attn.in_proj_weight,
                                     lkm.encoder.layers[1].self_attn.in_proj_weight))

    def test_pretraining_excludes_test_students_and_updates_all_heads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.npz"
            np.savez(path, student=np.repeat(np.arange(4), 6),
                     question=np.r_[np.tile(np.arange(3, 9), 3), np.full(6, 777)],
                     skills=np.tile([2, 3], (24, 1)), correct=np.arange(24) % 2,
                     difficulty=np.full(24, .4, dtype=np.float32),
                     split=np.repeat([0, 0, 0, 2], 6))
            store = InteractionStore(path)
            dataset = PretrainingDataset(store, max_events=8)
            items = [dataset[index] for index in range(len(dataset))]
            self.assertTrue(all(777 not in item["question"] for item in items))
            positives = sum(item["fbo_label"] for item in items)
            self.assertLessEqual(abs(positives-2*(len(items)-positives)), 1)
            batch = PretrainingCollator(17)(items)
            special = batch["special"].gt(0)
            self.assertTrue(batch["question_labels"][special].eq(-100).all())
            self.assertTrue(batch["response_labels"][special].eq(-100).all())
            model = LargeKnowledgeModel(configuration()).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
            before = model.pretraining_loss(batch)[0].item()
            for _ in range(20):
                optimizer.zero_grad()
                loss, _ = model.pretraining_loss(batch)
                loss.backward()
                optimizer.step()
            after = model.pretraining_loss(batch)[0].item()
            self.assertLess(after, before)
            for head in (model.question_head, model.response_head, model.order_head):
                self.assertGreater(head.weight.grad.norm().item(), 0)

    def test_device_batches_preserve_context_and_hide_target_or_future_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.npz"
            np.savez(path, student=np.repeat(np.arange(3), 6),
                     question=np.tile(np.arange(3, 9), 3), skills=np.tile([2, 3], (18, 1)),
                     correct=np.arange(18) % 2, difficulty=np.full(18, .4, dtype=np.float32),
                     split=np.zeros(18, dtype=np.int8))
            store = InteractionStore(path)
            dataset = PrefixDataset(store, 0, max_history=4)
            dataset.examples = [dataset.examples[index] for index in [3, 5, 9]]
            expected = collate_prefix([dataset[index] for index in range(len(dataset))])
            for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
                actual = next(iter(DevicePrefixBatches(dataset, 3, device)))
                for key in expected:
                    torch.testing.assert_close(actual[key].cpu(), expected[key])


if __name__ == "__main__":
    unittest.main()
