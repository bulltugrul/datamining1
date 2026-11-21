"""Makale eşitlikleri, iki aşamalı dikkat ve açıkça test edilen şablon gerilimi."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kt_project.models import ModelConfig
from kt_project.plkt_models import PaperLKM, PaperKnowledgeTracer, PaperPromptEncoder
from kt_project.plkt_sequences import PaperPretrainingDataset, PaperPretrainingCollator
from kt_project.sequences import InteractionStore, PrefixDataset, collate_prefix
from kt_project.tracking import FrozenTracer, snapshot_for


class PaperAlignmentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.config = ModelConfig(50, 12, hidden_size=16, layers=2, attention_heads=4,
            feedforward_size=32, max_history=8, prompt_length=3, prompt_layers=1, dropout=0.)
        self.batch = {
            "history_question": torch.tensor([[3, 4, 5], [4, 6, 0]]),
            "history_skills": torch.tensor([[[2], [3], [2]], [[2], [3], [0]]]),
            "history_response": torch.tensor([[1, 0, 1], [0, 1, 3]]),
            "history_difficulty": torch.tensor([[.3, .4, .3], [.4, .2, 0.]]),
            "history_padding": torch.tensor([[False, False, False], [False, False, True]]),
            "target_question": torch.tensor([7, 8]), "target_skills": torch.tensor([[2], [3]]),
            "target_difficulty": torch.tensor([.2, .3]), "label": torch.tensor([0., 1.]),
        }

    def test_paper_order_and_readout_indices(self):
        encoder = PaperPromptEncoder(self.config)
        q, c, r = [torch.randn(2, 16) for _ in range(3)]
        _, response_index = encoder(q, c, r, auxiliary=False)
        _, exercise_index = encoder(q, c, r, auxiliary=True)
        self.assertEqual(response_index, 2)
        self.assertEqual(exercise_index, 0)
        captured = []
        handle = encoder.norm.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        encoder(q, c, r)
        handle.remove()
        stripped = captured[0]-encoder.position(torch.arange(6))
        torch.testing.assert_close(stripped[:, 0], q)
        torch.testing.assert_close(stripped[:, 1], c)
        torch.testing.assert_close(stripped[:, 2], r)
        torch.testing.assert_close(stripped[:, 3:], encoder.soft_prompt[None].expand(2, -1, -1))

    def test_both_encoders_bidirectional_in_training_and_causal_in_prediction(self):
        model = PaperKnowledgeTracer(PaperLKM(self.config), "kpt")
        recorded = []
        hooks = [encoder.register_forward_pre_hook(
            lambda module, args, kwargs: recorded.append(kwargs.get("mask")), with_kwargs=True)
            for encoder in (model.prompt_encoder.encoder, model.lkm.encoder)]
        model.train()
        model(self.batch)
        self.assertTrue(all(mask is None for mask in recorded))
        recorded.clear()
        model.eval()
        model(self.batch)
        self.assertTrue(all(torch.equal(mask, torch.triu(torch.ones_like(mask), 1)) for mask in recorded))
        for hook in hooks:
            hook.remove()

    def test_target_correctness_not_used_in_prediction_or_response_training(self):
        model = PaperKnowledgeTracer(PaperLKM(self.config), "kmpt")
        changed = {**self.batch, "label": 1-self.batch["label"]}
        for training in (True, False):
            model.train(training)
            torch.testing.assert_close(model(self.batch), model(changed), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            model.loss(self.batch, auxiliary=True)

    def test_tail_prompts_have_training_gradient_but_no_strict_causal_prediction_effect(self):
        model = PaperKnowledgeTracer(PaperLKM(self.config), "kmpt", "paper_tail")
        model.train()
        model.loss(self.batch).backward()
        self.assertGreater(model.prompt_encoder.soft_prompt.grad.abs().sum().item(), 1e-8)
        model.eval()
        before = model(self.batch).detach()
        with torch.no_grad():
            model.prompt_encoder.soft_prompt.add_(torch.randn_like(model.prompt_encoder.soft_prompt)*10)
        torch.testing.assert_close(before, model(self.batch), rtol=0, atol=1e-7)

    def test_prefix_control_retains_inference_prompt_effect(self):
        model = PaperKnowledgeTracer(PaperLKM(self.config), "kmpt", "prefix_control").eval()
        before = model(self.batch).detach()
        with torch.no_grad():
            model.prompt_encoder.soft_prompt.add_(torch.randn_like(model.prompt_encoder.soft_prompt)*10)
        self.assertGreater((before-model(self.batch)).abs().max().item(), 1e-8)

    def test_prompt_template_excludes_target_difficulty(self):
        model = PaperKnowledgeTracer(PaperLKM(self.config), "kpt").eval()
        changed = {**self.batch, "target_difficulty": torch.ones(2)}
        torch.testing.assert_close(model(self.batch), model(changed), rtol=0, atol=0)

    def test_post_norm_bert_style_and_lkm_updates_are_explicit(self):
        model = PaperKnowledgeTracer(PaperLKM(self.config), "kmpt")
        self.assertFalse(model.lkm.encoder.layers[0].norm_first)
        self.assertIsNone(model.lkm.encoder.norm)
        self.assertTrue(model.lkm.encoder.layers[0].linear1.weight.requires_grad)

    def test_padding_and_batch_companions_do_not_change_prediction(self):
        model = PaperKnowledgeTracer(PaperLKM(self.config), "kmpt").eval()
        together = model(self.batch)[1]
        alone = {key: value[1:2, :2] if key.startswith("history") else value[1:2]
                 for key, value in self.batch.items()}
        torch.testing.assert_close(together, model(alone)[0], rtol=0, atol=1e-7)

    def test_fbo_uses_whole_segment_pool_and_never_labels_original_pair_negative(self):
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "events.npz"
            np.savez(path, student=np.repeat(np.arange(4), 10), split=np.repeat([0, 0, 0, 1], 10),
                question=np.arange(3, 43), skills=np.full((40, 1), 2), correct=np.arange(40)%2,
                difficulty=np.full(40, .3, dtype=np.float32))
            store = InteractionStore(path)
            dataset = PaperPretrainingDataset(store, 8)
            seen_first_second_half, seen_same_student_negative = False, False
            for epoch in range(30):
                dataset.set_epoch(epoch)
                for i in range(len(dataset)):
                    item = dataset[i]
                    a, b = item["segment_ids"]
                    self.assertTrue(np.all(store.arrays["split"][item["source_indices"]] == 0))
                    if item["fbo_label"]:
                        self.assertEqual(a % 2, 0)
                        self.assertEqual(b, a+1)
                    else:
                        self.assertFalse(a % 2 == 0 and b == a+1)
                        seen_first_second_half |= a % 2 == 1
                        seen_same_student_negative |= dataset.segments[a][0] == dataset.segments[b][0]
            self.assertTrue(seen_first_second_half)
            self.assertTrue(seen_same_student_negative)
            batch = PaperPretrainingCollator(17)([dataset[i] for i in range(4)])
            chosen = batch["question_labels"] != -100
            self.assertTrue(torch.equal(chosen, batch["response_labels"] != -100))
            self.assertFalse(bool((chosen & batch["special"].gt(0)).any()))
            model = PaperLKM(self.config)
            loss, parts = model.pretraining_loss(batch)
            per_token = parts["question_token_mean"]+parts["response_token_mean"]
            expected = per_token*chosen.sum()/len(chosen)+parts["order_loss"]
            torch.testing.assert_close(loss, expected)

    def test_corrected_checkpoint_loads_through_student_tracking_without_labels(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            (root / "data/processed").mkdir(parents=True)
            (root / "configs").mkdir()
            (root / "data/processed/metadata.json").write_text(json.dumps({"source_sha256": "toy"}))
            model = PaperKnowledgeTracer(PaperLKM(self.config), "kmpt")
            path = root / "weights.pt"
            torch.save({"config": self.config.to_dict(), "model_state": model.state_dict(),
                        "format": "plkt_method_v2", "variant": "kmpt", "method": "kmpt"}, path)
            selection = {"checkpoint": "weights.pt", "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            (root / "selection.json").write_text(json.dumps(selection))
            (root / "configs/active_model.json").write_text(json.dumps({"selection_file": "selection.json"}))
            store = SimpleNamespace(arrays={
                "question": np.arange(3, 8), "skills": np.full((5, 1), 2),
                "correct": np.array([1, 0, 1, 0, 1]), "difficulty": np.full(5, .3, dtype=np.float32)},
                spans={0: (0, 5)})
            tracer = FrozenTracer(root, store, device="cpu")
            self.assertIsInstance(tracer.model, PaperKnowledgeTracer)
            self.assertFalse(tracer.model.training)
            self.assertFalse(any(parameter.requires_grad for parameter in tracer.model.parameters()))
            request = (snapshot_for(store, 0, 4), {"question": 7, "skills": [2], "difficulty": .3})
            batch = tracer.query_batch([request])
            self.assertNotIn("label", batch)
            self.assertTrue(np.isfinite(tracer.predict_requests([request])).all())
            selection["checkpoint_sha256"] = "incorrect"
            (root / "selection.json").write_text(json.dumps(selection))
            with self.assertRaises(ValueError):
                FrozenTracer(root, store, device="cpu")


if __name__ == "__main__":
    unittest.main()
