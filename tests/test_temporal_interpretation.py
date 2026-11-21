"""Aynı adım yorumu geleceği gizlerken gerçek prompt erişimi sağlamalı."""

from pathlib import Path
import sys
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from kt_project.models import ModelConfig
from kt_project.plkt_models import PaperLKM,PaperKnowledgeTracer
from kt_project.plkt_temporal import build_temporal_tracer


class TemporalInterpretationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.config=ModelConfig(25,8,hidden_size=16,layers=1,attention_heads=4,
            feedforward_size=32,max_history=8,prompt_length=3,prompt_layers=1,dropout=0)
        self.batch={"history_question":torch.tensor([[3,4,5]]),
            "history_skills":torch.tensor([[[2],[2],[3]]]),
            "history_response":torch.tensor([[1,0,1]]),
            "history_difficulty":torch.tensor([[.3,.4,.3]]),
            "history_padding":torch.zeros(1,3,dtype=torch.bool),
            "target_question":torch.tensor([6]),"target_skills":torch.tensor([[2]]),
            "target_difficulty":torch.tensor([.4]),"label":torch.tensor([1.])}

    def test_same_state_dict_but_prompt_access_is_open(self):
        strict=PaperKnowledgeTracer(PaperLKM(self.config),"kmpt").eval()
        temporal=build_temporal_tracer(self.config,"kmpt").eval()
        temporal.load_state_dict(strict.state_dict())
        for key,value in strict.state_dict().items():
            torch.testing.assert_close(value,temporal.state_dict()[key],rtol=0,atol=0)
        before=temporal(self.batch).detach()
        with torch.no_grad():
            temporal.prompt_encoder.soft_prompt.add_(torch.randn_like(temporal.prompt_encoder.soft_prompt))
        self.assertGreater((before-temporal(self.batch)).abs().max().item(),1e-8)

    def test_history_cannot_see_later_events_but_template_can_see_its_known_tokens(self):
        model=build_temporal_tracer(self.config,"kmpt").eval()
        masks=[]
        hook=model.lkm.encoder.register_forward_pre_hook(
            lambda module,args,kwargs:masks.append(kwargs["mask"]),with_kwargs=True)
        model(self.batch)
        hook.remove()
        mask=masks[0]
        start=1+self.batch["history_question"].shape[1]
        self.assertTrue(bool(mask[1,2]))
        self.assertTrue(bool(mask[1,-1]))
        self.assertFalse(bool(mask[start+2,-1]))
        self.assertFalse(bool(mask[-1,start]))
        self.assertFalse(bool(mask[start+2,0]))

    def test_prediction_is_independent_of_target_label_and_accepts_no_label(self):
        model=build_temporal_tracer(self.config,"kmpt").eval()
        changed={**self.batch,"label":torch.tensor([0.])}
        unlabeled={key:value for key,value in self.batch.items() if key!="label"}
        torch.testing.assert_close(model(self.batch),model(changed),rtol=0,atol=0)
        torch.testing.assert_close(model(self.batch),model(unlabeled),rtol=0,atol=0)


if __name__=="__main__":
    unittest.main()
