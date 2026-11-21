"""Açık bir yorum kontrolü: hedefin bilinen şablon tokenları aynı zaman adımı.

Makalenin kesin maskesi iddiası değildir. Geçmiş konumları nedensel kalır;
hedefin şablonu içinde bütün tokenlar bilinen girdi veya MASK olduğundan ortak
erişim verilir. Gelecek öğrenci cevabı eklenmez.
"""

import torch

from .plkt_models import PaperLKM, PaperPromptEncoder, PaperKnowledgeTracer


class TemporalBlockLKM(PaperLKM):
    def encode(self, embeddings, padding, positions=None, causal=False):
        if not causal:
            return super().encode(embeddings,padding,positions,causal=False)
        batch,width=embeddings.shape[:2]
        if positions is None:
            positions=torch.arange(width,device=embeddings.device).expand(batch,-1)
        if positions.max().item() >= self.config.max_positions:
            raise ValueError("Konum kapasitesi aşıldı.")
        encoded=self.dropout(self.input_norm(embeddings+self.position_embedding(positions)))
        mask=torch.triu(torch.ones(width,width,device=embeddings.device,dtype=torch.bool),1)
        template_width=self.config.prompt_length+3
        # Past events still cannot see any later event or target-template key.
        # The current target's known/masked template is one semantic time block.
        mask[-template_width:,-template_width:]=False
        return self.encoder(encoded,mask=mask,src_key_padding_mask=padding)


class TemporalPromptEncoder(PaperPromptEncoder):
    def forward(self,question,concepts,response,auxiliary=False,causal=False):
        # All template elements belong to the same target step; no observed
        # target response is supplied during prediction.
        return super().forward(question,concepts,response,auxiliary,causal=False)


class TemporalKnowledgeTracer(PaperKnowledgeTracer):
    def __init__(self,lkm,method):
        if method not in {"kpt","kmpt"}:
            raise ValueError("Zaman bloğu yorumu yalnızca kuyruk şablonları için.")
        super().__init__(lkm,method,layout="paper_tail")
        self.prompt_encoder=TemporalPromptEncoder(lkm.config,"paper_tail")


def build_temporal_tracer(config,method):
    return TemporalKnowledgeTracer(TemporalBlockLKM(config),method)
