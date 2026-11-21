"""Makale denetimi sonrası PLKT: Denklem 10–13 ve Bölüm 3.5 dikkat kipleri.

Eski deneyin kaynakları ve ağırlıkları korunur. `paper_tail` metindeki sıralamayı
ve katı nedensel tahmini uygular. `prefix_control` aynı yeni kodda yalnızca
şablon sırasını değiştirir; makale tarifinin yerine geçirilmez.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .models import ModelConfig, LargeKnowledgeModel, initialize
from .sequences import QUESTION_MASK, RESPONSE_MASK


def bert_encoder(config, layers):
    layer = nn.TransformerEncoderLayer(
        config.hidden_size, config.attention_heads, config.feedforward_size,
        config.dropout, activation="gelu", batch_first=True, norm_first=False,
        layer_norm_eps=1e-12)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class PaperLKM(LargeKnowledgeModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = bert_encoder(config, config.layers)
        self.encoder.apply(initialize)

    def pretraining_loss(self, batch):
        embeddings = self.interaction_embeddings(batch["question"], batch["skills"],
                                                 batch["response"], batch["difficulty"])
        embeddings = torch.where(batch["special"].gt(0).unsqueeze(-1),
                                 self.special_embedding(batch["special"]), embeddings)
        hidden = self.encode(embeddings, batch["padding"], causal=False)
        chosen = batch["question_labels"].ne(-100)
        assert torch.equal(chosen, batch["response_labels"].ne(-100))
        q_losses = F.cross_entropy(self.question_head(hidden[chosen]),
                                  batch["question_labels"][chosen], reduction="none")
        r_losses = F.binary_cross_entropy_with_logits(self.response_head(hidden[chosen]).squeeze(-1),
                                                       batch["response_labels"][chosen].float(), reduction="none")
        fbo = F.binary_cross_entropy_with_logits(self.order_head(hidden[:, 0]).squeeze(-1), batch["fbo_label"])
        # Eq. 3–7 sum over masked positions and sequences. Divide the WHOLE sum
        # by sequence count; separately token-averaging RTM changes its FBO weight.
        batch_size = len(hidden)
        q_sum, r_sum = q_losses.sum()/batch_size, r_losses.sum()/batch_size
        return q_sum+r_sum+fbo, {"question_loss": q_sum.detach(), "response_loss": r_sum.detach(),
                               "order_loss": fbo.detach(), "question_token_mean": q_losses.mean().detach(),
                               "response_token_mean": r_losses.mean().detach()}


class PaperPromptEncoder(nn.Module):
    def __init__(self, config, layout="paper_tail"):
        super().__init__()
        if layout not in {"paper_tail", "prefix_control"}:
            raise ValueError(layout)
        self.layout = layout
        self.soft_prompt = nn.Parameter(torch.empty(config.prompt_length, config.hidden_size))
        self.position = nn.Embedding(config.prompt_length+3, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.encoder = bert_encoder(config, config.prompt_layers)
        self.apply(initialize)
        nn.init.normal_(self.soft_prompt, std=.02)

    def forward(self, question, concepts, response, auxiliary=False, causal=False):
        soft = self.soft_prompt.unsqueeze(0).expand(len(question), -1, -1)
        # Eq.12: e,c,MASK_r,prompt; Eq.13: MASK_e,c,r,prompt.
        if self.layout == "paper_tail":
            tokens = torch.cat([torch.stack([question, concepts, response], dim=1), soft], dim=1)
            mask_index = 0 if auxiliary else 2
        else:
            tail = torch.stack([concepts, response, question] if auxiliary else [question, concepts, response], dim=1)
            tokens = torch.cat([soft, tail], dim=1)
            mask_index = tokens.shape[1]-1
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        tokens = self.norm(tokens+self.position(positions))
        mask = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device), 1) if causal else None
        return self.encoder(tokens, mask=mask), mask_index


class PaperKnowledgeTracer(nn.Module):
    def __init__(self, lkm, method, layout="paper_tail"):
        super().__init__()
        if method not in {"baseline", "kpt", "kmpt"}:
            raise ValueError(method)
        self.lkm, self.method, self.layout = lkm, method, layout
        config = lkm.config
        self.prompt_encoder = None
        self.classification_head = None
        if method == "baseline":
            self.classification_head = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.hidden_size, 1))
            self.classification_head.apply(initialize)
        else:
            self.prompt_encoder = PaperPromptEncoder(config, layout)
        # Algorithm1 step24 tunes the pre-trained LKM; no freezing is prescribed.
        for parameter in self.lkm.order_head.parameters():
            parameter.requires_grad_(False)
        for parameter in self.lkm.question_head.parameters():
            parameter.requires_grad_(method == "kmpt")
        for parameter in self.lkm.response_head.parameters():
            parameter.requires_grad_(method != "baseline")

    def encode_prediction(self, batch, auxiliary=False):
        if auxiliary and (self.method != "kmpt" or not self.training):
            raise ValueError("Yardımcı şablon yalnızca KMPT eğitimi içindir.")
        question = batch["target_question"]
        response = batch["label"].long() if auxiliary else torch.full_like(question, RESPONSE_MASK)
        if auxiliary:
            question = torch.full_like(question, QUESTION_MASK)
        # Each batch is already a prefix ending immediately BEFORE its target.
        # Bidirectional attention over this observed prefix does not expose future labels.
        history = self.lkm.interaction_embeddings(batch["history_question"], batch["history_skills"],
                                                   batch["history_response"], batch["history_difficulty"])
        causal = not self.training
        if self.method == "baseline":
            template = self.lkm.interaction_embeddings(question, batch["target_skills"], response,
                                                        batch["target_difficulty"]).unsqueeze(1)
            local_index = 0
        else:
            # Eq.12–13 specify e,c,r embeddings in the template; target difficulty
            # is not added to e. History difficulty remains in Eq.2 embeddings.
            template, local_index = self.prompt_encoder(self.lkm.question_embedding(question),
                self.lkm.concept_embeddings(batch["target_skills"]), self.lkm.response_embedding(response),
                auxiliary=auxiliary, causal=causal)
        batch_size, width = history.shape[:2]
        cls = self.lkm.special_embedding.weight[1].expand(batch_size, 1, -1)
        embeddings = torch.cat([cls, history, template], dim=1)
        padding = torch.cat([torch.zeros(batch_size, 1, device=history.device, dtype=torch.bool),
                              batch["history_padding"],
                              torch.zeros(batch_size, template.shape[1], device=history.device, dtype=torch.bool)], dim=1)
        lengths = (~batch["history_padding"]).sum(dim=1)
        positions = torch.cat([torch.zeros(batch_size, 1, dtype=torch.long, device=history.device),
            torch.arange(1, width+1, device=history.device).expand(batch_size, -1),
            lengths[:, None]+torch.arange(1, template.shape[1]+1, device=history.device)], dim=1)
        hidden = self.lkm.encode(embeddings, padding, positions, causal=causal)
        return hidden[:, 1+width+local_index]

    def forward(self, batch):
        head = self.classification_head if self.method == "baseline" else self.lkm.response_head
        return head(self.encode_prediction(batch)).squeeze(-1)

    def loss(self, batch, auxiliary=False):
        if auxiliary:
            return F.cross_entropy(self.lkm.question_head(self.encode_prediction(batch, auxiliary=True)),
                                   batch["target_question"])
        return F.binary_cross_entropy_with_logits(self(batch), batch["label"])

    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)}
