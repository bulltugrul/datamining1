"""LKM, RTM/FBO ve ortak temelden türetilen üç uyarlama yöntemi."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .sequences import QUESTION_MASK, RESPONSE_MASK


@dataclass(frozen=True)
class ModelConfig:
    question_vocabulary_size: int
    skill_vocabulary_size: int
    hidden_size: int = 128
    layers: int = 2
    attention_heads: int = 4
    feedforward_size: int = 512
    max_history: int = 128
    prompt_length: int = 8
    prompt_layers: int = 1
    dropout: float = .2

    @property
    def max_positions(self):
        return self.max_history + self.prompt_length + 8

    def to_dict(self):
        return asdict(self)


def initialize(module):
    # TransformerEncoder klonlarının aynı ağırlıklarla kalmasını engeller.
    if isinstance(module, nn.MultiheadAttention):
        nn.init.xavier_uniform_(module.in_proj_weight)
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
    elif isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, std=.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding) and module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def make_encoder(config: ModelConfig, layers: int):
    layer = nn.TransformerEncoderLayer(
        d_model=config.hidden_size, nhead=config.attention_heads,
        dim_feedforward=config.feedforward_size, dropout=config.dropout,
        activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(config.hidden_size),
                                 enable_nested_tensor=False)


class LargeKnowledgeModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        size = config.hidden_size
        self.question_embedding = nn.Embedding(config.question_vocabulary_size, size, padding_idx=0)
        self.skill_embedding = nn.Embedding(config.skill_vocabulary_size, size, padding_idx=0)
        self.response_embedding = nn.Embedding(4, size, padding_idx=3)
        self.difficulty_embedding = nn.Linear(1, size)
        self.position_embedding = nn.Embedding(config.max_positions, size)
        self.special_embedding = nn.Embedding(3, size)
        self.input_norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(config.dropout)
        self.encoder = make_encoder(config, config.layers)
        self.question_head = nn.Linear(size, config.question_vocabulary_size)
        self.response_head = nn.Linear(size, 1)
        self.order_head = nn.Linear(size, 1)
        self.apply(initialize)
        # Eğitimde görülmeyen soru veya beceri rastgele anlam taşımamalı.
        with torch.no_grad():
            self.question_embedding.weight[1].zero_()
            self.skill_embedding.weight[1].zero_()

    def concept_embeddings(self, skills):
        valid = skills.ne(0).unsqueeze(-1)
        embeddings = self.skill_embedding(skills) * valid
        return embeddings.sum(dim=-2) / valid.sum(dim=-2).clamp_min(1)

    def interaction_embeddings(self, question, skills, response, difficulty):
        return (self.question_embedding(question) + self.concept_embeddings(skills)
                + self.response_embedding(response)
                + self.difficulty_embedding(difficulty.unsqueeze(-1)))

    def encode(self, embeddings, padding, positions=None, causal=False):
        batch, width = embeddings.shape[:2]
        if positions is None:
            positions = torch.arange(width, device=embeddings.device).expand(batch, -1)
        if positions.max().item() >= self.config.max_positions:
            raise ValueError("Dizi uzunluğu modelin konum kapasitesini aşıyor.")
        encoded = self.dropout(self.input_norm(embeddings + self.position_embedding(positions)))
        mask = torch.triu(torch.ones(width, width, device=embeddings.device, dtype=torch.bool), 1) if causal else None
        return self.encoder(encoded, mask=mask, src_key_padding_mask=padding)

    def pretraining_loss(self, batch):
        embeddings = self.interaction_embeddings(batch["question"], batch["skills"],
                                                 batch["response"], batch["difficulty"])
        embeddings = torch.where(batch["special"].gt(0).unsqueeze(-1),
                                 self.special_embedding(batch["special"]), embeddings)
        hidden = self.encode(embeddings, batch["padding"], causal=False)
        question_mask = batch["question_labels"].ne(-100)
        response_mask = batch["response_labels"].ne(-100)
        # Büyük soru sözlüğü için yalnızca maskeli konumların logits'leri üretilir.
        question_loss = F.cross_entropy(self.question_head(hidden[question_mask]),
                                       batch["question_labels"][question_mask])
        response_loss = F.binary_cross_entropy_with_logits(
            self.response_head(hidden[response_mask]).squeeze(-1),
            batch["response_labels"][response_mask].float())
        order_loss = F.binary_cross_entropy_with_logits(
            self.order_head(hidden[:, 0]).squeeze(-1), batch["fbo_label"])
        loss = question_loss + response_loss + order_loss
        return loss, {"question_loss": question_loss.detach(),
                      "response_loss": response_loss.detach(), "order_loss": order_loss.detach()}


class PromptEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.soft_prompt = nn.Parameter(torch.empty(config.prompt_length, config.hidden_size))
        self.position = nn.Embedding(config.prompt_length+3, config.hidden_size)
        self.encoder = make_encoder(config, config.prompt_layers)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.apply(initialize)
        nn.init.normal_(self.soft_prompt, std=.02)

    def forward(self, question, concepts, response, auxiliary=False):
        batch = question.shape[0]
        soft = self.soft_prompt.unsqueeze(0).expand(batch, -1, -1)
        # Tahmin edilen maskeli bileşen sonda: nedensel LKM promptları görebilir.
        tail = torch.stack([concepts, response, question] if auxiliary
                           else [question, concepts, response], dim=1)
        tokens = torch.cat([soft, tail], dim=1)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        tokens = self.norm(tokens + self.position(positions))
        # Bu blokta yalnızca hedef soru/konu ve maskeli cevap vardır; gelecek cevap yok.
        return self.encoder(tokens)


class KnowledgeTracer(nn.Module):
    def __init__(self, lkm: LargeKnowledgeModel, method: str):
        super().__init__()
        if method not in {"baseline", "kpt", "kmpt"}:
            raise ValueError(f"Bilinmeyen yöntem: {method}")
        self.lkm, self.method = lkm, method
        size = lkm.config.hidden_size
        self.prompt_encoder = None
        self.classification_head = None
        if method == "baseline":
            self.classification_head = nn.Sequential(nn.Linear(size, size), nn.GELU(),
                                                      nn.Dropout(lkm.config.dropout), nn.Linear(size, 1))
            self.classification_head.apply(initialize)
        else:
            self.prompt_encoder = PromptEncoder(lkm.config)
        # Ana karşılaştırma ortak LKM'nin uyarlanmasıdır; saf PEFT karşılaştırması değildir.
        for parameter in self.lkm.order_head.parameters():
            parameter.requires_grad_(False)
        for parameter in self.lkm.question_head.parameters():
            parameter.requires_grad_(method == "kmpt")
        if method == "baseline":
            for parameter in self.lkm.response_head.parameters():
                parameter.requires_grad_(False)

    def encode_prediction(self, batch, auxiliary=False):
        question, skills, difficulty = (batch[key] for key in
                                        ("target_question", "target_skills", "target_difficulty"))
        history = self.lkm.interaction_embeddings(
            batch["history_question"], batch["history_skills"],
            batch["history_response"], batch["history_difficulty"])
        batch_size, history_width = history.shape[:2]
        if auxiliary:
            if self.method != "kmpt" or not self.training:
                raise ValueError("Yardımcı soru tahmini yalnızca KMPT eğitimi içindir.")
            question = torch.full_like(question, QUESTION_MASK)
            response = batch["label"].long()
        else:
            # label alanı tahmin girdisine hiçbir yoldan alınmaz.
            response = torch.full_like(question, RESPONSE_MASK)
        if self.method == "baseline":
            template = self.lkm.interaction_embeddings(question, skills, response, difficulty).unsqueeze(1)
        else:
            q = self.lkm.question_embedding(question) + self.lkm.difficulty_embedding(difficulty.unsqueeze(-1))
            c = self.lkm.concept_embeddings(skills)
            r = self.lkm.response_embedding(response)
            template = self.prompt_encoder(q, c, r, auxiliary=auxiliary)
        cls = self.lkm.special_embedding.weight[1].expand(batch_size, 1, -1)
        embeddings = torch.cat([cls, history, template], dim=1)
        padding = torch.cat([torch.zeros(batch_size, 1, device=history.device, dtype=torch.bool),
                             batch["history_padding"],
                             torch.zeros(batch_size, template.shape[1], device=history.device, dtype=torch.bool)], dim=1)
        lengths = (~batch["history_padding"]).sum(dim=1)
        # Konumlar dolgu sayısından bağımsız; aynı öğrenci farklı bir grupta aynı tahmini alır.
        positions = torch.cat([
            torch.zeros(batch_size, 1, device=history.device, dtype=torch.long),
            torch.arange(1, history_width+1, device=history.device).expand(batch_size, -1),
            lengths[:, None] + torch.arange(1, template.shape[1]+1, device=history.device)], dim=1)
        return self.lkm.encode(embeddings, padding, positions, causal=True)[:, -1]

    def forward(self, batch):
        hidden = self.encode_prediction(batch, auxiliary=False)
        head = self.classification_head if self.method == "baseline" else self.lkm.response_head
        return head(hidden).squeeze(-1)

    def loss(self, batch, auxiliary=False):
        if auxiliary:
            hidden = self.encode_prediction(batch, auxiliary=True)
            logits = self.lkm.question_head(hidden)
            return F.cross_entropy(logits, batch["target_question"])
        return F.binary_cross_entropy_with_logits(self(batch), batch["label"])

    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)}
