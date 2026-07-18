"""Decoder-only transformer with pluggable positional encoding.

PE conditions (the experiment's independent variable), selected by config
string: {nope, learned, sinusoidal, rope, alibi}.

Two fixed architectures from the proposal (research_proposal.md §2):
  - 1-layer "Nanda-style": no LayerNorm, no attention-projection biases,
    untied embed/unembed, ReLU MLP. For circuit clarity.
  - 2-layer pre-LN variant for robustness checks.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

PE_TYPES = ("nope", "learned", "sinusoidal", "rope", "alibi")


@dataclass(frozen=True)
class ModelConfig:
    vocab: int  # p + 1
    seq_len: int = 3
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 512
    n_layers: int = 1
    pe: str = "learned"
    pre_ln: bool = False  # True only for the 2-layer variant

    def __post_init__(self):
        if self.pe not in PE_TYPES:
            raise ValueError(f"pe must be one of {PE_TYPES}, got {self.pe!r}")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must divide evenly into heads")


def sinusoidal_table(seq_len: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def alibi_slopes(n_heads: int) -> torch.Tensor:
    # Geometric slopes 2^(-8i/n) per Press et al. (ALiBi), for power-of-2 head counts.
    return torch.tensor([2 ** (-8.0 * (i + 1) / n_heads) for i in range(n_heads)])


def rope_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, Dh); cos/sin: (T, Dh/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.d_head = cfg.d_model // cfg.n_heads
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        t = cfg.seq_len
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)
        if cfg.pe == "rope":
            inv = 1.0 / (10000.0 ** (torch.arange(0, self.d_head, 2).float() / self.d_head))
            angles = torch.arange(t).float().unsqueeze(1) * inv.unsqueeze(0)  # (T, Dh/2)
            self.register_buffer("rope_cos", angles.cos(), persistent=False)
            self.register_buffer("rope_sin", angles.sin(), persistent=False)
        if cfg.pe == "alibi":
            pos = torch.arange(t)
            dist = pos.unsqueeze(0) - pos.unsqueeze(1)  # key_pos - query_pos (<=0 in causal part)
            bias = alibi_slopes(cfg.n_heads).view(-1, 1, 1) * dist.view(1, t, t)
            self.register_buffer("alibi_bias", bias, persistent=False)

    def forward(self, x: torch.Tensor, uniform_attn: bool = False) -> torch.Tensor:
        # When self.store_attn is set (analysis only), the softmaxed pattern is
        # kept on self.last_attn; never set during training.
        b, t, _ = x.shape
        h, dh = self.cfg.n_heads, self.d_head

        def split(w):
            return w(x).view(b, t, h, dh).transpose(1, 2)  # (B, H, T, Dh)

        q, k, v = split(self.wq), split(self.wk), split(self.wv)
        if self.cfg.pe == "rope":
            q = rope_rotate(q, self.rope_cos, self.rope_sin)
            k = rope_rotate(k, self.rope_cos, self.rope_sin)

        scores = q @ k.transpose(-2, -1) / math.sqrt(dh)  # (B, H, T, T)
        if self.cfg.pe == "alibi":
            scores = scores + self.alibi_bias.unsqueeze(0)
        scores = scores.masked_fill(self.causal_mask[:t, :t], float("-inf"))
        attn = scores.softmax(dim=-1)
        if getattr(self, "store_attn", False):
            self.last_attn = attn.detach()
        if uniform_attn:
            # Zhong et al.-style discriminator: replace attention with the
            # uniform-over-visible-positions pattern.
            uni = (~self.causal_mask[:t, :t]).float()
            attn = (uni / uni.sum(-1, keepdim=True)).expand_as(attn)
        out = (attn @ v).transpose(1, 2).reshape(b, t, h * dh)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.win = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.wout = nn.Linear(cfg.d_mlp, cfg.d_model)

    def forward(self, x):
        return self.wout(F.relu(self.win(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)
        if cfg.pre_ln:
            self.ln1 = nn.LayerNorm(cfg.d_model)
            self.ln2 = nn.LayerNorm(cfg.d_model)

    def forward(self, x, uniform_attn: bool = False):
        if self.cfg.pre_ln:
            x = x + self.attn(self.ln1(x), uniform_attn=uniform_attn)
            x = x + self.mlp(self.ln2(x))
        else:
            x = x + self.attn(x, uniform_attn=uniform_attn)
            x = x + self.mlp(x)
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        if cfg.pe == "learned":
            self.pos_embed = nn.Parameter(torch.randn(cfg.seq_len, cfg.d_model) * 0.02)
        elif cfg.pe == "sinusoidal":
            self.register_buffer("pos_table", sinusoidal_table(cfg.seq_len, cfg.d_model), persistent=False)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        if cfg.pre_ln:
            self.ln_f = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab - 1, bias=False)  # logits over residues only

    def forward(self, tokens: torch.Tensor, uniform_attn: bool = False) -> torch.Tensor:
        x = self.embed(tokens)
        if self.cfg.pe == "learned":
            x = x + self.pos_embed[: tokens.shape[1]]
        elif self.cfg.pe == "sinusoidal":
            x = x + self.pos_table[: tokens.shape[1]]
        for blk in self.blocks:
            x = blk(x, uniform_attn=uniform_attn)
        if self.cfg.pre_ln:
            x = self.ln_f(x)
        return self.unembed(x)  # (B, T, p) — train.py reads position -1 (the EQ slot)

    def n_params(self) -> int:
        return sum(t.numel() for t in self.parameters())


def build_model(p: int, pe: str, n_layers: int = 1) -> Transformer:
    """The two proposal architectures. Asserts the parameter-count bracket
    (0.2M–1.2M; see decisions.md D8)."""
    cfg = ModelConfig(vocab=p + 1, pe=pe, n_layers=n_layers, pre_ln=(n_layers == 2))
    model = Transformer(cfg)
    n = model.n_params()
    assert 200_000 <= n <= 1_200_000, f"param count {n} outside 0.2M-1.2M bracket"
    return model
