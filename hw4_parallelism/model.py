"""A small GPT-2-style decoder used by both parallelism problems.

Deliberately tiny (CPU-scale) but a real transformer: token+position
embeddings, ``n_layer`` pre-LN blocks with causal multi-head attention and a
GELU MLP, final LayerNorm and a tied-free LM head.  The blocks are exposed as an
``nn.Sequential`` so the pipeline-parallel code can split them across stages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Embed(nn.Module):
    """Token + position embedding — the first stage of the sequential model."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        return self.drop(self.tok(idx) + self.pos(pos))


class Head(nn.Module):
    """Final LayerNorm + LM head — the last stage of the sequential model."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.n_embd)
        self.lm = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, x):
        return self.lm(self.ln(x))


def build_sequential(cfg: GPTConfig) -> nn.Sequential:
    """The whole model as an ``nn.Sequential`` (Embed, *Blocks, Head).

    Pipeline parallelism partitions exactly this sequence across stages.
    """
    layers = [Embed(cfg)] + [Block(cfg) for _ in range(cfg.n_layer)] + [Head(cfg)]
    return nn.Sequential(*layers)


class GPT(nn.Module):
    """Convenience wrapper for data-parallel training (non-pipelined)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.net = build_sequential(cfg)

    def forward(self, idx, targets=None):
        logits = self.net(idx)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss
