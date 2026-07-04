"""KV-cache for autoregressive decoding.

Without a cache, generating token ``t`` re-runs attention over the whole prefix
``0..t`` — O(t^2) work to produce a sequence of length T.  A KV-cache stores the
key/value projections of every past token, so step ``t`` only computes the query
for the *new* token and attends it against the cached keys/values: O(t) work per
step.  This is the mechanism SGLang's RadixAttention reuses across requests.

We implement a minimal causal multi-head attention with an optional cache and
verify that cached incremental decoding produces exactly the same outputs as
full recomputation, while doing far less arithmetic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KVCache:
    """Rolling store of past keys/values, shape (B, n_head, T_so_far, head_dim)."""

    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
        return self.k, self.v

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.size(2)


class CachedAttention(nn.Module):
    """Causal multi-head self-attention supporting an incremental KV-cache."""

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

    def _shape(self, x, B, T):
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, cache: Optional[KVCache] = None) -> torch.Tensor:
        """Attend ``x`` (B, T, C).

        If ``cache`` is given, ``x`` is only the *new* tokens; queries attend over
        the concatenation of cached and new keys/values.  Causality is automatic:
        new queries can see all past positions and each other (lower-triangular
        within the new block).
        """
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = self._shape(q, B, T)
        k = self._shape(k, B, T)
        v = self._shape(v, B, T)

        past_len = 0
        if cache is not None:
            past_len = cache.length
            k, v = cache.append(k, v)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # causal mask over the full (past + new) key axis
        total = past_len + T
        row = torch.arange(past_len, total, device=x.device).unsqueeze(1)  # (T,1)
        col = torch.arange(total, device=x.device).unsqueeze(0)  # (1,total)
        mask = (col <= row).view(1, 1, T, total)
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


def decode_full(attn: CachedAttention, tokens_embed: torch.Tensor) -> torch.Tensor:
    """Baseline: recompute attention over the whole prefix at every step.

    Returns the per-step output (the vector for the last token at each step),
    stacked to (B, T, C).  This is the O(T^2) reference.
    """
    B, T, C = tokens_embed.shape
    outs = []
    for t in range(1, T + 1):
        y = attn(tokens_embed[:, :t, :], cache=None)  # no cache
        outs.append(y[:, -1, :])  # last-token output
    return torch.stack(outs, dim=1)


def decode_cached(attn: CachedAttention, tokens_embed: torch.Tensor) -> torch.Tensor:
    """Incremental decoding with a KV-cache: feed one token at a time."""
    B, T, C = tokens_embed.shape
    cache = KVCache()
    outs = []
    for t in range(T):
        y = attn(tokens_embed[:, t : t + 1, :], cache=cache)  # (B,1,C)
        outs.append(y[:, -1, :])
    return torch.stack(outs, dim=1)


def attention_flops(seq_len: int, n_head: int, head_dim: int, cached: bool) -> int:
    """Approx. score+context MACs to generate ``seq_len`` tokens one at a time.

    Full recompute: step t attends t x t  -> sum_t t^2.
    Cached:         step t attends 1 x t  -> sum_t t.
    (QKV/proj projection cost is identical for both and omitted.)
    """
    macs = 0
    for t in range(1, seq_len + 1):
        keys = t
        queries = t if not cached else 1
        macs += 2 * queries * keys * n_head * head_dim  # scores + context
    return macs
