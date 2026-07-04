"""CMU 11-868 Assignment 2 — decoder-only transformer modules.

Faithful re-implementation of the modules the assignment asks the student to
write (Problems 3 & 4):

* ``Linear``       — y = x W + b        (modules_basic_student.py)
* ``Embedding``    — token/position lookup
* ``LayerNorm1d``  — layer normalisation over the feature dim
* ``Dropout``      — inverted dropout
* ``MultiHeadAttention`` — causal self-attention (modules_transfomer_student.py)
* ``FeedForward``  — Linear -> GELU -> Linear with dropout
* ``TransformerLayer``   — pre-LN residual block
* ``DecoderLM``    — token+pos embed -> N layers -> LN -> LM head

Every module is autograd-aware via :mod:`hw2_transformer.autograd`; the whole
DecoderLM trains end-to-end on a real sequence task in
``hw2_transformer/train_lm.py`` and its forward is checked against a PyTorch
reference in the test-suite.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .autograd import Value, embedding_forward


class Parameter:
    def __init__(self, data: np.ndarray):
        self.value = Value(np.asarray(data, dtype=np.float64), requires_grad=True)

    def zero_grad(self):
        self.value.grad = None


class Module:
    def parameters(self) -> List[Parameter]:
        params: List[Parameter] = []
        for v in vars(self).values():
            if isinstance(v, Parameter):
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


def _linear_init(in_f, out_f, rng):
    bound = 1.0 / np.sqrt(in_f)
    return rng.uniform(-bound, bound, size=(in_f, out_f))


class Linear(Module):
    def __init__(self, in_features, out_features, rng, bias=True):
        self.weight = Parameter(_linear_init(in_features, out_features, rng))
        self.bias = Parameter(np.zeros(out_features)) if bias else None

    def __call__(self, x: Value) -> Value:
        out = x @ self.weight.value
        if self.bias is not None:
            out = out.add_bias(self.bias.value)
        return out


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, rng):
        self.weight = Parameter(rng.standard_normal((num_embeddings, embedding_dim)) * 0.02)

    def __call__(self, idx: np.ndarray) -> Value:
        return embedding_forward(idx, self.weight.value)


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5):
        self.gamma = Parameter(np.ones(dim))
        self.beta = Parameter(np.zeros(dim))
        self.eps = eps

    def __call__(self, x: Value) -> Value:
        return x.layernorm(self.gamma.value, self.beta.value, self.eps)


class Dropout:
    def __init__(self, p, rng):
        self.p = p
        self.rng = rng

    def __call__(self, x: Value, train: bool) -> Value:
        return x.dropout(self.p, train, self.rng)


class MultiHeadAttention(Module):
    """Causal multi-head self-attention (Problem 4.1)."""

    def __init__(self, n_embd, n_head, rng, p_dropout=0.1):
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.q = Linear(n_embd, n_embd, rng, bias=False)
        self.k = Linear(n_embd, n_embd, rng, bias=False)
        self.v = Linear(n_embd, n_embd, rng, bias=False)
        self.proj = Linear(n_embd, n_embd, rng, bias=False)
        self.dropout = Dropout(p_dropout, rng)

    def _split_heads(self, x: Value, B, T):
        # (B, T, n_embd) -> (B, n_head, T, head_dim)
        x = x.reshape(B, T, self.n_head, self.head_dim)
        return x.transpose(1, 2)

    def __call__(self, x: Value, train: bool) -> Value:
        B, T, C = x.data.shape
        q = self._split_heads(self.q(x), B, T)
        k = self._split_heads(self.k(x), B, T)
        v = self._split_heads(self.v(x), B, T)

        scores = (q @ k.transpose(-1, -2)).scale(1.0 / np.sqrt(self.head_dim))
        # causal mask: -inf above the diagonal
        mask = np.triu(np.full((T, T), -1e9), k=1)[None, None, :, :]
        attn = scores.softmax_masked(mask, axis=-1)
        attn = self.dropout(attn, train)
        out = attn @ v  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class FeedForward(Module):
    """Position-wise FFN: Linear -> GELU -> Linear (Problem 4.2)."""

    def __init__(self, n_embd, hidden, rng, p_dropout=0.1):
        self.fc1 = Linear(n_embd, hidden, rng)
        self.fc2 = Linear(hidden, n_embd, rng)
        self.dropout = Dropout(p_dropout, rng)

    def __call__(self, x: Value, train: bool) -> Value:
        h = self.fc1(x).gelu()
        h = self.dropout(h, train)
        return self.fc2(h)


class TransformerLayer(Module):
    """Pre-LN residual block (Problem 4.3)."""

    def __init__(self, n_embd, n_head, hidden, rng, p_dropout=0.1):
        self.ln1 = LayerNorm1d(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, rng, p_dropout)
        self.ln2 = LayerNorm1d(n_embd)
        self.ffn = FeedForward(n_embd, hidden, rng, p_dropout)

    def __call__(self, x: Value, train: bool) -> Value:
        x = x + self.attn(self.ln1(x), train)
        x = x + self.ffn(self.ln2(x), train)
        return x


class DecoderLM(Module):
    """Decoder-only language model (Problem 4.4)."""

    def __init__(
        self,
        vocab_size,
        n_embd=64,
        n_head=4,
        n_layer=2,
        hidden=256,
        max_len=64,
        p_dropout=0.1,
        seed=0,
    ):
        rng = np.random.default_rng(seed)
        self.tok_emb = Embedding(vocab_size, n_embd, rng)
        self.pos_emb = Embedding(max_len, n_embd, rng)
        self.layers = [
            TransformerLayer(n_embd, n_head, hidden, rng, p_dropout) for _ in range(n_layer)
        ]
        self.ln_f = LayerNorm1d(n_embd)
        self.head = Linear(n_embd, vocab_size, rng, bias=False)
        self.max_len = max_len

    def parameters(self):
        params = self.tok_emb.parameters() + self.pos_emb.parameters()
        for layer in self.layers:
            params += layer.parameters()
        params += self.ln_f.parameters() + self.head.parameters()
        return params

    def __call__(self, idx: np.ndarray, train: bool = True) -> Value:
        B, T = idx.shape
        pos = np.arange(T)[None, :].repeat(B, axis=0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x, train)
        x = self.ln_f(x)
        return self.head(x)  # (B, T, vocab)
