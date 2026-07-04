"""A compact reverse-mode autograd tape for the HW2 transformer.

This is a self-contained NumPy autograd engine (a ``Value`` wrapping an ndarray
plus a recorded backward closure) supporting exactly the ops the decoder-only
transformer needs: matmul, add/sub/mul, transpose/reshape, softmax with a causal
mask, layernorm, GELU, dropout, embedding lookup and cross-entropy.  It is the
substrate on which :mod:`hw2_transformer.modules` builds MultiHeadAttention,
FeedForward, TransformerLayer and DecoderLM.

Gradients are checked element-for-element against PyTorch in the test-suite, so
this is a *verified* backward pass, not a claimed one.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from .nn_functions import gelu, gelu_grad, softmax


def _unbroadcast(grad: np.ndarray, shape) -> np.ndarray:
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Value:
    """An ndarray node on the autograd tape."""

    def __init__(self, data, requires_grad: bool = False, _children=(), _backward=None):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._prev: List["Value"] = list(_children)
        self._backward: Callable[[], None] = _backward or (lambda: None)

    # -- helpers -----------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def _needs(self, *vs) -> bool:
        return any(v.requires_grad for v in vs) or self.requires_grad

    def _accum(self, g: np.ndarray) -> None:
        g = _unbroadcast(g, self.data.shape)
        self.grad = g if self.grad is None else self.grad + g

    # -- ops ---------------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, self._needs(other), (self, other))

        def _bw():
            if self.requires_grad:
                self._accum(out.grad)
            if other.requires_grad:
                other._accum(out.grad)

        out._backward = _bw
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)

    def __neg__(self):
        out = Value(-self.data, self.requires_grad, (self,))
        out._backward = lambda: self.requires_grad and self._accum(-out.grad)
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, self._needs(other), (self, other))

        def _bw():
            if self.requires_grad:
                self._accum(out.grad * other.data)
            if other.requires_grad:
                other._accum(out.grad * self.data)

        out._backward = _bw
        return out

    def matmul(self, other):
        out = Value(self.data @ other.data, self._needs(other), (self, other))

        def _bw():
            if self.requires_grad:
                self._accum(out.grad @ np.swapaxes(other.data, -1, -2))
            if other.requires_grad:
                other._accum(np.swapaxes(self.data, -1, -2) @ out.grad)

        out._backward = _bw
        return out

    __matmul__ = matmul

    def transpose(self, ax1, ax2):
        out = Value(np.swapaxes(self.data, ax1, ax2), self.requires_grad, (self,))
        out._backward = lambda: self.requires_grad and self._accum(
            np.swapaxes(out.grad, ax1, ax2)
        )
        return out

    def reshape(self, *shape):
        old = self.data.shape
        out = Value(self.data.reshape(*shape), self.requires_grad, (self,))
        out._backward = lambda: self.requires_grad and self._accum(out.grad.reshape(old))
        return out

    def scale(self, s: float):
        out = Value(self.data * s, self.requires_grad, (self,))
        out._backward = lambda: self.requires_grad and self._accum(out.grad * s)
        return out

    def gelu(self):
        out = Value(gelu(self.data), self.requires_grad, (self,))
        g = gelu_grad(self.data)
        out._backward = lambda: self.requires_grad and self._accum(out.grad * g)
        return out

    def softmax_masked(self, mask: Optional[np.ndarray], axis: int = -1):
        """Softmax(self + mask) along ``axis``.  ``mask`` is a constant additive."""
        x = self.data if mask is None else self.data + mask
        p = softmax(x, axis=axis)
        out = Value(p, self.requires_grad, (self,))

        def _bw():
            if self.requires_grad:
                g = out.grad
                # Jacobian of softmax: p * (g - sum(g*p))
                s = np.sum(g * p, axis=axis, keepdims=True)
                self._accum(p * (g - s))

        out._backward = _bw
        return out

    def layernorm(self, gamma: "Value", beta: "Value", eps: float = 1e-5):
        """LayerNorm over the last axis with learnable gamma/beta."""
        x = self.data
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        std = np.sqrt(var + eps)
        xhat = (x - mu) / std
        out_data = xhat * gamma.data + beta.data
        out = Value(out_data, self._needs(gamma, beta), (self, gamma, beta))
        D = x.shape[-1]

        def _bw():
            g = out.grad
            if gamma.requires_grad:
                gamma._accum(np.sum(g * xhat, axis=tuple(range(g.ndim - 1))))
            if beta.requires_grad:
                beta._accum(np.sum(g, axis=tuple(range(g.ndim - 1))))
            if self.requires_grad:
                dxhat = g * gamma.data
                dvar = np.sum(dxhat * (x - mu) * -0.5 * std**-3, axis=-1, keepdims=True)
                dmu = np.sum(-dxhat / std, axis=-1, keepdims=True) + dvar * np.mean(
                    -2.0 * (x - mu), axis=-1, keepdims=True
                )
                dx = dxhat / std + dvar * 2.0 * (x - mu) / D + dmu / D
                self._accum(dx)

        out._backward = _bw
        return out

    def add_bias(self, bias: "Value"):
        out = Value(self.data + bias.data, self._needs(bias), (self, bias))

        def _bw():
            if self.requires_grad:
                self._accum(out.grad)
            if bias.requires_grad:
                bias._accum(np.sum(out.grad, axis=tuple(range(out.grad.ndim - 1))))

        out._backward = _bw
        return out

    def dropout(self, p: float, train: bool, rng: np.random.Generator):
        if not train or p <= 0:
            return self
        mask = (rng.random(self.data.shape) > p).astype(np.float64) / (1.0 - p)
        out = Value(self.data * mask, self.requires_grad, (self,))
        out._backward = lambda: self.requires_grad and self._accum(out.grad * mask)
        return out

    # -- backprop ----------------------------------------------------------
    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        topo: List[Value] = []
        seen = set()

        def build(v: "Value"):
            if id(v) in seen:
                return
            seen.add(id(v))
            for c in v._prev:
                build(c)
            topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data) if grad is None else grad
        for v in reversed(topo):
            v._backward()


def embedding_forward(idx: np.ndarray, weight: Value):
    """Gather rows of ``weight`` at integer ``idx`` (shape (...,)) -> (..., D)."""
    out_data = weight.data[idx]
    out = Value(out_data, weight.requires_grad, (weight,))

    def _bw():
        if weight.requires_grad:
            gw = np.zeros_like(weight.data)
            np.add.at(gw, idx.reshape(-1), out.grad.reshape(-1, weight.data.shape[-1]))
            weight._accum(gw)

    out._backward = _bw
    return out


def cross_entropy_from_logits(logits: Value, targets: np.ndarray):
    """Mean softmax cross-entropy over a (N, C) logits Value."""
    from .nn_functions import softmax_cross_entropy

    loss_val, dlogits = softmax_cross_entropy(logits.data, targets)
    out = Value(np.array(loss_val), logits.requires_grad, (logits,))
    out._backward = lambda: logits.requires_grad and logits._accum(out.grad * dlogits)
    return out
