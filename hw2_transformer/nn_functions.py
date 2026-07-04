"""CMU 11-868 Assignment 2, Problem 2 — activation & loss functions.

GELU, logsumexp, one_hot and softmax cross-entropy loss, implemented on NumPy
with matching forward + backward (VJP) so they can be dropped into the transformer
autograd tape in :mod:`hw2_transformer.autograd`.  Each is checked against PyTorch
in ``tests/test_hw2_transformer.py``.
"""

from __future__ import annotations

import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU: x * Phi(x) with the tanh approximation used by GPT-2."""
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))


def gelu_grad(x: np.ndarray) -> np.ndarray:
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x**3)
    tanh_inner = np.tanh(inner)
    dinner = c * (1.0 + 3 * 0.044715 * x**2)
    sech2 = 1.0 - tanh_inner**2
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * dinner


def logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = True) -> np.ndarray:
    """Numerically-stable log-sum-exp reduction."""
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def one_hot(idx: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros(idx.shape + (num_classes,))
    it = np.nditer(idx, flags=["multi_index"])
    while not it.finished:
        out[it.multi_index + (int(it[0]),)] = 1.0
        it.iternext()
    return out


def softmax_cross_entropy(logits: np.ndarray, targets: np.ndarray):
    """Mean softmax cross-entropy.  ``logits``: (N, C), ``targets``: (N,) int.

    Returns ``(loss, dlogits)`` where ``dlogits`` is the gradient w.r.t. logits.
    """
    N = logits.shape[0]
    lse = logsumexp(logits, axis=-1, keepdims=False)  # (N,)
    correct = logits[np.arange(N), targets]
    loss = float(np.mean(lse - correct))
    probs = softmax(logits, axis=-1)
    dlogits = probs.copy()
    dlogits[np.arange(N), targets] -= 1.0
    dlogits /= N
    return loss, dlogits
