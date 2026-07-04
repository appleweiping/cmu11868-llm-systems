"""CMU 11-868 Assignment 3 — fused attention-softmax and layernorm kernels.

The assignment asks the student to write hand-tuned CUDA kernels for the two
operators that dominate a transformer's runtime:

* ``launch_attn_softmax`` / ``launch_attn_softmax_bw`` — a *fused* softmax over
  the attention scores that applies the additive mask, does the max-subtraction,
  exp and normalisation in a single kernel (``src/softmax_kernel.cu``), plus its
  backward pass.
* ``launch_layernorm`` / ``launch_layernorm_bw`` — a fused LayerNorm computing
  mean/variance, normalisation and the affine transform in one kernel
  (``src/layernorm_kernel.cu``), plus its backward.

The CUDA kernels (shipped under ``hw3_fused_kernels/cuda/``) need an NVIDIA GPU.
This module is the numerically-identical CPU reference — the same fused math the
kernels implement — which is exactly what the kernel unit tests
(``kernel_tests/test_softmax_fw.py`` etc.) check the GPU output against, to the
tolerances used by the course (atol/rtol 1e-3).  We verify this reference
against PyTorch and benchmark it in ``hw3_fused_kernels/benchmark.py``.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Fused attention softmax
# ---------------------------------------------------------------------------
def attn_softmax_fw(inp: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fused softmax over the last dim of attention scores with additive mask.

    ``inp``  : (batch, nhead, from_len, to_len) attention logits.
    ``mask`` : broadcastable additive mask (already scaled by -1e8 for padding).

    Matches the kernel: for each (batch, head, query) row, compute
    ``max``, subtract, ``exp``, and divide by the row sum — in one pass.
    """
    x = inp + mask
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def attn_softmax_bw(out_grad: np.ndarray, soft_out: np.ndarray) -> np.ndarray:
    """Backward of the fused softmax.

    Given the upstream gradient ``out_grad`` and the softmax output
    ``soft_out`` (both (b, h, q, k)), the row-local Jacobian gives

        dx = soft_out * (out_grad - sum(out_grad * soft_out, axis=-1)).

    This is exactly what ``ker_attn_softmax_bw`` computes per row.
    """
    s = np.sum(out_grad * soft_out, axis=-1, keepdims=True)
    return soft_out * (out_grad - s)


# ---------------------------------------------------------------------------
# Fused LayerNorm
# ---------------------------------------------------------------------------
def layernorm_fw(inp: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
    """Fused LayerNorm over the last dim, returning (out, mean, rstd, xhat).

    The kernel computes per-row mean & variance in a single reduction, then the
    normalised + affine output.  We return the saved stats so the backward pass
    can reuse them (as the kernel stashes ``mean`` and ``1/std`` for its bw).
    """
    mu = inp.mean(axis=-1, keepdims=True)
    var = inp.var(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = (inp - mu) * rstd
    out = xhat * gamma + beta
    return out, mu, rstd, xhat


def layernorm_bw(out_grad, inp, gamma, mu, rstd, xhat):
    """Backward of the fused LayerNorm.

    Returns ``(dinp, dgamma, dbeta)``.  ``dgamma``/``dbeta`` are summed over all
    rows (the kernel's cross-block reduction); ``dinp`` uses the standard
    LayerNorm VJP with the saved ``mean`` and ``rstd``.
    """
    D = inp.shape[-1]
    reduce_axes = tuple(range(out_grad.ndim - 1))
    dgamma = np.sum(out_grad * xhat, axis=reduce_axes)
    dbeta = np.sum(out_grad, axis=reduce_axes)

    dxhat = out_grad * gamma
    # dinp = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
    mean_dxhat = np.mean(dxhat, axis=-1, keepdims=True)
    mean_dxhat_xhat = np.mean(dxhat * xhat, axis=-1, keepdims=True)
    dinp = rstd * (dxhat - mean_dxhat - xhat * mean_dxhat_xhat)
    return dinp, dgamma, dbeta
