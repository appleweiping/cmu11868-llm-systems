"""Weight-only symmetric quantization (int8 / int4).

Large-model serving stores weights in low precision to cut memory and bandwidth.
We implement per-row (per-output-channel) *symmetric* quantization:

    scale = max(|W_row|) / qmax
    W_q   = round(W_row / scale)   clamped to [-qmax, qmax]
    W_hat = W_q * scale            (de-quantised approximation)

A ``QuantizedLinear`` stores int weights + per-row float scales and de-quantises
on the fly, giving the real ``float_bits / n_bits`` memory reduction with a small,
measurable accuracy cost — the core idea behind int8/int4 LLM inference.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def quantize_symmetric(w: torch.Tensor, n_bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric quantization of a 2-D weight ``w`` (out, in).

    Returns ``(q, scale)`` where ``q`` is int (dtype int8 for n_bits<=8) in
    ``[-qmax, qmax]`` and ``scale`` is (out, 1).
    """
    assert w.dim() == 2
    qmax = 2 ** (n_bits - 1) - 1
    max_abs = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = max_abs / qmax
    q = torch.round(w / scale).clamp(-qmax, qmax)
    q = q.to(torch.int8) if n_bits <= 8 else q.to(torch.int16)
    return q, scale


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Reconstruct the float weight from int codes and per-row scales."""
    return q.to(scale.dtype) * scale


class QuantizedLinear(nn.Module):
    """A Linear whose weight is stored quantized and de-quantized per forward.

    Drop-in for ``nn.Linear`` at inference: build from an existing float linear
    with :meth:`from_float`.
    """

    def __init__(self, q: torch.Tensor, scale: torch.Tensor, bias, n_bits: int):
        super().__init__()
        self.register_buffer("q", q)
        self.register_buffer("scale", scale)
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)
        self.n_bits = n_bits

    @classmethod
    def from_float(cls, linear: nn.Linear, n_bits: int = 8) -> "QuantizedLinear":
        q, scale = quantize_symmetric(linear.weight.data, n_bits)
        bias = None if linear.bias is None else linear.bias.data.clone()
        return cls(q, scale, bias, n_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = dequantize(self.q, self.scale)
        return torch.nn.functional.linear(x, w, self.bias)

    def memory_bytes(self) -> int:
        w_bytes = self.q.numel() * (1 if self.n_bits <= 8 else 2)
        s_bytes = self.scale.numel() * 4
        b_bytes = 0 if self.bias is None else self.bias.numel() * 4
        return w_bytes + s_bytes + b_bytes


def reconstruction_error(w: torch.Tensor, n_bits: int = 8) -> dict:
    """Quantize->dequantize ``w`` and report error + compression stats."""
    q, scale = quantize_symmetric(w, n_bits)
    w_hat = dequantize(q, scale)
    err = (w - w_hat)
    rel = err.norm() / w.norm()
    float_bytes = w.numel() * 4
    # theoretical packed storage: n_bits per weight (e.g. int4 packs 2/byte)
    packed_bytes = w.numel() * n_bits / 8 + scale.numel() * 4
    return {
        "n_bits": n_bits,
        "max_abs_err": err.abs().max().item(),
        "rel_frobenius_err": rel.item(),
        "float32_bytes": float_bytes,
        "packed_bytes": packed_bytes,
        "packed_compression_x": float_bytes / packed_bytes,
    }
