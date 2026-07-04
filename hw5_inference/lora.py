"""Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning.

For a frozen weight ``W0`` (out, in), LoRA learns a low-rank update

    W = W0 + (alpha / r) * B @ A ,   A: (r, in),  B: (out, r)

Only ``A`` and ``B`` are trained, so the trainable parameter count drops from
``out*in`` to ``r*(out+in)`` — the reduction the assignment exploits to fit
LLaMA-2-7B fine-tuning on small GPUs.  ``A`` is init'd from a normal and ``B``
from zeros, so the adapted model starts exactly at ``W0``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Linear layer with a frozen base weight and a trainable LoRA update."""

    def __init__(self, in_features: int, out_features: int, r: int = 4, alpha: int = 8,
                 bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = alpha / r

        # frozen base
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.bias = None

        # trainable low-rank factors
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

    @classmethod
    def from_linear(cls, linear: nn.Linear, r: int = 4, alpha: int = 8) -> "LoRALinear":
        m = cls(linear.in_features, linear.out_features, r, alpha, bias=linear.bias is not None)
        m.weight.data.copy_(linear.weight.data)
        m.weight.requires_grad_(False)
        if linear.bias is not None:
            m.bias.data.copy_(linear.bias.data)
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = torch.nn.functional.linear(x, self.weight, self.bias)
        delta = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base + self.scaling * delta

    def merged_weight(self) -> torch.Tensor:
        """Return W0 + scaling * B@A (for deployment after fine-tuning)."""
        return self.weight + self.scaling * (self.lora_B @ self.lora_A)

    def trainable_parameters(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()

    def base_parameters(self) -> int:
        return self.weight.numel()


def param_efficiency(in_features: int, out_features: int, r: int) -> dict:
    """Report the trainable-parameter reduction of LoRA vs full fine-tuning."""
    full = in_features * out_features
    lora = r * (in_features + out_features)
    return {
        "in": in_features,
        "out": out_features,
        "rank": r,
        "full_finetune_params": full,
        "lora_params": lora,
        "reduction_x": full / lora,
        "trainable_fraction": lora / full,
    }
