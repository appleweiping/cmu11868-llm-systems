"""Tests for Assignment 6 — KV-cache, quantization, LoRA."""

from __future__ import annotations

import torch
import torch.nn as nn

from hw5_inference.kv_cache import (
    CachedAttention,
    KVCache,
    attention_flops,
    decode_cached,
    decode_full,
)
from hw5_inference.lora import LoRALinear, param_efficiency
from hw5_inference.quantization import (
    QuantizedLinear,
    dequantize,
    quantize_symmetric,
    reconstruction_error,
)


# --------------------------------------------------------------------------- #
# KV-cache                                                                    #
# --------------------------------------------------------------------------- #
def test_kv_cache_matches_full_recompute():
    torch.manual_seed(0)
    attn = CachedAttention(64, 4).eval()
    x = torch.randn(2, 24, 64)
    with torch.no_grad():
        full = decode_full(attn, x)
        cached = decode_cached(attn, x)
    assert torch.allclose(full, cached, atol=1e-5)


def test_kv_cache_length_grows():
    c = KVCache()
    assert c.length == 0
    k = torch.randn(1, 2, 1, 8)
    c.append(k, k)
    c.append(k, k)
    assert c.length == 2


def test_cached_attention_single_step_shape():
    attn = CachedAttention(32, 2).eval()
    cache = KVCache()
    x = torch.randn(1, 1, 32)
    with torch.no_grad():
        y = attn(x, cache=cache)
    assert y.shape == (1, 1, 32)
    assert cache.length == 1


def test_flop_reduction_is_large():
    full = attention_flops(64, 4, 16, cached=False)
    cached = attention_flops(64, 4, 16, cached=True)
    assert full > cached
    assert full / cached > 10  # quadratic vs linear over 64 steps


# --------------------------------------------------------------------------- #
# Quantization                                                                #
# --------------------------------------------------------------------------- #
def test_quant_dequant_roundtrip_small_error():
    torch.manual_seed(0)
    w = torch.randn(128, 128)
    q, scale = quantize_symmetric(w, 8)
    w_hat = dequantize(q, scale)
    rel = (w - w_hat).norm() / w.norm()
    assert rel < 0.02  # int8 per-row is accurate
    assert q.dtype == torch.int8
    assert q.abs().max() <= 127


def test_int4_worse_than_int8():
    torch.manual_seed(0)
    w = torch.randn(256, 256)
    e8 = reconstruction_error(w, 8)["rel_frobenius_err"]
    e4 = reconstruction_error(w, 4)["rel_frobenius_err"]
    assert e4 > e8
    assert reconstruction_error(w, 4)["packed_compression_x"] > 7.0


def test_quantized_linear_close_to_float():
    torch.manual_seed(0)
    lin = nn.Linear(256, 256)
    qlin = QuantizedLinear.from_float(lin, 8)
    x = torch.randn(16, 256)
    with torch.no_grad():
        ref = lin(x)
        got = qlin(x)
    rel = (ref - got).norm() / ref.norm()
    assert rel < 0.02
    assert qlin.memory_bytes() < lin.weight.numel() * 4  # smaller than float


# --------------------------------------------------------------------------- #
# LoRA                                                                        #
# --------------------------------------------------------------------------- #
def test_lora_starts_as_identity_of_base():
    torch.manual_seed(0)
    base = nn.Linear(64, 64)
    lora = LoRALinear.from_linear(base, r=4, alpha=8)
    x = torch.randn(8, 64)
    # B is zero-init -> LoRA output equals the frozen base output at init
    with torch.no_grad():
        assert torch.allclose(lora(x), base(x), atol=1e-6)


def test_lora_only_factors_trainable():
    lora = LoRALinear(128, 128, r=8, alpha=16)
    trainable = {name for name, p in lora.named_parameters() if p.requires_grad}
    assert trainable == {"lora_A", "lora_B"}
    assert not lora.weight.requires_grad


def test_lora_param_reduction():
    eff = param_efficiency(4096, 4096, r=8)
    assert eff["reduction_x"] > 100  # r=8 vs 4096x4096
    assert eff["lora_params"] == 8 * (4096 + 4096)


def test_lora_finetune_reduces_loss():
    torch.manual_seed(0)
    base = nn.Linear(64, 64)
    target = nn.Linear(64, 64)
    for p in target.parameters():
        p.requires_grad_(False)
    lora = LoRALinear.from_linear(base, r=8, alpha=16)
    opt = torch.optim.Adam([p for p in lora.parameters() if p.requires_grad], lr=1e-2)
    X = torch.randn(64, 64)
    Y = target(X).detach()
    first = last = None
    for i in range(150):
        opt.zero_grad()
        loss = ((lora(X) - Y) ** 2).mean()
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    assert last < first
    assert torch.allclose(lora.weight.data, base.weight.data)  # base stayed frozen
