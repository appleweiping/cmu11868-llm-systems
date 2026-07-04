"""Run and measure the Assignment 6 inference/efficient-training techniques.

Produces ``results/hw5_inference.json`` with real numbers for:
  1. KV-cache: correctness vs full recompute + measured FLOP reduction & latency.
  2. Quantization: int8/int4 reconstruction error, memory compression, and the
     end-to-end error on a real linear layer's outputs.
  3. LoRA: trainable-parameter reduction + a real fine-tuning run reducing loss.

Run:  python -m hw5_inference.run_inference
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kv_cache import CachedAttention, attention_flops, decode_cached, decode_full
from .lora import LoRALinear, param_efficiency
from .quantization import QuantizedLinear, reconstruction_error


def bench_kv_cache() -> dict:
    torch.manual_seed(0)
    n_embd, n_head, T, B = 128, 4, 64, 2
    attn = CachedAttention(n_embd, n_head).eval()
    x = torch.randn(B, T, n_embd)

    with torch.no_grad():
        full = decode_full(attn, x)
        cached = decode_cached(attn, x)
    max_err = (full - cached).abs().max().item()

    def _time(fn, iters=10):
        with torch.no_grad():
            fn()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
        return (time.perf_counter() - t0) / iters * 1e3

    full_ms = _time(lambda: decode_full(attn, x))
    cached_ms = _time(lambda: decode_cached(attn, x))
    hd = n_embd // n_head
    return {
        "seq_len": T,
        "matches_full_recompute": bool(torch.allclose(full, cached, atol=1e-5)),
        "max_abs_err": max_err,
        "full_recompute_ms": round(full_ms, 3),
        "cached_ms": round(cached_ms, 3),
        "flops_full": attention_flops(T, n_head, hd, cached=False),
        "flops_cached": attention_flops(T, n_head, hd, cached=True),
        "flop_reduction_x": attention_flops(T, n_head, hd, False)
        / attention_flops(T, n_head, hd, True),
    }


def bench_quantization() -> dict:
    torch.manual_seed(0)
    lin = nn.Linear(512, 512)
    x = torch.randn(64, 512)
    with torch.no_grad():
        ref = lin(x)

    results = {}
    for n_bits in (8, 4):
        rec = reconstruction_error(lin.weight.data, n_bits)
        qlin = QuantizedLinear.from_float(lin, n_bits)
        with torch.no_grad():
            out = qlin(x)
        out_rel = ((out - ref).norm() / ref.norm()).item()
        rec["output_rel_err"] = out_rel
        # actual stored bytes in this (unpacked) QuantizedLinear implementation
        rec["linear_stored_bytes_unpacked"] = qlin.memory_bytes()
        results[f"int{n_bits}"] = rec
    return results


def bench_lora() -> dict:
    torch.manual_seed(0)
    in_f, out_f, r = 256, 256, 8
    eff = param_efficiency(in_f, out_f, r)

    # real fine-tuning: fit a fixed random target linear map using ONLY the LoRA
    # factors on top of a frozen base — loss must drop.
    base = nn.Linear(in_f, out_f)
    target = nn.Linear(in_f, out_f)
    for p in target.parameters():
        p.requires_grad_(False)

    lora = LoRALinear.from_linear(base, r=r, alpha=16)
    trainable = [p for p in lora.parameters() if p.requires_grad]
    frozen = [p for p in lora.parameters() if not p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=1e-2)

    X = torch.randn(128, in_f)
    with torch.no_grad():
        Y = target(X)

    losses = []
    for _ in range(200):
        opt.zero_grad()
        pred = lora(X)
        loss = F.mse_loss(pred, Y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # confirm the frozen base weight never changed
    base_unchanged = torch.allclose(lora.weight.data, base.weight.data)
    eff.update(
        {
            "num_trainable_tensors": len(trainable),
            "num_frozen_tensors": len(frozen),
            "trainable_params_counted": sum(p.numel() for p in trainable),
            "frozen_params_counted": sum(p.numel() for p in frozen),
            "finetune_initial_loss": losses[0],
            "finetune_final_loss": losses[-1],
            "loss_reduced": bool(losses[-1] < losses[0]),
            "base_weight_unchanged": bool(base_unchanged),
        }
    )
    return eff


def main(out_dir="results"):
    torch.set_num_threads(3)
    kv = bench_kv_cache()
    quant = bench_quantization()
    lora = bench_lora()

    report = {
        "device": "cpu",
        "note": "CPU-scale verification of the systems techniques behind the GPU-only "
        "assignment (DeepSpeed-ZeRO+LoRA training, SGLang/RadixAttention serving). "
        "7B-scale ZeRO and the SGLang server throughput are documented as partial.",
        "kv_cache": kv,
        "quantization": quant,
        "lora": lora,
    }
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / "hw5_inference.json").write_text(json.dumps(report, indent=2))

    print("== KV-cache ==")
    print(f"  matches full recompute: {kv['matches_full_recompute']} (max_err {kv['max_abs_err']:.2e})")
    print(f"  FLOP reduction: {kv['flop_reduction_x']:.2f}x  "
          f"latency {kv['full_recompute_ms']:.2f} -> {kv['cached_ms']:.2f} ms")
    print("== Quantization ==")
    for k, v in quant.items():
        print(f"  {k}: rel_err {v['rel_frobenius_err']:.4f}, out_rel {v['output_rel_err']:.4f}, "
              f"packed compression {v['packed_compression_x']:.2f}x")
    print("== LoRA ==")
    print(f"  param reduction: {lora['reduction_x']:.1f}x "
          f"({lora['lora_params']} vs {lora['full_finetune_params']} trainable)")
    print(f"  fine-tune loss {lora['finetune_initial_loss']:.4f} -> {lora['finetune_final_loss']:.4f}, "
          f"base frozen: {lora['base_weight_unchanged']}")
    print(f"[HW6] wrote {out/'hw5_inference.json'}")
    return report


if __name__ == "__main__":
    main()
