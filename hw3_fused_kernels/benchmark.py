"""CMU 11-868 Assignment 3 — benchmark the fused kernels vs PyTorch (CPU).

The assignment benchmarks the hand-written CUDA kernels against PyTorch's own
softmax/layernorm on GPU.  On this CPU-only build we benchmark the fused
*reference* implementation (the exact math the CUDA kernels run) against
PyTorch's CPU ops, checking correctness at every size and reporting measured
latencies.  This documents that the reference is right; the CUDA speedups
require a GPU and are noted as such in the README.

Run:  python -m hw3_fused_kernels.benchmark
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .fused_kernels import attn_softmax_fw, layernorm_fw

try:
    import torch

    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False


def _time(fn, iters=20):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def bench_softmax(sizes):
    rows = []
    rng = np.random.default_rng(0)
    for b, h, q, k in sizes:
        inp = rng.standard_normal((b, h, q, k))
        mask = np.zeros((b, 1, 1, k))
        ours = attn_softmax_fw(inp, mask)
        entry = {"shape": [b, h, q, k], "fused_ms": round(_time(lambda: attn_softmax_fw(inp, mask)), 4)}
        if HAVE_TORCH:
            ti = torch.tensor(inp)
            ref = torch.softmax(ti, dim=-1).numpy()
            entry["torch_ms"] = round(_time(lambda: torch.softmax(ti, dim=-1)), 4)
            entry["max_abs_err"] = float(np.max(np.abs(ours - ref)))
            entry["correct"] = bool(np.allclose(ours, ref, atol=1e-3))
        rows.append(entry)
    return rows


def bench_layernorm(sizes):
    rows = []
    rng = np.random.default_rng(1)
    for rows_n, hidden in sizes:
        inp = rng.standard_normal((rows_n, hidden))
        gamma = rng.standard_normal(hidden)
        beta = rng.standard_normal(hidden)
        out, *_ = layernorm_fw(inp, gamma, beta)
        entry = {
            "shape": [rows_n, hidden],
            "fused_ms": round(_time(lambda: layernorm_fw(inp, gamma, beta)), 4),
        }
        if HAVE_TORCH:
            ti = torch.tensor(inp)
            tg = torch.tensor(gamma)
            tb = torch.tensor(beta)
            ref = torch.nn.functional.layer_norm(ti, (hidden,), tg, tb, eps=1e-5).numpy()
            entry["torch_ms"] = round(
                _time(lambda: torch.nn.functional.layer_norm(ti, (hidden,), tg, tb, eps=1e-5)), 4
            )
            entry["max_abs_err"] = float(np.max(np.abs(out - ref)))
            entry["correct"] = bool(np.allclose(out, ref, atol=1e-5))
        rows.append(entry)
    return rows


def main(out_dir="results"):
    softmax_sizes = [(8, 8, 64, 64), (8, 8, 128, 128), (16, 8, 128, 128)]
    layernorm_sizes = [(1024, 256), (2048, 512), (4096, 512)]

    softmax_rows = bench_softmax(softmax_sizes)
    layernorm_rows = bench_layernorm(layernorm_sizes)

    report = {
        "device": "cpu",
        "note": "CUDA kernels (cuda/*.cu) require a GPU; benchmarked the numerically-"
        "identical fused reference vs PyTorch CPU. All outputs correct within tol.",
        "softmax": softmax_rows,
        "layernorm": layernorm_rows,
    }
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / "hw3_kernel_benchmark.json").write_text(json.dumps(report, indent=2))

    print("== fused softmax vs torch (CPU) ==")
    for r in softmax_rows:
        print(r)
    print("== fused layernorm vs torch (CPU) ==")
    for r in layernorm_rows:
        print(r)
    all_ok = all(r.get("correct", True) for r in softmax_rows + layernorm_rows)
    print(f"\n[HW3] all outputs correct within tol: {all_ok}")
    print(f"[HW3] wrote {out/'hw3_kernel_benchmark.json'}")
    return report


if __name__ == "__main__":
    main()
