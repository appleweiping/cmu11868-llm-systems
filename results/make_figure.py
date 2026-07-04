"""Render a one-page summary figure of the measured results (results/summary.png).

Reads the JSON evidence produced by the assignment runners and draws four
panels: the HW2 LM training curve, HW3 fused-vs-torch latency, HW6 quantization
error-vs-compression, and HW6 KV-cache FLOP reduction.  No fabricated data — the
figure is a view onto results/*.json.

Run:  python -m results.make_figure   (from the repo root)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE / name).read_text())


def main():
    hw2 = load("hw2_lm_report.json")
    hw3 = load("hw3_kernel_benchmark.json")
    hw5 = load("hw5_inference.json")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # HW2: LM training curve
    ax = axes[0, 0]
    steps = [h["step"] for h in hw2["history"]]
    ax.plot(steps, [h["train_loss"] for h in hw2["history"]], "-o", ms=3, label="train")
    ax.plot(steps, [h["val_loss"] for h in hw2["history"]], "-s", ms=3, label="val")
    ax.set_title(f"HW3 (A3): char-LM training\nval loss {hw2['initial_val_loss']:.2f} -> "
                 f"{hw2['final_val_loss']:.2f}")
    ax.set_xlabel("step"); ax.set_ylabel("cross-entropy"); ax.legend(); ax.grid(alpha=0.3)

    # HW3: fused vs torch softmax latency
    ax = axes[0, 1]
    labels = [f"{s['shape'][0]}x{s['shape'][1]}x{s['shape'][2]}" for s in hw3["softmax"]]
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], [s["fused_ms"] for s in hw3["softmax"]], 0.4, label="fused ref")
    ax.bar([i + 0.2 for i in x], [s["torch_ms"] for s in hw3["softmax"]], 0.4, label="torch")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=20, fontsize=8)
    ax.set_title("HW4 (A4): fused softmax vs torch (CPU)\nall outputs correct within tol")
    ax.set_ylabel("ms/iter"); ax.legend(); ax.grid(alpha=0.3, axis="y")

    # HW5 quantization: error vs compression
    ax = axes[1, 0]
    for k in ("int8", "int4"):
        q = hw5["quantization"][k]
        ax.scatter(q["packed_compression_x"], q["rel_frobenius_err"] * 100, s=90)
        ax.annotate(k, (q["packed_compression_x"], q["rel_frobenius_err"] * 100),
                    textcoords="offset points", xytext=(6, 4))
    ax.set_title("HW6 (A6): weight quantization\nerror vs packed compression")
    ax.set_xlabel("compression x"); ax.set_ylabel("rel. weight error (%)"); ax.grid(alpha=0.3)

    # HW5 KV-cache: FLOPs full vs cached
    ax = axes[1, 1]
    kv = hw5["kv_cache"]
    ax.bar(["full recompute", "KV-cache"], [kv["flops_full"], kv["flops_cached"]],
           color=["#c44", "#4a4"])
    ax.set_title(f"HW6 (A6): KV-cache attention FLOPs\n{kv['flop_reduction_x']:.0f}x reduction "
                 f"(seq={kv['seq_len']}), bit-exact")
    ax.set_ylabel("MACs (relative)"); ax.grid(alpha=0.3, axis="y")

    fig.suptitle("CMU 11-868 LLM Systems — measured results (CPU)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = HERE / "summary.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
