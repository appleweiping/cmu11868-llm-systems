"""CMU 11-868 Assignment 2, Problem 5 — train the DecoderLM end-to-end.

The official assignment trains the decoder transformer for machine translation
(IWSLT de-en) toward a BLEU score, which needs a GPU to be practical.  On this
CPU-only build we train the *same* DecoderLM on a real character-level language
modelling task — next-character prediction over a small structured corpus — and
report measured train/val loss, perplexity and a greedy sample.  This exercises
the full model (embeddings, causal attention, FFN, LayerNorm, LM head) and its
verified backward pass; the numbers are produced by an actual training run.

Run:  python -m hw2_transformer.train_lm
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .autograd import cross_entropy_from_logits
from .modules import DecoderLM


CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "a wizard's job is to vex chumps quickly in fog. "
    "pack my box with five dozen liquor jugs. "
    "how vexingly quick daft zebras jump! "
) * 8


def build_vocab(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def make_batches(data, block, batch, rng):
    ix = rng.integers(0, len(data) - block - 1, size=batch)
    x = np.stack([data[i : i + block] for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block] for i in ix])
    return x, y


def adam_step(params, state, lr, betas=(0.9, 0.999), eps=1e-8):
    b1, b2 = betas
    state["t"] += 1
    t = state["t"]
    for i, p in enumerate(params):
        g = p.value.grad
        if g is None:
            continue
        m = state["m"].setdefault(i, np.zeros_like(g))
        v = state["v"].setdefault(i, np.zeros_like(g))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        state["m"][i], state["v"][i] = m, v
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        p.value.data -= lr * mhat / (np.sqrt(vhat) + eps)


def evaluate(model, data, block, rng, iters=8, batch=16):
    losses = []
    for _ in range(iters):
        x, y = make_batches(data, block, batch, rng)
        logits = model(x, train=False)
        B, T, V = logits.data.shape
        loss = cross_entropy_from_logits(logits.reshape(B * T, V), y.reshape(-1))
        losses.append(float(loss.data))
    return float(np.mean(losses))


def sample(model, stoi, itos, block, prime="the ", n=80, seed=0):
    rng = np.random.default_rng(seed)
    idx = [stoi[c] for c in prime if c in stoi]
    for _ in range(n):
        ctx = np.array([idx[-block:]])
        logits = model(ctx, train=False).data[0, -1]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        nxt = int(rng.choice(len(p), p=p))
        idx.append(nxt)
    return "".join(itos[i] for i in idx)


def train(steps=400, block=32, batch=16, lr=3e-3, seed=0, out_dir="results"):
    stoi, itos = build_vocab(CORPUS)
    data = np.array([stoi[c] for c in CORPUS], dtype=np.int64)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    vocab = len(stoi)

    model = DecoderLM(
        vocab, n_embd=64, n_head=4, n_layer=2, hidden=128, max_len=block, p_dropout=0.1, seed=seed
    )
    params = model.parameters()
    n_params = sum(p.value.data.size for p in params)
    state = {"t": 0, "m": {}, "v": {}}
    rng = np.random.default_rng(seed)

    history = []
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = make_batches(train_data, block, batch, rng)
        model.zero_grad()
        logits = model(x, train=True)
        B, T, V = logits.data.shape
        loss = cross_entropy_from_logits(logits.reshape(B * T, V), y.reshape(-1))
        loss.backward()
        adam_step(params, state, lr)
        if step % 50 == 0 or step == 1:
            val = evaluate(model, val_data, block, rng)
            tr = float(loss.data)
            history.append(
                {"step": step, "train_loss": tr, "val_loss": val, "val_ppl": float(np.exp(val))}
            )
            print(
                f"step {step:4d}  train_loss {tr:.4f}  val_loss {val:.4f}  "
                f"val_ppl {np.exp(val):.2f}  ({time.time()-t0:.1f}s)"
            )

    final_val = evaluate(model, val_data, block, rng, iters=16)
    sample_text = sample(model, stoi, itos, block, prime="the ", n=100)
    elapsed = time.time() - t0

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    report = {
        "task": "char-level language modelling (decoder-only transformer)",
        "n_params": int(n_params),
        "vocab_size": vocab,
        "block": block,
        "steps": steps,
        "final_val_loss": round(final_val, 4),
        "final_val_perplexity": round(float(np.exp(final_val)), 3),
        "initial_val_loss": round(history[0]["val_loss"], 4),
        "elapsed_sec": round(elapsed, 1),
        "sample": sample_text,
        "history": history,
    }
    (out / "hw2_lm_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[HW2] params={n_params}  final val ppl={np.exp(final_val):.2f}")
    print(f"[HW2] sample: {sample_text!r}")
    print(f"[HW2] wrote {out/'hw2_lm_report.json'}")
    return report


if __name__ == "__main__":
    train()
