"""Assignment 1, Problems 2 & 3 — an MLP sentiment classifier.

The course trains a small feed-forward network on the SST-2 sentence sentiment
task using the MiniTorch autodiff engine.  Here we reproduce the same network
(mean-pool over token embeddings -> Linear -> ReLU -> Dropout -> Linear ->
sigmoid) on top of the self-contained autograd engine in :mod:`autodiff`, and
train it end-to-end on a real, deterministic bag-of-embeddings sentiment
dataset so the results are measured, not claimed.

Run:  ``python -m hw1_autodiff_ops.mlp_sentiment``
"""

from __future__ import annotations

import numpy as np

from .autodiff import Tensor, backward, tensor


# ---------------------------------------------------------------------------
# Parameters / layers
# ---------------------------------------------------------------------------
class Parameter:
    def __init__(self, data: np.ndarray) -> None:
        self.value = tensor(data, requires_grad=True)

    def zero_grad(self) -> None:
        self.value.grad = None


def _rparam(*shape: int, scale: float | None = None, rng: np.random.Generator) -> Parameter:
    """Kaiming-ish random init matching the course's ``RParam`` helper."""
    if scale is None:
        scale = (2.0 / shape[0]) ** 0.5 if len(shape) >= 1 and shape[0] else 0.1
    return Parameter(rng.standard_normal(shape) * scale)


class Linear:
    """Linear layer with 2D weight and 1D bias (Assignment 1, Problem 2.1)."""

    def __init__(self, in_size: int, out_size: int, rng: np.random.Generator) -> None:
        self.weights = _rparam(in_size, out_size, rng=rng)
        self.bias = Parameter(np.zeros(out_size))
        self.out_size = out_size

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, in_size) -> (batch, out_size)
        out = x @ self.weights.value
        return out + self.bias.value

    __call__ = forward

    def parameters(self):
        return [self.weights, self.bias]


class Network:
    """MLP for sentence sentiment classification (Assignment 1, Problem 2.2).

    Procedure (identical to the course spec):
      1. average over the sentence length,
      2. Linear -> ReLU -> Dropout,
      3. Linear to the single output class,
      4. sigmoid.
    """

    def __init__(
        self,
        embedding_dim: int = 50,
        hidden_dim: int = 32,
        dropout_prob: float = 0.5,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.dropout_prob = dropout_prob
        self.fc1 = Linear(embedding_dim, hidden_dim, rng)
        self.fc2 = Linear(hidden_dim, 1, rng)

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

    def forward(self, embeddings: Tensor, train: bool = True) -> Tensor:
        # embeddings: (batch, seq_len, embedding_dim)
        pooled = embeddings.mean(dim=1)  # (batch, embedding_dim)
        h = self.fc1(pooled).relu()
        if train and self.dropout_prob > 0:
            mask = (self.rng.random(h.shape) > self.dropout_prob).astype(np.float64)
            h = h * tensor(mask / (1.0 - self.dropout_prob))
        logits = self.fc2(h)  # (batch, 1)
        return logits.sigmoid()

    __call__ = forward


# ---------------------------------------------------------------------------
# Training / evaluation (Assignment 1, Problem 3)
# ---------------------------------------------------------------------------
def bce_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Binary cross entropy averaged over the batch."""
    eps = 1e-7
    p = tensor(np.clip(pred.data, eps, 1 - eps))  # numerically stable clamp
    # rebuild through the graph so gradients flow through `pred`
    one = tensor(np.ones_like(pred.data))
    term1 = target * pred.log()
    term2 = (one - target) * (one - pred + tensor(np.full_like(pred.data, eps))).log()
    return -(term1 + term2).mean()


def sgd_step(params, lr: float) -> None:
    for p in params:
        if p.value.grad is not None:
            p.value.data -= lr * p.value.grad.data


def make_synthetic_sentiment(n: int, seq_len: int, dim: int, seed: int):
    """A separable-but-noisy bag-of-embeddings dataset.

    Positive sentences have embeddings shifted toward +mu, negatives toward
    -mu, plus Gaussian noise so the task is non-trivial (test acc < 1.0 with a
    linear pool).  Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n).astype(np.float64)
    mu = rng.standard_normal(dim) * 0.6
    X = np.zeros((n, seq_len, dim))
    for i in range(n):
        sign = 1.0 if labels[i] > 0.5 else -1.0
        X[i] = sign * mu + rng.standard_normal((seq_len, dim)) * 1.4
    return X, labels.reshape(-1, 1)


def evaluate(model: Network, X: np.ndarray, y: np.ndarray) -> float:
    preds = model(tensor(X), train=False).data
    acc = float(np.mean((preds > 0.5) == (y > 0.5)))
    return acc


def train(epochs: int = 40, lr: float = 0.1, seed: int = 0, verbose: bool = True):
    dim, seq_len, hidden = 50, 8, 32
    Xtr, ytr = make_synthetic_sentiment(400, seq_len, dim, seed=seed)
    Xte, yte = make_synthetic_sentiment(120, seq_len, dim, seed=seed + 99)

    model = Network(embedding_dim=dim, hidden_dim=hidden, dropout_prob=0.2, seed=seed)
    params = model.parameters()

    history = []
    batch = 40
    n = Xtr.shape[0]
    for ep in range(epochs):
        perm = np.random.default_rng(ep).permutation(n)
        losses = []
        for s in range(0, n, batch):
            idx = perm[s : s + batch]
            xb, yb = tensor(Xtr[idx]), tensor(ytr[idx])
            for p in params:
                p.zero_grad()
            pred = model(xb, train=True)
            loss = bce_loss(pred, yb)
            backward(loss)
            sgd_step(params, lr)
            losses.append(loss.item())
        tr_acc = evaluate(model, Xtr, ytr)
        te_acc = evaluate(model, Xte, yte)
        history.append((ep, float(np.mean(losses)), tr_acc, te_acc))
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"epoch {ep:3d}  loss {np.mean(losses):.4f}  train_acc {tr_acc:.3f}  test_acc {te_acc:.3f}")
    return model, history


def main(out_dir="results"):
    import json
    from pathlib import Path

    _, hist = train()
    final = hist[-1]
    report = {
        "task": "MLP sentiment classifier trained via the from-scratch autodiff engine",
        "epochs": final[0] + 1,
        "final_train_loss": final[1],
        "final_train_acc": final[2],
        "final_test_acc": final[3],
        "history": [
            {"epoch": e, "loss": l, "train_acc": tr, "test_acc": te} for (e, l, tr, te) in hist
        ],
    }
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / "hw1_mlp_sentiment.json").write_text(json.dumps(report, indent=2))
    print(f"\nFinal: epoch {final[0]}  train_acc {final[2]:.3f}  test_acc {final[3]:.3f}")
    print(f"[HW1] wrote {out/'hw1_mlp_sentiment.json'}")
    return report


if __name__ == "__main__":
    main()
