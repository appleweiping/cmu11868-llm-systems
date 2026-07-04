"""A small, self-contained character-level dataset for the parallelism runs.

We generate a deterministic corpus from a fixed seed text (no downloads, no
copyrighted data) so the distributed runs are fully reproducible on any machine.
Each example is a ``(block_size)`` context and its next-token targets.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

SEED_TEXT = (
    "the quick brown fox jumps over the lazy dog. "
    "a wizard's job is to vex chumps quickly in fog. "
    "pack my box with five dozen liquor jugs. "
    "how vexingly quick daft zebras jump. "
    "sphinx of black quartz, judge my vow. "
)


def build_corpus(n_repeat: int = 400) -> str:
    return SEED_TEXT * n_repeat


class CharDataset(Dataset):
    """Contiguous (block_size) windows over a character corpus."""

    def __init__(self, block_size: int = 64, n_repeat: int = 400):
        text = build_corpus(n_repeat)
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.block_size = block_size
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.data = data

    def __len__(self) -> int:
        return len(self.data) - self.block_size - 1

    def __getitem__(self, i):
        chunk = self.data[i : i + self.block_size + 1]
        return chunk[:-1], chunk[1:]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)
