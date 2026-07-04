"""Assignment 5, Problem 1 — Data parallelism.

Each worker holds a full replica of the model and a disjoint shard of the
training data.  After every backward pass the workers average their gradients
with an all-reduce so all replicas take the identical optimiser step — exactly
the synchronous-SGD scheme the assignment implements with ``torch.distributed``.

Faithful to the spec, this file provides:

* ``Partition`` / ``DataPartitioner`` — random, reproducible sharding of a
  dataset across ``world_size`` workers.
* ``partition_dataset`` — returns this rank's ``DataLoader`` and per-device
  batch size.
* ``average_gradients`` — all-reduce (SUM) then divide by ``world_size``.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset


class Partition(Dataset):
    """A view onto a subset of ``data`` given by ``index``."""

    def __init__(self, data: Dataset, index: Sequence[int]):
        self.data = data
        self.index = list(index)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i):
        return self.data[self.index[i]]


class DataPartitioner:
    """Split a dataset into fractional partitions with a fixed shuffle seed."""

    def __init__(self, data: Dataset, sizes: Sequence[float] = (0.5, 0.5), seed: int = 1234):
        self.data = data
        self.partitions: List[List[int]] = []
        rng = torch.Generator().manual_seed(seed)
        n = len(data)
        indexes = torch.randperm(n, generator=rng).tolist()
        for frac in sizes:
            part_len = int(round(frac * n))
            self.partitions.append(indexes[:part_len])
            indexes = indexes[part_len:]
        # any rounding remainder goes to the last partition
        if indexes:
            self.partitions[-1].extend(indexes)

    def use(self, partition_id: int) -> Partition:
        return Partition(self.data, self.partitions[partition_id])


def partition_dataset(
    dataset: Dataset,
    world_size: int,
    rank: int,
    global_batch_size: int,
    seed: int = 1234,
):
    """Return (loader, per_device_batch_size) for this ``rank``.

    The global batch is divided equally across workers so the *effective*
    batch size matches single-GPU training with ``global_batch_size``.
    """
    sizes = [1.0 / world_size] * world_size
    partitioner = DataPartitioner(dataset, sizes, seed=seed)
    partition = partitioner.use(rank)
    per_device_bs = max(1, global_batch_size // world_size)
    loader = DataLoader(partition, batch_size=per_device_bs, shuffle=True, drop_last=True)
    return loader, per_device_bs


def average_gradients(model: torch.nn.Module) -> None:
    """All-reduce (SUM) every parameter's gradient, then divide by world size.

    This is the synchronisation step of synchronous data-parallel SGD.
    """
    world_size = dist.get_world_size()
    for param in model.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= world_size
