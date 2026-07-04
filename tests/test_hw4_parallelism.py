"""Tests for Assignment 5 — data & pipeline parallelism.

The heavy multi-process data-parallel run is exercised by
``hw4_parallelism.run_data_parallel``; here we unit-test the algorithmic pieces
(partitioning, all-reduce math, module split, GPipe schedule, pipeline
equivalence) which run fast in-process.  A single 2-process gloo all-reduce is
also checked so the distributed path itself is covered.
"""

from __future__ import annotations

import os

import pytest
import torch

from hw4_parallelism.data import CharDataset
from hw4_parallelism.data_parallel import DataPartitioner, Partition, partition_dataset
from hw4_parallelism.model import GPTConfig, build_sequential
from hw4_parallelism.pipeline import Pipe, _clock_cycles, _split_module


# --------------------------------------------------------------------------- #
# Data partitioning                                                           #
# --------------------------------------------------------------------------- #
def test_partitioner_disjoint_and_complete():
    data = list(range(100))
    part = DataPartitioner(data, sizes=[0.5, 0.5], seed=1)
    a = set(part.partitions[0])
    b = set(part.partitions[1])
    assert a.isdisjoint(b)
    assert a | b == set(range(100))


def test_partition_view_indexes_underlying():
    data = list(range(10))
    p = Partition(data, [3, 1, 7])
    assert len(p) == 3
    assert [p[i] for i in range(3)] == [3, 1, 7]


def test_partition_dataset_batch_split():
    ds = CharDataset(block_size=16, n_repeat=20)
    loader, per_bs = partition_dataset(ds, world_size=4, rank=0, global_batch_size=32)
    assert per_bs == 8
    xb, yb = next(iter(loader))
    assert xb.shape[0] == 8 and xb.shape[1] == 16


# --------------------------------------------------------------------------- #
# Module split & GPipe schedule                                               #
# --------------------------------------------------------------------------- #
def test_split_module_preserves_layers():
    cfg = GPTConfig(vocab_size=20, block_size=16, n_layer=4, n_embd=32, n_head=4)
    model = build_sequential(cfg)  # Embed + 4 blocks + Head = 6 layers
    stages = _split_module(model, 3)
    assert len(stages) == 3
    flat = [m for st in stages for m in st.children()]
    orig = list(model.children())
    assert len(flat) == len(orig)
    assert all(a is b for a, b in zip(flat, orig))


def test_clock_cycles_cover_all_pairs_once():
    n_micro, n_stages = 4, 3
    ticks = list(_clock_cycles(n_micro, n_stages))
    assert len(ticks) == n_micro + n_stages - 1
    seen = [pair for tick in ticks for pair in tick]
    assert len(seen) == len(set(seen)) == n_micro * n_stages
    # every pair present
    assert set(seen) == {(m, s) for m in range(n_micro) for s in range(n_stages)}


def test_pipeline_forward_matches_unpipelined():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=20, block_size=16, n_layer=4, n_embd=32, n_head=4)
    model = build_sequential(cfg).eval()
    pipe = Pipe(build_sequential(cfg), n_partitions=3, n_micro=4)
    # sync weights: copy split stages into the pipe
    stages = _split_module(model, 3)
    state = {}
    for si, st in enumerate(stages):
        for k, v in st.state_dict().items():
            state[f"partitions.{si}.{k}"] = v
    pipe.load_state_dict(state)
    pipe.eval()
    x = torch.randint(0, 20, (8, 16))
    with torch.no_grad():
        ref = model(x)
        got = pipe(x)
    assert torch.allclose(ref, got, atol=1e-5)


def test_pipeline_backward_flows():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=20, block_size=16, n_layer=3, n_embd=32, n_head=4)
    pipe = Pipe(build_sequential(cfg), n_partitions=2, n_micro=2)
    x = torch.randint(0, 20, (4, 16))
    out = pipe(x)
    out.sum().backward()
    grads = [p.grad for p in pipe.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


# --------------------------------------------------------------------------- #
# Real 2-process gloo all-reduce                                              #
# --------------------------------------------------------------------------- #
def _allreduce_worker(rank, world, q):
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29566"
    dist.init_process_group("gloo", rank=rank, world_size=world)
    t = torch.tensor([float(rank + 1)])
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= world
    if rank == 0:
        q.put(t.item())
    dist.destroy_process_group()


def test_gloo_all_reduce_average():
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_allreduce_worker, args=(r, 2, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
    val = q.get(timeout=5)
    # average of [1, 2] = 1.5
    assert abs(val - 1.5) < 1e-6
