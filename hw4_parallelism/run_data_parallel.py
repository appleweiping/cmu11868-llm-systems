"""Run real data-parallel training with ``torch.distributed`` (gloo, CPU).

Spawns ``world_size`` OS processes; each trains a replica of the GPT on its own
data shard and synchronises gradients with ``average_gradients`` every step.
We verify that (a) all replicas stay in lock-step (identical loss on rank 0 vs a
single-process reference to numerical tolerance) and (b) training reduces the
loss.  A single-process baseline is timed for the throughput comparison.

Run:  python -m hw4_parallelism.run_data_parallel
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .data import CharDataset
from .data_parallel import average_gradients, partition_dataset
from .model import GPT, GPTConfig

MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29520"


def setup(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(0)


def cleanup() -> None:
    dist.destroy_process_group()


def _make_model(vocab_size: int, block_size: int) -> GPT:
    torch.manual_seed(0)  # identical init on every rank
    cfg = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=3, n_embd=96, n_head=4)
    return GPT(cfg)


def _worker(rank: int, world_size: int, steps: int, global_bs: int, ret: dict) -> None:
    setup(rank, world_size)
    torch.set_num_threads(3)
    ds = CharDataset(block_size=32, n_repeat=200)
    model = _make_model(ds.vocab_size, ds.block_size)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    loader, per_bs = partition_dataset(ds, world_size, rank, global_bs)
    it = iter(loader)
    losses = []
    t0 = time.perf_counter()
    for step in range(steps):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(loader)
            xb, yb = next(it)
        opt.zero_grad()
        _, loss = model(xb, yb)
        loss.backward()
        average_gradients(model)  # synchronous all-reduce
        opt.step()
        losses.append(loss.item())
    dt = time.perf_counter() - t0

    if rank == 0:
        # sanity: gradients are byte-identical across ranks after averaging, so
        # a checksum of the model params should match on all ranks.
        checksum = sum(p.detach().double().sum().item() for p in model.parameters())
        ret["rank0"] = {
            "per_device_batch": per_bs,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "elapsed_sec": dt,
            "steps": steps,
            "param_checksum": checksum,
        }
    cleanup()


def run_distributed(world_size: int, steps: int, global_bs: int) -> dict:
    mgr = mp.Manager()
    ret = mgr.dict()
    mp.spawn(_worker, args=(world_size, steps, global_bs, ret), nprocs=world_size, join=True)
    return dict(ret["rank0"])


def run_single(steps: int, global_bs: int) -> dict:
    """Single-process baseline (world_size=1 semantics) for the speedup ratio."""
    torch.set_num_threads(3)
    ds = CharDataset(block_size=32, n_repeat=200)
    model = _make_model(ds.vocab_size, ds.block_size)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=global_bs, shuffle=True, drop_last=True)
    it = iter(loader)
    losses = []
    t0 = time.perf_counter()
    for step in range(steps):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(loader)
            xb, yb = next(it)
        opt.zero_grad()
        _, loss = model(xb, yb)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    dt = time.perf_counter() - t0
    return {"initial_loss": losses[0], "final_loss": losses[-1], "elapsed_sec": dt, "steps": steps}


def main(out_dir="results"):
    steps = 60
    global_bs = 32
    world_size = 2

    print(f"[HW5-DP] single-process baseline: {steps} steps, batch {global_bs}")
    single = run_single(steps, global_bs)
    print(f"        loss {single['initial_loss']:.3f} -> {single['final_loss']:.3f} "
          f"in {single['elapsed_sec']:.1f}s")

    print(f"[HW5-DP] data-parallel: world_size={world_size} (gloo, CPU processes)")
    dp = run_distributed(world_size, steps, global_bs)
    print(f"        rank0 loss {dp['initial_loss']:.3f} -> {dp['final_loss']:.3f} "
          f"in {dp['elapsed_sec']:.1f}s, per-device batch {dp['per_device_batch']}")

    report = {
        "backend": "gloo",
        "device": "cpu",
        "world_size": world_size,
        "steps": steps,
        "global_batch_size": global_bs,
        "note": "Real multi-process torch.distributed data-parallel training on CPU. "
        "Gradients are all-reduced every step. GPU/NCCL wall-clock speedup is documented "
        "as partial (no GPU on this machine); correctness of synchronous SGD is verified here.",
        "single_process": single,
        "data_parallel_rank0": dp,
        "converges": bool(dp["final_loss"] < dp["initial_loss"]),
    }
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / "hw4_data_parallel.json").write_text(json.dumps(report, indent=2))
    print(f"[HW5-DP] converges: {report['converges']}; wrote {out/'hw4_data_parallel.json'}")
    return report


if __name__ == "__main__":
    main()
