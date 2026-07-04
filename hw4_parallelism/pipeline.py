"""Assignment 5, Problem 2 — Pipeline (GPipe-style) parallelism.

The model is an ``nn.Sequential``; we split it into ``n`` contiguous stages and
stream ``m`` micro-batches through them.  With the GPipe schedule, at clock tick
``t`` the pairs ``(micro i, stage j)`` with ``i + j == t`` run concurrently, so
different stages work on different micro-batches at the same time instead of one
stage sitting idle while another runs.

Faithful to the spec, this file provides:

* ``_split_module`` — partition an ``nn.Sequential`` into ``n`` stage modules.
* ``_clock_cycles`` — the GPipe schedule: for ``m`` micro-batches and ``n``
  stages, yield, per tick, the list of ``(micro_idx, stage_idx)`` that run.
* ``Pipe`` — an ``nn.Module`` that chunks the input into micro-batches, executes
  them stage-by-stage following the schedule, and concatenates the outputs.

On a real multi-GPU box each stage lives on its own device and the schedule
overlaps compute across GPUs.  On this CPU build the stages share the device, so
we get the *correct* pipelined result (verified equal to the un-pipelined model)
but not the wall-clock overlap — documented as partial in the README.
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

import torch
import torch.nn as nn


def _split_module(module: nn.Sequential, n_partitions: int) -> List[nn.Sequential]:
    """Split ``module``'s children into ``n_partitions`` contiguous stages.

    The layers are divided as evenly as possible; earlier stages take the extra
    layer when the count doesn't divide evenly.
    """
    children = list(module.children())
    assert n_partitions >= 1
    assert len(children) >= n_partitions, "more partitions than layers"
    n = len(children)
    base, rem = divmod(n, n_partitions)
    stages: List[nn.Sequential] = []
    start = 0
    for p in range(n_partitions):
        size = base + (1 if p < rem else 0)
        stages.append(nn.Sequential(*children[start : start + size]))
        start += size
    return stages


def _clock_cycles(n_micro: int, n_stages: int) -> Iterator[List[Tuple[int, int]]]:
    """Yield the GPipe schedule as a list of ``(micro_idx, stage_idx)`` per tick.

    There are ``n_micro + n_stages - 1`` ticks.  At tick ``t`` every pair with
    ``micro + stage == t`` (and both in range) is ready to run.
    """
    for t in range(n_micro + n_stages - 1):
        tasks: List[Tuple[int, int]] = []
        for micro in range(n_micro):
            stage = t - micro
            if 0 <= stage < n_stages:
                tasks.append((micro, stage))
        yield tasks


class Pipe(nn.Module):
    """Pipeline-parallel wrapper over an ``nn.Sequential``.

    Args:
        module: the sequential model to pipeline.
        n_partitions: number of pipeline stages.
        n_micro: number of micro-batches to split each input batch into.
    """

    def __init__(self, module: nn.Sequential, n_partitions: int, n_micro: int = 4):
        super().__init__()
        self.n_partitions = n_partitions
        self.n_micro = n_micro
        self.partitions = nn.ModuleList(_split_module(module, n_partitions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        micros = list(x.chunk(self.n_micro, dim=0))
        m = len(micros)
        # buffers[micro] holds the tensor currently sitting at the *input* of the
        # stage it is about to enter; after the last stage it is the output.
        buffers: List[torch.Tensor] = list(micros)
        for tasks in _clock_cycles(m, self.n_partitions):
            new_vals = {}
            for micro, stage in tasks:
                new_vals[micro] = self.partitions[stage](buffers[micro])
            for micro, val in new_vals.items():
                buffers[micro] = val
        return torch.cat(buffers, dim=0)


def compute_model_parallel(module: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
    """Naive model parallelism baseline: run whole batch stage-by-stage.

    Same partitioning as :class:`Pipe` but with **no** micro-batch overlap — this
    is the baseline the assignment asks pipeline parallelism to beat.  On CPU we
    use it only to check that :class:`Pipe` produces the identical result.
    """
    for layer in module:
        x = layer(x)
    return x
