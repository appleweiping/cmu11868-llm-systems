"""CMU 11-868 Assignment 5 — Distributed Training & Parallelism.

Two distributed-training methodologies for a GPT-style model, implemented
faithfully to the assignment spec and run for real on this machine:

* :mod:`hw4_parallelism.data_parallel` — data parallelism: partition the dataset
  across workers and synchronise gradients with an all-reduce every step
  (``partition_dataset``, ``DataPartitioner``, ``average_gradients``).
* :mod:`hw4_parallelism.pipeline` — pipeline (GPipe-style) parallelism: split an
  ``nn.Sequential`` model across stages and stream micro-batches through it with
  a clock-cycle schedule (``_split_module``, ``_clock_cycles``, ``Pipe``).

The official course runs these on multiple GPUs with the NCCL backend.  This
build has no GPU, so we run *real* multi-process training on CPU with the
``gloo`` backend (genuine ``torch.distributed`` collectives across OS
processes) — the same code path, only the backend and device differ.  The
GPU-only wall-clock speedups are documented as partial in the README; the
correctness of the distributed algorithms is verified here on CPU.
"""
