"""CMU 11-868 Assignment 1 — parallel tensor primitives (map / zip / reduce / matmul).

The course asks the student to implement the four low-level operators that every
higher-level tensor operation is built out of, first as strided CPU loops and
then as CUDA kernels (`minitorch/cuda_ops.py`, `src/combine.cu`).  A CUDA
device is required to *run* the GPU kernels, but the algorithm each kernel
implements is exactly the strided, broadcasting loop reproduced here in NumPy.

We implement the operators directly on ``TensorData`` (a flat storage + strides
+ shape triple, matching MiniTorch's layout) so that:

* the indexing / broadcasting logic is the *real* assignment logic, not a
  ``numpy`` one-liner, and
* the result is numerically identical to what the CUDA kernel computes, which
  is what the kernel unit tests check on a GPU.

The four primitives:

``tensor_map(fn)``    element-wise unary map with broadcasting of the out shape.
``tensor_zip(fn)``    element-wise binary map with NumPy-style broadcasting.
``tensor_reduce(fn)`` reduction along one dimension with an identity ``start``.
``matrix_multiply``   batched matmul over the last two dims (the tiled kernel).

Run the self-check:  ``python -m hw1_autodiff_ops.tensor_ops``
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np

Shape = Tuple[int, ...]
Strides = Tuple[int, ...]


# ---------------------------------------------------------------------------
# TensorData: flat storage + shape + strides (MiniTorch's core layout)
# ---------------------------------------------------------------------------
def strides_from_shape(shape: Shape) -> Strides:
    """Row-major (C-contiguous) strides for ``shape``."""
    strides = [1]
    for s in reversed(shape[1:]):
        strides.insert(0, strides[0] * s)
    return tuple(strides)


def index_to_position(index: Sequence[int], strides: Strides) -> int:
    """Flatten a multi-dimensional ``index`` into a storage offset."""
    pos = 0
    for i, s in zip(index, strides):
        pos += i * s
    return pos


def to_index(ordinal: int, shape: Shape, out_index: list) -> None:
    """Inverse of :func:`index_to_position` for a contiguous ordinal."""
    cur = ordinal
    for i in range(len(shape) - 1, -1, -1):
        sh = shape[i]
        out_index[i] = int(cur % sh)
        cur //= sh


def broadcast_index(
    big_index: Sequence[int], big_shape: Shape, shape: Shape, out_index: list
) -> None:
    """Map an index in the (larger) broadcasted shape back to ``shape``."""
    offset = len(big_shape) - len(shape)
    for i in range(len(shape)):
        if shape[i] == 1:
            out_index[i] = 0
        else:
            out_index[i] = big_index[i + offset]


def shape_broadcast(a: Shape, b: Shape) -> Shape:
    """NumPy broadcasting rules for two shapes."""
    ra, rb = list(reversed(a)), list(reversed(b))
    out = []
    for i in range(max(len(ra), len(rb))):
        da = ra[i] if i < len(ra) else 1
        db = rb[i] if i < len(rb) else 1
        if da == 1:
            out.append(db)
        elif db == 1:
            out.append(da)
        elif da == db:
            out.append(da)
        else:
            raise ValueError(f"cannot broadcast {a} with {b}")
    return tuple(reversed(out))


class TensorData:
    """Flat float storage addressed by (shape, strides)."""

    def __init__(self, storage, shape: Shape, strides: Strides | None = None):
        self._storage = np.asarray(storage, dtype=np.float64).reshape(-1)
        self.shape = tuple(int(s) for s in shape)
        self.strides = strides or strides_from_shape(self.shape)
        self.size = int(np.prod(self.shape)) if self.shape else 1

    @staticmethod
    def zeros(shape: Shape) -> "TensorData":
        return TensorData(np.zeros(int(np.prod(shape))), shape)

    def get(self, index: Sequence[int]) -> float:
        return float(self._storage[index_to_position(index, self.strides)])

    def set(self, index: Sequence[int], value: float) -> None:
        self._storage[index_to_position(index, self.strides)] = value

    def to_numpy(self) -> np.ndarray:
        out = np.empty(self.size)
        idx = [0] * len(self.shape)
        for ordinal in range(self.size):
            to_index(ordinal, self.shape, idx)
            out[ordinal] = self.get(idx)
        return out.reshape(self.shape)


# ---------------------------------------------------------------------------
# Primitive 1 — map (unary, broadcasting the output shape).  CUDA: one thread
# per output element, each reads its (broadcast) input and writes one output.
# ---------------------------------------------------------------------------
def tensor_map(fn: Callable[[float], float]):
    def _map(out: TensorData, a: TensorData) -> TensorData:
        out_index = [0] * len(out.shape)
        in_index = [0] * len(a.shape)
        for ordinal in range(out.size):
            to_index(ordinal, out.shape, out_index)
            broadcast_index(out_index, out.shape, a.shape, in_index)
            val = a.get(in_index)
            out.set(out_index, fn(val))
        return out

    return _map


# ---------------------------------------------------------------------------
# Primitive 2 — zip (binary, NumPy broadcasting).  CUDA: one thread per output
# element, broadcasting *both* inputs down to their own shapes.
# ---------------------------------------------------------------------------
def tensor_zip(fn: Callable[[float, float], float]):
    def _zip(out: TensorData, a: TensorData, b: TensorData) -> TensorData:
        out_index = [0] * len(out.shape)
        a_index = [0] * len(a.shape)
        b_index = [0] * len(b.shape)
        for ordinal in range(out.size):
            to_index(ordinal, out.shape, out_index)
            broadcast_index(out_index, out.shape, a.shape, a_index)
            broadcast_index(out_index, out.shape, b.shape, b_index)
            out.set(out_index, fn(a.get(a_index), b.get(b_index)))
        return out

    return _zip


# ---------------------------------------------------------------------------
# Primitive 3 — reduce along one dim.  CUDA: a block per output element does a
# parallel tree reduction over the reduced dimension; here a serial fold with
# the same identity ``start`` gives the identical result.
# ---------------------------------------------------------------------------
def tensor_reduce(fn: Callable[[float, float], float], start: float):
    def _reduce(a: TensorData, dim: int) -> TensorData:
        out_shape = list(a.shape)
        reduce_size = out_shape[dim]
        out_shape[dim] = 1
        out = TensorData.zeros(tuple(out_shape))
        out_index = [0] * len(out.shape)
        for ordinal in range(out.size):
            to_index(ordinal, out.shape, out_index)
            acc = start
            a_index = list(out_index)
            for j in range(reduce_size):
                a_index[dim] = j
                acc = fn(acc, a.get(a_index))
            out.set(out_index, acc)
        return out

    return _reduce


# ---------------------------------------------------------------------------
# Primitive 4 — batched matmul over the last two dims (the tiled CUDA kernel).
# ---------------------------------------------------------------------------
def matrix_multiply(a: TensorData, b: TensorData) -> TensorData:
    """Batched matmul: (..., I, K) @ (..., K, J) -> (..., I, J)."""
    assert a.shape[-1] == b.shape[-2], f"inner dims mismatch {a.shape} {b.shape}"
    batch_shape = shape_broadcast(a.shape[:-2], b.shape[:-2])
    I, K, J = a.shape[-2], a.shape[-1], b.shape[-1]
    out_shape = batch_shape + (I, J)
    out = TensorData.zeros(out_shape)

    out_index = [0] * len(out_shape)
    a_index = [0] * len(a.shape)
    b_index = [0] * len(b.shape)
    for ordinal in range(out.size):
        to_index(ordinal, out_shape, out_index)
        # broadcast batch dims onto a and b
        broadcast_index(out_index[:-2], batch_shape, a.shape[:-2], a_index)
        broadcast_index(out_index[:-2], batch_shape, b.shape[:-2], b_index)
        i, j = out_index[-2], out_index[-1]
        a_index[-2] = i
        b_index[-1] = j
        acc = 0.0
        for k in range(K):
            a_index[-1] = k
            b_index[-2] = k
            acc += a.get(a_index) * b.get(b_index)
        out.set(out_index, acc)
    return out


# convenient concrete operators built from the primitives
add_zip = tensor_zip(lambda x, y: x + y)
mul_zip = tensor_zip(lambda x, y: x * y)
neg_map = tensor_map(lambda x: -x)
relu_map = tensor_map(lambda x: x if x > 0 else 0.0)
sum_reduce = tensor_reduce(lambda x, y: x + y, 0.0)
max_reduce = tensor_reduce(lambda x, y: x if x > y else y, -np.inf)


def _self_check() -> None:
    rng = np.random.default_rng(0)
    # map
    a = TensorData(rng.standard_normal(24), (2, 3, 4))
    out = TensorData.zeros((2, 3, 4))
    got = relu_map(out, a).to_numpy()
    assert np.allclose(got, np.maximum(a.to_numpy(), 0)), "map failed"

    # zip with broadcasting (2,3,4) + (4,)
    a = TensorData(rng.standard_normal(24), (2, 3, 4))
    b = TensorData(rng.standard_normal(4), (4,))
    out = TensorData.zeros((2, 3, 4))
    got = add_zip(out, a, b).to_numpy()
    assert np.allclose(got, a.to_numpy() + b.to_numpy()), "zip broadcast failed"

    # reduce along dim=1
    a = TensorData(rng.standard_normal(24), (2, 3, 4))
    got = sum_reduce(a, 1).to_numpy()
    assert np.allclose(got, a.to_numpy().sum(axis=1, keepdims=True)), "reduce failed"

    # batched matmul (5,2,3) @ (5,3,4)
    a = TensorData(rng.standard_normal(30), (5, 2, 3))
    b = TensorData(rng.standard_normal(60), (5, 3, 4))
    got = matrix_multiply(a, b).to_numpy()
    assert np.allclose(got, a.to_numpy() @ b.to_numpy()), "matmul failed"

    # broadcast batched matmul (2,3) @ (5,3,4)
    a = TensorData(rng.standard_normal(6), (2, 3))
    b = TensorData(rng.standard_normal(60), (5, 3, 4))
    got = matrix_multiply(a, b).to_numpy()
    assert np.allclose(got, a.to_numpy() @ b.to_numpy()), "broadcast matmul failed"

    print("all tensor-op primitives match NumPy references.")


if __name__ == "__main__":
    _self_check()
