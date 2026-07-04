"""Verify the HW1 tensor primitives (map/zip/reduce/matmul) against NumPy.

These are the CPU-reference versions of the CUDA kernels in
``hw1_autodiff_ops/cuda/combine.cu``.  Correctness here is what the GPU kernel
unit tests assert on a CUDA device.
"""

import numpy as np
import pytest

from hw1_autodiff_ops.tensor_ops import (
    TensorData,
    add_zip,
    matrix_multiply,
    max_reduce,
    mul_zip,
    relu_map,
    sum_reduce,
)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_map_relu(seed):
    rng = np.random.default_rng(seed)
    a = TensorData(rng.standard_normal(24), (2, 3, 4))
    out = TensorData.zeros((2, 3, 4))
    got = relu_map(out, a).to_numpy()
    assert np.allclose(got, np.maximum(a.to_numpy(), 0))


def test_zip_add_broadcast():
    rng = np.random.default_rng(1)
    a = TensorData(rng.standard_normal(24), (2, 3, 4))
    b = TensorData(rng.standard_normal(4), (4,))
    out = TensorData.zeros((2, 3, 4))
    got = add_zip(out, a, b).to_numpy()
    assert np.allclose(got, a.to_numpy() + b.to_numpy())


def test_zip_mul_broadcast_middle():
    rng = np.random.default_rng(2)
    a = TensorData(rng.standard_normal(24), (2, 3, 4))
    b = TensorData(rng.standard_normal(8), (2, 1, 4))
    out = TensorData.zeros((2, 3, 4))
    got = mul_zip(out, a, b).to_numpy()
    assert np.allclose(got, a.to_numpy() * b.to_numpy())


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_reduce_sum(dim):
    rng = np.random.default_rng(3)
    a = TensorData(rng.standard_normal(24), (2, 3, 4))
    got = sum_reduce(a, dim).to_numpy()
    assert np.allclose(got, a.to_numpy().sum(axis=dim, keepdims=True))


def test_reduce_max():
    rng = np.random.default_rng(4)
    a = TensorData(rng.standard_normal(24), (2, 3, 4))
    got = max_reduce(a, 2).to_numpy()
    assert np.allclose(got, a.to_numpy().max(axis=2, keepdims=True))


def test_matmul_batched():
    rng = np.random.default_rng(5)
    a = TensorData(rng.standard_normal(30), (5, 2, 3))
    b = TensorData(rng.standard_normal(60), (5, 3, 4))
    got = matrix_multiply(a, b).to_numpy()
    assert np.allclose(got, a.to_numpy() @ b.to_numpy())


def test_matmul_broadcast_batch():
    rng = np.random.default_rng(6)
    a = TensorData(rng.standard_normal(6), (2, 3))
    b = TensorData(rng.standard_normal(60), (5, 3, 4))
    got = matrix_multiply(a, b).to_numpy()
    assert np.allclose(got, a.to_numpy() @ b.to_numpy())
