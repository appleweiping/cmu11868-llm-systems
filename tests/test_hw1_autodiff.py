"""Verify the HW2 autodiff engine against finite differences and PyTorch."""

import numpy as np
import pytest

from hw1_autodiff_ops.autodiff import (
    Tensor,
    backward,
    grad_check,
    tensor,
    topological_sort,
)

torch = pytest.importorskip("torch")


def test_topological_sort_orders_output_first_leaves_last():
    a = tensor([2.0], requires_grad=True)
    b = tensor([3.0], requires_grad=True)
    c = a * b
    d = c + a
    order = list(topological_sort(d))
    # output d must come before its parents; leaves a,b come after internal c
    ids = [n.unique_id for n in order]
    assert ids[0] == d.unique_id
    assert order[-1].is_leaf()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_grad_check_relu_mlp(seed):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((4, 3))

    def f(x):
        return (x @ tensor(W)).relu().sum()

    x = tensor(rng.standard_normal((2, 4)))
    err = grad_check(f, x)
    assert err < 1e-4, f"grad mismatch {err}"


def test_matches_pytorch_matmul_relu_sigmoid():
    rng = np.random.default_rng(7)
    xn = rng.standard_normal((5, 6))
    Wn = rng.standard_normal((6, 3))

    # our engine
    x = tensor(xn, requires_grad=True)
    W = tensor(Wn, requires_grad=True)
    out = (x @ W).relu().sigmoid().sum()
    backward(out)

    # torch reference
    xt = torch.tensor(xn, requires_grad=True)
    Wt = torch.tensor(Wn, requires_grad=True)
    ot = torch.sigmoid(torch.relu(xt @ Wt)).sum()
    ot.backward()

    assert np.allclose(out.item(), ot.item(), atol=1e-8)
    assert np.allclose(x.grad.data, xt.grad.numpy(), atol=1e-8)
    assert np.allclose(W.grad.data, Wt.grad.numpy(), atol=1e-8)


def test_broadcast_bias_gradient():
    rng = np.random.default_rng(3)
    xn = rng.standard_normal((4, 3))
    bn = rng.standard_normal((3,))
    x = tensor(xn, requires_grad=True)
    b = tensor(bn, requires_grad=True)
    out = (x + b).sum()
    backward(out)
    # d/db of sum(x+b) is number of rows, per element
    assert np.allclose(b.grad.data, np.full(3, 4.0))
