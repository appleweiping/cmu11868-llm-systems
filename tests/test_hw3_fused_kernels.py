"""Verify HW3 fused softmax/layernorm reference kernels against PyTorch.

These CPU references are numerically identical to the CUDA kernels in
``hw3_fused_kernels/cuda/`` and are what the GPU kernel unit tests check against
(at the course's atol/rtol 1e-3).  Here we hold them to a much tighter atol.
"""

import numpy as np
import pytest

from hw3_fused_kernels.fused_kernels import (
    attn_softmax_bw,
    attn_softmax_fw,
    layernorm_bw,
    layernorm_fw,
)

torch = pytest.importorskip("torch")


def test_attn_softmax_fw_matches_torch():
    rng = np.random.default_rng(0)
    b, h, q, k = 2, 4, 5, 7
    inp = rng.standard_normal((b, h, q, k))
    mask = (rng.random((b, 1, 1, k)) < 0.3).astype(np.float64) * -1e8
    ours = attn_softmax_fw(inp, mask)
    ref = torch.softmax(torch.tensor(inp + mask), dim=-1).numpy()
    assert np.allclose(ours, ref, atol=1e-9)


def test_attn_softmax_bw_matches_torch():
    rng = np.random.default_rng(1)
    b, h, q, k = 2, 3, 4, 6
    inp = torch.tensor(rng.standard_normal((b, h, q, k)), requires_grad=True)
    mask = torch.zeros((b, 1, 1, k))
    soft = torch.softmax(inp + mask, dim=-1)
    seed = rng.standard_normal((b, h, q, k))
    soft.backward(torch.tensor(seed))
    ref_grad = inp.grad.numpy()

    ours = attn_softmax_bw(seed, soft.detach().numpy())
    assert np.allclose(ours, ref_grad, atol=1e-9)


def test_layernorm_fw_matches_torch():
    rng = np.random.default_rng(2)
    rows, hidden = 8, 16
    inp = rng.standard_normal((rows, hidden))
    gamma = rng.standard_normal(hidden)
    beta = rng.standard_normal(hidden)
    out, _, _, _ = layernorm_fw(inp, gamma, beta)
    ref = torch.nn.functional.layer_norm(
        torch.tensor(inp), (hidden,), torch.tensor(gamma), torch.tensor(beta), eps=1e-5
    ).numpy()
    assert np.allclose(out, ref, atol=1e-6)


def test_layernorm_bw_matches_torch():
    rng = np.random.default_rng(3)
    rows, hidden = 6, 12
    inp_t = torch.tensor(rng.standard_normal((rows, hidden)), requires_grad=True)
    gamma_t = torch.tensor(rng.standard_normal(hidden), requires_grad=True)
    beta_t = torch.tensor(rng.standard_normal(hidden), requires_grad=True)
    out_t = torch.nn.functional.layer_norm(inp_t, (hidden,), gamma_t, beta_t, eps=1e-5)
    seed = rng.standard_normal((rows, hidden))
    out_t.backward(torch.tensor(seed))

    inp = inp_t.detach().numpy()
    gamma = gamma_t.detach().numpy()
    beta = beta_t.detach().numpy()
    _, mu, rstd, xhat = layernorm_fw(inp, gamma, beta)
    dinp, dgamma, dbeta = layernorm_bw(seed, inp, gamma, mu, rstd, xhat)

    assert np.allclose(dinp, inp_t.grad.numpy(), atol=1e-6)
    assert np.allclose(dgamma, gamma_t.grad.numpy(), atol=1e-6)
    assert np.allclose(dbeta, beta_t.grad.numpy(), atol=1e-6)


def test_softmax_roundtrip_gradcheck():
    """Finite-difference check of the fused softmax backward."""
    rng = np.random.default_rng(4)
    b, h, q, k = 1, 1, 3, 5
    inp = rng.standard_normal((b, h, q, k))
    mask = np.zeros((b, 1, 1, k))
    w = rng.standard_normal((b, h, q, k))  # scalar loss = sum(w * softmax(inp))

    soft = attn_softmax_fw(inp, mask)
    analytic = attn_softmax_bw(w, soft)

    eps = 1e-6
    numeric = np.zeros_like(inp)
    it = np.nditer(inp, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = inp[idx]
        inp[idx] = orig + eps
        lp = np.sum(w * attn_softmax_fw(inp, mask))
        inp[idx] = orig - eps
        lm = np.sum(w * attn_softmax_fw(inp, mask))
        inp[idx] = orig
        numeric[idx] = (lp - lm) / (2 * eps)
        it.iternext()
    assert np.allclose(analytic, numeric, atol=1e-5)
