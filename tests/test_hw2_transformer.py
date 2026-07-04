"""Verify HW2 transformer modules against PyTorch (forward + backward)."""

import numpy as np
import pytest

from hw2_transformer.autograd import Value, cross_entropy_from_logits
from hw2_transformer.nn_functions import gelu, logsumexp, softmax, one_hot
from hw2_transformer import modules as M

torch = pytest.importorskip("torch")


def test_gelu_matches_torch():
    x = np.linspace(-4, 4, 50)
    ours = gelu(x)
    ref = torch.nn.functional.gelu(torch.tensor(x), approximate="tanh").numpy()
    assert np.allclose(ours, ref, atol=1e-6)


def test_logsumexp_matches_torch():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 7))
    ours = logsumexp(x, axis=-1, keepdims=False)
    ref = torch.logsumexp(torch.tensor(x), dim=-1).numpy()
    assert np.allclose(ours, ref, atol=1e-8)


def test_softmax_matches_torch():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 5))
    ours = softmax(x, axis=-1)
    ref = torch.softmax(torch.tensor(x), dim=-1).numpy()
    assert np.allclose(ours, ref, atol=1e-8)


def test_one_hot():
    idx = np.array([0, 2, 1])
    oh = one_hot(idx, 3)
    assert np.array_equal(oh, np.eye(3)[idx])


def test_linear_backward_matches_torch():
    rng = np.random.default_rng(2)
    xn = rng.standard_normal((4, 6))
    lin = M.Linear(6, 3, rng)
    x = Value(xn, requires_grad=True)
    out = lin(x)
    out.backward(np.ones_like(out.data))

    xt = torch.tensor(xn, requires_grad=True)
    Wt = torch.tensor(lin.weight.value.data, requires_grad=True)
    bt = torch.tensor(lin.bias.value.data, requires_grad=True)
    ot = xt @ Wt + bt
    ot.sum().backward()
    assert np.allclose(out.data, ot.detach().numpy(), atol=1e-8)
    assert np.allclose(x.grad, xt.grad.numpy(), atol=1e-8)
    assert np.allclose(lin.weight.value.grad, Wt.grad.numpy(), atol=1e-8)


def test_layernorm_backward_matches_torch():
    rng = np.random.default_rng(3)
    xn = rng.standard_normal((5, 8))
    ln = M.LayerNorm1d(8)
    x = Value(xn, requires_grad=True)
    out = ln(x)
    seed_grad = rng.standard_normal(out.data.shape)
    out.backward(seed_grad)

    xt = torch.tensor(xn, requires_grad=True)
    gt = torch.tensor(ln.gamma.value.data, requires_grad=True)
    bt = torch.tensor(ln.beta.value.data, requires_grad=True)
    ot = torch.nn.functional.layer_norm(xt, (8,), gt, bt, eps=1e-5)
    ot.backward(torch.tensor(seed_grad))
    assert np.allclose(out.data, ot.detach().numpy(), atol=1e-6)
    assert np.allclose(x.grad, xt.grad.numpy(), atol=1e-5)
    assert np.allclose(ln.gamma.value.grad, gt.grad.numpy(), atol=1e-5)


def test_attention_causal_shape_and_grad():
    rng = np.random.default_rng(4)
    B, T, C, H = 2, 5, 16, 4
    xn = rng.standard_normal((B, T, C))
    mha = M.MultiHeadAttention(C, H, rng, p_dropout=0.0)
    x = Value(xn, requires_grad=True)
    out = mha(x, train=False)
    assert out.data.shape == (B, T, C)
    out.backward(np.ones_like(out.data))
    assert x.grad is not None and x.grad.shape == (B, T, C)
    assert np.all(np.isfinite(x.grad))


def test_decoder_lm_forward_and_loss_grad():
    rng = np.random.default_rng(5)
    vocab, B, T = 20, 3, 6
    model = M.DecoderLM(vocab, n_embd=32, n_head=4, n_layer=2, hidden=64, max_len=T, seed=0)
    idx = rng.integers(0, vocab, size=(B, T))
    logits = model(idx, train=False)
    assert logits.data.shape == (B, T, vocab)
    targets = rng.integers(0, vocab, size=(B * T,))
    loss = cross_entropy_from_logits(logits.reshape(B * T, vocab), targets)
    loss.backward()
    # every parameter should receive a finite gradient
    for p in model.parameters():
        assert p.value.grad is not None
        assert np.all(np.isfinite(p.value.grad))


def test_cross_entropy_matches_torch():
    rng = np.random.default_rng(6)
    logits = rng.standard_normal((7, 11))
    targets = rng.integers(0, 11, size=7)
    lv = Value(logits, requires_grad=True)
    loss = cross_entropy_from_logits(lv, targets)
    loss.backward()

    lt = torch.tensor(logits, requires_grad=True)
    ref = torch.nn.functional.cross_entropy(lt, torch.tensor(targets))
    ref.backward()
    assert np.allclose(loss.data, ref.item(), atol=1e-8)
    assert np.allclose(lv.grad, lt.grad.numpy(), atol=1e-8)
