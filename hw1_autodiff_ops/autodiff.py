"""Reverse-mode automatic differentiation over n-dimensional tensors.

This is a self-contained re-implementation of the core of the CMU 11-868
Assignment 1 automatic-differentiation problem.  The two functions the course
asks the student to write are :func:`topological_sort` and
:func:`backpropagate`; both are implemented here and exercised end-to-end by a
small tensor autograd engine so that gradients can be checked numerically
against finite differences and against PyTorch.

The engine deliberately mirrors the MiniTorch design used by the course:

* every ``Tensor`` records the ``Function`` (op) that produced it plus the
  input tensors (its parents) in a ``History`` object,
* the backward pass walks the computation graph in reversed topological order
  (Problem 1.1) and accumulates leaf derivatives via the chain rule
  (Problem 1.2).

Only NumPy is used for the numerics so the whole thing runs on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Autograd graph bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class History:
    """Records how a tensor was produced so we can replay the backward pass."""

    last_fn: Optional["Function"] = None
    ctx: Optional["Context"] = None
    inputs: Sequence["Tensor"] = ()


class Context:
    """Stores values saved during the forward pass for use in ``backward``."""

    def __init__(self, no_grad: bool = False) -> None:
        self.no_grad = no_grad
        self.saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values


_UNIQUE_COUNTER = 0


def _next_id() -> int:
    global _UNIQUE_COUNTER
    _UNIQUE_COUNTER += 1
    return _UNIQUE_COUNTER


class Tensor:
    """A minimal autograd tensor backed by a NumPy array."""

    def __init__(
        self,
        data: np.ndarray,
        history: Optional[History] = None,
        requires_grad: bool = False,
        name: Optional[str] = None,
    ) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.history = history
        self.requires_grad = requires_grad
        self.grad: Optional["Tensor"] = None
        self.unique_id = _next_id()
        self.name = name or f"t{self.unique_id}"

    # -- graph predicates ---------------------------------------------------
    def is_leaf(self) -> bool:
        """A leaf is a user-created tensor (no producing function)."""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        return self.history is None

    @property
    def parents(self) -> Iterable["Tensor"]:
        assert self.history is not None
        return [p for p in self.history.inputs if isinstance(p, Tensor)]

    def accumulate_derivative(self, d: "Tensor") -> None:
        """Accumulate a gradient onto a leaf tensor (Problem 1.2)."""
        assert self.is_leaf(), "only leaf tensors accumulate gradients"
        if self.grad is None:
            self.grad = Tensor(np.zeros_like(self.data))
        self.grad.data = self.grad.data + _unbroadcast(d.data, self.data.shape)

    def chain_rule(self, d_output: "Tensor") -> List[Tuple["Tensor", "Tensor"]]:
        """Apply the local derivative of the producing function.

        Returns a list of ``(input_tensor, d_input)`` pairs for every input
        that requires a gradient.
        """
        h = self.history
        assert h is not None and h.last_fn is not None and h.ctx is not None
        grads = h.last_fn.backward(h.ctx, d_output)
        if not isinstance(grads, (tuple, list)):
            grads = (grads,)
        result: List[Tuple[Tensor, Tensor]] = []
        for inp, g in zip(h.inputs, grads):
            if isinstance(inp, Tensor) and not inp.is_constant():
                result.append((inp, g))
        return result

    # -- convenience --------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def item(self) -> float:
        return float(self.data.reshape(-1)[0])

    def numpy(self) -> np.ndarray:
        return self.data

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Tensor({self.data!r}, requires_grad={self.requires_grad})"

    # operator overloads delegate to the Function subclasses defined below
    def __add__(self, other):
        return Add.apply(self, _as_tensor(other))

    def __radd__(self, other):
        return Add.apply(_as_tensor(other), self)

    def __sub__(self, other):
        return Add.apply(self, Neg.apply(_as_tensor(other)))

    def __mul__(self, other):
        return Mul.apply(self, _as_tensor(other))

    def __rmul__(self, other):
        return Mul.apply(_as_tensor(other), self)

    def __neg__(self):
        return Neg.apply(self)

    def __matmul__(self, other):
        return MatMul.apply(self, _as_tensor(other))

    def sum(self, dim: Optional[int] = None):
        return Sum.apply(self, dim)

    def mean(self, dim: Optional[int] = None):
        return Mean.apply(self, dim)

    def relu(self):
        return ReLU.apply(self)

    def sigmoid(self):
        return Sigmoid.apply(self)

    def log(self):
        return Log.apply(self)


def _as_tensor(x: Any) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return Tensor(np.asarray(x, dtype=np.float64))


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reduce ``grad`` back to ``shape`` after NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


# ---------------------------------------------------------------------------
# THE TWO CORE ASSIGNMENT FUNCTIONS
# ---------------------------------------------------------------------------
def topological_sort(variable: Tensor) -> Iterable[Tensor]:
    """Return nodes of the computation graph in reversed topological order.

    Assignment 1, Problem 1.1.  We run a post-order depth-first search from the
    output ``variable``; when all of a node's children have been visited we
    prepend the node to the ordering.  The result therefore lists ``variable``
    first and its leaves last, which is the order the backward pass needs.
    """
    order: List[Tensor] = []
    visited: set[int] = set()

    def visit(node: Tensor) -> None:
        if node.unique_id in visited or node.is_constant():
            return
        if not node.is_leaf():
            for parent in node.parents:
                visit(parent)
        visited.add(node.unique_id)
        order.insert(0, node)

    visit(variable)
    return order


def backpropagate(variable: Tensor, deriv: Tensor) -> None:
    """Run backprop from ``variable`` seeded with ``deriv`` (Problem 1.2).

    Walk the graph in reversed topological order.  Maintain a table of the
    accumulated output-derivative for every node.  For a leaf, accumulate the
    derivative onto ``.grad``; for an internal node, push the derivative to its
    inputs via the chain rule.
    """
    derivs: dict[int, Tensor] = {variable.unique_id: deriv}

    for node in topological_sort(variable):
        d_output = derivs.get(node.unique_id)
        if d_output is None:
            continue
        if node.is_leaf():
            node.accumulate_derivative(d_output)
            continue
        for inp, d_input in node.chain_rule(d_output):
            if inp.unique_id in derivs:
                derivs[inp.unique_id] = Tensor(derivs[inp.unique_id].data + d_input.data)
            else:
                derivs[inp.unique_id] = d_input


def backward(variable: Tensor, grad: Optional[Tensor] = None) -> None:
    """Public entry point: seed the backward pass from a (scalar) output."""
    if grad is None:
        assert variable.data.size == 1, "grad required for non-scalar output"
        grad = Tensor(np.ones_like(variable.data))
    backpropagate(variable, grad)


# ---------------------------------------------------------------------------
# Function base class + concrete ops (provide the local derivatives)
# ---------------------------------------------------------------------------
class Function:
    @classmethod
    def apply(cls, *inputs: Any) -> Tensor:
        """Run the forward pass.

        ``inputs`` may mix :class:`Tensor` objects (graph nodes) with plain
        Python values (e.g. a reduction ``dim``).  Only the tensors become
        graph parents; scalars are passed straight through to ``forward``.
        """
        tensor_inputs = [t for t in inputs if isinstance(t, Tensor)]
        needs_grad = any(t.requires_grad for t in tensor_inputs)
        ctx = Context(no_grad=not needs_grad)
        raw = [t.data if isinstance(t, Tensor) else t for t in inputs]
        out_data = cls.forward(ctx, *raw)
        out = Tensor(out_data, requires_grad=needs_grad)
        out.history = History(last_fn=cls, ctx=ctx, inputs=tensor_inputs)
        return out

    @staticmethod
    def forward(ctx: Context, *args: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad: Tensor):  # pragma: no cover
        raise NotImplementedError


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        return a + b

    @staticmethod
    def backward(ctx, grad):
        return Tensor(grad.data), Tensor(grad.data)


class Neg(Function):
    @staticmethod
    def forward(ctx, a):
        return -a

    @staticmethod
    def backward(ctx, grad):
        return (Tensor(-grad.data),)


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return Tensor(grad.data * b), Tensor(grad.data * a)


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a @ b

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return (
            Tensor(grad.data @ np.swapaxes(b, -1, -2)),
            Tensor(np.swapaxes(a, -1, -2) @ grad.data),
        )


class Sum(Function):
    @staticmethod
    def forward(ctx, a, dim):
        ctx.save_for_backward(a.shape, dim)
        if dim is None:
            return np.array(a.sum())
        return a.sum(axis=dim)

    @staticmethod
    def backward(ctx, grad):
        shape, dim = ctx.saved_tensors
        g = grad.data
        if dim is None:
            g = np.broadcast_to(g, shape)
        else:
            g = np.expand_dims(g, axis=dim)
            g = np.broadcast_to(g, shape)
        return Tensor(np.array(g)), None


class Mean(Function):
    @staticmethod
    def forward(ctx, a, dim):
        ctx.save_for_backward(a.shape, dim)
        if dim is None:
            return np.array(a.mean())
        return a.mean(axis=dim)

    @staticmethod
    def backward(ctx, grad):
        shape, dim = ctx.saved_tensors
        g = grad.data
        if dim is None:
            n = int(np.prod(shape))
            g = np.broadcast_to(g, shape) / n
        else:
            n = shape[dim]
            g = np.expand_dims(g, axis=dim)
            g = np.broadcast_to(g, shape) / n
        return Tensor(np.array(g)), None


class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return np.maximum(a, 0.0)

    @staticmethod
    def backward(ctx, grad):
        (a,) = ctx.saved_tensors
        return (Tensor(grad.data * (a > 0)),)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        out = 1.0 / (1.0 + np.exp(-a))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad):
        (out,) = ctx.saved_tensors
        return (Tensor(grad.data * out * (1.0 - out)),)


class Log(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return np.log(a)

    @staticmethod
    def backward(ctx, grad):
        (a,) = ctx.saved_tensors
        return (Tensor(grad.data / a),)


# ---------------------------------------------------------------------------
# helpers used by tests / the MLP
# ---------------------------------------------------------------------------
def tensor(data, requires_grad: bool = False) -> Tensor:
    t = Tensor(np.asarray(data, dtype=np.float64), requires_grad=requires_grad)
    t.history = History()  # mark as a leaf (last_fn is None)
    return t


def grad_check(f: Callable[[Tensor], Tensor], x: Tensor, eps: float = 1e-6) -> float:
    """Return max abs difference between autodiff and finite-difference grads."""
    x = tensor(x.data.copy(), requires_grad=True)
    out = f(x)
    backward(out.sum() if out.data.size > 1 else out)
    analytic = x.grad.data.copy()

    numeric = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x.data[idx]
        x.data[idx] = orig + eps
        plus = f(tensor(x.data.copy())).data.sum()
        x.data[idx] = orig - eps
        minus = f(tensor(x.data.copy())).data.sum()
        x.data[idx] = orig
        numeric[idx] = (plus - minus) / (2 * eps)
        it.iternext()
    return float(np.max(np.abs(analytic - numeric)))
