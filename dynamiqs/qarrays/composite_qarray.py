from __future__ import annotations

import equinox as eqx
from jaxtyping import ArrayLike

from ..time_qarray import TimeQArray
from .qarray import QArray

__all__ = ['CompositeTerm', 'CompositeQArray']


class CompositeTerm(eqx.Module):
    r"""One separable term in a :class:`CompositeQArray`.

    Represents a single term of the form

    $$
        c \, A_0 \otimes A_1 \otimes \cdots \otimes A_{N-1}
    $$

    where $c$ is a scalar coefficient and each $A_k$ is a local operator acting
    on subsystem $k$.

    Attributes:
        operators: Local operators $(A_0, \ldots, A_{N-1})$, one per subsystem.
            Each is a square :class:`QArray` or :class:`TimeQArray` of shape
            $(\ldots, d_k, d_k)$. Use the factory functions :func:`composite_term`
            or :func:`composite` to construct, which validate dimensions automatically.
        coeff: Scalar coefficient $c$ multiplying the full tensor-product operator.
            Can be a Python scalar or a broadcastable JAX array for batched use.
            Defaults to $1$.
    """

    operators: tuple[QArray | TimeQArray, ...]
    coeff: ArrayLike = 1.0


class CompositeQArray(eqx.Module):
    r"""Lazy sum of separable tensor-product operators.

    Represents an operator acting on a composite Hilbert space
    $\mathcal{H} = \mathcal{H}_0 \otimes \cdots \otimes \mathcal{H}_{N-1}$
    of total dimension $n = \prod_k d_k$, written as a sum of separable terms:

    $$
        H = \sum_{j} c_j \, A_{j,0} \otimes A_{j,1} \otimes \cdots \otimes A_{j,N-1}
    $$

    where each term $j$ is a :class:`CompositeTerm`.

    Storing the operator in this factored form — rather than materializing the full
    $n \times n$ Kronecker product — enables efficient matrix-vector products via
    per-subsystem contractions, avoiding the exponential memory cost of the dense
    representation.

    Attributes:
        dims: Hilbert space dimensions $(d_0, d_1, \ldots, d_{N-1})$ of each
            subsystem.
        terms: Tuple of :class:`CompositeTerm` objects whose sum defines the
            operator. All terms must have ``len(operators) == len(dims)``.
    """

    dims: tuple[int, ...]
    terms: tuple[CompositeTerm, ...]
