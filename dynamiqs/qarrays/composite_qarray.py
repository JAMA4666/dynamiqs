from __future__ import annotations

from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, Device
from jaxtyping import ArrayLike
from qutip import Qobj

from .layout import Layout
from .qarray import QArray, QArrayLike

__all__ = ['CompositeTerm', 'CompositeQArray', 'composite_term', 'composite']


class CompositeTerm(eqx.Module):
    """One separable term in a CompositeQArray.

    Represents a single term of the form

        coeff * ⊗ A ⊗ B ⊗ C

    Only the active non-identity local operators are stored. The location of
    each active operator in the full tensor-product Hilbert space is stored in
    `locs`.

    Example
    -------
    For dims=(2, 3, 4), the term

        0.5 * A ⊗ I_3 ⊗ B

    is stored as

        CompositeTerm(coeff=0.5, operators=(A, B), locs=(0, 2))
    """

    coeff: ArrayLike
    """Scalar or broadcastable array multiplying this term."""

    operators: tuple[QArray, ...]
    """Active non-identity operators appearing in this term."""

    locs: tuple[int, ...]
    """Subsystem location of each active operator."""

    # ==========================================
    # === Initialization / validation
    # ==========================================

    def __check_init__(self):
        # === Structural integrity ===
        if len(self.operators) != len(self.locs):
            raise ValueError(
                'CompositeTerm must have same number of operators and locs, '
                f'but got {len(self.operators)} operators and {len(self.locs)} locs.'
            )
        
        # === No duplicate locations ===
        if len(set(self.locs)) != len(self.locs):
            raise ValueError(
                'CompositeTerm cannot have two operators on the same subsystem, '
                f'but got duplicate locs={self.locs}.'
            )
        
        # === All operators are square ===
        for i, op in enumerate(self.operators):
            if op.shape[-1] != op.shape[-2]:
                raise ValueError(
                    f'All operators in CompositeTerm must be square. '
                    f'Operator {i} has shape {op.shape}.'
                )
        
        # === Broadcastable batch shapes ===
        try:
            bshapes = [jnp.shape(self.coeff)]
            bshapes += [op.shape[:-2] for op in self.operators]
            jnp.broadcast_shapes(*bshapes)
        except ValueError as e:
            raise ValueError(
                'CompositeTerm coeff and operators must have broadcastable batch shapes.'
            ) from e
        
        # === Optional: all operators on same device (helps catch errors early) ===
        devices = set()
        for op in self.operators:
            devices.update(op.devices())
        if len(devices) > 1:
            raise ValueError(
                f'All operators in CompositeTerm must be on the same device, '
                f'but got devices {devices}.'
            )

    def validate_for_dims(self, dims: tuple[int, ...]) -> None:
        """Check that locs are in range and operator dims match `dims[loc]`."""
        pass

    def sorted(self) -> CompositeTerm:
        """Return a canonicalized term with operators ordered by ascending loc."""
        pass

    # ==========================================
    # === Properties
    # ==========================================

    @property
    def n_active(self) -> int:
        """Number of active (non-identity) operators in this term."""
        pass

    @property
    def dtype(self) -> jnp.dtype:
        """Promoted dtype across coeff and all operators."""
        pass

    @property
    def bshape(self) -> tuple[int, ...]:
        """Broadcasted batch shape across coeff and all operators."""
        pass

    @property
    def mT(self) -> CompositeTerm:
        """Transpose each operator (coeff unchanged)."""
        pass

    # ==========================================
    # === Per-term transformations
    # ==========================================

    def conj(self) -> CompositeTerm:
        """Conjugate coeff and each operator."""
        pass

    def dag(self) -> CompositeTerm:
        """Dagger (transpose + conjugate) each operator."""
        pass

    def scale(self, y: ArrayLike) -> CompositeTerm:
        """Multiply coeff by a scalar."""
        pass

    # ==========================================
    # === Per-term computation
    # ==========================================

    def trace(self, dims: tuple[int, ...]) -> Array:
        """Trace of this term.

        trace = coeff * prod_i tr(op_i) * prod_{j not in locs} d_j
        """
        pass

    def _apply_dense(self, rho_tensor: Array) -> Array:
        """Apply this term to rho (in tensor form) via dense tensordot contractions.

        Implements the sequential contraction:
            for op, axis in zip(operators, locs):
                rho = jnp.tensordot(op, rho, axes=(1, axis))
                rho = jnp.moveaxis(rho, 0, axis)
        """
        pass

    def _apply_sparse(self, rho_tensor: Array) -> Array:
        """Apply this term to rho via sparse-dense matmul on contracted axes."""
        pass

    # ==========================================
    # === Per-term materialization
    # ==========================================

    def devices(self) -> set[Device]:
        """Union of devices across active operators."""
        pass

    def _full_factors_dense(self, dims: tuple[int, ...]) -> tuple[QArray, ...]:
        """Return the full list of factors as dense QArrays (identities filled in)."""
        pass

    def _full_factors_sparsedia(self, dims: tuple[int, ...]) -> tuple[QArray, ...]:
        """Return the full list of factors as sparse-dia QArrays (identities filled in)."""
        pass

    def asdense(self, dims: tuple[int, ...]) -> QArray:
        """Build the full Kronecker product operator for this term as a dense QArray."""
        pass

    def assparsedia(self, dims: tuple[int, ...]) -> QArray:
        """Build the full Kronecker product operator for this term as a sparsedia QArray."""
        pass


class CompositeQArray(QArray):
    """Lazy sum of separable tensor-product terms.

    Represents an operator of the form

        sum_k coeff_k * O_{k,0} ⊗ O_{k,1} ⊗ ... ⊗ O_{k,N-1}

    where each term stores only its active non-identity local operators.
    Optimized separable matmul applies each term via per-axis contractions on
    the tensor form of the right-hand-side operator, avoiding materialization
    of the full Kronecker product.
    """

    terms: tuple[CompositeTerm, ...]
    """Terms in the lazy sum."""

    # ==========================================
    # === Initialization
    # ==========================================

    def __check_init__(self):
        # Validate:
        # - len(terms) >= 1
        # - each term.validate_for_dims(self.dims)
        # - all term.bshape are mutually broadcastable
        # - all operators across all terms share a single device
        # NOTE: do NOT mutate self.terms here (equinox modules are frozen during
        # tracing). Canonicalization (sorting per-term locs) should be done in
        # the factory functions `composite_term` / `composite`.
        pass

    # ==========================================
    # === Properties
    # ==========================================

    @property
    def nterms(self) -> int:
        """Number of terms in the lazy sum."""
        pass

    @property
    def _all_operators(self) -> tuple[QArray, ...]:
        """Flattened iterator over every active operator across all terms."""
        pass

    @property
    def dtype(self) -> jnp.dtype:
        pass

    @property
    def layout(self) -> Layout:
        # TODO: consider introducing a dedicated `composite` layout. For now,
        # the layout of a CompositeQArray is ambiguous since individual terms
        # may mix dense and sparse operators.
        pass

    @property
    def shape(self) -> tuple[int, ...]:
        """Broadcast batch shape + (prod(dims), prod(dims))."""
        pass

    @property
    def ndim(self) -> int:
        pass

    @property
    def mT(self) -> CompositeQArray:
        """Transpose each term."""
        pass

    # ==========================================
    # === Core composite-native operations
    # ==========================================

    def conj(self) -> CompositeQArray:
        """Conjugate each term."""
        pass

    def dag(self) -> CompositeQArray:
        """Dagger each term."""
        pass

    def trace(self) -> Array:
        """Sum of per-term traces (no materialization)."""
        pass

    def isherm(self, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Sufficient (not necessary) check: real coeffs + all operators Hermitian."""
        pass

    # ==========================================
    # === Arithmetic (composite-aware)
    # ==========================================

    def __mul__(self, y: ArrayLike) -> CompositeQArray:
        """Scale every term's coefficient by the scalar `y`."""
        pass

    def __rmul__(self, y: ArrayLike) -> CompositeQArray:
        pass

    def __add__(self, y: QArrayLike) -> QArray:
        """Composite + Composite → merge terms; Composite + other → materialize."""
        pass

    def __radd__(self, y: QArrayLike) -> QArray:
        pass

    # ==========================================
    # === Matrix multiplication (the key feature)
    # ==========================================

    def __matmul__(self, y: QArrayLike) -> QArray | Array:
        """Composite @ Composite → lazy term-product; Composite @ rho → separable matmul."""
        pass

    def __rmatmul__(self, y: QArrayLike) -> QArray | Array:
        """For y @ Composite (y not composite): materialize fallback."""
        pass

    def _separable_matmul_dense(self, rho: QArray) -> QArray:
        """Apply self @ rho using dense per-axis tensordot contractions.

        Reshapes rho to tensor form (..., d_0, ..., d_{N-1}, D), applies each
        term via `CompositeTerm._apply_dense`, weights by coeff, sums, and
        reshapes back.
        """
        pass

    def _separable_matmul_sparse(self, rho: QArray) -> QArray:
        """Apply self @ rho using the sparse-dense matmul kernel for each axis."""
        pass

    # ==========================================
    # === Tensor product
    # ==========================================

    def __and__(self, y: QArray) -> QArray:
        """Tensor product: extend each term with `y` on the new subsystem(s)."""
        pass

    # ==========================================
    # === Materialize-and-delegate fallbacks
    # ==========================================

    def norm(self, *, psd: bool = False) -> Array:
        pass

    def powm(self, n: int) -> QArray:
        """n==0 → identity; n==1 → self; n>=2 → repeated self @ self."""
        pass

    def expm(self, *, max_squarings: int = 16) -> QArray:
        pass

    def sum(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        pass

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        pass

    def _reshape_unchecked(self, *shape: int) -> QArray:
        pass

    def broadcast_to(self, *shape: int) -> QArray:
        pass

    def _eig(self) -> tuple[Array, QArray]:
        pass

    def _eigh(self) -> tuple[Array, Array]:
        pass

    def _eigvals(self) -> Array:
        pass

    def _eigvalsh(self) -> Array:
        pass

    def ptrace(self, *keep: int) -> QArray:
        """Partial trace. Could be done term-by-term; fallback to materialize for now."""
        pass

    def addscalar(self, y: ArrayLike) -> QArray:
        pass

    def elmul(self, y: QArrayLike) -> QArray:
        pass

    def elpow(self, power: int) -> QArray:
        pass

    # ==========================================
    # === Conversion methods
    # ==========================================

    def to_jax(self) -> Array:
        pass

    def to_qutip(self) -> Qobj | list[Qobj]:
        pass

    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # noqa: ANN001
        pass

    def asdense(self) -> QArray:
        """Materialize: sum over terms of their dense Kronecker product."""
        pass

    def assparsedia(self, offsets: tuple[int, ...] | None = None) -> QArray:
        """Materialize: sum over terms of their sparse-dia Kronecker product."""
        pass

    def devices(self) -> set[Device]:
        pass

    def block_until_ready(self) -> CompositeQArray:
        pass

    # ==========================================
    # === Indexing
    # ==========================================

    def __getitem__(self, key) -> QArray:
        """Indexing into the batch dimensions. Falls back to materialization."""
        pass

    # ==========================================
    # === Repr
    # ==========================================

    def __repr__(self) -> str:
        pass


# ==========================================
# === Module-level helpers
# ==========================================


def _mul_terms(
    left: CompositeTerm, right: CompositeTerm, dims: tuple[int, ...]
) -> CompositeTerm:
    """Lazy product of two terms acting on the same composite space.

    On each subsystem: compose operators if both terms are active, otherwise
    keep the single active operator. Coefficients multiply.
    """
    pass


def _tensor_terms(
    left: CompositeTerm, right: CompositeTerm, shift: int
) -> CompositeTerm:
    """Lazy tensor product of two terms (concatenate operators, shift right locs)."""
    pass

def composite_term(
    operators: tuple[QArray, ...] | list[QArray],
    locs: tuple[int, ...] | list[int],
    *,
    dims: tuple[int, ...] | list[int],
    coeff: ArrayLike = 1,
    vectorized: bool = False,
) -> CompositeQArray:
    """Factory: build a single-term CompositeQArray."""
    pass


def composite(
    terms: tuple[CompositeTerm, ...] | list[CompositeTerm],
    *,
    dims: tuple[int, ...] | list[int],
    vectorized: bool = False,
) -> CompositeQArray:
    """Factory: build a multi-term CompositeQArray (sorts each term canonically)."""
    pass
