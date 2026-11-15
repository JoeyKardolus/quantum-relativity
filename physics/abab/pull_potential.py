"""Pull potential: each energy volume's energy spread through the ABAB structure, fading (bible rule 7).

Contract: `pull_potential(state)` returns one potential per volume, >= 0:
  conductance  c = 2 / l_a on each contact (a, b), l_a = size_a ** (1/3): a path from
               energy volume a through a joint to energy volume c then has
               conductance 2 / (l_a + l_c), as between neighbours in the old cells engine,
               so smaller, more packed volumes pass more pull. A joint's own size
               does not enter: it is thin, and its swing must not drive itself
               through the pull (a lone joint keeps its swing)
  solve        (L + GROUND * I) p = source, L the Laplacian of the contacts with those
               conductances, source an energy volume's energy and 0 for a joint (a
               joint does not pull; pull passes through it). GROUND (0.1, as in the
               old cells engine) leaks pull at every volume, so it fades the more volumes
               lie in between
  numerics     conjugate gradients, Jacobi preconditioner, relative residual 1e-12
  guard        RuntimeError when a size is not finite and positive or the solve does
               not converge; the message names the count, worst value or residual
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg

from physics.abab.state import KIND_A, State

GROUND = 0.1


def pull_potential(state: State) -> np.ndarray:
    """Grounded graph-Poisson potential over the contacts with each energy volume's energy as source."""
    _check_sizes(state.size)
    heads, tails = state.contacts[:, 0], state.contacts[:, 1]
    conductance = 2.0 / np.cbrt(state.size[heads])
    count = len(state.kind)
    coupling = sp.coo_matrix((np.concatenate([conductance, conductance]),
                              (np.concatenate([heads, tails]), np.concatenate([tails, heads]))),
                             shape=(count, count)).tocsr()
    total = np.asarray(coupling.sum(axis=1)).ravel() + GROUND
    system = (sp.diags(total) - coupling).tocsr()
    source = np.where(state.kind == KIND_A, state.energy, 0.0)
    potential, failed = cg(system, source, x0=source / GROUND, rtol=1e-12,
                           M=sp.diags(1.0 / total), maxiter=10 * count)
    if failed:
        residual = np.linalg.norm(system @ potential - source) / np.linalg.norm(source)
        raise RuntimeError(f"pull potential did not converge (relative residual {residual:.3g})")
    return potential


def _check_sizes(size: np.ndarray) -> None:
    """Raise when any size is not finite and positive."""
    bad = ~(np.isfinite(size) & (size > 0))
    if bad.any():
        raise RuntimeError(f"pull potential: {int(bad.sum())} volumes have a size that is not finite and "
                           f"positive (lowest {np.nanmin(size):.3g}, highest {np.nanmax(size):.3g})")
