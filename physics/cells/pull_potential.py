"""Pull potential: each energy volume's energy spread over the joint graph (bible rule 7).

Contract: `pull_potential(grid)` returns one potential per energy volume, >= 0:
  conductance  c_ab = 2 / (l_a + l_b) on joint (a, b), l = volume ** (1/3): a
               smaller, more packed pair of volumes sits closer and passes more pull.
  solve        (L + GROUND * I) p = energy, L the graph Laplacian with those
               conductances. GROUND (0.1) leaks pull at every volume, so the
               potential fades the more volumes lie in between (screening length
               about sqrt(4 / GROUND) ~ 6 volumes on a 4-joint grid) and the system
               is symmetric positive definite. The full energy is the source, not
               its unevenness: even space feels a background pull of
               energy / GROUND everywhere, which drift balances.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from physics.cells.grid import Grid

GROUND = 0.1


def pull_potential(grid: Grid) -> np.ndarray:
    """Grounded graph-Poisson potential with each energy volume's energy as source."""
    side = np.cbrt(grid.volume)
    heads, tails = grid.joints[:, 0], grid.joints[:, 1]
    conductance = 2.0 / (side[heads] + side[tails])
    count = grid.cell_count
    coupling = sp.coo_matrix((np.concatenate([conductance, conductance]),
                              (np.concatenate([heads, tails]), np.concatenate([tails, heads]))),
                             shape=(count, count)).tocsr()
    total = np.asarray(coupling.sum(axis=1)).ravel() + GROUND
    return spsolve((sp.diags(total) - coupling).tocsc(), grid.energy)
