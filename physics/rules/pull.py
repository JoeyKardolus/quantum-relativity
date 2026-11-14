"""Pull: how hard each joint is pulled, from energy volume spread through the connections.

Contract: `pull(network, energy, length)` returns one pull value per joint
(>= 0), using no coordinates (bible rule 7):
  source   each joint's region holds energy E_j = 1/2 * sum of its connections'
           energy and takes up volume V_j = 1/2 * sum of their lengths; with
           length = LENGTH_SCALE / energy, more energy means a smaller, more
           packed region. Its source is s_j = E_j / V_j (0 for a joint with no
           connections), so a small packed region pulls harder.
  spread   (L + I / LENGTH_SCALE) p = s, where L is the graph Laplacian with
           conductance 1 / length on each connection (shorter = closer) and
           every joint leaks to ground with conductance 1 / LENGTH_SCALE. The
           ground makes the system solvable (symmetric positive definite) and
           makes the pull fade the more connections lie between a joint and a
           packed region; p >= 0 wherever s >= 0. Solved exactly with a sparse
           direct solver.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from physics.network import LENGTH_SCALE, Network


def pull(network: Network, energy: np.ndarray, length: np.ndarray) -> np.ndarray:
    """Per-joint pull: the grounded graph-Poisson spread of energy per volume."""
    return spsolve(_grounded_laplacian(network, length), _source(network, energy, length))


def _source(network: Network, energy: np.ndarray, length: np.ndarray) -> np.ndarray:
    ends = network.pairs.ravel()
    region_energy = np.bincount(ends, np.repeat(energy, 2), network.joint_count) / 2
    region_volume = np.bincount(ends, np.repeat(length, 2), network.joint_count) / 2
    held = region_volume > 0
    return np.divide(region_energy, region_volume, out=np.zeros(network.joint_count), where=held)


def _grounded_laplacian(network: Network, length: np.ndarray) -> sp.csc_matrix:
    conductance = network.adjacency(1.0 / length)
    total = np.asarray(conductance.sum(axis=1)).ravel() + 1.0 / LENGTH_SCALE
    return (sp.diags(total) - conductance).tocsc()
