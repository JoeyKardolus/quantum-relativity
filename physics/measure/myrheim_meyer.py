"""Myrheim-Meyer dimension of a causal network from its ordering fraction.

Contract: `myrheim_meyer_dimension(network, joints)` counts the related pairs R
among `joints` (default: all; they should form an interval, e.g. a sprinkled
diamond), takes the ordering fraction f = R / C(n, 2), and solves
f = Gamma(d+1) Gamma(d/2) / (2 Gamma(3d/2)) for the spacetime dimension d
(f = 1/2 at d = 2, 1/10 at d = 4). Returns {"dimension", "ordering_fraction"}.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln

from physics.network import Network


def myrheim_meyer_dimension(network: Network, joints: np.ndarray | None = None) -> dict:
    """Spacetime dimension that a flat interval with this ordering fraction would have."""
    relation = network.causal
    if relation is None:
        raise ValueError("myrheim_meyer_dimension needs a causal network")
    if joints is not None:
        relation = relation[joints][:, joints]
    size = relation.shape[0]
    fraction = relation.nnz / (size * (size - 1) / 2)
    dimension = brentq(lambda dim: _flat_fraction(dim) - fraction, 1.0, 12.0)
    return {"dimension": float(dimension), "ordering_fraction": float(fraction)}


def _flat_fraction(dimension: float) -> float:
    return float(np.exp(gammaln(dimension + 1) + gammaln(dimension / 2)
                        - gammaln(1.5 * dimension)) / 2)
