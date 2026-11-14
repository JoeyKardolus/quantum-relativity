"""Spectral dimension from the return probability of a random walk.

Contract: `spectral_dimension(network, sources, steps, seed, window)` runs an
exact lazy random walk (stay with probability 1/2, else step along a
connection with probability proportional to its energy, i.e. its conductance)
from `sources` random joints for `steps` steps, averages the probability P(t)
of being back at the start, and fits P(t) ~ t^(-ds/2) over the step window
`window` (a fixed default, not tuned per network). Returns {"dimension",
"steps", "return_probability", "window"}.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from physics.network import Network


def spectral_dimension(network: Network, sources: int = 64, steps: int = 200,
                       seed: int = 0, window: tuple[int, int] = (20, 200)) -> dict:
    """Fitted ds of return probability P(t) ~ t^(-ds/2) for a lazy energy-weighted walk."""
    rng = np.random.default_rng(seed)
    chosen = rng.choice(network.joint_count, size=min(sources, network.joint_count),
                        replace=False)
    weights = network.adjacency(network.energy)
    strength = np.asarray(weights.sum(axis=1)).ravel()
    strength[strength == 0] = 1.0
    move = sp.diags(1.0 / strength) @ weights
    walk = (0.5 * sp.identity(network.joint_count) + 0.5 * move).T.tocsr()
    density = np.zeros((network.joint_count, len(chosen)))
    density[chosen, np.arange(len(chosen))] = 1.0
    returns = np.empty(steps)
    for step in range(steps):
        density = walk @ density
        returns[step] = density[chosen, np.arange(len(chosen))].mean()
    times = np.arange(1, steps + 1)
    fit = (times >= window[0]) & (times <= window[1])
    slope, _ = np.polyfit(np.log(times[fit]), np.log(returns[fit]), 1)
    return {"dimension": float(-2 * slope), "steps": times,
            "return_probability": returns, "window": window}
