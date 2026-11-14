"""Step a network under the rules and measure its dimension as it goes.

Contract: `track(network, settings, steps, checkpoints, seed)` applies
`physics.rules.step` `steps` times and, at each step listed in `checkpoints`
(0 = before any step), records the ball-growth and spectral dimension (seed fixed, default fit
windows), the steps across the network (mean largest distance / median
connection length) and the moves the rewiring accepted since the last record.
A dimension is recorded as None, never as a number, when its measurement has
no valid range, by rules fixed before any run:
  ball      the fit range spans less than a factor 2 in radius (or none);
  spectral  the walk's return probability at the window end is within a factor
            2 of its equilibrium floor 1/N (the walk has already filled the
            network, so the slope says nothing about dimension).
Returns {"step", "ball", "spectral", "across", "accepted"} as lists.
"""
from __future__ import annotations

import numpy as np

from physics.measure import ball_dimension, distances, spectral_dimension
from physics.network import Network
from physics.rules import RuleSettings, step


def track(network: Network, settings: RuleSettings, steps: int,
          checkpoints: list[int], seed: int) -> dict:
    """Dimension, size and rewiring activity over `steps` steps of the rules."""
    rng = np.random.default_rng(seed)
    history = {"step": [], "ball": [], "spectral": [], "across": [], "accepted": []}
    accepted = 0
    for index in range(steps + 1):
        if index in checkpoints:
            history["step"].append(index)
            history["ball"].append(_ball(network, seed))
            history["spectral"].append(_spectral(network, seed))
            history["across"].append(_across(network, seed))
            history["accepted"].append(accepted)
            accepted = 0
        if index < steps:
            stepped = step(network, settings, rng)
            accepted += int(np.any(stepped.pairs != network.pairs, axis=1).sum())
            network = stepped
    return history


def _ball(network: Network, seed: int) -> float | None:
    try:
        result = ball_dimension(network, seed=seed)
    except ValueError:
        return None
    low, high = result["fit_range"]
    return float(result["dimension"]) if high >= 2 * low else None


def _spectral(network: Network, seed: int) -> float | None:
    result = spectral_dimension(network, seed=seed)
    floor = 1.0 / network.joint_count
    return None if result["return_probability"][-1] < 2 * floor else float(result["dimension"])


def _across(network: Network, seed: int) -> float:
    chosen = np.random.default_rng(seed).choice(network.joint_count, size=32, replace=False)
    reach = distances(network, chosen)
    largest = np.nanmax(np.where(np.isfinite(reach), reach, np.nan), axis=1)
    return float(np.mean(largest) / np.median(network.length))
