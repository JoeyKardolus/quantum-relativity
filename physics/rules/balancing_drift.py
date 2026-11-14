"""Balancing drift: the drift that keeps even space from shrinking or growing on average.

Contract: `balancing_drift(network, settings, rng)` takes an even reference
network (every connection holding the scene's background energy) and the
scene's settings, and returns the drift d for which one step leaves the total
(so the mean) connection length exactly unchanged. This is a calibration, like
Einstein's cosmological constant balancing the even background pull of a
static universe; like his, the balance is unstable (a slightly larger network
keeps growing, a slightly smaller one keeps shrinking), and scenes report it.
  formula  drift multiplies every length by (1 + d) and keeps energy * length,
           so one full step (rewiring off) gives connection c the new length
           L'_c = (1 + d) * q_c(d) * L_c, where q_c is what flow, pull, joint
           push, repulsion and the energy-length rule do to the drifted
           network. Even space keeps its size when sum_c L'_c = sum_c L_c:
               d = sum_c L_c / sum_c (q_c(d) * L_c) - 1,
           the length-weighted mean fractional shortening s = 1 - Q undone,
           d = 1 / Q - 1 with Q = sum q_c L_c / sum L_c. With energy_length 1
           and every pull equal, q = exp(-(a - push) / 2) for the log-shortening
           a and push of one step, so d ~ (a - push) / 2. q depends weakly on d
           (pull reads the drifted energies), so d is found by fixed-point
           iteration from d = 0 until it moves less than 1e-12 (at most 50
           rounds). On a network whose energies are not even, energy flow's
           one-off evening-out would be counted as a steady pull; use an even one.
Raises ValueError if a joint collapses in the calibration step: the network
is then not even, and no single drift balances it.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.network import Network
from physics.rules.settings import RuleSettings
from physics.rules.step import step

_TOLERANCE = 1e-12
_ROUNDS = 50


def balancing_drift(network: Network, settings: RuleSettings, rng: np.random.Generator) -> float:
    """Drift that makes one step leave the total connection length unchanged."""
    drift = 0.0
    for _ in range(_ROUNDS):
        stepped = step(network, replace(settings, drift=drift, rewiring=0), rng)
        if not np.array_equal(stepped.merged_into, np.arange(network.joint_count)):
            raise ValueError("joints collapsed in the balancing step: the network is not even")
        kept = float(stepped.length.sum() / network.length.sum()) / (1.0 + drift)
        balanced = 1.0 / kept - 1.0
        if abs(balanced - drift) < _TOLERANCE:
            return balanced
        drift = balanced
    return drift
