"""The balancing drift keeps an even 3D network the same size under all rules.

WHAT IT PINS. physics/rules balancing_drift and all_rules_on: on an even 3D
network (every connection holding the same energy), the drift balancing_drift
returns, run with every other rule all_rules_on switches on, keeps the mean
connection length within 1% of its start over 20 steps; all_rules_on leaves
rewiring and drift at 0 and switches every other rule on.

HOW THE EXPECTED ANSWER IS DERIVED WITHOUT THE CODE UNDER TEST. On an even
network every joint's source and so its pull is the same, so pull and push
shorten every connection by the same factor; a drift that undoes one step's
shortening undoes every step's. Without drift the same run must shrink by more
than 1%, so the check can fail.

GUARDS. v2 balancing drift (owner, 2026-11-14).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.network import Network, spatial_reference
from physics.rules import all_rules_on, balancing_drift, pull, step

STEPS = 20


def _even_network():
    base, _ = spatial_reference(800, 3, seed=3)
    return Network.from_lengths(base.joint_count, base.pairs,
                                np.full(base.connection_count, base.length.mean()))


def _size_after(network, settings, rng):
    current = network
    for _ in range(STEPS):
        current = step(current, settings, rng)
    return current.length.mean() / network.length.mean()


def test_balancing_drift_keeps_even_space_the_same_size():
    network = _even_network()
    rng = np.random.default_rng(5)
    background = float(np.median(pull(network, network.energy, network.length)))
    settings = all_rules_on(attraction=0.01 / background)
    assert settings.rewiring == 0 and settings.drift == 0
    assert all(getattr(settings, name) > 0 for name in
               ("energy_length", "energy_flow", "attraction", "joint_push", "push_limit",
                "repulsion"))
    drift = balancing_drift(network, settings, rng)
    assert drift > 0
    assert abs(_size_after(network, replace(settings, drift=drift), rng) - 1) < 0.01
    assert _size_after(network, settings, rng) < 0.99
