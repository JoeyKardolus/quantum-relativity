"""The even version spreads a scene's energy evenly over the same connections.

WHAT IT PINS. physics/experiments/expansion through `even_version`: the same
joints and connections, every connection holding the scene's mean energy, the
total energy unchanged, energy * length = 1, and every joint the same pull.

HOW THE EXPECTED ANSWER IS DERIVED WITHOUT THE CODE UNDER TEST. Spreading a
total E over C connections gives each E / C; on an even network every joint's
source is energy^2 (pull.py), and a constant pull solves the grounded Poisson
equation with that source, so the pull is (E / C)^2 at every joint.
"""
from __future__ import annotations

import numpy as np

from physics.experiments.expansion.clustered_reference import clustered_reference
from physics.experiments.expansion.even_version import even_version
from physics.rules import pull


def test_the_even_version_spreads_a_scenes_energy_evenly_over_the_same_connections() -> None:
    """Same pairs, equal energies, same total, energy-length kept, pull (E/C)^2 everywhere."""
    network, _ = clustered_reference(600, 2, 100, 0.03, 8, seed=2)
    even = even_version(network)
    mean_energy = network.energy.sum() / network.connection_count
    assert even.joint_count == network.joint_count
    assert np.array_equal(even.pairs, network.pairs)
    assert np.allclose(even.energy, mean_energy)
    assert np.isclose(even.energy.sum(), network.energy.sum())
    assert np.allclose(even.energy * even.length, 1.0)
    assert np.allclose(pull(even, even.energy, even.length), mean_energy ** 2)
