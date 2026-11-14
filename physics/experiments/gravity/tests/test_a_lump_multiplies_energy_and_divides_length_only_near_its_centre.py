"""A lump multiplies energy and divides length only near its centre.

WHAT IT PINS. physics/experiments/gravity through `add_lumps`: connections with
an end within network distance `radius` of a centre get energy x (1 + a) and
length / (1 + a); every other connection is untouched; energy x length stays
the same everywhere.

HOW THE EXPECTED ANSWER IS DERIVED WITHOUT THE CODE UNDER TEST. The joints near
the centre are found here with scipy's Dijkstra on the length-weighted network,
independently of the code; the factor 1 + a is the lump's definition.

ONE FUNCTION. This module's one test.

WHAT IT WAS PROVED AGAINST. add_lumps leaving the length unchanged, which breaks
the energy-length rule at the start.

GUARDS. First version of the physics package, 2026-11-14.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import dijkstra

from physics.experiments.gravity.lumps import add_lumps
from physics.network import spatial_reference


def test_a_lump_multiplies_energy_and_divides_length_only_near_its_centre() -> None:
    """Touched connections scaled by 1 + a, the rest equal, energy x length kept."""
    network = spatial_reference(1500, 3, seed=4)[0]
    radius = 2.0 * network.length.mean()
    near = dijkstra(network.adjacency(), indices=[0]).ravel() <= radius
    touched = near[network.pairs[:, 0]] | near[network.pairs[:, 1]]
    after = add_lumps(network, [0], radius, 4.0)
    assert 0 < touched.sum() < network.connection_count
    np.testing.assert_allclose(after.energy[touched], network.energy[touched] * 5.0)
    np.testing.assert_allclose(after.length[touched], network.length[touched] / 5.0)
    np.testing.assert_array_equal(after.energy[~touched], network.energy[~touched])
    np.testing.assert_allclose(after.energy * after.length, network.energy * network.length)
