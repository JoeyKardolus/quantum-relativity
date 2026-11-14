"""A display layout follows network distances and leaves the network alone.

WHAT IT PINS. physics/experiments/picture through `display_layout`: drawn
distances between joints track their network distances, and the Network keeps
no positions (its fields are unchanged after a layout).

HOW THE EXPECTED ANSWER IS DERIVED WITHOUT THE CODE UNDER TEST. On a network
scattered in a flat 2D torus, a patch much smaller than the torus is flat, so
its network distances fit in a plane up to the detour of paths along
connections: their correlation with the drawn distances should be near 1.
Network distances here come from scipy's Dijkstra directly.

ONE FUNCTION. This module's one test.

WHAT IT WAS PROVED AGAINST. Taking the two smallest eigenvalues instead of the
largest, which scrambles the layout.

GUARDS. v2 results page, 2026-11-14.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import dijkstra

from physics.experiments.picture import display_layout
from physics.network import spatial_reference


def test_a_display_layout_follows_network_distances_and_leaves_the_network_alone() -> None:
    """Drawn vs network distance correlation above 0.95 on a flat patch; network untouched."""
    network = spatial_reference(3000, 2, seed=2)[0]
    from_zero = dijkstra(network.adjacency(), indices=0)
    patch = np.flatnonzero(from_zero <= 0.15)
    lengths_before = network.length.copy()
    positions = display_layout(network, patch)
    table = dijkstra(network.adjacency(), indices=patch)[:, patch]
    drawn = np.linalg.norm(positions[:, None] - positions[None], axis=2)
    upper = np.triu_indices(len(patch), 1)
    assert positions.shape == (len(patch), 2)
    assert np.corrcoef(table[upper], drawn[upper])[0, 1] > 0.95
    np.testing.assert_array_equal(network.length, lengths_before)
    assert not hasattr(network, "positions")
