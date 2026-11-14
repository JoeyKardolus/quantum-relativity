"""Pull is strongest at a packed lump and fades with network distance.

WHAT IT PINS. physics/rules through step and pull (bible rule 7): on a random
network of equal-energy connections with one packed lump (every connection at
five joints holds four times the energy, so it is four times shorter), the
connections in the lump's neighbourhood shorten more than those far away, and
the mean shortening falls with every extra connection between a connection and
the lump (hop distance measured on the network, no coordinates).

HOW THE EXPECTED ANSWER IS DERIVED WITHOUT THE CODE UNDER TEST. Rule 7: a
smaller, more packed region pulls harder, and pull spreads through the
connections, weaker the more connections lie between. A grounded graph Poisson
solution with a non-negative source peaked at the lump decays away from it.

ONE FUNCTION. One seeded network, checked shell by shell.

WHAT IT WAS PROVED AGAINST. Replacing the spread with the local source alone
(pull = s) leaves shells 1 to 3 equally shortened, which fails the fade check.

GUARDS. v2 attraction from energy volume, 2026-11-14.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import shortest_path

from physics.network import Network, random_network
from physics.rules import RuleSettings, step

_LUMP = 5


def test_pull_is_strongest_at_a_packed_lump_and_fades_with_network_distance() -> None:
    """Shortening falls with hop distance from the lump; near beats far."""
    base = random_network(600, mean_degree=6, seed=1)
    energy = np.where((base.pairs < _LUMP).any(axis=1), 4.0, 1.0)
    network = Network.from_lengths(base.joint_count, base.pairs, 1.0 / energy)
    after = step(network, RuleSettings(attraction=0.01))
    shortening = -np.log(after.length / network.length)
    assert (shortening > 0).all()
    hops = shortest_path(network.adjacency(np.ones(network.connection_count)), unweighted=True,
                         indices=np.arange(_LUMP)).min(axis=0)
    shell = np.minimum(hops[network.pairs[:, 0]], hops[network.pairs[:, 1]])
    means = [shortening[shell == distance].mean() for distance in range(4)]
    assert all(near > far for near, far in zip(means, means[1:])), means
    assert shortening[shell <= 1].mean() > shortening[shell >= 3].mean()
