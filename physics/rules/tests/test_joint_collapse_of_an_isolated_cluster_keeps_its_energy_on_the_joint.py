"""Joint collapse of a cluster with nowhere else to go keeps its energy on the joint.

WHAT IT PINS. physics/rules collapse and physics/network Network.held: when
every joint of an isolated cluster gives way, the merged joint has no
connection left; collapse does not raise, the joint stays as an isolated joint
holding exactly the cluster's energy, total_energy() is unchanged, and a later
collapse elsewhere carries the held energy with its joint.

HOW THE EXPECTED ANSWER IS DERIVED WITHOUT THE CODE UNDER TEST. Bible rule 8:
collapse keeps all energy. An isolated triangle of energies 1, 2, 3 falls into
one joint with no connection, so that joint holds 6; the rest of the network
keeps its own energy.

GUARDS. v2 run stopped on "joint collapse left 1 joint(s) holding energy with
no connection" (2026-11-14).
"""
from __future__ import annotations

import numpy as np

from physics.network import Network
from physics.rules import collapse


def test_joint_collapse_of_an_isolated_cluster_keeps_its_energy_on_the_joint():
    pairs = np.array([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [5, 6]])
    lengths = 1.0 / np.array([1.0, 2.0, 3.0, 1.0, 1.0, 1.0])
    network = Network.from_lengths(7, pairs, lengths)
    total = network.total_energy()
    triangle = np.zeros(7, dtype=bool)
    triangle[:3] = True
    fallen = collapse(network, triangle)
    assert fallen.joint_count == 5
    assert fallen.total_energy() == total
    isolated = fallen.merged_into[0]
    assert fallen.held[isolated] == 6.0
    assert not (fallen.pairs == isolated).any()
    end = np.zeros(5, dtype=bool)
    end[fallen.merged_into[6]] = True
    again = collapse(fallen, end)
    assert again.total_energy() == total
    assert again.held[again.merged_into[isolated]] == 6.0
