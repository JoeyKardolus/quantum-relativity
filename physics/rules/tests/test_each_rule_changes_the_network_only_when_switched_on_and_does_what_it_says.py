"""Each rule changes the network only when switched on and does what it says.

WHAT IT PINS. physics/rules through step and RuleSettings: default settings
leave a consistent network alone; drift lengthens every connection by exactly
(1 + drift) per step and lowers its energy by the same factor; energy flow
conserves total energy and evens it out; attraction (pull from energy volume)
shortens the connections in the energy-dense part more; repulsion stretches only the charged joint's
connections and lowers their energy; rewiring keeps the connection count and
total energy, makes no self or double connections and lowers the action.

HOW THE EXPECTED ANSWER IS DERIVED WITHOUT THE CODE UNDER TEST. From the rule
definitions in the bible (rules 4, 6, 7, 8) on small hand-built or seeded
networks: (1.01)^10 is the drift factor after 10 steps; a diffusion conserves
its total and cannot raise the spread; a joint with no charge has nothing to push.

ONE FUNCTION. Parametrized over the rules.

WHAT IT WAS PROVED AGAINST. Halving the drift's lengthening makes mean length
grow by less than (1.01)^10.

GUARDS. First version of the physics package, 2026-11-14.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from physics.network import Network, random_network, spatial_reference
from physics.rules import RuleSettings, step


def _base():
    return spatial_reference(2000, 3, seed=2)[0]


def _default_leaves_alone():
    network = _base()
    after = step(network)
    np.testing.assert_allclose(after.length, network.length, rtol=1e-12)
    np.testing.assert_allclose(after.energy, network.energy, rtol=1e-12)
    np.testing.assert_array_equal(after.merged_into, np.arange(network.joint_count))


def _drift_lengthens():
    network = after = _base()
    for _ in range(10):
        after = step(after, RuleSettings(drift=0.01))
    np.testing.assert_allclose(after.length, network.length * 1.01 ** 10, rtol=1e-9)
    np.testing.assert_allclose(after.energy, network.energy / 1.01 ** 10, rtol=1e-9)
    assert after.length.mean() > network.length.mean()


def _flow_conserves():
    network = _base()
    after = step(network, RuleSettings(energy_flow=0.5, energy_length=0.0))
    assert np.isclose(after.energy.sum(), network.energy.sum(), rtol=1e-12)
    assert after.energy.std() < network.energy.std()


def _attraction_pulls_dense_regions():
    network = random_network(500, mean_degree=6, seed=2)
    dense = np.zeros(network.joint_count, bool)
    dense[:100] = True
    hot = dense[network.pairs[:, 0]] & dense[network.pairs[:, 1]]
    cold = ~dense[network.pairs[:, 0]] & ~dense[network.pairs[:, 1]]
    energy = np.where(hot, 3.0, 1.0)
    network = Network.from_lengths(network.joint_count, network.pairs, 1.0 / energy)
    after = step(network, RuleSettings(attraction=0.05))
    shrink = after.length / network.length
    assert (shrink < 1).all()
    assert shrink[hot].mean() < shrink[cold].mean()
    assert (after.energy > network.energy).all()


def _repulsion_stretches_own_connections():
    network = _base()
    charge = np.zeros(network.joint_count)
    charge[0] = 1.0
    network = replace(network, charge=charge)
    after = step(network, RuleSettings(repulsion=0.1))
    touched = (network.pairs == 0).any(axis=1)
    assert (after.length[touched] > network.length[touched]).all()
    assert (after.energy[touched] < network.energy[touched]).all()
    np.testing.assert_allclose(after.length[~touched], network.length[~touched], rtol=1e-12)


def _rewiring_lowers_action():
    network = random_network(800, mean_degree=6, seed=2)
    settings = RuleSettings(rewiring=5000)
    after = step(network, settings, np.random.default_rng(2))
    assert after.connection_count == network.connection_count
    assert np.isclose(after.energy.sum(), network.energy.sum())
    unordered = np.sort(after.pairs, axis=1)
    assert (unordered[:, 0] != unordered[:, 1]).all()
    assert len(np.unique(unordered, axis=0)) == len(unordered)
    assert _action(after, settings) < _action(network, settings)
    assert (after.pairs != network.pairs).any()


def _action(network, settings):
    """The documented rewiring action, recomputed from scratch with matrix powers."""
    adjacency = network.adjacency(np.ones(network.connection_count))
    triangles = (adjacency @ adjacency).multiply(adjacency).sum(axis=1).A.ravel() / 2
    degree_term = ((network.degree() - settings.target_degree) ** 2).sum()
    triangle_term = ((triangles - settings.target_triangles) ** 2).sum()
    return degree_term + settings.triangle_weight * triangle_term


CASES = [_default_leaves_alone, _drift_lengthens, _flow_conserves,
         _attraction_pulls_dense_regions, _repulsion_stretches_own_connections,
         _rewiring_lowers_action]


@pytest.mark.parametrize("case", CASES, ids=[case.__name__.strip("_") for case in CASES])
def test_each_rule_changes_the_network_only_when_switched_on_and_does_what_it_says(case) -> None:
    """The rule's documented effect, and nothing else."""
    case()
