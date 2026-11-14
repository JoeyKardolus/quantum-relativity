"""Joint push stays under its limit, and joint collapse keeps all energy.

WHAT IT PINS. physics/rules through step and collapse (bible rule 8):
  push      push(squeeze) follows joint_push * squeeze and never passes
            push_limit; through step, the stretch push adds to a surviving
            connection, log(length with push / length without), is positive and
            never more than push_limit;
  collapse  joints past the limit give way: joint_count drops, merged_into maps
            every old joint to a joint of the new network and every new joint is
            reached; collapse keeps total energy exactly, leaves no self or
            double connection, and changed connections re-follow
            energy * length = LENGTH_SCALE; with no joint past the limit,
            merged_into is the identity.

HOW THE EXPECTED ANSWER IS DERIVED WITHOUT THE CODE UNDER TEST. Rule 8: a
joint pushes back up to a limit, past it the joint gives way keeping all
energy; the push of each end is at most push_limit, so their mean is too.
Merging joints of a connected network can only lower the joint count.

ONE FUNCTION. Parametrized over the push cap, push through step (on
connections whose joints did not merge), collapse through step, and collapse directly (each joint
that gives way ends up with the far end of its shortest connection).

WHAT IT WAS PROVED AGAINST. Dropping the loop-energy share in collapse loses
the energy of connections between merged joints, so the total falls.

GUARDS. v2 joint push and joint collapse, 2026-11-14.
"""
from __future__ import annotations

import numpy as np
import pytest

from physics.network import LENGTH_SCALE, Network, random_network
from physics.rules import RuleSettings, collapse, push, step


def _lumpy():
    base = random_network(600, mean_degree=6, seed=1)
    energy = np.where((base.pairs < 5).any(axis=1), 4.0, 1.0)
    return Network.from_lengths(base.joint_count, base.pairs, 1.0 / energy)


def _well_formed(network):
    unordered = np.sort(network.pairs, axis=1)
    assert (unordered[:, 0] != unordered[:, 1]).all()
    assert len(np.unique(unordered, axis=0)) == len(unordered)


def _push_is_capped():
    squeeze = np.linspace(0.0, 0.05, 51)
    given = push(squeeze, RuleSettings(joint_push=2.0, push_limit=0.03))
    assert given.max() == 0.03 and (given <= 0.03).all()
    np.testing.assert_allclose(given[squeeze <= 0.015], 2.0 * squeeze[squeeze <= 0.015])


def _push_stretches_surviving_connections_within_the_limit():
    network = _lumpy()
    limit = 0.015
    pulled = step(network, RuleSettings(attraction=0.01, energy_length=0.0))
    pushed = step(network, RuleSettings(attraction=0.01, energy_length=0.0, joint_push=1.0,
                                        push_limit=limit))
    assert pushed.joint_count < network.joint_count
    alone = np.bincount(pushed.merged_into)[pushed.merged_into] == 1
    untouched = alone[network.pairs[:, 0]] & alone[network.pairs[:, 1]]
    new_index = {tuple(pair): index for index, pair in enumerate(np.sort(pushed.pairs, axis=1))}
    ends = np.sort(pushed.merged_into[network.pairs[untouched]], axis=1)
    rows = [new_index[tuple(pair)] for pair in ends]
    stretch = np.log(pushed.length[rows] / pulled.length[untouched])
    assert (stretch > 0).all() and stretch.max() <= limit + 1e-12


def _collapse_through_step_keeps_bookkeeping():
    network = _lumpy()
    after = step(network, RuleSettings(attraction=0.01, joint_push=1.0, push_limit=0.015))
    assert after.joint_count < network.joint_count
    assert len(after.merged_into) == network.joint_count
    assert set(after.merged_into.tolist()) == set(range(after.joint_count))
    _well_formed(after)
    calm = step(network, RuleSettings(attraction=0.01, joint_push=1.0, push_limit=1.0))
    np.testing.assert_array_equal(calm.merged_into, np.arange(network.joint_count))


def _collapse_keeps_all_energy():
    network = _lumpy()
    giving_way = np.zeros(network.joint_count, bool)
    giving_way[:40] = True
    after = collapse(network, giving_way)
    assert after.joint_count < network.joint_count
    assert np.isclose(after.energy.sum(), network.energy.sum(), rtol=1e-12, atol=0)
    _well_formed(after)
    np.testing.assert_allclose(after.energy * after.length, LENGTH_SCALE, rtol=1e-12)
    for joint in range(40):
        nearest = np.argmin(np.where((network.pairs == joint).any(axis=1), network.length, np.inf))
        other = network.pairs[nearest][network.pairs[nearest] != joint][0]
        assert after.merged_into[joint] == after.merged_into[other]


CASES = [_push_is_capped, _push_stretches_surviving_connections_within_the_limit,
         _collapse_through_step_keeps_bookkeeping, _collapse_keeps_all_energy]


@pytest.mark.parametrize("case", CASES, ids=[case.__name__.strip("_") for case in CASES])
def test_joint_push_stays_under_its_limit_and_joint_collapse_keeps_all_energy(case) -> None:
    """The documented push cap and collapse bookkeeping."""
    case()
