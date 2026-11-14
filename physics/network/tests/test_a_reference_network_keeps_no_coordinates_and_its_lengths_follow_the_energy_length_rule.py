"""A reference network keeps no coordinates and its lengths follow the energy-length rule.

WHAT IT PINS. physics/network through its public builders: spatial_reference,
causal_reference and random_network return a Network whose only fields are
joint count, connection pairs, energy, length, charge and the causal order
(no place to hide a coordinate), whose connections are well formed, and whose
energy * length equals LENGTH_SCALE on every connection.

HOW THE EXPECTED ANSWER IS DERIVED WITHOUT THE CODE UNDER TEST. Properties that
hold whatever the seed: rule 4 of the bible (energy * length is the one
constant); a causal relation is a strict partial order (irreflexive,
antisymmetric, transitive) and the diamond's bottom tip precedes every other
event and the top tip follows every other; a k-nearest network gives every
joint at least k connections; a random network with mean degree m on n joints
has n*m/2 connections.

ONE FUNCTION. Parametrized over the three builders.

WHAT IT WAS PROVED AGAINST. Breaking the energy-length rule to
LENGTH_SCALE * value makes energy * length = length^2, not LENGTH_SCALE.

GUARDS. First version of the physics package, 2026-11-14.
"""
from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from physics.network import LENGTH_SCALE, causal_reference, random_network, spatial_reference

_FIELDS = {"joint_count", "pairs", "energy", "length", "charge", "causal"}


def _spatial():
    network, known = spatial_reference(3000, 2, neighbours=8, seed=3)
    assert known["dimension"] == 2
    assert network.degree().min() >= 8
    return network


def _causal():
    network, known = causal_reference(400, 2, seed=3)
    relation = network.causal.toarray()
    assert not relation.diagonal().any()
    assert not (relation & relation.T).any()
    closure = (relation.astype(int) @ relation.astype(int)) > 0
    assert not (closure & ~relation).any()
    others = np.arange(2, network.joint_count)
    assert relation[known["bottom"], others].all() and relation[others, known["top"]].all()
    assert relation[network.pairs[:, 0], network.pairs[:, 1]].all()
    return network


def _random():
    network = random_network(1000, mean_degree=6, seed=3)
    assert network.connection_count == 3000
    return network


@pytest.mark.parametrize("build", [_spatial, _causal, _random], ids=["spatial", "causal", "random"])
def test_a_reference_network_keeps_no_coordinates_and_its_lengths_follow_the_energy_length_rule(build) -> None:
    """Only relational fields, well-formed connections, and energy * length = LENGTH_SCALE."""
    network = build()
    assert {field.name for field in fields(network)} == _FIELDS
    assert not hasattr(network, "__dict__") or set(vars(network)) <= _FIELDS
    pairs = network.pairs
    assert (pairs[:, 0] != pairs[:, 1]).all()
    unordered = np.sort(pairs, axis=1)
    assert len(np.unique(unordered, axis=0)) == len(pairs)
    np.testing.assert_allclose(network.energy * network.length, LENGTH_SCALE, rtol=1e-12)
    assert len(network.charge) == network.joint_count and not network.charge.any()
