"""Measurements recover the known dimension and proper time of reference spacetimes.

WHAT IT PINS. physics/measure through its public functions: ball_dimension and
spectral_dimension on spatial reference networks, longest_chain and
myrheim_meyer_dimension on causal reference networks, expansion_rate on a
scale-factor history. Each measurement reads only the network.

HOW THE EXPECTED ANSWER IS DERIVED WITHOUT THE CODE UNDER TEST. The known
answers come from the flat space or spacetime the points were scattered in:
a 2- or 3-torus has dimension 2 or 3; a 1+1 or 3+1 diamond has spacetime
dimension 2 or 4; in 1+1 the longest chain between the tips approaches
sqrt(2 * density) * T (Brightwell and Gregory 1991). The tolerance is 10%,
fixed before running; at these sizes the longest chain sits a few percent
below its limit (finite-size correction ~ N^(1/6)). An exponential history
a = exp(0.01 t) has rate exactly 0.01 and deceleration parameter -1.

ONE FUNCTION. Parametrized over the measurements, seeded.

WHAT IT WAS PROVED AGAINST. Counting two steps per chain link in
longest_chain doubles L, far outside the 10% band.

GUARDS. First version of the physics package, 2026-11-14.
"""
from __future__ import annotations

import numpy as np
import pytest

from physics.measure import (ball_dimension, expansion_rate, longest_chain,
                             myrheim_meyer_dimension, spectral_dimension)
from physics.network import causal_reference, spatial_reference


def _ball(dimension, size):
    network, known = spatial_reference(size, dimension, seed=1)
    return ball_dimension(network, seed=1)["dimension"], known["dimension"]


def _spectral(dimension, size):
    network, known = spatial_reference(size, dimension, seed=1)
    return spectral_dimension(network, seed=1)["dimension"], known["dimension"]


def _chain(dimension, size):
    network, known = causal_reference(size, dimension, seed=0)
    expected = np.sqrt(2 * known["density"]) * known["proper_time"]
    return longest_chain(network, known["bottom"], known["top"]), expected


def _ordering(dimension, size):
    network, known = causal_reference(size, dimension, seed=0)
    return myrheim_meyer_dimension(network)["dimension"], known["dimension"]


def _expansion(_dimension, _size):
    history = np.exp(0.01 * np.arange(50))
    result = expansion_rate(history)
    np.testing.assert_allclose(result["deceleration"][2:-2], -1.0, rtol=1e-3)
    return float(np.median(result["rate"])), 0.01


CASES = [
    ("ball 2D", _ball, 2, 20000),
    ("ball 3D", _ball, 3, 30000),
    ("spectral 2D", _spectral, 2, 20000),
    ("spectral 3D", _spectral, 3, 30000),
    ("chain 1+1", _chain, 2, 2000),
    ("myrheim-meyer 1+1", _ordering, 2, 2000),
    ("myrheim-meyer 3+1", _ordering, 4, 2000),
    ("expansion", _expansion, 0, 0),
]


@pytest.mark.parametrize("label,measure,dimension,size", CASES, ids=[case[0] for case in CASES])
def test_measurements_recover_the_known_dimension_and_proper_time_of_reference_spacetimes(
        label, measure, dimension, size) -> None:
    """The measured value lies within 10% of the known value."""
    measured, known = measure(dimension, size)
    print(f"{label}: measured {measured:.3f} known {known:.3f} size {size}")
    assert abs(measured - known) <= 0.1 * known
