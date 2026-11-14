"""The far-background ruler does not count a shrink of all space as an approach.

WHAT IT PINS. physics/experiments/gravity through `track_lumps` and
`lump_curves`: when every connection of a lumped network is shortened by the
same factor (all of space shrinks), the raw lump-centre distance shrinks by
that factor, the far-background mean length shrinks by it too, and the
background-relative distance stays exactly 1.

HOW THE EXPECTED ANSWER IS DERIVED WITHOUT THE CODE UNDER TEST. Shortening
every length by a factor scales every network distance and every mean length
by it, and leaves the energy pattern (so the lump centres) as it was; the
ratio of the two is unchanged.

GUARDS. v3 gravity test, all rules on with the balancing drift (owner, 2026-11-14).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.experiments.gravity.curves import lump_curves
from physics.experiments.gravity.lumps import add_lumps
from physics.experiments.gravity.tracking import track_lumps
from physics.measure import distances, mean_length
from physics.network import Network, spatial_reference

SHRINK = 0.6


def test_the_far_background_ruler_does_not_count_a_shrink_of_all_space_as_an_approach() -> None:
    """Uniform shrink: raw distance and background both x SHRINK, relative distance 1."""
    scattered = spatial_reference(1500, 3, seed=4)[0]
    even = Network.from_lengths(scattered.joint_count, scattered.pairs,
                                np.full(scattered.connection_count, scattered.length.mean()))
    unit = mean_length(even)
    from_first = distances(even, [0])[0]
    joints = (0, int(np.argmin(np.abs(from_first - 5 * unit))))
    ruler = distances(even, list(joints)) / unit
    far = np.flatnonzero(ruler.min(axis=0) >= 3)
    lumped = add_lumps(even, list(joints), 1.5 * unit, 4.0)
    shrunk = replace(lumped, length=lumped.length * SHRINK)
    records = [track_lumps(network, joints, far) for network in (lumped, shrunk)]
    for record in records:
        record["origin"] = [np.array([lump["centre"]]) for lump in record["lumps"]]
    curves = lump_curves(records, ruler, unit)
    assert len(far) > 0
    np.testing.assert_allclose(curves["background"], [1.0, SHRINK])
    np.testing.assert_allclose(curves["raw"][1] / curves["raw"][0], SHRINK)
    np.testing.assert_allclose(curves["centre"], [1.0, 1.0])
