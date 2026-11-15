"""Energy sets size: lengths, distance, dimension and a sealed region's radius are measured from energy.

A box holding half the start's mean energy is twice as long; the flat reference (every
box equal) measures dimension 3 within 7% between proper radii 8 and 12 (paths along
6 face neighbours; the known bias of that ball's surface terms); on the same boxes with
every energy halved every distance doubles; a sealed ball of known energy measures the
radius of a sphere of its summed length cubed, smaller when more energy packs it.
"""
import math

import numpy as np
import pytest

from physics.fluid import Settings
from physics.fluid.dimension import dimension
from physics.fluid.local_length import local_length
from physics.fluid.proper_distance import proper_distance
from physics.fluid.sealed_radius import sealed_radius


def test_a_thinner_box_is_longer():
    length = local_length(np.array([1.0, 0.5, 4.0, 0.0]), np.array([0.03, 0.015, 0.12, 0.0]), 1.0, 0.03)
    assert np.allclose(length[:3], [1.0, 2.0, 0.25])
    assert math.isinf(length[3])


def test_the_flat_reference_measures_three_dimensions_and_distance_scales_with_length():
    side = 31
    occupied = np.ones((side,) * 3, dtype=bool)
    origin = (15, 15, 15)
    flat = local_length(np.ones((side,) * 3), np.full((side,) * 3, 0.03), 1.0, 0.03)
    distance = proper_distance(flat, occupied, origin)
    assert distance[origin] == 0.0
    assert distance[20, 15, 15] == pytest.approx(5.0)
    assert dimension(distance, flat, 8.0, 12.0) == pytest.approx(3.0, rel=0.07)
    thin = local_length(np.full((side,) * 3, 0.5), np.full((side,) * 3, 0.015), 1.0, 0.03)
    assert np.allclose(proper_distance(thin, occupied, origin), 2.0 * distance)


def test_a_sealed_ball_measures_the_radius_of_its_lengths():
    settings = Settings.all_on()
    axis = np.arange(21) - 10.0
    ball = np.sqrt(axis[:, None, None] ** 2 + axis[None, :, None] ** 2 + axis[None, None, :] ** 2) <= 5.0
    radii = []
    for packing in (2.0, 4.0):
        compression = np.where(ball, 1.1 * settings.seal_compression, 1.0)
        length = local_length(np.where(ball, packing, 1.0), np.where(ball, packing * 0.03, 0.03), 1.0, 0.03)
        radii.append(sealed_radius(compression, length, settings.seal_compression))
        expected = (3.0 * ball.sum() * (1.0 / packing) ** 3 / (4.0 * math.pi)) ** (1.0 / 3.0)
        assert radii[-1] == pytest.approx(expected, rel=1e-9)
    assert radii[1] == pytest.approx(radii[0] / 2.0)
    assert sealed_radius(np.ones((3, 3, 3)), np.ones((3, 3, 3)), settings.seal_compression) == 0.0
