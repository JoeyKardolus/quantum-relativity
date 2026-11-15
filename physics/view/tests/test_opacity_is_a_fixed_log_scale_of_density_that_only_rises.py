"""A dot's opacity is a fixed log scale of its density over the start's mean, never a rank."""
from __future__ import annotations

import numpy as np

from physics.abab.state import KIND_A, KIND_B
from physics.view import density_reference, opacity, point_colours
from physics.view.opacity import FAINTEST, PACKED, THINNEST
from physics.view.point_colours import BLUE, RED


def test_denser_is_never_fainter_and_rises_inside_the_scale():
    ratios = np.geomspace(0.01, 100.0, 400)
    alpha = opacity(ratios, np.ones_like(ratios))
    assert np.all(np.diff(alpha) >= 0)
    inside = (ratios > THINNEST) & (ratios < PACKED)
    assert np.all(np.diff(alpha[inside]) > 0)


def test_the_scale_is_fixed_ends_and_start_mean_near_invisible():
    alpha = opacity(np.array([0.01, THINNEST, 1.0, PACKED, 1e6, 0.0, np.nan]), np.ones(7))
    assert np.allclose(alpha[[0, 1, 5, 6]], FAINTEST)
    assert np.allclose(alpha[[3, 4]], 1.0)
    assert alpha[2] < 0.05


def test_log_of_opacity_is_linear_in_log_density():
    ratios = np.array([1.0, 2.0, 4.0, 8.0])
    steps = np.diff(np.log(opacity(ratios, np.ones(4))))
    assert np.allclose(steps, steps[0])


def test_a_dot_does_not_depend_on_the_others_so_growing_mass_shows():
    alone = opacity(np.array([3.0]), np.array([1.0]))
    among_thin = opacity(np.array([3.0, 0.1, 0.2]), np.ones(3))[0]
    among_packed = opacity(np.array([3.0, 50.0, 90.0]), np.ones(3))[0]
    assert alone[0] == among_thin == among_packed
    everyone = np.array([1.0, 1.5, 2.0])
    assert np.all(opacity(everyone * 4.0, np.ones(3)) > opacity(everyone, np.ones(3)))


def test_colours_are_red_and_blue_each_against_its_own_kinds_start():
    kind = np.array([KIND_A, KIND_B, KIND_A, KIND_B])
    start = np.array([1.0, 30.0, 1.0, 30.0])
    reference = density_reference(kind, start)
    assert np.allclose(reference, [1.0, 30.0])
    colours = point_colours(kind, np.array([1.0, 30.0, 16.0, 480.0]), reference)
    assert colours.dtype == np.uint8 and colours.shape == (4, 4)
    assert tuple(colours[0, :3]) == RED and tuple(colours[1, :3]) == BLUE
    assert colours[0, 3] == colours[1, 3] < 13
    assert colours[2, 3] == colours[3, 3] == 255
