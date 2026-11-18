"""`uneven_field`: one value per energy volume, mean 0, the asked spread, the same for the same seed."""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform_rows
from physics.engine.axiom_1_energy.definitions.passes import START_FIELD_A, START_FIELD_B
import pytest

from physics.engine.axiom_1_energy.logic.uneven_field import uneven_field


def test_one_value_per_volume_with_mean_0_and_the_asked_spread():
    field = uneven_field(16, 1, -3.0, 0.1)
    assert field.shape == (8, 8, 8) and field.dtype == np.float64
    assert field.mean() == pytest.approx(0.0, abs=1e-12)
    assert np.sqrt(np.mean(field ** 2)) == pytest.approx(0.1, rel=1e-12)


def test_the_same_seed_gives_the_same_field_and_another_seed_another():
    assert np.array_equal(uneven_field(16, 7, -3.0, 0.3), uneven_field(16, 7, -3.0, 0.3))
    assert not np.array_equal(uneven_field(16, 7, -3.0, 0.3), uneven_field(16, 8, -3.0, 0.3))


def test_a_steeper_slope_puts_more_of_the_spread_in_the_longest_ripples():
    def longest_share(slope: float) -> float:
        power = np.abs(np.fft.fftn(uneven_field(32, 3, slope, 1.0))) ** 2
        longest = power[1, 0, 0] + power[-1, 0, 0] + power[0, 1, 0] + power[0, -1, 0] + power[0, 0, 1] + power[0, 0, -1]
        return float(longest / power.sum())

    assert longest_share(-4.0) > longest_share(-3.0) > longest_share(-1.0)


def test_no_unevenness_is_an_even_field():
    assert not uneven_field(16, 1, -3.0, 0.0).any()
    assert not uneven_field(4, 1, -3.0, 0.0).any()


@pytest.mark.parametrize("side", [4, 8, 12, 14])
def test_a_lattice_too_small_for_ripples_gets_plain_seeded_noise_with_mean_0_and_the_asked_spread(side):
    field = uneven_field(side, 5, -3.0, 0.2)
    places = np.arange((side // 2) ** 3)
    noise = (np.sqrt(-2.0 * np.log(1.0 - draw_uniform_rows(5, 0, START_FIELD_A, places)))
             * np.cos(2.0 * np.pi * draw_uniform_rows(5, 0, START_FIELD_B, places))).reshape((side // 2,) * 3)
    assert field.shape == (side // 2,) * 3
    assert abs(field.mean()) < 1e-12 and abs(np.sqrt(np.mean(field ** 2)) - 0.2) < 1e-12
    assert np.allclose(field, (noise - noise.mean()) * (0.2 / (noise - noise.mean()).std()))


@pytest.mark.parametrize("side, slope, unevenness", [(5, -3.0, 0.1), (2, -3.0, 0.1), (8, float("nan"), 0.1),
                                                     (8, -3.0, -0.1)])
def test_it_refuses_what_it_cannot_make(side, slope, unevenness):
    with pytest.raises(ValueError):
        uneven_field(side, 1, slope, unevenness)
