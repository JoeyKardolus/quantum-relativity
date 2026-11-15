"""The start's ripples follow the asked spectral slope within 10%, and the start is that field.

The uneven field's shell-averaged power spectrum, fitted as a power law over every
wavenumber from 1 to side / 2 (each shell weighted by its mode count), has the asked
slope within 10%; new_grid's log A-energy inside the filled cube is that field plus a
constant, and B is the joint share of A.
"""
import numpy as np
import pytest

from physics.fluid import QUANTUM, new_grid
from physics.fluid.new_grid import JOINT_SHARE
from physics.fluid.uneven_field import uneven_field


def _fitted_slope(field: np.ndarray) -> float:
    side = field.shape[0]
    power = np.abs(np.fft.rfftn(field)) ** 2
    full = np.fft.fftfreq(side) * side
    half = np.fft.rfftfreq(side) * side
    wavenumber = np.sqrt(full[:, None, None] ** 2 + full[None, :, None] ** 2 + half[None, None, :] ** 2)
    shell = np.rint(wavenumber).astype(int)
    weight = np.where(half[None, None, :] > 0, 2.0, 1.0) * np.ones_like(power)
    shells = np.arange(1, side // 2 + 1)
    counts = np.array([weight[shell == number].sum() for number in shells])
    means = np.array([(power * weight)[shell == number].sum() for number in shells]) / counts
    slope, _ = np.polyfit(np.log(shells), np.log(means), 1, w=np.sqrt(counts))
    return float(slope)


@pytest.mark.parametrize("slope", [-3.0, -2.0, -4.0])
def test_the_uneven_field_has_the_asked_spectral_slope_within_ten_percent(slope):
    field = uneven_field(64, seed=1, slope=slope, unevenness=0.3)
    assert np.sqrt(np.mean(field ** 2)) == pytest.approx(0.3, rel=1e-9)
    assert abs(_fitted_slope(field) - slope) <= 0.1 * abs(slope)


def test_the_start_is_the_uneven_field_in_a_filled_cube():
    side, unevenness = 32, 0.3
    grid = new_grid(side, seed=2, unevenness=unevenness, device="cpu")
    a = grid.a_energy.numpy() * QUANTUM
    b = grid.b_energy.numpy() * QUANTUM
    margin = side // 16
    inner = (slice(margin, side - margin),) * 3
    assert np.all(a[inner] > 0)
    assert a.sum() - a[inner].sum() == 0
    assert a[inner].mean() == pytest.approx(1.0, rel=1e-6)
    offset = np.log(a[inner]) - uneven_field(side, 2, -3.0, unevenness)[inner]
    assert offset.std() < 1e-6
    assert np.allclose(b[inner], JOINT_SHARE * a[inner], rtol=1e-6)
