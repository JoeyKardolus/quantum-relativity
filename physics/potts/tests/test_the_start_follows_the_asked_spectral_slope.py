"""The start's ripples follow the asked spectral slope within 10%, and the start is that field.

The uneven field's shell-averaged power spectrum, fitted as a power law over every
wavenumber from 1 to side / 2 (each shell weighted by its mode count), has the asked
slope within 10%; new_universe's log energy over the filled cube is that field plus a
constant per kind, with the mean A box at 1 and a B box JOINT_SHARE times its field.
"""
import numpy as np
import pytest

from physics.potts import JOINT, QUANTUM, new_universe, uneven_field
from physics.potts.new_universe import JOINT_SHARE
from physics.potts.tests._gpu import needs_gpu


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


@needs_gpu
def test_the_start_is_the_uneven_field_in_the_filled_cube():
    cells, boxes, unevenness = 4, 4, 0.3
    universe = new_universe(cells, boxes, seed=2, unevenness=unevenness)
    owner = universe.owner.numpy()
    energy = universe.energy.numpy() * QUANTUM
    filled = cells * boxes
    low = (universe.side - filled) // 2
    inner = (slice(low, low + filled),) * 3
    assert np.all(owner[inner] >= 0) and energy.sum() == pytest.approx(energy[inner].sum())
    joint = universe.kind.numpy()[owner[inner]] == JOINT
    field = uneven_field(filled, 2, -3.0, unevenness)
    log_energy = np.log(np.where(joint, energy[inner] / JOINT_SHARE, energy[inner]))
    assert (log_energy - field).std() < 1e-6
    assert energy[inner][~joint].mean() == pytest.approx(1.0, rel=1e-6)
