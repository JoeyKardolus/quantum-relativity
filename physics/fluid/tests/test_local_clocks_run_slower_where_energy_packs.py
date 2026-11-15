"""Local clocks run slower where energy packs, joints included, and everything in a box keeps to its clock.

Hand-built boxes: the start's mean box runs at 1, a box four times as packed at
(s + 1) / (s + 4), squeezed joints alone slow a box holding little A, and an empty
box runs fastest. Then identical moving blobs one, four and sixteen times as packed
as the reference (all rules on): over 1 time unit each centre moves at the start
speed times its clock rate, within 6%.
"""
import numpy as np
import pytest

from physics.fluid import Grid, Settings, step
from physics.fluid.clock_rate import clock_rate
from physics.fluid.new_grid import JOINT_SHARE


def _rate(concentration: float, settings: Settings) -> float:
    return (settings.clock_scale + 1.0) / (settings.clock_scale + concentration)


def test_a_box_clock_follows_its_energy_concentration_joints_included():
    settings = Settings.all_on()
    a = np.zeros((4, 4, 4))
    b = np.zeros((4, 4, 4))
    a[0, 0, 0], b[0, 0, 0] = 1.0, settings.b_rest
    a[1, 0, 0], b[1, 0, 0] = 4.0, 4.0 * settings.b_rest
    a[2, 0, 0], b[2, 0, 0] = 0.1, 6.0 * settings.b_rest
    grid = Grid.from_host(a, b, device="cpu")
    grid.start_a_mean = 1.0
    clock_rate(grid, settings)
    clock = grid.clock.numpy()
    assert clock[0, 0, 0] == pytest.approx(1.0, rel=1e-5)
    assert clock[1, 0, 0] == pytest.approx(_rate(4.0, settings), rel=1e-5)
    assert clock[2, 0, 0] == pytest.approx(_rate(3.05, settings), rel=1e-5)
    assert clock[3, 3, 3] == pytest.approx(_rate(0.0, settings), rel=1e-5)
    assert clock[1, 0, 0] < clock[2, 0, 0] < clock[0, 0, 0] < clock[3, 3, 3]


@pytest.mark.parametrize("packing", [1.0, 4.0, 16.0])
def test_a_packed_blob_moves_slower_by_its_clock(packing):
    side, speed = 32, 0.5
    settings = Settings.all_on(step_time=0.5)
    axis = np.arange(side) - (side - 1) / 2.0
    inside = np.sqrt(axis[:, None, None] ** 2 + axis[None, :, None] ** 2 + axis[None, None, :] ** 2) <= 3.0
    velocity = np.zeros(inside.shape + (3,))
    velocity[inside, 0] = speed
    grid = Grid.from_host(np.where(inside, packing, 0.0), np.where(inside, packing * JOINT_SHARE, 0.0),
                          velocity, velocity, device="cpu")
    grid.start_a_mean = 1.0
    index = np.arange(side)

    def centre() -> float:
        energy = (grid.a_energy.numpy() + grid.b_energy.numpy()).astype(np.float64)
        return float((energy.sum(axis=(1, 2)) * index).sum() / energy.sum())

    start = centre()
    step(grid, settings)
    step(grid, settings)
    measured = (centre() - start) / (2 * settings.step_time)
    assert measured / (speed * _rate(packing, settings)) == pytest.approx(1.0, abs=0.06)
