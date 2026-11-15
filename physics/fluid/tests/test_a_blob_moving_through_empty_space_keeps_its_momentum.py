"""A blob moving through empty space keeps its momentum: its centre moves at constant speed.

All rules on, the centre moves at the start speed over the first 5 time units, within
3%, while the blob's clock is still about 1. Later the blob packs under its own pull,
its clock slows (clock_rate.py) and the counter sees it move slower: the pull acts in
each box's own time, so where clocks differ the pull's action and reaction differ in
the counter's time (measured: total momentum 0.91 to 1.03 of the start over 50 time
units, seed-free scene below). So the momentum invariant itself is checked in a
side-by-side comparison run with every clock even (clock_scale 1e30): over 50 time
units the centre speed stays within 2% and nothing reaches the wall.
"""
import numpy as np

from physics.fluid import Grid, Settings, stats, step
from physics.fluid.new_grid import JOINT_SHARE
from physics.fluid.tests._gpu import needs_gpu


def _centre(grid: Grid) -> np.ndarray:
    energy = (grid.a_energy.numpy() + grid.b_energy.numpy()).astype(np.float64)
    index = np.arange(grid.side, dtype=np.float64)
    return np.array([(energy.sum(axis=(1, 2)) * index).sum(), (energy.sum(axis=(0, 2)) * index).sum(),
                     (energy.sum(axis=(0, 1)) * index).sum()]) / energy.sum()


def _speeds(settings: Settings, steps: int, window: int) -> tuple[np.ndarray, float]:
    side, radius, speed = 64, 4.0, 0.5
    axis = np.arange(side, dtype=np.float64)
    offsets = [axis[:, None, None] - 16.0, axis[None, :, None] - (side - 1) / 2.0,
               axis[None, None, :] - (side - 1) / 2.0]
    inside = np.sqrt(sum(offset ** 2 for offset in offsets)) <= radius
    velocity = np.zeros(inside.shape + (3,))
    velocity[inside, 0] = speed
    grid = Grid.from_host(np.where(inside, 1.0, 0.0), np.where(inside, JOINT_SHARE, 0.0), velocity, velocity)
    centres = [_centre(grid)]
    for _ in range(steps):
        step(grid, settings)
        centres.append(_centre(grid))
    centres = np.array(centres)
    moved = np.linalg.norm(centres[window::window] - centres[:-window:window], axis=1)
    return moved / (window * settings.step_time) / speed, stats(grid, settings)["edge_share"]


@needs_gpu
def test_the_centre_of_a_moving_blob_moves_at_its_start_speed_while_its_clock_is_even():
    early, _ = _speeds(Settings.all_on(step_time=0.5), 10, 10)
    assert abs(early[0] - 1.0) < 0.03
    even, edge_share = _speeds(Settings.all_on(step_time=0.5, clock_scale=1e30), 100, 10)
    assert edge_share == 0.0
    assert np.max(np.abs(even - 1.0)) < 0.02
