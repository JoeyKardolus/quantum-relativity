"""A blob moving through empty space keeps its momentum: its centre moves at constant speed."""
import numpy as np

from physics.fluid import Grid, Settings, stats, step
from physics.fluid.new_grid import JOINT_SHARE
from physics.fluid.tests._gpu import needs_gpu


def _centre(grid: Grid) -> np.ndarray:
    energy = (grid.a_energy.numpy() + grid.b_energy.numpy()).astype(np.float64)
    index = np.arange(grid.side, dtype=np.float64)
    return np.array([(energy.sum(axis=(1, 2)) * index).sum(), (energy.sum(axis=(0, 2)) * index).sum(),
                     (energy.sum(axis=(0, 1)) * index).sum()]) / energy.sum()


@needs_gpu
def test_the_centre_of_a_moving_blob_moves_at_constant_speed_over_100_steps():
    side, radius, speed = 64, 4.0, 0.5
    settings = Settings.all_on(step_time=0.5)
    axis = np.arange(side, dtype=np.float64)
    offsets = [axis[:, None, None] - 16.0, axis[None, :, None] - (side - 1) / 2.0,
               axis[None, None, :] - (side - 1) / 2.0]
    inside = np.sqrt(sum(offset ** 2 for offset in offsets)) <= radius
    velocity = np.zeros(inside.shape + (3,))
    velocity[inside, 0] = speed
    grid = Grid.from_host(np.where(inside, 1.0, 0.0), np.where(inside, JOINT_SHARE, 0.0), velocity, velocity)
    centres = [_centre(grid)]
    for _ in range(100):
        step(grid, settings)
        centres.append(_centre(grid))
    centres = np.array(centres)
    assert stats(grid, settings)["edge_share"] == 0.0
    window = 10
    moved = np.linalg.norm(centres[window::window] - centres[:-window:window], axis=1)
    measured = moved / (window * settings.step_time)
    assert np.max(np.abs(measured / speed - 1.0)) < 0.02
