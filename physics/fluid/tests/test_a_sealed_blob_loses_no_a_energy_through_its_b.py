"""A sealed blob loses no A-energy through its B: A cannot pass B that is past the seal point.

Scene: a ball of A-energy in B that is past the seal point everywhere, falling in under
its own pull for 40 steps. A that cannot pass through B moves only with it, so no box
ever holds more A per B than the most any box held at the start (each box's new share
is a mix of the shares that flowed in). With the seal off (a side-by-side comparison
run) A sinks through B towards the centre and that share is passed by more than 1%;
that pair is run with every clock even (clock_scale 1e30), since with all rules on the
squeezed B runs its clock at about half speed and the unsealed peak is 0.76%
(measured), the ball collapsing before A sinks further.
"""
import numpy as np

from physics.fluid import Grid, Settings, step
from physics.fluid.tests._gpu import needs_gpu


def _highest_share_over_start(settings: Settings, steps: int) -> float:
    side, radius = 32, 5.0
    seal = Settings.all_on().seal_compression
    axis = np.arange(side) - (side - 1) / 2.0
    distance = np.sqrt(axis[:, None, None] ** 2 + axis[None, :, None] ** 2 + axis[None, None, :] ** 2)
    a = np.where(distance <= radius, 1.0, 0.0)
    grid = Grid.from_host(a, np.full(a.shape, 1.2 * seal * settings.b_rest))

    def share() -> float:
        return float((grid.a_energy.numpy() / np.maximum(grid.b_energy.numpy(), 1)).max())

    start = share()
    highest = start
    for _ in range(steps):
        step(grid, settings)
        highest = max(highest, share())
    return highest / start


@needs_gpu
def test_with_the_seal_no_box_gains_a_share_of_a_while_without_it_a_sinks_through_b():
    assert _highest_share_over_start(Settings.all_on(), 40) < 1.0 + 1e-6
    assert _highest_share_over_start(Settings.all_on(clock_scale=1e30), 40) < 1.0 + 1e-6
    assert _highest_share_over_start(Settings.all_on(clock_scale=1e30, seal_compression=float("inf")), 40) > 1.01
