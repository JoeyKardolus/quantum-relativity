"""The three directions behave alike: the start turned gives the same course; same seed, same run."""
import numpy as np

from physics.engine import step
from physics.engine.axiom_3_structure.logic.lay_grid import calm_grid

CALM = 648_000


def _course(swirl: int, seed: int = 1, steps: int = 40) -> tuple:
    """The volumes' quanta and each direction's joints' after `steps`, told as the swirl lies."""
    store = calm_grid(reds=(2, 2, 2), uneven=0.1, swirl=swirl, seed=seed)
    for _ in range(steps):
        step(store)
    turned = [(swirl + 1) % 3, (swirl + 2) % 3, swirl]
    held = store.cell_quanta.numpy().reshape(store.grid).transpose(turned) / CALM
    joints = [held[1::2, ::2, ::2], held[::2, 1::2, ::2], held[::2, ::2, 1::2]]
    return held[::2, ::2, ::2], np.array(joints), store.box_quanta.numpy()


def test_the_three_directions_behave_alike():
    volumes, joints, _boxes = zip(*(_course(swirl) for swirl in range(3)))
    assert np.ptp(volumes[0]) > 0.005 and joints[0].min() < -1.04  # mid course: things happen
    for turned in (1, 2):  # the start turned so the swirls lie along another direction
        assert np.abs(volumes[turned] - volumes[0]).max() < 0.0003  # measured: 0.00005
        assert np.abs(joints[turned] - joints[0]).max() < 0.001     # measured: 0.0002
    across, along = joints[0][:2].mean(), joints[0][2].mean()
    assert 0.01 < along - across < 0.02  # the swirls' own direction differs: its joints lag (0.013)


def test_the_same_seed_gives_the_same_grid_run():
    first, again, other = (_course(2, seed, 60)[2] for seed in (1, 1, 2))
    assert np.array_equal(first, again) and not np.array_equal(first, other)
