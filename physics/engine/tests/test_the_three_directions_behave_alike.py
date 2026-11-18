"""The three directions behave alike: the start turned gives the same course; same seed, same run."""
import numpy as np

from physics.engine import step, three_cells
from physics.engine.axiom_3_structure.logic.lay_grid import calm_grid

CALM = 648_000
TURNED = ((0, 1, 2), (1, 2, 0), (2, 0, 1))


def _course(turned: tuple, seed: int = 1, steps: int = 40) -> tuple:
    """The volumes' quanta and each direction's joints' after `steps`, told as the start was turned."""
    store = calm_grid(reds=(2, 2, 2), uneven=0.1, turned=turned, seed=seed)
    for _ in range(steps):
        step(store)
    held = store.cell_quanta.numpy().reshape(store.grid).transpose(turned) / CALM
    joints = [held[1::2, ::2, ::2], held[::2, 1::2, ::2], held[::2, ::2, 1::2]]
    return held[::2, ::2, ::2], np.array(joints), store.box_quanta.numpy()


def test_the_same_start_turned_onto_another_direction_gives_the_same_course():
    volumes, joints, _boxes = zip(*(_course(turned) for turned in TURNED))
    assert np.ptp(volumes[0]) > 0.005 and joints[0].min() < -1.1  # mid course: things happen
    for turned in (1, 2):
        assert np.abs(volumes[turned] - volumes[0]).max() < 0.0005  # measured: 0.0002
        assert np.abs(joints[turned] - joints[0]).max() < 0.0005    # measured: 0.00013


def test_the_joints_along_each_direction_are_squeezed_on_the_same_course():
    store, seen = calm_grid(reds=(2, 2, 2)), []
    for steps in (20, 40, 160):  # early, mid course, near the balance
        while store.step < steps:
            step(store)
        held = store.cell_quanta.numpy().reshape(store.grid) / CALM
        seen.append([held[1::2, ::2, ::2].mean(), held[::2, 1::2, ::2].mean(), held[::2, ::2, 1::2].mean()])
    seen = -np.array(seen)
    assert 1.09 < seen[0].mean() < 1.12 and 1.16 < seen[1].mean() < 1.19 and seen[2].mean() > 1.27
    assert np.ptp(seen, axis=1).max() < 0.0005  # measured: 0.00005; before the ball whirl 0.013


def test_a_line_of_cells_runs_the_same_course_along_each_direction():
    held = []
    for along in range(3):
        store = three_cells(along=along)
        for _ in range(150):
            step(store)
        held.append(store.cell_quanta.numpy()[:3] / CALM)
    assert held[0][0] < 1.2 and held[0][1] < -1.1  # things happened: A from 1.5, the joint from -1
    assert np.abs(np.array(held) - held[0]).max() < 0.003  # measured: 0.0007


def test_the_same_seed_gives_the_same_grid_run():
    first, again, other = (_course(TURNED[2], seed, 60)[2] for seed in (1, 1, 2))
    assert np.array_equal(first, again) and not np.array_equal(first, other)
