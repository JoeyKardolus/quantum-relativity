"""A small grid that wraps round: laid alike from every volume, exact books, and it stands calm."""
import numpy as np

from physics.engine import step
from physics.engine.axiom_3_structure.definitions.grid import joints_of
from physics.engine.axiom_3_structure.logic.boxes_of import swirl_of
from physics.engine.axiom_3_structure.logic.lay_grid import calm_grid
from physics.engine.axiom_5_packing.logic.squeeze import squeeze

CALM = 648_000


def _books(store) -> np.ndarray:
    """Every cell's quanta, checked against the sum of its own boxes and what joints made."""
    held, cells = store.cell_quanta.numpy(), store.own_cell.numpy()[:store.count]
    assert (store.box_quanta.numpy()[:store.count].astype(np.int64).sum(axis=1) == held[cells]).all()
    assert (held[held < 0] == -CALM - store.cell_made.numpy()[held < 0]).all()  # joints only make
    return held


def test_every_volume_sees_the_same_laying_mirrors_across_the_swirl_and_its_like_along_it():
    for swirl in range(3):
        store = calm_grid(reds=(4, 4, 4), swirl=swirl)
        held, laid = store.cell_quanta.numpy(), store.cell_laid.numpy()
        assert (held > 0).sum() == 64 and (held < 0).sum() == 192 and store.count == 256
        for volume in np.flatnonzero(held > 0):
            joints = joints_of(store.grid, int(volume))
            beyond = [joints_of(store.grid, joint)[way] for way, joint in enumerate(joints)]
            other = [int(laid[one] != laid[volume]) for one in beyond]
            assert (held[joints] == -CALM).all() and (held[beyond] == CALM).all()
            assert other == [int(way // 2 != swirl) for way in range(6)] and laid[volume] // 2 == swirl


def test_the_calm_grid_comes_to_balance_and_stands_with_exact_books_and_whole_swirls():
    store, seen = calm_grid(reds=(2, 2, 2)), []
    for done in range(3000):
        step(store)
        if done < 1000 or done >= 2500:
            held = _books(store)
            assert held[held > 0].sum() == 8 * CALM  # plus energy only moves
            seen.append(held / CALM)
    last = np.array(seen[-500:])
    volumes, joints = last[:, last[0] > 0], -last[:, last[0] < 0]
    assert np.abs(volumes - 1).max() < 0.001                    # measured: 0.0001
    assert np.ptp(joints, axis=0).max() < 0.0002 and np.abs(joints - 1.2805).max() < 0.001
    squeeze(store)
    own = store.own_cell.numpy()[:store.count]
    pull, push, apart, _axis = store.own_squeeze.numpy()[:store.count][last[0][own] < 0].T
    assert np.abs(pull - push).max() < 0.001 * push.min() and np.abs(apart - 1.25).max() < 0.001
    for volume in np.flatnonzero(last[0] > 0):  # every volume still turns as one whole swirl
        swirl = swirl_of(store, int(volume))
        assert abs(swirl["turns"]) == 1 and swirl["core"] < 0.2 and swirl["off_rule"] < 1


def test_an_uneven_start_evens_out_and_nothing_runs_away():
    store = calm_grid(reds=(2, 2, 2), uneven=0.1)
    start = store.cell_quanta.numpy() / CALM
    assert np.ptp(start[start > 0]) > 0.1
    for _ in range(1000):
        step(store)
    held = _books(store) / CALM
    assert np.ptp(held[held > 0]) < 0.001 and abs(held[held > 0].mean() - start[start > 0].mean()) < 1e-9
    made = store.cell_made.numpy().sum()
    for _ in range(300):
        step(store)
    assert store.cell_made.numpy().sum() - made < 0.0005 * made  # the joints have all but stopped
    assert -1.31 < (_books(store) / CALM).min()                   # near the even grid's 1.28
