"""A lone volume leaks whole quanta toward the free places round it; a thinner one a bigger share."""
import numpy as np

from physics.engine import first_volume, step, three_cells

CALM = 648_000


def _leaked(quanta: int, steps: int = 46):
    """A lone volume after about a turn of its swirl: the store, and what gathered at each place round it."""
    store = first_volume(layers=1, quanta=quanta)
    for _ in range(steps):
        step(store)
    held = store.cell_quanta.numpy().astype(np.int64)
    middle = store.side ** 3 // 2
    assert int(store.box_quanta.numpy()[0].sum()) == held[middle]  # the volume's books
    assert held.sum() == quanta and held.min() >= 0               # plus that left it arrived
    return store, np.delete(held, middle)[np.delete(held, middle) > 0]


def test_a_thinner_volume_leaks_a_bigger_share_of_itself_in_a_turn_than_a_fuller_one():
    shares = [_leaked(calms * CALM)[1].sum() / (calms * CALM) for calms in (4, 1)]
    assert shares[1] > 1.25 * shares[0] > 0.1  # measured: 0.088 and 0.118 (the law: 1/11, 1/8)
    assert shares[1] < 1 / 7  # and never a seventh: the volume is left standing


def test_the_leak_gathers_alike_at_the_26_places_and_nowhere_else():
    store, gathered = _leaked(40 * CALM)
    assert len(gathered) == 26 and gathered.max() < 1.5 * gathered.min()  # measured: 1.32
    assert store.count == 1  # nothing gathered is a cell yet


def test_the_crossing_bench_does_not_leak():
    store = three_cells()
    for _ in range(50):
        step(store)
    assert (store.cell_quanta.numpy() != 0).sum() == 3
