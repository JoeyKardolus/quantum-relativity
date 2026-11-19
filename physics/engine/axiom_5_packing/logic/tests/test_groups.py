"""The card's pull against the law read pair by pair on the host, and what reading far volumes by groups leaves out."""
import numpy as np
import pytest

from physics.engine import first_volume, step
from physics.engine.axiom_3_structure.logic.spare import spare
from physics.engine.axiom_5_packing.definitions.groups import wide_of
from physics.engine.axiom_5_packing.definitions.pull import REACH
from physics.engine.axiom_5_packing.logic.line import calm_block
from physics.engine.axiom_5_packing.logic.pairs import pulls
from physics.engine.axiom_5_packing.logic.pull import pull
from physics.engine.axiom_5_packing.logic.staging import copy_of


def _held_still(grown_patch, near: int) -> tuple:
    """The ball held still until every pull has arrived, read with `near`: its depths and the pulls on its volumes."""
    store = copy_of(grown_patch)
    store.near, store.cell_past = near, None
    for _ in range(REACH):
        pull(store)
        store.step += 1
    return store, store.cell_pull.numpy(), spare(store, "drawn", wide_of(store.side) ** 3, None).numpy()


@pytest.fixture(scope="module")
def read(grown_patch):
    """The ball's pull read on the host pair by pair, and on the card with `near` 0, 1 and 2."""
    store, deep, drawn = _held_still(grown_patch, 0)
    cells, host_deep, host_drawn = pulls(store)
    at = np.stack(np.unravel_index(cells, (store.side,) * 3), axis=-1) // 2
    column = np.ravel_multi_index(tuple(at.T), (wide_of(store.side),) * 3)
    by = {near: _held_still(grown_patch, near)[1:] for near in (1, 2)}
    return cells, column, (host_deep, host_drawn), (deep, drawn), by


def test_all_pairs_on_the_card_is_the_host_s_pair_by_pair_pull(read):
    cells, column, (host_deep, host_drawn), (deep, drawn), _by = read
    off_deep = np.abs(deep[cells] / host_deep - 1).max()
    off_drawn = (np.linalg.norm(drawn[column] - host_drawn, axis=1) / np.linalg.norm(host_drawn, axis=1)).max()
    print(f"{len(cells)} volumes: the depth is off by {off_deep:.1e} of itself at most, the pull by {off_drawn:.1e}")
    assert len(cells) > 100 and off_deep < 1e-6 and off_drawn < 1e-9  # a 32-bit depth; the sums' order


def test_reading_far_volumes_by_groups_leaves_out_little(read):
    cells, column, _host, (deep, drawn), by = read
    worst = {}
    for near, (grouped_deep, grouped_drawn) in by.items():
        off = np.linalg.norm(grouped_drawn[column] - drawn[column], axis=1) / np.linalg.norm(drawn[column], axis=1)
        worst[near] = (float(off.max()), float(np.median(off)), float(np.abs(grouped_deep[cells] / deep[cells] - 1).max()))
        print(f"near {near}: the pull is off by {worst[near][0]:.1e} of itself at most, {worst[near][1]:.1e} in the "
              f"middle volume; the depth by {worst[near][2]:.1e}")
    assert worst[1][0] < 1e-3 and worst[1][2] < 1e-3  # 2.6e-04 and 1.4e-04: the first volume holds nearly all of it
    assert worst[2][0] == 0  # the ball is small: with two groups near, every volume is read by itself


def test_in_an_uneven_block_of_512_volumes_groups_leave_out_under_a_hundredth_of_the_largest_pull():
    drawn = {}
    for near in (0, 1):
        store = calm_block((8, 8, 8), {(1, 2, 3): 4, (6, 5, 1): 9, (3, 3, 4): 0.3, (0, 7, 7): 20})
        store.near = near
        for _ in range(REACH):
            pull(store)
            store.step += 1
        drawn[near] = spare(store, "drawn", 0, None).numpy()
    off, size = np.linalg.norm(drawn[1] - drawn[0], axis=1), np.linalg.norm(drawn[0], axis=1)
    print(f"off by {off.max() / size.max():.1e} of the largest pull at most, {np.median(off / size):.1e} of itself in the middle volume")
    assert 0 < off.max() < 0.01 * size.max() and np.median(off / size) < 0.01  # 4.5e-03 and 3.6e-03


def test_a_universe_grown_reading_by_groups_keeps_its_books_and_repeats():
    runs = [first_volume(layers=7, near=1) for _ in range(2)]
    for _ in range(120):
        for store in runs:
            step(store)
    store, held = runs[0], runs[0].cell_quanta.numpy()
    own = store.own_cell.numpy()[:store.count]
    assert store.count > 27 and (store.box_quanta.numpy()[:store.count].astype(np.int64).sum(axis=1) == held[own]).all()
    assert held[held > 0].sum() - held[held < 0].sum() - store.cell_made.numpy().sum() == 2 ** 30
    assert np.array_equal(held, runs[1].cell_quanta.numpy())
    assert np.array_equal(store.box_quanta.numpy(), runs[1].box_quanta.numpy())
