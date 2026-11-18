"""`settle`, `draw_whole` and `react` on a staged split: exact whole quanta that come to rest, the cuts booked."""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_1_energy.logic.count_cells import count_cells
from physics.engine.axiom_7_quantum.definitions.split_store import MOMENTUM, REACTION
from physics.engine.axiom_7_quantum.logic.draw_whole import draw_whole
from physics.engine.axiom_7_quantum.logic.react import react
from physics.engine.axiom_7_quantum.logic.settle import settle
from physics.engine.axiom_7_quantum.logic.tests.conftest import HELD, PER_BOX, held_by, open_fork, send, set_cells


def _split(grid, home: int, halves: dict, lost: int = 0) -> None:
    """Stage a fork: `home` sent `halves` ({cell: quanta}), `lost` of each without direction; open the split."""
    for cell, quanta in halves.items():
        send(grid, home, cell, quanta, lost=lost)
    open_fork(grid)


def _draw(grid, giver: int, taker: int, amount: int) -> None:
    """One direct pull of `amount` quanta from one cell into another."""
    plan = {name: np.zeros(grid.count, dtype=np.int64) for name in ("take", "give", "heat_change")}
    plan["take"][giver], plan["give"][taker] = amount, amount
    draw_whole(grid, plan["take"], plan["give"], plan["heat_change"])


def test_nothing_settles_while_nothing_reacted_and_momentum_is_left(grid, volumes):
    home, first, second = volumes[:3]
    _split(grid, home, {first: 216, second: 432})
    before = grid.quanta.numpy().copy()
    settle(grid)
    assert len(grid.quantum["source"]) == 1 and np.array_equal(before, grid.quanta.numpy())


def test_a_reacted_split_settles_whole_in_the_pointer_s_cell_and_the_total_stands(grid, volumes):
    home, first, second, third = volumes[:4]
    _split(grid, home, {first: 216, second: 432, third: 100})
    store = grid.quantum
    target = int(store["pointer"][0])
    store["reacted"][0] = True
    settle(grid)
    for cell, part in ((first, 216), (second, 432), (third, 100)):
        assert held_by(grid, cell) == (HELD + 748 - part if cell == target else HELD - part), cell
    assert held_by(grid, home) == HELD and int(grid.quanta.numpy().min()) >= 0
    assert len(store["source"]) == 0 and len(store["part_cell"]) == 0, "a settled split is gone from the store"
    settled = store["settled"]
    assert (settled["source"].tolist(), settled["cell"].tolist(), settled["amount"].tolist(), settled["short"].tolist(),
            settled["cause"].tolist()) == ([home], [target], [748], [0], [REACTION])
    assert store["step"]["settled_reaction"] == 1 and store["step"]["settled_quanta"] == 748


def test_a_split_all_of_whose_direction_friction_took_settles_by_momentum(grid, volumes):
    home, first, second = volumes[:3]
    _split(grid, home, {first: 50, second: 50}, lost=50)
    assert grid.quantum["directed"].tolist() == [0]
    settle(grid)
    assert grid.quantum["settled"]["cause"].tolist() == [MOMENTUM] and held_by(grid, first) + held_by(grid, second) == 2 * HELD
    assert sorted((held_by(grid, first), held_by(grid, second))) == [HELD - 50, HELD + 50]


def test_a_caller_s_detector_in_a_half_s_cell_settles_that_split_and_no_other(grid, volumes):
    home, first, second, other, third, fourth = volumes[:6]
    _split(grid, home, {first: 216, second: 432})
    _split(grid, other, {third: 10, fourth: 20})
    react(grid, second)
    settle(grid)
    assert grid.quantum["source"].tolist() == [other] and grid.quantum["settled"]["source"].tolist() == [home]
    assert grid.quantum["detected"] == set()


def test_a_half_whose_cell_no_longer_holds_it_is_cut_and_the_cut_is_booked(grid, volumes):
    home, first, second = volumes[:3]
    _split(grid, home, {first: 216, second: 432})
    store = grid.quantum
    store["pointer"][0], store["reacted"][0] = first, True
    kept = np.zeros((6, 6, 6), dtype=np.int32)
    kept[0, 0, 0] = 32
    grid.set_block("quanta", second, kept)
    count_cells(grid)
    settle(grid)
    assert held_by(grid, second) == 0 and held_by(grid, first) == HELD + 32
    assert store["settled"]["amount"].tolist() == [216 + 32] and store["settled"]["short"].tolist() == [400]
    assert store["step"]["short_quanta"] == 400


def test_heat_moves_with_the_quanta_that_are_drawn(grid, volumes):
    home, first, second = volumes[:3]
    _split(grid, home, {first: 216, second: 432})
    set_cells(grid, "heat_quanta", {second: HELD // 2})
    grid.quantum["pointer"][0], grid.quantum["reacted"][0] = first, True
    settle(grid)
    heat = grid.host("heat_quanta")
    assert (int(heat[second]), int(heat[first])) == (HELD // 2 - 216, 216)


def test_what_is_drawn_comes_to_rest_so_the_taker_s_own_phases_stand(grid, volumes):
    giver, taker = volumes[:2]
    before = grid.block("phase", taker).copy()
    _draw(grid, giver, taker, HELD)
    assert np.all(grid.block("quanta", taker) == 2 * PER_BOX) and held_by(grid, giver) == 0
    assert np.array_equal(grid.block("phase", taker), before), "settling makes no motion and wipes no swirl"


def test_quanta_owed_after_the_whole_shares_leave_and_arrive_one_a_box(grid, volumes):
    giver, taker = volumes[:2]
    _draw(grid, giver, taker, 100)
    gave, got = PER_BOX - grid.block("quanta", giver), grid.block("quanta", taker) - PER_BOX
    assert gave.sum() == 100 and got.sum() == 100 and set(np.unique(gave)) <= {0, 1} and set(np.unique(got)) <= {0, 1}
