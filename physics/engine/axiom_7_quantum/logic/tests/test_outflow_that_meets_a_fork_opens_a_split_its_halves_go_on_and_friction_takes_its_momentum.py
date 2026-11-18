"""`open_splits` and `carry_on` on a staged crossing: what the fluid sent is all they read."""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_7_quantum.definitions.split_store import amount_of
from physics.engine.axiom_7_quantum.logic.carry_on import carry_on
from physics.engine.axiom_7_quantum.logic.open_splits import open_splits
from physics.engine.axiom_7_quantum.logic.read_sent import read_sent
from physics.engine.axiom_7_quantum.logic.tests.conftest import HELD, all_arrays, next_step, send, set_cells


def _open(grid) -> None:
    """What `meet_fork` does after the turn: carry the halves on, then open what is new."""
    rows = read_sent(grid)
    open_splits(grid, rows, carry_on(grid, rows))


def _same(before: dict, after: dict) -> bool:
    return all(np.array_equal(before[name], after[name]) for name in before)


def test_read_sent_adds_up_what_each_volume_sent_into_each_other(grid, volumes):
    home, first, second = volumes[:3]
    send(grid, home, first, 216, lost=100)
    send(grid, second, home, 5)
    rows = read_sent(grid)
    assert list(zip(rows["giver"].tolist(), rows["taker"].tolist(), rows["sent"].tolist(), rows["lost"].tolist())) \
        == sorted([(home, first, 216, 100), (second, home, 5, 0)])


def test_outflow_into_two_volumes_is_one_split_with_its_halves_where_the_fluid_went(grid, volumes):
    home, first, second = volumes[:3]
    send(grid, home, first, 216, lost=100)
    send(grid, home, second, 432, lost=200)
    before = all_arrays(grid)
    _open(grid)
    store = grid.quantum
    assert store["source"].tolist() == [home] and store["born"].tolist() == [grid.step]
    assert list(zip(store["part_cell"].tolist(), store["part_quanta"].tolist())) == sorted([(first, 216), (second, 432)])
    assert store["directed"].tolist() == [648 - 300] and amount_of(store).tolist() == [648]
    assert store["pointer"][0] in (first, second) and not store["reacted"][0]
    assert store["step"]["opened"] == 1 and store["total"]["opened"] == 1
    assert _same(before, all_arrays(grid)), "opening a split writes no array: the quanta stay in the boxes"


def test_outflow_into_one_volume_met_no_fork_and_opens_nothing(grid, volumes):
    send(grid, volumes[0], volumes[1], 216)
    _open(grid)
    assert len(grid.quantum["source"]) == 0 and grid.quantum["step"]["opened"] == 0


def test_every_forking_volume_gets_its_own_split_and_later_outflow_feeds_the_open_one(grid, volumes):
    home, first, second, other = volumes[:4]
    for giver in (home, other):
        send(grid, giver, first, 100)
        send(grid, giver, second, 50, lost=50)
    _open(grid)
    assert grid.quantum["source"].tolist() == sorted([home, other])
    born = grid.quantum["born"].copy()
    next_step(grid)
    send(grid, home, first, 10)
    send(grid, home, second, 20)
    _open(grid)
    store = grid.quantum
    assert store["born"].tolist() == born.tolist(), "a fed split keeps the step it was born at"
    assert amount_of(store).tolist() == [180 if source == home else 150 for source in store["source"]]
    assert store["step"]["joined"] == 1


def _held_before(grid, volumes) -> None:
    set_cells(grid, "quanta_before", {cell: HELD for cell in volumes})


def test_a_half_goes_on_with_what_its_cell_sent_and_no_quantum_is_claimed_twice(grid, volumes):
    home, first, second, third, fourth = volumes[:5]
    _held_before(grid, volumes)
    send(grid, home, first, 1080)
    send(grid, home, second, 540)
    _open(grid)
    next_step(grid)
    send(grid, first, third, HELD // 4)
    send(grid, first, fourth, HELD // 4)
    before = all_arrays(grid)
    _open(grid)
    store = grid.quantum
    halves = {(int(source), int(cell)): int(quanta) for source, cell, quanta in
              zip(store["part_source"], store["part_cell"], store["part_quanta"])}
    gone = halves.get((home, third), 0) + halves.get((home, fourth), 0)
    assert halves[(home, first)] == 540 and gone == 540, "every quantum that went on lies in one taking volume"
    assert amount_of(store)[store["source"] == home].tolist() == [1620], "the split still holds every quantum"
    assert halves.get((first, third), 0) == HELD // 4 - halves.get((home, third), 0), "only what was no split's opens a new one"
    assert store["step"]["carried"] == 540 and _same(before, all_arrays(grid))


def test_friction_takes_a_split_s_direction_crossing_by_crossing_until_none_is_left(grid, volumes):
    home, first, second, third = volumes[:4]
    _held_before(grid, volumes)
    send(grid, home, first, 1080)
    send(grid, home, second, 1080)
    _open(grid)
    assert grid.quantum["directed"].tolist() == [2160]
    next_step(grid)
    send(grid, first, third, HELD, lost=HELD // 2)
    _open(grid)
    store = grid.quantum
    at_home = store["source"] == home
    assert store["directed"][at_home].tolist() == [2160 - 540] and store["step"]["cooled"] == 540
    next_step(grid)
    send(grid, second, third, HELD, lost=HELD)
    send(grid, third, first, HELD, lost=HELD)
    _open(grid)
    assert grid.quantum["directed"][grid.quantum["source"] == home].tolist() == [0], "no momentum left: it settles next"


def test_a_cell_that_sent_nothing_keeps_its_half_and_a_crossing_elsewhere_changes_no_split(grid, volumes):
    home, first, second, other, far = volumes[:5]
    _held_before(grid, volumes)
    send(grid, home, first, 100)
    send(grid, home, second, 50)
    _open(grid)
    kept = {name: grid.quantum[name].copy() for name in ("part_cell", "part_quanta", "directed")}
    next_step(grid)
    send(grid, other, far, 40, lost=40)
    _open(grid)
    assert all(np.array_equal(kept[name], grid.quantum[name]) for name in kept)
    assert not grid.quantum["reacted"].any(), "friction in another cell is no reaction with a half"


def test_a_pointer_whose_half_went_on_whole_jumps_to_a_cell_its_split_lies_in_now(grid, volumes):
    home, first, second, third = volumes[:4]
    _held_before(grid, volumes)
    send(grid, home, first, 100)
    send(grid, home, second, 50)
    _open(grid)
    grid.quantum["pointer"][0] = first
    next_step(grid)
    send(grid, first, third, HELD)
    _open(grid)
    store = grid.quantum
    mine = store["part_source"] == home
    assert first not in store["part_cell"][mine].tolist(), "the half went on whole"
    assert int(store["pointer"][store["source"] == home][0]) in store["part_cell"][mine].tolist()
