"""`carry_events` and `follow` on staged event records: the tables follow the cells."""
from __future__ import annotations

from physics.engine.axiom_7_quantum.definitions.split_store import REACTION, amount_of
from physics.engine.axiom_7_quantum.logic.carry_events import carry_events
from physics.engine.axiom_7_quantum.logic.follow import follow
from physics.engine.axiom_7_quantum.logic.tests.conftest import open_fork, HELD, held_by, send


def _split(grid, home: int, halves: dict) -> None:
    for cell, quanta in halves.items():
        send(grid, home, cell, quanta)
    open_fork(grid)


def _parts(store: dict) -> list:
    return list(zip(store["part_source"].tolist(), store["part_cell"].tolist(), store["part_quanta"].tolist()))


def test_nothing_changed_nothing_carried(grid, volumes):
    home, first, second = volumes[:3]
    _split(grid, home, {first: 216, second: 432})
    before = _parts(grid.quantum)
    carry_events(grid)
    assert _parts(grid.quantum) == before and not grid.quantum["reacted"].any()


def test_a_collapse_moves_a_half_and_a_pointer_into_the_kept_volume_and_reacts(grid, volumes):
    home, first, second, third = volumes[:4]
    _split(grid, home, {first: 216, second: 432})
    grid.quantum["pointer"][0] = second
    grid.changed = [{"what": "collapse", "joint": 1, "kept": third, "gone": second}]
    carry_events(grid)
    store = grid.quantum
    assert _parts(store) == sorted([(home, first, 216), (home, third, 432)])
    assert store["pointer"].tolist() == [third] and store["reacted"].tolist() == [True]


def test_two_halves_that_come_to_lie_in_one_volume_are_one_half(grid, volumes):
    home, first, second = volumes[:3]
    _split(grid, home, {first: 216, second: 432})
    grid.changed = [{"what": "collapse", "joint": 1, "kept": first, "gone": second}]
    carry_events(grid)
    assert _parts(grid.quantum) == [(home, first, 648)] and grid.quantum["pointer"].tolist() == [first]


def test_a_split_whose_source_merged_away_is_the_kept_volume_s(grid, volumes):
    home, first, second, third = volumes[:4]
    _split(grid, home, {first: 216, second: 432})
    grid.changed = [{"what": "collapse", "joint": 1, "kept": third, "gone": home}]
    carry_events(grid)
    assert grid.quantum["source"].tolist() == [third] and amount_of(grid.quantum).tolist() == [648]


def test_a_collapse_elsewhere_and_one_with_no_volume_left_touch_no_split(grid, volumes):
    home, first, second, third, fourth = volumes[:5]
    _split(grid, home, {first: 216, second: 432})
    grid.changed = [{"what": "collapse", "joint": 1, "kept": third, "gone": fourth},
                    {"what": "collapse", "joint": 3, "kept": None, "gone": None}]
    carry_events(grid)
    assert not grid.quantum["reacted"].any() and _parts(grid.quantum) == sorted([(home, first, 216), (home, second, 432)])


def test_a_tear_shares_a_half_over_the_pieces_and_the_joint_s_share_leaves_the_split(grid, volumes):
    home, first, second = volumes[:3]
    new_joint, new_volume = 200, 201
    _split(grid, home, {first: 300, second: 432})
    grid.quantum["pointer"][0] = second
    grid.changed = [{"what": "tear", "volume": first, "joint": new_joint, "new_volume": new_volume,
                     "quanta": [720, 720, 720]}]
    carry_events(grid)
    store = grid.quantum
    assert _parts(store) == sorted([(home, first, 100), (home, new_volume, 100), (home, second, 432)])
    assert store["step"]["torn_quanta"] == 100 and store["reacted"].tolist() == [True]
    assert store["directed"].tolist() == [632], "direction is never more than the quanta left"
    assert store["pointer"].tolist() == [second]


def test_a_pointer_on_a_torn_volume_lands_on_one_of_its_two_plus_pieces(grid, volumes):
    home, first, second = volumes[:3]
    _split(grid, home, {first: 300, second: 432})
    grid.quantum["pointer"][0] = first
    grid.changed = [{"what": "tear", "volume": first, "joint": 200, "new_volume": 201, "quanta": [1, 1, 5000]}]
    carry_events(grid)
    assert grid.quantum["pointer"].tolist() == [201], "all but nothing of the half went to the new piece"


def test_follow_settles_in_the_same_step_the_split_an_event_reacted_with(grid, volumes):
    home, first, second, third = volumes[:4]
    _split(grid, home, {first: 216, second: 432})
    grid.changed = [{"what": "collapse", "joint": 1, "kept": first, "gone": first}]
    follow(grid)
    store = grid.quantum
    assert len(store["source"]) == 0 and store["settled"]["cause"].tolist() == [REACTION]
    assert held_by(grid, first) + held_by(grid, second) == 2 * HELD and store["settled"]["amount"].tolist() == [648]
