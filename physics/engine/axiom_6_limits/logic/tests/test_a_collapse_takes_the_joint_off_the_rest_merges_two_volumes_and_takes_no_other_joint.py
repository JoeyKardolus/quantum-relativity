"""Collapses on a real small universe: exact books, two volumes merged, no other joint taken, none left past its limit."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from physics import engine
from physics.engine.axiom_1_energy.logic.clear_ledgers import clear_ledgers
from physics.engine.axiom_1_energy.logic.count_cells import count_cells
from physics.engine.axiom_6_limits.definitions.held import Held
from physics.engine.axiom_6_limits.logic.collapse import collapse
from physics.engine.axiom_6_limits.logic.find_limits import find_limits
from physics.engine.axiom_6_limits.logic.make_minus import make_minus
from physics.engine.axiom_6_limits.logic.run_events import run_events

wp.config.quiet = True


def _universe(side: int = 4):
    settings = engine.Settings(pull_strength=1e-13)
    return engine.new_universe(settings, cells_per_side=side, seed=1, device="cpu"), settings


def _put_back(state, held) -> None:
    for cell in held.cells:
        state.set_neighbours(cell, held.entries[cell])
    for cell in held.quanta:
        state.set_block("quanta", cell, held.quanta[cell])
        state.set_block("phase", cell, held.phase[cell])
    engine.relink(state, held.cells)
    count_cells(state)


def _joint_with(state, ways: int) -> int:
    return next(int(cell) for cell in np.flatnonzero(state.host("kind") == engine.JOINT)
                if len(state.ways_of(int(cell))) == ways)


EXACT = {"amount": 0, "plus": 0, "minus": 0, "taken": 0}
UNBROKEN = ("touching", "bare", "one_sided", "dead_links", "loose_ways", "missing_ways", "shared_mouths", "flickered",
            "lone_joints")


def _count(state, kind: int) -> int:
    return int((state.host("kind") == kind).sum())


def _pair(way: tuple) -> frozenset:
    return frozenset(way[1:3])


def test_a_face_joint_goes_its_two_volumes_become_one_and_as_much_again_comes_off_them():
    state, settings = _universe(side=8)
    joint = _joint_with(state, 1)
    (slot, way_a, way_b, mouth_a, mouth_b), = state.ways_of(joint)
    held_joint = int(state.host("cell_quanta")[joint])
    both = int(state.host("cell_quanta")[way_a]) + int(state.host("cell_quanta")[way_b])
    held = Held()
    record = collapse(state, held, joint)
    _put_back(state, held)
    assert (record["kept"], record["gone"], record["joint_type"], record["volumes"]) == (way_a, way_b, "face", 2)
    assert record["taken_joint"] == held_joint == record["taken_volume"]
    assert int(state.host("cell_quanta")[way_a]) == both - held_joint
    assert state.host("kind")[[joint, way_b]].tolist() == [0, 0] and state.host("fell_into")[way_b] == way_a
    books = engine.read_books(state)
    assert engine.book_law(state.start_books, books) == EXACT
    assert books["wrong_boxes"] == books["wrong_cells"] == 0 and books["short_takes"] == 0
    pattern = engine.check_pattern(state)
    assert pattern["web_pieces"] == 1 and not any(pattern[key] for key in UNBROKEN)


def test_only_the_collapsed_joint_goes_every_joint_round_the_merged_pair_stays():
    state, settings = _universe(side=8)
    joints, volumes = _count(state, engine.JOINT), _count(state, engine.ENERGY_VOLUME)
    held = Held()
    record = collapse(state, held, _joint_with(state, 1))
    _put_back(state, held)
    assert _count(state, engine.JOINT) == joints - 1 and _count(state, engine.ENERGY_VOLUME) == volumes - 1
    assert state.collapses == 1 and state.merges == 1 and "swallowed" not in record
    round_kept = {one for one, code in state.neighbours(record["kept"])}
    assert len(round_kept) == 26 + 26 - 9 - 1           # both volumes' joints, the 9 they shared once, less the one gone


def test_the_joints_that_lay_on_opposite_sides_of_the_gone_joint_now_touch_each_other():
    state, settings = _universe(side=8)
    joint = _joint_with(state, 1)
    kind = state.host("kind")
    beside = {code: one for one, code in state.neighbours(joint) if kind[one] == engine.JOINT}
    pairs = [(beside[code], beside[26 - code], code) for code in beside if code < 13 and 26 - code in beside]
    assert len(pairs) == 12 and not any((far, 26 - code) in state.neighbours(near) for near, far, code in pairs)
    held = Held()
    collapse(state, held, joint)
    _put_back(state, held)
    for near, far, code in pairs:
        assert (far, 26 - code) in state.neighbours(near) and (near, code) in state.neighbours(far)
    assert engine.check_pattern(state)["one_sided"] == 0


def test_a_corner_joint_merges_only_the_two_volumes_of_its_hardest_way_and_no_other_cell_gains_a_relation():
    state, settings = _universe(side=8)
    joint = _joint_with(state, 4)
    hardest = state.ways_of(joint)[2]
    state.set_cell("squeezed_way", joint, hardest[0])
    others = [way for way in state.ways_of(joint) if way[0] != hardest[0]]
    before = {volume: set(state.neighbours(volume)) for way in others for volume in way[1:3]}
    held = Held()
    record = collapse(state, held, joint)
    _put_back(state, held)
    assert record["volumes"] == 8 and record["joint_type"] == "corner"
    assert {record["kept"], record["gone"]} == set(hardest[1:3]) and record["kept"] < record["gone"]
    assert _count(state, engine.ENERGY_VOLUME) == 63 and _count(state, engine.JOINT) == 447
    for volume, entries in before.items():
        assert set(state.neighbours(volume)) == {entry for entry in entries if entry[0] != joint}
    pattern = engine.check_pattern(state)
    assert pattern["fewest_joints"] == 25 and pattern["web_pieces"] == 1
    assert not any(pattern[key] for key in UNBROKEN)


def _merged_three(state) -> tuple:
    """Two face joints a-b and a-c, and a joint with a way b-c and another way: `(first, second, third, its ways)`."""
    faces = {_pair(state.ways_of(int(cell))[0]): int(cell) for cell in np.flatnonzero(state.host("kind") == engine.JOINT)
             if len(state.ways_of(int(cell))) == 1}
    for cell in np.flatnonzero(state.host("kind") == engine.JOINT).tolist():
        ways = state.ways_of(cell)
        if len(ways) != 2:
            continue
        for way in ways:
            low, high = way[1:3]
            for middle in range(state.count):
                if frozenset((middle, low)) in faces and frozenset((middle, high)) in faces:
                    return faces[frozenset((middle, low))], faces[frozenset((middle, high))], cell, way, ways
    raise AssertionError("no such three joints in this lattice")


def test_the_hardest_way_is_followed_through_the_merges_of_the_same_step():
    state, settings = _universe(side=8)
    first, second, third, way, ways = _merged_three(state)
    other = next(one for one in ways if one[0] != way[0])
    low, high = way[1:3]
    held = Held()
    merged = collapse(state, held, first)["kept"]
    state.set_cell("squeezed_way", third, other[0])
    held.values.setdefault("squeezed_way", {})[third] = other[0]
    record = collapse(state, held, third)
    _put_back(state, held)
    # the way table is stale inside a step, yet the two that merge are the hardest way's two as they stand now
    stands = {volume: (merged if state.host("fell_into")[volume] == merged else volume) for volume in other[1:3]}
    assert {record["kept"], record["gone"]} <= set(stands.values()) | set(other[1:3])
    assert record["kept"] != record["gone"] and record["joint_type"] == "edge"
    assert engine.book_law(state.start_books, engine.read_books(state)) == EXACT


def test_when_the_hardest_way_s_two_volumes_are_one_by_now_the_next_hardest_way_s_two_merge():
    state, settings = _universe(side=8)
    first, second, third, way, ways = _merged_three(state)
    other = next(one for one in ways if one[0] != way[0])
    held = Held()
    collapse(state, held, first)
    collapse(state, held, second)                      # now the two volumes of `way` are one volume
    state.set_cell("squeezed_way", third, way[0])
    held.values.setdefault("squeezed_way", {})[third] = way[0]
    record = collapse(state, held, third)
    _put_back(state, held)
    kept = int(state.host("fell_into")[way[2]]) if state.host("kind")[way[2]] == 0 else way[2]
    now = {int(state.host("fell_into")[volume]) if state.host("kind")[volume] == 0 else volume for volume in other[1:3]}
    assert record["kept"] != record["gone"]
    assert record["kept"] in now | {kept} and state.host("fell_into")[record["gone"]] == record["kept"]
    assert state.merges == 3 and state.collapses == 3


def test_a_joint_none_of_whose_ways_still_has_two_volumes_only_goes_and_its_energy_is_still_taken_off_the_rest():
    state, settings = _universe(side=4)
    joint = _joint_with(state, 1)
    (slot, way_a, way_b, mouth_a, mouth_b), = state.ways_of(joint)
    twin = next(int(cell) for cell in np.flatnonzero(state.host("kind") == engine.JOINT)
                if cell != joint and len(state.ways_of(int(cell))) == 1 and _pair(state.ways_of(int(cell))[0])
                == frozenset((way_a, way_b)))            # 4 a side wraps: two face joints lie between the same two
    held = Held()
    collapse(state, held, joint)
    held_twin = int(state.host("cell_quanta")[twin])
    record = collapse(state, held, twin)
    _put_back(state, held)
    assert record["joint_type"] == "closed" and record["kept"] == record["gone"] == way_a
    assert record["taken_joint"] == held_twin == record["taken_volume"]
    assert state.merges == 1 and state.collapses == 2
    assert engine.book_law(state.start_books, engine.read_books(state)) == EXACT


def test_a_collapse_that_finds_less_than_its_joint_held_takes_all_there_is_and_books_the_rest_as_short():
    state, settings = _universe(side=8)
    joint = _joint_with(state, 1)
    (slot, way_a, way_b, mouth_a, mouth_b), = state.ways_of(joint)
    for volume in (way_a, way_b):
        state.set_block("quanta", volume, np.ones((6, 6, 6), dtype=np.int32))
    count_cells(state)
    held_joint = int(state.host("cell_quanta")[joint])
    held = Held()
    record = collapse(state, held, joint)
    _put_back(state, held)
    assert record["taken_joint"] == held_joint and record["taken_volume"] == 432
    assert state.short_takes == 1 and state.short_quanta == held_joint - 432
    assert int(state.host("cell_quanta")[way_a]) == 0
    assert engine.book_law(state.start_books, engine.read_books(state))["taken"] == 0


def test_a_squeezed_joint_makes_whole_minus_quanta_in_its_own_boxes_and_an_unsqueezed_one_nothing():
    state, settings = _universe()
    joints = np.flatnonzero(state.host("kind") == engine.JOINT)
    squeezed, loose = int(joints[0]), int(joints[1])
    clear_ledgers(state)
    state.set_cell("squeeze", squeezed, 16.0)
    state.set_cell("squeeze", loose, 1.0)
    block = state.block("quanta", squeezed)
    block[0, 0, 0] = 0
    state.set_block("quanta", squeezed, block)
    count_cells(state)
    before = state.quanta.numpy().copy()
    make_minus(state, settings)
    after = state.quanta.numpy()
    made = int(state.host("made")[squeezed])
    holds = int(state.host("cell_quanta")[squeezed])
    rate = float(state.host("rate")[squeezed])
    assert abs(made - holds * (1.0 - np.exp(-0.25 * rate))) <= 1 and state.made_total == made
    assert int(before[squeezed].sum() - after[squeezed].sum()) == made and (after[squeezed] <= before[squeezed]).all()
    assert after[squeezed][0, 0, 0] == 0
    others = np.ones(state.capacity, dtype=bool)
    others[squeezed] = False
    assert np.array_equal(after[others], before[others]) and int(state.host("made").sum()) == made


def test_a_making_larger_than_a_box_can_hold_is_refused_and_writes_nothing():
    state, settings = _universe()
    squeezed = int(np.flatnonzero(state.host("kind") == engine.JOINT)[0])
    state.set_block("quanta", squeezed, np.full((engine.BLOCK,) * 3, -2 ** 30, dtype=np.int32))     # a joint whose boxes are half full
    count_cells(state)
    clear_ledgers(state)
    state.set_cell("squeeze", squeezed, 1.0e8)
    before = state.quanta.numpy().copy()
    with pytest.raises(OverflowError):
        make_minus(state, settings)
    assert np.array_equal(before, state.quanta.numpy()) and state.made_total == 0


def test_every_joint_past_its_limit_collapses_in_the_same_step_also_one_an_earlier_event_touched():
    state, settings = _universe()
    joints = np.flatnonzero(state.host("kind") == engine.JOINT)
    for joint in joints[:6].tolist():
        state.made_total += 2 * int(-state.block("quanta", joint).sum())    # staged as made, so the books hold
        state.set_block("quanta", joint, state.block("quanta", joint) * 3)
    count_cells(state)
    clear_ledgers(state)
    find_limits(state, settings)
    assert sorted(np.flatnonzero(state.host("flags") == 1).tolist()) == joints[:6].tolist()
    assert state.found.numpy().tolist() == [6, 0]
    engine.relink(state, run_events(state, settings))
    count_cells(state)
    kind, quanta, fresh = state.host("kind"), state.host("cell_quanta"), state.host("fresh_quanta")
    assert not kind[joints[:6]].any() and int((kind == engine.JOINT).sum()) == len(joints) - 6
    living = kind == engine.JOINT
    assert not (living & (quanta >= 3 * fresh)).any()
    assert engine.book_law(state.start_books, engine.read_books(state)) == EXACT
    pattern = engine.check_pattern(state)
    assert pattern["one_sided"] == 0 and pattern["touching"] == 0 and pattern["loose_ways"] == 0
