"""Axiom 7, Quantum: energy split over several routes settles whole in one cell, chosen by each cell's share.

The owner's axiom (bible physics/quantum-relativity, "The axioms", 2026-11-16): each box's
fluid also carries a phase, so waves add and cancel. Moving energy can split over several
routes at once. Every split gets a direct pull between its halves, private to them, and
its own oscillation, a shared state that keeps turning, starting at the split at a speed
nobody can know. At each turn a pointer jumps to one of the halves' cells, with that
cell's share of the energy as its chance. When something reacts with a half, or the
energy loses its momentum, it settles whole in the cell the pointer is on.

Staged (quantum_hand.py: the engine's own crossing takes all direction off what crosses,
so a splitter and a route are placed by the scene): the bench is new_universe(32), all
rules on, defaults; DRAWS = 256 squares of four energy volumes S, A, R1, R2 that share
no volume, each with its own packet of 1000 of its S's quanta and its own draws (a
moving energy's draws are seeded by the state's seed and its serial). S-A is the
splitter, S-R1 and A-R2 the two routes, R1-R2 where the routes are brought back together.
Bands were fixed before the first run, 3 standard deviations of 256 draws: a share of
0.5 must land in 0.40 .. 0.60 of the draws, 0.7 in 0.61 .. 0.79, 0.3 in 0.21 .. 0.39.
Not built: filter-angle (Bell) tests on pairs of whirls.
"""
import math

import numpy as np
import pytest

from physics.engine import (Settings, new_universe, quantum_hand, quantum_move, quantum_packet, quantum_react,
                            quantum_store, step, totals)
from physics.engine.tests._gpu import needs_gpu

DRAWS = 256
SIDE = 32
PACKET = 1000
HALF = math.pi / 2.0
THIRTY = 2.0 * math.asin(math.sqrt(0.3))


def _squares(count: int = DRAWS) -> list:
    """`count` squares (S, A, R1, R2) of energy volumes sharing no volume: A beside S along the first axis, R beside along the second."""
    found = []
    for first in range(0, SIDE, 4):
        for second in range(0, SIDE, 4):
            for third in range(0, SIDE, 2):
                found.append(tuple((one * SIDE + two) * SIDE + third for one, two in
                                   ((first, second), (first + 2, second), (first, second + 2), (first + 2, second + 2))))
    return found[:count]


def _bench(seed: int, count: int = DRAWS) -> tuple:
    """The bench universe with a packet named in the S of each of `count` squares."""
    state = new_universe(SIDE, seed=seed)
    squares = _squares(count)
    for square in squares:
        quantum_packet(state, square[0], PACKET)
    return state, squares


def _run(state, steps: int) -> None:
    for _ in range(steps):
        step(state, Settings())


def _hand(state, squares: list, giver: int, taker: int, turn: float) -> list:
    """The same handover in every square, between its corners number `giver` and `taker`."""
    return [moved for square in squares for _, moved in quantum_hand(state, square[giver], square[taker], turn)]


def _react(state, squares: list, corner: int) -> list:
    return [event for square in squares for event in quantum_react(state, square[corner])]


def _landed(events: list, squares: list, corner: int) -> float:
    """The share of the events that settled in corner number `corner` of their square."""
    homes = {square[corner] for square in squares}
    return sum(event["cell"] in homes for event in events) / len(events)


@needs_gpu
def test_a_packet_split_50_50_settles_whole_in_one_side_about_half_the_draws_each():
    state, squares = _bench(1000)
    assert all(499 <= moved <= 501 for moved in _hand(state, squares, 0, 1, HALF))
    _run(state, 2)
    assert all(len(record["cells"]) == 2 and record["jumps"] >= 2 for record in quantum_store(state)["moving"])
    before = state.host("cell_quanta").copy()
    books = totals(state)
    events = _react(state, squares, 1)
    change = state.host("cell_quanta") - before
    after = totals(state)
    assert len(events) == DRAWS and not quantum_store(state)["moving"]
    for event, square in zip(events, squares):
        other = square[0] if event["cell"] == square[1] else square[1]
        assert event["amount"] == PACKET and event["drawn"] == event["parts"][other]
        assert change[event["cell"]] == event["drawn"] and change[other] == -event["drawn"]
    assert np.count_nonzero(change) == 2 * DRAWS
    assert after["amount"] == books["amount"] and after["wrong_cells"] == 0 and after["wrong_boxes"] == 0
    share = _landed(events, squares, 0)
    print(f"50/50 split, reaction in A after 2 steps: whole in S in {share:.3f} of {DRAWS} draws, whole in A in the rest")
    assert 0.40 <= share <= 0.60


@needs_gpu
def test_the_pointer_rides_with_the_energy_so_a_reaction_right_at_the_split_also_gives_the_shares():
    state, squares = _bench(6000)
    _hand(state, squares, 0, 1, HALF)
    share = _landed(_react(state, squares, 0), squares, 0)
    print(f"50/50 split, reaction in S with no step in between: whole in S in {share:.3f} of {DRAWS} draws")
    assert 0.40 <= share <= 0.60


@needs_gpu
def test_a_70_30_split_settles_in_the_70_side_in_about_70_percent_never_by_which_half_reacted():
    shares = {}
    for reacting in (0, 1):
        state, squares = _bench(11000 + 10000 * reacting)
        assert all(299 <= moved <= 301 for moved in _hand(state, squares, 0, 1, THIRTY))
        _run(state, 2)
        shares[reacting] = _landed(_react(state, squares, reacting), squares, 0)
    print(f"70/30 split: whole in the 70% cell in {shares[0]:.3f} of {DRAWS} draws when the 70% half reacts, "
          f"{shares[1]:.3f} when the 30% half reacts")
    assert 0.61 <= shares[0] <= 0.79 and 0.61 <= shares[1] <= 0.79


def _two_routes(state, squares: list) -> None:
    """Split 50/50 over S and A, hand each half on along its route (S to R1, A to R2) and run 3 steps."""
    _hand(state, squares, 0, 1, HALF)
    _run(state, 1)
    _hand(state, squares, 0, 2, math.pi)
    _hand(state, squares, 1, 3, math.pi)
    _run(state, 3)


@needs_gpu
def test_two_routes_brought_back_together_interfere_and_one_exit_gets_nearly_all():
    exits = {}
    for turn in (HALF, -HALF):
        state, squares = _bench(31000)
        _two_routes(state, squares)
        moving = quantum_store(state)["moving"]
        assert all(sorted(record["cells"]) == sorted(square[2:]) for record, square in zip(moving, squares))
        _hand(state, squares, 2, 3, turn)
        parts = [dict(zip(record["cells"], record["parts"])).get(square[3], 0) for record, square in zip(moving, squares)]
        merged = sum(event["cause"] == "merge" for event in quantum_store(state)["ended"])
        events = _react(state, squares, 2) + _react(state, squares, 3)
        assert len(events) == DRAWS
        exits[turn] = (min(parts), max(parts), _landed(events, squares, 3), merged)
    low, high, share, merged = exits[HALF]
    print(f"two routes rejoined in step: exit R2 holds {low} .. {high} of {PACKET} quanta, whole in R2 in {share:.3f} "
          f"of {DRAWS} draws, {merged} merged back before any reaction")
    back_low, back_high, back_share, back_merged = exits[-HALF]
    print(f"rejoined with the opposite turn: exit R2 holds {back_low} .. {back_high}, whole in R2 in {back_share:.3f}, "
          f"{back_merged} merged back")
    assert low >= 0.99 * PACKET and share >= 0.97
    assert back_high <= 0.01 * PACKET and back_share <= 0.03


@needs_gpu
def test_a_reaction_in_one_route_takes_the_interference_away():
    state, squares = _bench(41000)
    _two_routes(state, squares)
    reacted = _react(state, squares, 2)
    assert len(reacted) == DRAWS and not quantum_store(state)["moving"]
    for event in reacted:
        quantum_packet(state, event["cell"], PACKET)
    route = _landed(reacted, squares, 2)
    _run(state, 1)
    _hand(state, squares, 2, 3, HALF)
    moving = quantum_store(state)["moving"]
    by_cell = {cell: part for record in moving for cell, part in zip(record["cells"], record["parts"])}
    assert all(499 <= by_cell.get(square[3], 0) <= 501 for square in squares)
    events = _react(state, squares, 2) + _react(state, squares, 3)
    share = _landed(events, squares, 3)
    print(f"reaction in route R1: it settled there in {route:.3f} of {DRAWS} draws; sent on from where it settled and "
          f"brought to the same joining, exit R2 holds 499 .. 501 of {PACKET} and gets it whole in {share:.3f} of the draws")
    assert len(events) == DRAWS and 0.40 <= route <= 0.60 and 0.40 <= share <= 0.60


@needs_gpu
def test_no_message_the_far_sides_odds_are_the_same_with_or_without_a_reaction_on_the_near_side():
    found, spread = {}, {}
    for near_reacts in (True, False):
        state, squares = _bench(51000 + 10000 * near_reacts)
        _hand(state, squares, 0, 1, THIRTY)
        _run(state, 1)
        if near_reacts:
            _react(state, squares, 0)
        _run(state, 2)
        far = {square[1] for square in squares}
        held = [sum(part for cell, part in zip(record["cells"], record["parts"]) if cell in far)
                for record in quantum_store(state)["moving"]]
        held += [PACKET * (event["cell"] in far) for event in quantum_store(state)["ended"]]
        spread[near_reacts] = (min(held), max(held), float(np.mean(held)))
        _react(state, squares, 1)
        ended = quantum_store(state)["ended"]
        assert len(ended) == DRAWS
        found[near_reacts] = _landed(ended, squares, 1)
    print(f"far side (the 30% half) finds the energy whole in {found[True]:.3f} of {DRAWS} draws with a reaction on the "
          f"near side first, {found[False]:.3f} without")
    print(f"the far cell's part of the energy just before its own reaction: {spread[True][0]} .. {spread[True][1]} "
          f"(mean {spread[True][2]:.0f}) with the near reaction, {spread[False][0]} .. {spread[False][1]} "
          f"(mean {spread[False][2]:.0f}) without")
    assert 0.21 <= found[True] <= 0.39 and 0.21 <= found[False] <= 0.39
    assert abs(found[True] - found[False]) <= 0.12


@needs_gpu
def test_quanta_stay_exact_through_a_settle_whatever_the_distance():
    state, squares = _bench(71000, 16)
    far = [square[0] + 12 * SIDE * SIDE for square in squares]
    _hand(state, squares, 0, 1, THIRTY)
    for square, cell in zip(squares, far):
        record = next(record for record in quantum_store(state)["moving"] if square[1] in record["cells"])
        index = record["cells"].index(square[1])
        quantum_move(state, [(square[1], cell, record["parts"][index])])
        record["cells"][index] = cell
    _run(state, 2)
    before, held = totals(state), state.host("cell_quanta").copy()
    boxes = state.quanta.numpy().copy()
    events = [event for cell in far for event in quantum_react(state, cell)]
    after, change = totals(state), state.host("cell_quanta") - held
    touched = np.flatnonzero((state.quanta.numpy() != boxes).reshape(state.capacity, -1).any(axis=1))
    assert len(events) == 16
    assert all(after[name] == before[name] for name in ("amount", "plus", "minus"))
    assert after["wrong_boxes"] == 0 and after["wrong_cells"] == 0
    for event, square, cell in zip(events, squares, far):
        other = square[0] if event["cell"] == cell else cell
        assert change[event["cell"]] == event["parts"][other] and change[other] == -event["parts"][other]
        assert event["step"] == state.step
    print(f"settle between cells 6 volumes apart, 16 draws: amount, plus and minus unchanged, the two cells' quanta "
          f"changed by exactly the drawn part, cells whose boxes changed: {len(touched)} (two per draw)")
    assert len(touched) == 32 and np.count_nonzero(change) == 32


@needs_gpu
def test_a_collapse_that_changes_a_halfs_cell_is_a_reaction_and_the_energy_settles_whole():
    state, settings = new_universe(8, unevenness=0.0), Settings(crossing_rate=0.0)
    low, gone = (0 * 8 + 2) * 8 + 2, (2 * 8 + 2) * 8 + 2
    beside_low, beside_gone = (0 * 8 + 4) * 8 + 2, (2 * 8 + 4) * 8 + 2
    for volume in (low, gone):
        state.fill(volume, 5 * 648000)
    start = totals(state)
    quantum_packet(state, gone, PACKET)
    quantum_hand(state, gone, beside_gone, HALF)
    quantum_packet(state, 4 * 64 + 4 * 8 + 4, PACKET)
    store = quantum_store(state)
    for _ in range(60):
        step(state, settings)
        if state.collapses:
            break
    assert state.collapses >= 1 and state.kind_h[gone] == 0
    events = {event["serial"]: event for event in store["ended"]}
    books = totals(state)
    print(f"joint collapse in step {state.step}: the split with a half in the volume that fell in settled by "
          f"'{events[0]['cause']}' whole in cell {events[0]['cell']}; the whole packet far away moves on: "
          f"{[record['serial'] for record in store['moving']]}")
    assert events[0]["cause"] == "collapse or tear" and events[0]["cell"] in (low, beside_gone)
    assert events[0]["amount"] == PACKET and [record["serial"] for record in store["moving"]] == [1]
    assert books["wrong_cells"] == 0 and books["wrong_boxes"] == 0
    assert books["amount"] == start["amount"] + books["made"] - books["taken_joint"] - books["taken_volumes"]


@needs_gpu
def test_friction_takes_direction_off_a_split_and_with_none_left_it_settles_whole():
    """The crossing's friction rubs direction off in whole quanta over 20 steps; the last step is staged.

    Losing all of 1000 quanta's direction takes thousands of steps, so the end is staged: in half the draws the
    split's directed count is set to 0 by hand, where the rub would bring it; in the next step each of those
    settles whole by "momentum" and the others move on.
    """
    state, squares = _bench(81000, 32)
    start = totals(state)["plus"]
    _hand(state, squares, 0, 1, HALF)
    moving = quantum_store(state)["moving"]
    directed = [[record["directed"] for record in moving]]
    for _ in range(20):
        _run(state, 1)
        directed.append([record["directed"] for record in moving])
    falls = np.diff(np.array(directed), axis=0)
    lost = PACKET - np.array(directed[-1])
    print(f"20 steps, 32 draws: a 50/50 split of {PACKET} quanta lost the direction of {lost.min()} .. "
          f"{lost.max()} quanta (mean {lost.mean():.1f}), none settled")
    assert len(moving) == 32 and np.all(falls <= 0) and lost.sum() > 0
    spent = [record for record in moving if record["serial"] % 2 == 0]
    for record in spent:
        record["directed"] = 0
    _run(state, 1)
    ended = quantum_store(state)["ended"]
    assert sorted(event["serial"] for event in ended) == list(range(0, 32, 2))
    assert all(event["cause"] == "momentum" and event["drawn"] in event["parts"].values() for event in ended)
    assert len(quantum_store(state)["moving"]) == 16
    assert totals(state)["plus"] == start


@needs_gpu
def test_a_handover_needs_two_volumes_with_a_joint_between_and_a_move_never_empties_a_box():
    state, squares = _bench(1, 1)
    source, beside, _, across = squares[0]
    with pytest.raises(ValueError):
        quantum_hand(state, source, across, HALF)
    with pytest.raises(ValueError):
        quantum_packet(state, source + 1, 10)
    held = int(state.host("cell_quanta")[source])
    with pytest.raises(ValueError):
        quantum_move(state, [(source, beside, held)])
