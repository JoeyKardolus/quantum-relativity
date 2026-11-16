"""Axiom 7, Quantum: energy split over several routes settles whole in one cell, chosen by each cell's share.

The owner's axiom (bible physics/quantum-relativity, "The axioms", 2026-11-16): each box's
fluid also carries a phase, so waves add and cancel. Moving energy can split over several
routes at once. Every split gets a direct pull between its halves, private to them, and
its own oscillation, a shared state that keeps turning, starting at the split at a speed
nobody can know. At each turn a pointer jumps to one of the halves' cells, with that
cell's share of the energy as its chance. When something reacts with a half, or the
energy loses its momentum, it settles whole in the cell the pointer is on.

Staged (quantum_hand.py: the engine's own crossing takes all direction off what crosses,
so a splitter and a route are placed by the scene): the bench is new_universe(8, 6,
seed=1), 64 energy volumes of 6 boxes across wrapping round, all rules on, defaults,
copied SEEDS times side by side with seeds FIRST_SEED + u (the seed sets each
universe's rounding and each moving energy's own draws). A packet is 1000 of volume S's
quanta. S, A, R1, R2 are four volumes in a square: S-A the splitter, S-R1 and A-R2 the
two routes, R1-R2 where the routes are brought back together.
Bands were fixed before the first run, 3 standard deviations of SEEDS draws: a share of
0.5 must land in 0.40 .. 0.60 of the seeds, 0.7 in 0.61 .. 0.79, 0.3 in 0.21 .. 0.39.
Not built: filter-angle (Bell) tests on pairs of whirls.
"""
import functools
import math

import numpy as np
import pytest

from physics.engine.tests._gpu import needs_gpu

SEEDS = 256
FIRST_SEED = 1000
PACKET = 1000
S, A, R1, R2 = 222, 242, 422, 442
HALF = math.pi / 2.0
THIRTY = 2.0 * math.asin(math.sqrt(0.3))


@functools.lru_cache(maxsize=1)
def _one():
    """The bench universe's boxes and cell graph, built once."""
    from physics.engine import host_boxes, new_universe
    return host_boxes(new_universe(8, 6, seed=1))


def _bench(universes: int = SEEDS, first_seed: int = FIRST_SEED):
    """`universes` copies of the bench side by side, seeds first_seed + u, a packet named in S of each."""
    from physics.engine import State, quantum_packet
    boxes = _one()
    graph = {name: boxes[name] for name in ("owner", "kind", "between", "place", "size", "size_energy")}
    quanta = np.repeat(boxes["quanta"][None], universes, axis=0)
    state = State.from_host(quanta, np.zeros(quanta.shape, dtype=np.float32), seed=first_seed,
                            lattice=[graph] * universes, wrap=True)
    for universe in range(universes):
        quantum_packet(state, S, PACKET, universe)
    return state


def _amounts(state) -> np.ndarray:
    """Every cell's quanta ignoring sign, (universes, capacity), counted from the boxes."""
    from physics.engine import cell_count
    cell_count(state)
    return state.cell_quanta.copy()


def _run(state, steps: int) -> None:
    from physics.engine import Settings, step
    for _ in range(steps):
        step(state, Settings())


def _landed(events: list, cell: int) -> float:
    """The share of the events that settled in `cell`."""
    return sum(event["cell"] == cell for event in events) / len(events)


@needs_gpu
def test_a_packet_split_50_50_settles_whole_in_one_side_about_half_the_seeds_each():
    from physics.engine import quantum_hand, quantum_react, quantum_store
    state = _bench()
    handed = quantum_hand(state, S, A, HALF)
    assert all(499 <= moved <= 501 for _, moved in handed)
    _run(state, 2)
    assert all(len(record["cells"]) == 2 and record["jumps"] >= 2 for record in quantum_store(state)["moving"])
    before = _amounts(state)
    events = quantum_react(state, A)
    change = _amounts(state) - before
    assert len(events) == SEEDS and not quantum_store(state)["moving"]
    for event in events:
        other = S if event["cell"] == A else A
        assert event["amount"] == PACKET and event["drawn"] == event["parts"][other]
        assert change[event["universe"], event["cell"]] == event["drawn"]
        assert change[event["universe"], other] == -event["drawn"]
    assert np.count_nonzero(change) == 2 * SEEDS
    share = _landed(events, S)
    print(f"50/50 split, reaction in A after 2 steps: whole in S in {share:.3f} of {SEEDS} seeds, whole in A in the rest")
    assert 0.40 <= share <= 0.60


@needs_gpu
def test_the_pointer_rides_with_the_energy_so_a_reaction_right_at_the_split_also_gives_the_shares():
    from physics.engine import quantum_hand, quantum_react
    state = _bench(first_seed=FIRST_SEED + 5000)
    quantum_hand(state, S, A, HALF)
    share = _landed(quantum_react(state, S), S)
    print(f"50/50 split, reaction in S with no step in between: whole in S in {share:.3f} of {SEEDS} seeds")
    assert 0.40 <= share <= 0.60


@needs_gpu
def test_a_70_30_split_settles_in_the_70_side_in_about_70_percent_never_by_which_half_reacted():
    from physics.engine import quantum_hand, quantum_react
    shares = {}
    for reacting in (S, A):
        state = _bench(first_seed=FIRST_SEED + 10000 * (1 + (reacting == A)))
        handed = quantum_hand(state, S, A, THIRTY)
        assert all(299 <= moved <= 301 for _, moved in handed)
        _run(state, 2)
        events = quantum_react(state, reacting)
        assert all(event["drawn"] in (event["parts"][S], event["parts"][A]) for event in events)
        shares[reacting] = _landed(events, S)
    print(f"70/30 split: whole in the 70% cell in {shares[S]:.3f} of {SEEDS} seeds when the 70% half reacts, "
          f"{shares[A]:.3f} when the 30% half reacts")
    assert 0.61 <= shares[S] <= 0.79 and 0.61 <= shares[A] <= 0.79


def _two_routes(state) -> None:
    """Split 50/50 over S and A, hand each half on along its route (S to R1, A to R2) and run 3 steps."""
    from physics.engine import quantum_hand
    quantum_hand(state, S, A, HALF)
    _run(state, 1)
    quantum_hand(state, S, R1, math.pi)
    quantum_hand(state, A, R2, math.pi)
    _run(state, 3)


@needs_gpu
def test_two_routes_brought_back_together_interfere_and_one_exit_gets_nearly_all():
    from physics.engine import quantum_hand, quantum_react, quantum_store
    exits = {}
    for turn in (HALF, -HALF):
        state = _bench(first_seed=FIRST_SEED + 30000)
        _two_routes(state)
        moving = quantum_store(state)["moving"]
        assert all(sorted(record["cells"]) == [R1, R2] for record in moving)
        quantum_hand(state, R1, R2, turn)
        parts = [dict(zip(record["cells"], record["parts"])) for record in moving]
        merged = sum(event["cause"] == "merge" for event in quantum_store(state)["ended"])
        events = quantum_react(state, R1) + quantum_react(state, R2)
        assert len(events) == SEEDS
        exits[turn] = (min(part.get(R2, 0) for part in parts), max(part.get(R2, 0) for part in parts),
                       _landed(events, R2), merged)
    low, high, share, merged = exits[HALF]
    print(f"two routes rejoined in step: exit R2 holds {low} .. {high} of {PACKET} quanta, whole in R2 in {share:.3f} "
          f"of {SEEDS} seeds, {merged} merged back before any reaction")
    back_low, back_high, back_share, back_merged = exits[-HALF]
    print(f"rejoined with the opposite turn: exit R2 holds {back_low} .. {back_high}, whole in R2 in {back_share:.3f}, "
          f"{back_merged} merged back")
    assert low >= 0.99 * PACKET and share >= 0.97
    assert back_high <= 0.01 * PACKET and back_share <= 0.03


@needs_gpu
def test_a_reaction_in_one_route_takes_the_interference_away():
    from physics.engine import quantum_hand, quantum_packet, quantum_react, quantum_store
    state = _bench(first_seed=FIRST_SEED + 40000)
    _two_routes(state)
    reacted = quantum_react(state, R1)
    assert len(reacted) == SEEDS and not quantum_store(state)["moving"]
    for event in reacted:
        quantum_packet(state, event["cell"], PACKET, event["universe"])
    route = _landed(reacted, R1)
    _run(state, 1)
    quantum_hand(state, R1, R2, HALF)
    parts = [dict(zip(record["cells"], record["parts"])) for record in quantum_store(state)["moving"]]
    assert all(499 <= part.get(R2, 0) <= 501 for part in parts)
    events = quantum_react(state, R1) + quantum_react(state, R2)
    share = _landed(events, R2)
    print(f"reaction in route R1: it settled there in {route:.3f} of {SEEDS} seeds; sent on from where it settled and "
          f"brought to the same joining, exit R2 holds 499 .. 501 of {PACKET} and gets it whole in {share:.3f} of the seeds")
    assert len(events) == SEEDS and 0.40 <= route <= 0.60 and 0.40 <= share <= 0.60


@needs_gpu
def test_no_message_the_far_sides_odds_are_the_same_with_or_without_a_reaction_on_the_near_side():
    from physics.engine import quantum_hand, quantum_react, quantum_store
    found, spread = {}, {}
    for near_reacts in (True, False):
        state = _bench(first_seed=FIRST_SEED + 50000 + 10000 * near_reacts)
        quantum_hand(state, S, A, THIRTY)
        _run(state, 1)
        if near_reacts:
            quantum_react(state, S)
        _run(state, 2)
        held = [dict(zip(record["cells"], record["parts"])).get(A, 0) for record in quantum_store(state)["moving"]]
        held += [PACKET * (event["cell"] == A) for event in quantum_store(state)["ended"]]
        spread[near_reacts] = (min(held), max(held), float(np.mean(held)))
        quantum_react(state, A)
        ended = quantum_store(state)["ended"]
        assert len(ended) == SEEDS
        found[near_reacts] = _landed(ended, A)
    print(f"far side (the 30% half) finds the energy whole in {found[True]:.3f} of {SEEDS} seeds with a reaction on the "
          f"near side first, {found[False]:.3f} without")
    print(f"the far cell's part of the energy just before its own reaction: {spread[True][0]} .. {spread[True][1]} "
          f"(mean {spread[True][2]:.0f}) with the near reaction, {spread[False][0]} .. {spread[False][1]} "
          f"(mean {spread[False][2]:.0f}) without")
    assert 0.21 <= found[True] <= 0.39 and 0.21 <= found[False] <= 0.39
    assert abs(found[True] - found[False]) <= 0.12


@needs_gpu
def test_quanta_stay_exact_through_a_settle_whatever_the_distance():
    from physics.engine import quantum_hand, quantum_react, totals
    universes = 16
    state = _bench(universes, FIRST_SEED + 70000)
    far = 262
    quantum_hand(state, S, A, THIRTY)
    quantum_hand(state, A, far, math.pi)
    _run(state, 2)
    start = [universe["amount"] for universe in state.start]
    before = totals(state)
    quanta_before = state.quanta.numpy().copy()
    owner_before = state.owner.numpy().copy()
    amounts_before = _amounts(state)
    events = quantum_react(state, far)
    after = totals(state)
    quanta = state.quanta.numpy()
    change = _amounts(state) - amounts_before
    assert len(events) == universes
    assert [universe["amount"] for universe in after["universes"]] == start
    assert all(after[name] == before[name] for name in ("amount", "plus", "minus", "boxes"))
    assert np.array_equal(np.sign(quanta), np.sign(quanta_before)) and np.array_equal(state.owner.numpy(), owner_before)
    assert np.abs(quanta[quanta != 0]).min() >= 1
    for event in events:
        other = S if event["cell"] == far else far
        expected = np.zeros(state.capacity, dtype=np.int64)
        expected[event["cell"]] = event["parts"][other]
        expected[other] = -event["parts"][other]
        assert np.array_equal(change[event["universe"]], expected)
        assert event["step"] == state.step
    moved = (quanta != quanta_before) & ~np.isin(owner_before, (S, far))
    print(f"settle between cells two volumes apart, {universes} seeds: amount, plus, minus and boxes unchanged, the two "
          f"cells' quanta changed by exactly the drawn part, boxes of other cells changed: {int(moved.sum())}")
    assert not moved.any()
    _run(state, 2)
    assert [universe["amount"] for universe in totals(state)["universes"]] == start


@needs_gpu
def test_a_collapse_that_changes_a_halfs_cell_is_a_reaction_and_the_energy_settles_whole():
    from physics.engine import State, cells, host_boxes, new_universe, quantum_hand, quantum_packet, quantum_store, totals
    joint, low, gone, beside_low, beside_gone = (3 * 7 + 2) * 7 + 2, (2 * 7 + 2) * 7 + 2, (4 * 7 + 2) * 7 + 2, \
        (2 * 7 + 4) * 7 + 2, (4 * 7 + 4) * 7 + 2
    boxes = host_boxes(new_universe(5, 10, seed=1, unevenness=0.0, margin=8))
    quanta = boxes["quanta"].copy()
    quanta[boxes["owner"] == joint] = -30000
    graph = {name: boxes[name] for name in ("owner", "kind", "between", "place", "size", "size_energy")}
    state = State.from_host(quanta, np.zeros(quanta.shape, dtype=np.float32), seed=1, lattice=graph)
    start = totals(state)["amount"]
    quantum_packet(state, gone, PACKET)
    quantum_hand(state, gone, beside_gone, HALF)
    quantum_packet(state, beside_low, PACKET)
    store = quantum_store(state)
    for _ in range(30):
        _run(state, 1)
        if state.collapses[0]:
            break
    assert state.collapses[0] >= 1 and state.kind[0, gone] == 0
    _run(state, 1)
    events = {event["serial"]: event for event in store["ended"]}
    print(f"joint collapse by step {state.step - 1}: the split with a half in the volume that fell in settled by "
          f"'{events[0]['cause']}' whole in cell {events[0]['cell']} during step {events[0]['step'] + 1}; the whole packet "
          f"beside it moves on: {[record['serial'] for record in store['moving']]}")
    assert events[0]["cause"] == "collapse or tear" and events[0]["cell"] in (low, beside_gone)
    assert events[0]["amount"] == PACKET and [record["serial"] for record in store["moving"]] == [1]
    assert totals(state)["amount"] == start and cells(host_boxes(state))["volumes"] == 7


@needs_gpu
def test_friction_takes_direction_off_a_split_and_with_none_left_it_settles_whole():
    """The crossing's friction rubs direction off in whole quanta over 20 steps; the last step is staged.

    Losing all of 1000 quanta's direction takes thousands of steps (a separate run of this bench: 119 quanta in
    the first 500 steps, 43 in the next 500, 28 in the next), so the end is staged: in half the seeds the
    split's directed count is set to 0 by hand, where the rub would bring it; in the next step each of those
    settles whole by "momentum" and the others move on.
    """
    from physics.engine import quantum_hand, quantum_store, totals
    universes = 32
    state = _bench(universes, FIRST_SEED + 80000)
    start = [universe["amount"] for universe in state.start]
    quantum_hand(state, S, A, HALF)
    moving = quantum_store(state)["moving"]
    directed = [[record["directed"] for record in moving]]
    for _ in range(20):
        _run(state, 1)
        directed.append([record["directed"] for record in moving])
    falls = np.diff(np.array(directed), axis=0)
    lost = PACKET - np.array(directed[-1])
    print(f"20 steps, {universes} seeds: a 50/50 split of {PACKET} quanta lost the direction of {lost.min()} .. "
          f"{lost.max()} quanta (mean {lost.mean():.1f}), none settled")
    assert len(moving) == universes and np.all(falls <= 0) and lost.sum() > 0
    spent = [record for record in moving if record["universe"] % 2 == 0]
    for record in spent:
        record["directed"] = 0
    _run(state, 1)
    ended = quantum_store(state)["ended"]
    assert sorted(event["universe"] for event in ended) == list(range(0, universes, 2))
    assert all(event["cause"] == "momentum" and event["drawn"] == event["parts"][S if event["cell"] == A else A]
               for event in ended)
    assert len(quantum_store(state)["moving"]) == universes // 2
    assert [universe["amount"] for universe in totals(state)["universes"]] == start


@needs_gpu
def test_a_handover_needs_two_volumes_sharing_a_joint_and_a_move_never_empties_a_box():
    from physics.engine import quantum_hand, quantum_move, quantum_packet
    state = _bench(1)
    with pytest.raises(ValueError):
        quantum_hand(state, S, R2, HALF)
    with pytest.raises(ValueError):
        quantum_packet(state, 223, 10)
    held = int(_amounts(state)[0, S])
    with pytest.raises(ValueError):
        quantum_move(state, [(0, S, A, held)])
