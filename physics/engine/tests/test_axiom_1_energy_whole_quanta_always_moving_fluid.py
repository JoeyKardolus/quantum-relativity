"""Axiom 1, Energy: an always-moving quantum fluid, one whole signed number of quanta per Planck box at each moment of local time; the total is fixed; zero is empty space, which does nothing.

The owner's axiom (bible physics/quantum-relativity, "The axioms", 2026-11-16), checked in 3 parts, each
kept whole from the file it came from (names that two parts shared carry a _P<n> suffix).

Part 1 (was axiom 01). Energy is whole signed quanta per Planck box, and the total ignoring sign is exactly conserved.

Bible rule 4. The default universe (new_universe(seed=1), Settings(), axiom_run.py) runs
300 steps; after every step the boxes hold int32 whole numbers and the sum of |quanta|
equals the start's to the quantum.

Part 2 (was test_the_total_amount_of_quanta_is_exact_over_500_steps). The amount of energy ignoring sign is exact, to the quantum, over 500 steps with every rule on.

Bible rule 4: energy is the only countable thing, in whole quanta, and the total amount
ignoring sign stays fixed. A start of the forced cell lattice (new_universe, seed 1,
5**3 cells of 8**3 boxes, 8 energy volumes of uneven plus quanta, joints -1000 per box,
an empty margin of 8 boxes) runs 500 steps of `step` with the default settings: pull
wave, moving cell corners, fluid hops, the leak with friction and drift spreading, boxes
changing cell, collapse, clocks. The sum of |quanta| is identical after each of the
first 100 steps and after every 100 steps after, and no plus box has an empty box or a
wall among its 26 neighbours (every energy volume stays wrapped).

Part 3 (was test_a_plane_wave_moves_at_one_speed_along_an_axis_and_both_diagonals). The fluid has no favourite direction: a wave packet of whole quanta keeps one speed along an axis and both diagonals.

The hops (hop.py) run only along x, y and z, and move whole quanta. A direction-free
fluid still carries a packet at one speed whichever way it points. Plain fluid fills a
40-box cube (all plus: no joints, no empty boxes, no pull). A Gaussian packet (width 4
boxes, 100000 quanta at its peak, at least 1 quantum everywhere) gets a phase rising
0.3 per box along x, along the face diagonal (1, 1, 0) / sqrt 2 and along the body
diagonal (1, 1, 1) / sqrt 3. Settings chosen before the run: (an energy volume has no push of its own since
2026-11-16, so the fluid is free; before, volume stiffness 1e-6 made it near free; a
linear wave, whose known speed is the lattice group velocity, sum over axes of
u_i sin(0.3 u_i) for direction u) and local clocks off (clock_packing 0; on, the packed
peak would run slower than its tail, a separate rule). After 48 steps (12 time units) the
packet's quanta centre has moved that far along its direction within 5%, the three
speeds agree within 5%, it drifts sideways less than 0.05 boxes, and the amount of
quanta is identical.
Splits are off (split_across 0): one plus cube this big is one energy volume past the
split size, and this test watches the fluid alone.
"""
import numpy as np

from physics.engine.tests._gpu import needs_gpu
from physics.engine import Settings, State, new_universe, step, totals


@needs_gpu
def test_the_amount_ignoring_sign_is_the_same_whole_number_after_every_step():
    from physics.engine.tests.axiom_run import axiom_run
    run = axiom_run()
    start = run["start"]["amount"]
    drifts = np.array([row["amount"] - start for row in run["steps"]])
    print(f"start amount {start}; steps {len(drifts)}; largest drift {np.abs(drifts).max()} quanta")
    assert len(drifts) >= 300
    assert not drifts.any()


def _unwrapped(quanta: np.ndarray) -> int:
    """How many plus boxes have an empty box (or the wall) among their 26 neighbours."""
    padded = np.pad(quanta, 1)
    side = quanta.shape[0]
    empty_near = np.zeros(quanta.shape, dtype=bool)
    for shift in np.ndindex(3, 3, 3):
        if shift != (1, 1, 1):
            empty_near |= padded[shift[0]:shift[0] + side, shift[1]:shift[1] + side, shift[2]:shift[2] + side] == 0
    return int(((quanta > 0) & empty_near).sum())


@needs_gpu
def test_the_amount_ignoring_sign_holds_to_the_quantum_and_volumes_stay_wrapped():
    state = new_universe(5, 8, seed=1, margin=8)
    settings = Settings()
    start = totals(state)
    for count in range(500):
        step(state, settings)
        if count < 100:
            assert totals(state)["amount"] == start["amount"]
        if count % 100 == 99:
            now = totals(state)
            spread, collapsed, rehomes, _ = state.tallies.numpy()[0]
            print(f"step {count + 1}: amount {now['amount']} (start {start['amount']}), plus {now['plus']}, "
                  f"minus {now['minus']}, boxes {now['boxes']}, spread {spread}, collapsed {collapsed}, "
                  f"boxes that changed cell {rehomes}")
            assert now["amount"] == start["amount"]
    assert _unwrapped(state.quanta.numpy()) == 0


SIDE = 40


WAVE = 0.3


WIDTH = 4.0


STEPS = 48


DIRECTIONS = {"axis": (1.0, 0.0, 0.0), "face_diagonal": (1.0, 1.0, 0.0), "body_diagonal": (1.0, 1.0, 1.0)}


PLACES = np.stack(np.meshgrid(*[np.arange(SIDE, dtype=np.float64)] * 3, indexing="ij"), axis=-1)


def _centre(quanta: np.ndarray) -> np.ndarray:
    weight = np.abs(quanta).astype(np.float64)
    return (PLACES * weight[..., None]).sum(axis=(0, 1, 2)) / weight.sum()


def _run(unit: np.ndarray, settings: Settings) -> tuple:
    """How far the packet's quanta centre went along `unit` and across it, its known distance, the amount change."""
    offsets = PLACES - ((SIDE - 1) / 2.0 - 0.15 * SIDE * unit)
    density = 100000.0 * np.exp(-(offsets ** 2).sum(axis=-1) / (2.0 * WIDTH ** 2))
    quanta = np.maximum(np.rint(density), 1).astype(np.int32)
    state = State.from_host(quanta, offsets @ (WAVE * unit))
    start = _centre(quanta)
    amount = totals(state)["amount"]
    for _ in range(STEPS):
        step(state, settings)
    moved = _centre(state.quanta.numpy()) - start
    along = float(moved @ unit)
    known = float((unit * np.sin(WAVE * unit)).sum()) * STEPS * settings.step_time
    return along, float(np.linalg.norm(moved - along * unit)), known, totals(state)["amount"] - amount


@needs_gpu
def test_a_packet_goes_as_fast_along_an_axis_as_along_either_diagonal():
    settings = Settings(pull_strength=0.0, clock_packing=0.0, split_across=0.0)
    speeds = {}
    for name, direction in DIRECTIONS.items():
        unit = np.asarray(direction) / np.linalg.norm(direction)
        along, across, known, change = _run(unit, settings)
        speeds[name] = along / (STEPS * settings.step_time)
        print(f"{name}: along {along:.4f} known {known:.4f} ratio {along / known:.4f} across {across:.2e} "
              f"amount change {change}")
        assert abs(along / known - 1.0) <= 0.05
        assert across < 0.05
        assert change == 0
    assert max(speeds.values()) <= 1.05 * min(speeds.values())
