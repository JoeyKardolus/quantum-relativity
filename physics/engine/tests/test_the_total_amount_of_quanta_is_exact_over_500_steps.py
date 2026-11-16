"""The amount of energy ignoring sign is exact, to the quantum, over 500 steps with every rule on.

Bible rule 4: energy is the only countable thing, in whole quanta, and the total amount
ignoring sign stays fixed. A start of the forced cell lattice (new_universe, seed 1,
5**3 cells of 8**3 boxes, 8 energy volumes of uneven plus quanta, joints -1000 per box,
an empty margin of 8 boxes) runs 500 steps of `step` with the default settings: pull
wave, moving cell corners, fluid hops, the leak with friction and drift spreading, boxes
changing cell, collapse, clocks. The sum of |quanta| is identical after each of the
first 100 steps and after every 100 steps after, and no plus box has an empty box or a
wall among its 26 neighbours (every energy volume stays wrapped).
"""
import numpy as np

from physics.engine import Settings, new_universe, step, totals
from physics.engine.tests._gpu import needs_gpu


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
