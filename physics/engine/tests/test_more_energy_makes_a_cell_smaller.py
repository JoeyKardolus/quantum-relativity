"""More energy makes a cell smaller: the volume given extra energy loses boxes, its neighbour with less gains them.

Owner, 2026-11-16: "more energy, smaller cell" (bible rule 3). The rule here
(move_corners.py): a cell of q quanta wants to be size_energy / q boxes across and
presses by how much smaller than that it is, so a cell with more energy presses less
and its neighbours take room from it; no set size, no minimum. A forced cell lattice
(new_universe, seed 1, 5**3 cells of 10**3 boxes, margin 8, no unevenness: every box
1000 quanta, the default start's cell, at rest) and its twin with one energy volume,
A (lattice 2, 2, 2), at 2000 quanta a box run 30 steps each with the default
settings. Measured, not tuned: in the twin, A owns fewer boxes than in the even run and
fewer than the 1000 it started with, its face-neighbouring joint J (lattice 3, 2, 2,
half A's energy) owns more boxes than in the even run, and the amount ignoring sign is
the same to the quantum before and after. Every other rule acts too: the outer cells
take the empty margin, so the even run grows, and the leak drains A into its
neighbours (by about step 40 A holds no more than the others and grows back), so the
run stops at 30 steps and the numbers are printed beside the asserts.
"""
import numpy as np

from physics.engine import Settings, State, cells, host_boxes, new_universe, step, totals
from physics.engine.tests._gpu import needs_gpu

A = (2 * 7 + 2) * 7 + 2


def _boxes_of(found: dict, place: list) -> int:
    """The box count of the cell at extended lattice place `place`."""
    return int(found["boxes"][int(np.flatnonzero((found["place"] == place).all(axis=1))[0])])


def _run(packed: bool) -> tuple:
    """A's and J's box counts after 30 steps, and the amount before and after."""
    boxes = host_boxes(new_universe(5, 10, seed=1, unevenness=0.0, margin=8))
    quanta = boxes["quanta"].copy()
    if packed:
        quanta[boxes["owner"] == A] = 2000
    lattice = {"cells_per_side": 5, "owner": boxes["owner"], "corners": boxes["corners"]}
    state = State.from_host(quanta, np.zeros(quanta.shape, dtype=np.float32), seed=1, lattice=lattice)
    before = totals(state)["amount"]
    settings = Settings()
    for _ in range(30):
        step(state, settings)
    found = cells(host_boxes(state))
    return _boxes_of(found, [2, 2, 2]), _boxes_of(found, [3, 2, 2]), before, totals(state)["amount"]


@needs_gpu
def test_the_volume_with_more_energy_gets_smaller_and_its_neighbour_bigger():
    even = _run(False)
    packed = _run(True)
    print(f"even start: A {even[0]} boxes, J {even[1]} boxes; A at 2000 a box: A {packed[0]} boxes, "
          f"J {packed[1]} boxes (all started at 1000); amount {packed[2]} -> {packed[3]}")
    assert packed[0] < even[0] and packed[0] < 1000
    assert packed[1] > even[1]
    assert packed[3] == packed[2] and even[3] == even[2]
