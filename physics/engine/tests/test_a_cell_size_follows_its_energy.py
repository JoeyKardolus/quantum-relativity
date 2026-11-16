"""A cell's size follows its energy: the moving corners give room to the cell whose energy presses harder.

Owner, 2026-11-16: the cells' corners move; each cell's size follows its energy, no set
size, no minimum, the rules decide. The rules here (move_corners.py): a cell presses
with -d(size * box_energy at its mean packing)/d(size), an energy volume with its
fluid's own push, a joint with its steep push, and the pull draws volumes. A forced
cell lattice (new_universe, seed 1, 5**3 cells of 8**3 boxes, margin 8, no unevenness:
every box 1000 quanta) and its twin with one energy volume, A (lattice 2, 2, 2), at
2000 quanta a box run 60 steps each with the default settings. Measured, not tuned:
A's size, in Planck boxes it owns and in the volume its corners span, ends larger in
the twin where it holds more energy than in the even start, and larger than the mean of
the other seven volumes of its own run; its packing (quanta per box) falls from the
2000 it started at. Every other rule acts too (the leak drains A into its
neighbours), so the numbers are printed beside the asserts.
"""
import numpy as np

from physics.engine import Settings, State, cells, host_boxes, new_universe, step
from physics.engine.tests._gpu import needs_gpu

A = (2 * 7 + 2) * 7 + 2


def _corner_volume(corners: np.ndarray, place: np.ndarray) -> float:
    """The volume of a cell's eight corners' hull as six Kuhn tetrahedra."""
    total = 0.0
    origin = corners[tuple(place)]
    far = corners[tuple(place + 1)]
    for first, second in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)):
        one = place.copy()
        one[first] += 1
        two = one.copy()
        two[second] += 1
        total += abs(np.linalg.det(np.stack([corners[tuple(one)] - origin, corners[tuple(two)] - origin,
                                             far - origin]))) / 6.0
    return total


def _run(packed: bool) -> tuple:
    """A's box count, corner volume and packing after 60 steps, and the other volumes' mean box count."""
    boxes = host_boxes(new_universe(5, 8, seed=1, unevenness=0.0, margin=8))
    quanta = boxes["quanta"].copy()
    if packed:
        quanta[boxes["owner"] == A] = 2000
    lattice = {"cells_per_side": 5, "owner": boxes["owner"], "corners": boxes["corners"]}
    state = State.from_host(quanta, np.zeros(quanta.shape, dtype=np.float32), seed=1, lattice=lattice)
    settings = Settings()
    for _ in range(60):
        step(state, settings)
    after = host_boxes(state)
    found = cells(after)
    volumes = found["volumes"]
    a_id = int(np.flatnonzero((found["place"][:volumes] == [2, 2, 2]).all(axis=1))[0])
    others = [index for index in range(volumes) if index != a_id]
    size = int(found["boxes"][a_id])
    spanned = _corner_volume(after["corners"], np.array([2, 2, 2]))
    return size, spanned, found["quanta"][a_id] / size, float(found["boxes"][others].mean())


@needs_gpu
def test_the_volume_with_more_energy_takes_more_room():
    even = _run(False)
    packed = _run(True)
    print(f"even start: A {even[0]} boxes, corners span {even[1]:.1f}, packing {even[2]:.0f}, others "
          f"{even[3]:.1f} boxes; A at 2000 a box: A {packed[0]} boxes, corners span {packed[1]:.1f}, "
          f"packing {packed[2]:.0f}, others {packed[3]:.1f} boxes (all started at 512)")
    assert packed[0] > even[0] and packed[1] > even[1]
    assert packed[0] > packed[3]
    assert packed[2] < 2000
