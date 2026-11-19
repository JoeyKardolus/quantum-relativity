"""Lay tracers: put tracers at rest in cells, by the quanta of their boxes, so the dots lie
as the fluid lies. `even` gives every shown cell as many, whatever it holds: in a cluster
whose cells hold very different quanta each cell is still seen. `move_into` hands cells
that are new their share of the tracers there are, taken at random from all the others.
"""
import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.places import steps_of
from physics.engine.axiom_3_structure.definitions.store import peek
from physics.engine.axiom_3_structure.logic.boxes_of import boxes_of
from physics.view.logic.tracers import MARKS, Tracers

COUNT = 8000  # tracers a cell, a drawing number


def places(store, cell: int, count: int, seed: int = 1, cells: int = 1, shown=1,
           even: bool = False) -> np.ndarray:
    """`count` places at rest in the `cells` own cells from `cell`'s slot on, by the quanta of boxes `shown`."""
    box_quanta, _phase, slot = boxes_of(store, cell)
    quanta = np.abs(box_quanta[slot:slot + cells].numpy()) * shown
    if even:
        quanta = quanta / np.maximum(quanta.sum(axis=1, keepdims=True), 1)
    chance = np.random.default_rng(seed)
    boxes = chance.choice(quanta.size, size=count, p=quanta.ravel() / quanta.sum())
    corner = np.stack(np.unravel_index(boxes % store.across ** 3, (store.across,) * 3), axis=1)
    place = (corner + chance.random((count, 3))).astype(np.float32)
    place[:, 1] += (slot + boxes // store.across ** 3) * store.across  # the cell's slot
    return place


def lay_tracers(store, cell: int, count: int = COUNT, seed: int = 1, cells: int = 1,
                shown=1, even: bool = False) -> Tracers:
    """`count` tracers at rest, as `places` lays them, each with an empty tail."""
    place = places(store, cell, count, seed, cells, shown, even)
    return Tracers(wp.array(np.tile(place, (MARKS, 1, 1)), dtype=wp.vec3,
                            device=store.box_quanta.device))


def shown(store, flat: bool) -> np.ndarray:
    """1 for each own slot whose cell is drawn (count, 1): all, or the middle layer's; never a
    slot a collapse emptied."""
    own = store.own_cell.numpy()[:store.count]
    third = steps_of(own, max(store.side, 1))[:, 2:]
    return ((third == store.side // 2) | (not flat)) & (store.cell_quanta.numpy()[own, None] != 0)


def trace(store, flat: bool = False) -> Tracers:
    """Fresh tracers in the own cells that are drawn: by the quanta, or evenly in a growing cluster."""
    cells = max(store.count, 1)  # a calm cell alone is drawn too
    first = peek(store.own_cell, 0) if store.count else 0
    return lay_tracers(store, first, COUNT * min(store.cell_quanta.shape[0], 4), cells=cells,
                       shown=shown(store, flat) if store.count else 1, even=store.leaks)


@wp.kernel
def _move(pick: wp.array(dtype=wp.int32), place: wp.array(dtype=wp.vec3),
          tail: wp.array2d(dtype=wp.vec3)):
    mark, one = wp.tid()
    tail[mark, pick[one]] = place[one]


def move_into(store, tracers: Tracers, since: int, shown: np.ndarray) -> None:
    """Give the shown cells in slots from `since` on their even share of the tracers."""
    number = int(tracers.tail.shape[1] * shown[since:].sum() / max(shown.sum(), 1))
    if number == 0:
        return
    cell = int(store.own_cell[since:since + 1].numpy()[0])
    place = places(store, cell, number, store.step, store.count - since, shown[since:], True)
    pick = np.random.default_rng(store.step).choice(tracers.tail.shape[1], number, replace=False)
    device = tracers.tail.device
    wp.launch(_move, dim=(MARKS, number), device=device,
              inputs=[wp.array(pick, dtype=wp.int32, device=device),
                      wp.array(place, dtype=wp.vec3, device=device), tracers.tail])
