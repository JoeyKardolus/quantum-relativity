"""Lumps: a small grown patch to stage a check in, and the two things a check stages by hand.

`grown` steps a first volume until a volume stands `layers` layers out: the patch is made
by the leak, as every grid is. `fuller` makes chosen volumes hold `times` their quanta, box
for box, so each whirl keeps its shape; `moving` sets their fluid going one way along a
direction. `region` names a volume and the six at its sides. Only a check or a scene's
start stages; no step ever does. `two_lumps` is the scene: two fuller regions in a patch.
"""
import numpy as np

from physics.engine import first_volume, step
from physics.engine.axiom_3_structure.definitions.kinds import ACROSS
from physics.engine.axiom_3_structure.definitions.places import steps_of
from physics.engine.axiom_3_structure.definitions.store import Store

FULLER = 4  # how many times fuller a staged region is: clearly, chosen before any run


def grown(layers: int = 3, room: int = 0, across: int = ACROSS, device: str = "cpu") -> Store:
    """A universe grown by its leak until a volume stands `layers` layers out."""
    store = first_volume(across, device, room or layers + 1)
    middle = store.side // 2
    while np.abs(steps_of(store.own_cell.numpy()[:store.count], store.side) - middle).max() < 2 * layers:
        step(store)
    return store


def region(store: Store, out: tuple) -> list:
    """The volume `out` volumes from the middle along each direction, and the six at its sides."""
    at = np.array(out) * 2 + store.side // 2
    sides = [at] + [at + 2 * way * np.eye(3, dtype=int)[axis] for axis in range(3) for way in (-1, 1)]
    return [int(np.ravel_multi_index(tuple(one), (store.side,) * 3)) for one in sides]


def _slots(store: Store, cells: list) -> np.ndarray:
    """The slots of own cells."""
    own = store.own_cell.numpy()[:store.count]
    return np.array([int(np.flatnonzero(own == cell)[0]) for cell in cells])


def fuller(store: Store, cells: list, times: int = FULLER) -> None:
    """Make each of the own volumes `cells` hold `times` its quanta, every box alike."""
    held, boxes, slots = store.cell_quanta.numpy(), store.box_quanta.numpy(), _slots(store, cells)
    boxes[slots] *= times
    held[cells] = boxes[slots].sum(axis=1)
    store.cell_quanta.assign(held)
    store.box_quanta.assign(boxes)


def moving(store: Store, cells: list, axis: int, speed: float) -> None:
    """Set the fluid of the volumes `cells` going up direction `axis` at `speed` calm lengths a
    turn of a calm swirl: each one's flow that way becomes its quanta times that speed."""
    flow, held = store.cell_flow.numpy(), store.cell_quanta.numpy()
    flow[cells, axis] = np.floor(held[cells] * speed).astype(np.int64)
    store.cell_flow.assign(flow)


def two_lumps(across: int = ACROSS, device: str = "cpu", layers: int = 4, out: int = 2) -> Store:
    """A patch grown `layers` layers out with two fuller regions, `out` volumes either side of the middle."""
    store = grown(layers, layers + 4, across, device)
    for way in (-out, out):
        fuller(store, region(store, (0, way, 0)))
    return store
