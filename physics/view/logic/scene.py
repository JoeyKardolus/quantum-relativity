"""Scene: what the window shows. One cell is layer 1's energy volume; `crossing` a red
volume, a blue joint and a red volume in a line; `calm` a small full grid that wraps round.

A grid is shown as a flat cut through one layer of its volumes: the tracers are laid in
the boxes the cut passes, the two middle layers of boxes of that layer's volumes and of
the joints between them. Or in three dimensions, a slab `slab` layers thick from `layer`
on: every layer's cut, and the joints from layer to layer each cut both ways along its
length. A slab of all the layers is the whole grid, every volume and every joint.
"""
import numpy as np
import warp as wp

from physics.engine import one_cell, three_cells
from physics.engine.axiom_3_structure.definitions.store import Store
from physics.engine.axiom_3_structure.logic.lay_grid import calm_grid
from physics.view.definitions.camera import DISTANCE, LINE_DISTANCE
from physics.view.logic.line_streaks import line_streaks
from physics.view.logic.placed import placed
from physics.view.logic.streaks import streaks
from physics.view.logic.title_line import cells_title, title_line
from physics.view.logic.tracers import COUNT, Tracers, lay_tracers

SCENES = {"one": one_cell, "crossing": three_cells, "calm": calm_grid}


def layers(store: Store) -> int:
    """How many layers of volumes a cut can pass: 0 when the scene is no grid."""
    return store.grid[2] // 2


def in_cut(store: Store, layer: int | None, slab: int = 0) -> np.ndarray | int:
    """1 for each box of each own cell that is shown: the flat cut through `layer`, or the slab."""
    if not layers(store):
        return 1
    deep = np.unravel_index(store.own_cell.numpy()[:store.count], store.grid)[2]
    box = np.indices((store.across,) * 3).reshape(3, -1)
    middle = np.abs(box - (store.across - 1) / 2) < 1  # the two middle layers, each direction
    if not slab:
        return np.outer(deep == 2 * layer, middle[2])
    last = store.grid[2] if slab == layers(store) else 2 * (layer + slab - 1)  # all: the wrap's too
    cuts = np.outer(deep % 2 == 0, middle[2]) | np.outer(deep % 2 == 1, middle[0] | middle[1])
    return cuts & ((deep >= 2 * layer) & (deep <= last))[:, None]


def trace(store: Store, layer: int | None, slab: int = 0) -> Tracers:
    """Fresh tracers in what is shown: `COUNT` a cell, shared out past four cells."""
    cells = max(store.count, 1)  # a calm cell alone is drawn too
    return lay_tracers(store, 0, COUNT * min(cells, 4), cells=cells,
                       shown=in_cut(store, layer, slab))


def drawn(store: Store, tracers: Tracers, layer: int | None = None, slab: int = 0) -> wp.array:
    """The tracers' tails as the card draws them now, what is shown at the picture's middle."""
    if store.grid == (1, 1, 1):
        return streaks(tracers)
    middle = layer if layer is None else layer + max(slab - 1, 0) / 2
    return line_streaks(tracers, store.across, placed(store, middle))


def title(store: Store) -> str:
    """The scene's numbers in words."""
    return title_line(store) if store.grid == (1, 1, 1) else cells_title(store)


def eye_distance(store: Store) -> float:
    """How far the eye starts from the middle of the scene: a grid's by its volumes along."""
    return DISTANCE * layers(store) if layers(store) else (DISTANCE, LINE_DISTANCE)[store.count > 1]
