"""Scene: what the window shows. One cell is layer 1's energy volume; `crossing` a red
volume, a blue joint and a red volume in a line; `calm` a small full grid that wraps round.

A grid is shown looking along one of its three directions at one layer of its volumes and
the joints between them, whole: balls with the blue all round them. Or in three
dimensions, a slab `slab` layers thick from `layer` on, the joints from layer to layer
too, each joint cut both ways along its length as it is drawn (`line_streaks`): balls in
a web. A slab of all the layers is the whole grid.
"""
import numpy as np
import warp as wp

from physics.engine import one_cell, three_cells
from physics.engine.axiom_3_structure.definitions.store import Store
from physics.engine.axiom_3_structure.logic.lay_grid import calm_grid
from physics.view.definitions.camera import DISTANCE, LINE_DISTANCE
from physics.view.logic.line_streaks import line_streaks
from physics.view.logic.placed import placed
from physics.view.logic.title_line import cells_title, title_line
from physics.view.logic.tracers import COUNT, Tracers, lay_tracers

SCENES = {"one": one_cell, "crossing": three_cells, "calm": calm_grid}


def layers(store: Store, along: int = 2) -> int:
    """How many layers of volumes lie along a direction: 0 when the scene is no grid."""
    return store.grid[along] // 2


def in_cut(store: Store, layer: int | None, slab: int = 0, along: int = 2) -> np.ndarray | int:
    """1 for each own cell that is shown: those of `layer` along `along`, or of the slab."""
    if not layers(store):
        return 1
    deep = np.unravel_index(store.own_cell.numpy()[:store.count], store.grid)[along]
    last = 2 * layer if not slab else 2 * (layer + slab - 1)
    if slab == layers(store, along):
        last = store.grid[along]  # all the layers: the joints round the wrap too
    return ((deep >= 2 * layer) & (deep <= last))[:, None]


def trace(store: Store, layer: int | None, slab: int = 0, along: int = 2) -> Tracers:
    """Fresh tracers in what is shown: `COUNT` a cell, shared out past four cells."""
    cells = max(store.count, 1)  # a calm cell alone is drawn too
    return lay_tracers(store, 0, COUNT * min(cells, 4), cells=cells,
                       shown=in_cut(store, layer, slab, along))


def drawn(store: Store, tracers: Tracers, layer: int | None = None, slab: int = 0,
          along: int = 2) -> wp.array:
    """The tracers' tails as the card draws them now, what is shown at the picture's middle."""
    if store.grid == (1, 1, 1):  # one volume, calm or own: a ball 1 wide at the middle
        return line_streaks(tracers, store.across, wp.array(
            [[[0, 0, 0], [1, 0, 0], [0, 0, 1]]], dtype=wp.mat33, device=tracers.tail.device))
    if layer is None:
        return line_streaks(tracers, store.across, placed(store))
    middle = layer + max(slab - 1, 0) / 2
    return line_streaks(tracers, store.across, placed(store, middle, along), slab > 0)


def title(store: Store) -> str:
    """The scene's numbers in words."""
    return title_line(store) if store.grid == (1, 1, 1) else cells_title(store)


def eye_distance(store: Store) -> float:
    """How far the eye starts from the middle of the scene: a grid's by its volumes along."""
    return DISTANCE * layers(store) if layers(store) else (DISTANCE, LINE_DISTANCE)[store.count > 1]
