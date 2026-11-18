"""Scene: what the window shows. One cell is layer 1's energy volume; `crossing` a red
volume, a blue joint and a red volume in a line; `calm` a small full grid that wraps round.

A grid is shown as a flat cut through one layer of its volumes: the tracers are laid in
the boxes the cut passes, the two middle layers of boxes of that layer's volumes and of
the joints between them. With no layer named the cuts through all layers are shown at
once; the joints that lie from layer to layer are in none of them.
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


def in_cut(store: Store, layer: int | None) -> np.ndarray | int:
    """1 for each box of each own cell that the flat cut through `layer` passes; None: every cut."""
    if not layers(store):
        return 1
    deep = np.unravel_index(store.own_cell.numpy()[:store.count], store.grid)[2]
    middle = np.abs(np.arange(store.across ** 3) % store.across - (store.across - 1) / 2) < 1
    return np.outer(deep % 2 == 0 if layer is None else deep == 2 * layer, middle)


def trace(store: Store, layer: int | None) -> Tracers:
    """Fresh tracers in what is shown: `COUNT` a cell, shared out past four cells."""
    cells = max(store.count, 1)  # a calm cell alone is drawn too
    return lay_tracers(store, 0, COUNT * min(cells, 4), cells=cells, shown=in_cut(store, layer))


def drawn(store: Store, tracers: Tracers, layer: int | None = None) -> wp.array:
    """The tracers' tails as the card draws them now, the cut's layer at the picture's middle."""
    if store.grid == (1, 1, 1):
        return streaks(tracers)
    return line_streaks(tracers, store.across, placed(store, layer))


def title(store: Store) -> str:
    """The scene's numbers in words."""
    return title_line(store) if store.grid == (1, 1, 1) else cells_title(store)


def eye_distance(store: Store) -> float:
    """How far the eye starts from the middle of the scene: a grid's by its volumes along."""
    return DISTANCE * layers(store) if layers(store) else (DISTANCE, LINE_DISTANCE)[store.count > 1]
