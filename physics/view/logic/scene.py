"""Scene: what the window shows. `one` is layer 1's energy volume; `crossing` a red volume,
a blue joint and a red volume in a line; `grow` one overstretched volume that tears, and
tears again: red balls with blue joints between them, a cluster growing outward.

Every cell is drawn where the relations put it and as wide as its size (`placed.py`), so a
volume that gains energy is seen to shrink and one that loses it to grow. `Seen` keeps what
the picture needs from frame to frame: the tracers, the middles, how much smaller it is drawn.
"""
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from physics.engine import first_volume, one_cell, three_cells
from physics.engine.axiom_3_structure.definitions.size_law import size_of
from physics.engine.axiom_3_structure.definitions.store import Store, peek
from physics.view.definitions.camera import DISTANCE, LINE_DISTANCE
from physics.view.logic.line_streaks import line_streaks
from physics.view.logic.placed import placed, settle
from physics.view.logic.title_line import cells_title, title_line
from physics.view.logic.tracers import COUNT, MARK_EVERY, Tracers, lay_tracers, step_carrying

SCENES = {"one": one_cell, "crossing": three_cells, "grow": first_volume}
FITS = 1.5  # how far from the middle the window shows at the cells' true sizes; only draws


@dataclass
class Seen:
    tracers: Tracers
    middle: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    span: float = 1.0  # the picture is drawn this many times smaller than the cells are


def trace(store: Store) -> Tracers:
    """Fresh tracers in every own cell, laid by the quanta: `COUNT` a cell, shared out past four."""
    return lay_tracers(store, 0, COUNT * min(store.cell_quanta.shape[0], 4),
                       cells=max(store.count, 1))  # a calm cell alone is drawn too


def advance(store: Store, seen: Seen, boxes: tuple = None) -> None:
    """One step with the tracers carried along; after a tear they are laid anew in the new cells."""
    cells = store.count
    step_carrying(store, seen.tracers, 0, store.step % MARK_EVERY == 0, boxes, max(cells, 1))
    if store.count != cells:
        seen.tracers = trace(store)


def drawn(store: Store, seen: Seen, web: bool = False) -> wp.array:
    """The tracers' tails as the card draws them now; `web`: the joints cut along their length."""
    if store.cells == 1:  # one volume, calm or own: a ball as wide as its size at the middle
        wide = size_of(peek(store.cell_quanta, 0), store.across)
        return line_streaks(seen.tracers, store.across, wp.array(
            [[[0, 0, 0], [wide, 0, 0], [0, 0, 1]]], dtype=wp.mat33, device=seen.tracers.tail.device))
    seen.middle, size, joints = settle(store, seen.middle)
    volumes = np.setdiff1d(np.arange(store.cells), joints[:, 0])
    reach = (np.abs(seen.middle[volumes]) + size[volumes, None] / 2).max()
    seen.span = max(1.0, reach / FITS)
    return line_streaks(seen.tracers, store.across,
                        placed(store, seen.middle, size, joints, seen.span), web)


def title(store: Store, seen: Seen = None) -> str:
    """The scene's numbers in words."""
    if store.cell_quanta.shape[0] == 1:
        return title_line(store)
    return cells_title(store) + (f"  drawn 1 : {seen.span:.1f}" if seen else "")


def eye_distance(store: Store) -> float:
    """How far the eye starts from the middle of the scene."""
    return DISTANCE if store.cell_quanta.shape[0] == 1 else LINE_DISTANCE
