"""Scene: what the window shows. `one` is one cell; `crossing` a red volume, a blue joint and a
red volume in a line; `grow` one volume whose leak tears off layer after layer of new
volumes round it; `lumps` a small grown patch with two fuller regions.
Every cell is drawn as wide as its size where its joints put it (`placed.py`). A growing
cluster above `WHOLE_UP_TO` cells shows only its flat cut, the middle layer (`strays.py`).
"""
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import warp as wp

from physics.engine import first_volume, one_cell, three_cells
from physics.engine.axiom_3_structure.definitions.size_law import size_of
from physics.engine.axiom_3_structure.definitions.store import Store, peek
from physics.engine.axiom_3_structure.logic.boxes_of import boxes_of
from physics.engine.axiom_5_packing.logic.lumps import two_lumps
from physics.view.logic.lay_tracers import move_into, shown, trace  # noqa: F401, trace handed on
from physics.view.logic.line_streaks import line_streaks
from physics.view.logic.placed import placed, settle
from physics.view.logic.strays import bring_back
from physics.view.logic.title_line import cells_title, squeeze_read, title_line
from physics.view.logic.tracers import MARK_EVERY, Tracers, step_carrying

SCENES = {"one": one_cell, "crossing": three_cells, "grow": partial(first_volume, layers=12), "lumps": two_lumps}
FITS, WHOLE_UP_TO = 1.5, 500  # drawing numbers: how far out the window shows; cells drawn whole


@dataclass
class Seen:
    tracers: Tracers
    middle: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    span: float = 1.0  # the picture is drawn this many times smaller than the cells are
    whole: bool | None = None  # None: whole while the cells are few, then the flat cut
    traced: tuple = (1, False)  # the slots that hold tracers, and whether as the flat cut
    ring: int = -1  # 0, 1, 2: that ring's fluid alone is drawn


def cut(store: Store, seen: Seen) -> bool:
    """Whether a growing cluster is drawn as its flat cut now."""
    return store.leaks and (store.count > WHOLE_UP_TO if seen.whole is None else not seen.whole)


def advance(store: Store, seen: Seen, boxes: tuple = None) -> None:
    """One step with the tracers carried along."""
    own = (store.box_quanta, store.box_phase, 0) if store.side else boxes
    step_carrying(store, seen.tracers, 0, store.step % MARK_EVERY == 0, own, max(store.count, 1))


def drawn(store: Store, seen: Seen) -> wp.array:
    """The tracers' tails as the card draws them now; new cells get their tracers first."""
    if store.side == 0:  # one cell, calm or own: a ball as wide as its size at the middle
        held = peek(store.cell_quanta, 0)
        ball = wp.array([[[0, 0, 0], [size_of(held, store.across), 0, 0], [0, 0, 1 if held > 0 else -0.5]]],
                        dtype=wp.mat33, device=seen.tracers.tail.device)
        return line_streaks(seen.tracers, store.across, ball, *boxes_of(store, 0)[1:], seen.ring)
    flat = cut(store, seen)
    if store.leaks and flat != seen.traced[1]:
        seen.tracers = trace(store, flat)
    elif store.leaks and store.count != seen.traced[0]:
        move_into(store, seen.tracers, seen.traced[0], shown(store, flat))
    seen.traced = (store.count, flat)
    bring_back(store, seen.tracers, shown(store, flat))
    seen.middle, size, joints, volumes, _apart = settle(store, seen.middle)
    reach = (np.abs(seen.middle[volumes]) + size[volumes, None] / 2).max() / FITS
    seen.span = reach if store.leaks else max(1.0, reach)
    return line_streaks(seen.tracers, store.across, placed(
        store, seen.middle, size, joints, seen.span, shown(store, flat)), store.box_phase, 0, seen.ring)


def title(store: Store, seen: Seen = None) -> str:
    """The scene's numbers in words."""
    if store.side == 0:
        return title_line(store)
    if not seen:
        return cells_title(store)
    flat = "  flat cut through the middle (C: whole)" if cut(store, seen) else ""
    return cells_title(store) + squeeze_read(store, seen.middle) + f"  drawn 1 : {seen.span:.3g}{flat}"
