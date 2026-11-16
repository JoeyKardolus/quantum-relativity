"""New universe: energy volumes wrapped in a web of joints, blocks of Planck boxes of uneven whole quanta, in an empty margin.

Contract: `new_universe(cells_per_side=7, boxes_per_cell_side=10, seed=1, unevenness=0.3,
slope=-3, margin=None, box_quanta=1000, universes=1, device="cuda", joint_unevenness=0.0)`
returns a State
(state.py) at rest, phase 0. The filled cube is cells_per_side**3 blocks of k**3 boxes
(k = boxes_per_cell_side); block (cx, cy, cz) is an energy volume when cx, cy and cz are
all odd and none is n - 1 (the outer layer), else joint. So every energy volume is
wrapped on all 26 sides by joint boxes and the joints form one web (bible "The
foundation"); energy volumes start about 10 boxes across (owner, 2026-11-16). An
energy volume's box holds box_quanta * exp(g) / mean(exp(g)) plus quanta rounded to the
nearest whole, at least 1, g the uneven field over the filled cube (uneven_field.py:
seed, slope, rms unevenness; ripples at every size); a joint's box holds -box_quanta,
its rest packing with the default settings (joint_rest), times exp(h) / mean(exp(h))
when joint_unevenness > 0, h a second uneven field (same slope, rms joint_unevenness,
its own seed: default_rng((seed + u, 1))). The two unevennesses are separate settings
for the fit to search (owner, 2026-11-16, decision D); joint_unevenness 0, the default,
starts every joint at rest, so none starts squeezed. An uneven joint packing can put
the field's tail past the seal or the collapse limit before any rule acted (rulefix
decisions.md); claims then move borders so sizes follow energy. The empty margin (0, no space) is `margin` boxes on every side (default: the
filled cube's own width), rounded up so the side is a multiple of 8 (whole ladder
blocks). With universes > 1 the batch holds that many universes side by side, universe
u built from seed + u (its own uneven field) and stepped with seed + u. Each
universe's start reference counts its cells, the energy volumes and the joints between
them (cells.py), when the filled cube is at most 128 boxes a side (None above: reading
cells is a host pass too slow for the largest starts); every universe of a batch has
the same signs, so one count serves all.
Raises ValueError for cells_per_side < 3, k < 2, a negative margin, box_quanta < 1, a
negative joint_unevenness and
the uneven field's errors (filled side under 8).
"""
from __future__ import annotations

import numpy as np

from physics.engine.cells import cells
from physics.engine.state import State
from physics.engine.uneven_field import uneven_field


_COUNTED_SIDE = 128


def new_universe(cells_per_side: int = 7, boxes_per_cell_side: int = 10, seed: int = 1, unevenness: float = 0.3,
                 slope: float = -3.0, margin: int | None = None, box_quanta: int = 1000, universes: int = 1,
                 device: str = "cuda", joint_unevenness: float = 0.0) -> State:
    """Seeded wrapped energy volumes in a joint web, uneven whole quanta, at rest, inside an empty margin."""
    if cells_per_side < 3 or boxes_per_cell_side < 2:
        raise ValueError(f"need at least 3 cells per side of at least 2 boxes, got {cells_per_side}, "
                         f"{boxes_per_cell_side}")
    if not joint_unevenness >= 0.0:
        raise ValueError(f"joint_unevenness must be >= 0, got {joint_unevenness}")
    if box_quanta < 1:
        raise ValueError(f"box_quanta must be at least 1, got {box_quanta}")
    filled = cells_per_side * boxes_per_cell_side
    margin = filled if margin is None else int(margin)
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    side = -(-(filled + 2 * margin) // 8) * 8
    low = (side - filled) // 2
    cell_index = np.arange(cells_per_side)
    inner_odd = (cell_index % 2 == 1) & (cell_index < cells_per_side - 1)
    volume_cells = inner_odd[:, None, None] & inner_odd[None, :, None] & inner_odd[None, None, :]
    volume_boxes = np.repeat(np.repeat(np.repeat(volume_cells, boxes_per_cell_side, 0), boxes_per_cell_side, 1),
                             boxes_per_cell_side, 2)
    quanta = np.zeros((universes, side, side, side), dtype=np.int32)
    inside = (slice(low, low + filled),) * 3
    for index in range(universes):
        raw = np.exp(uneven_field(filled, seed + index, slope, unevenness))
        whole = np.maximum(np.rint(box_quanta * raw / raw.mean()), 1).astype(np.int32)
        quanta[index][inside] = np.where(volume_boxes, whole, -_joint_quanta(filled, seed + index, slope,
                                                                            joint_unevenness, box_quanta))
    phase = np.zeros(quanta.shape, dtype=np.float32)
    state = State.from_host(quanta, phase, seed=seed, device=device)
    count = _cell_count(quanta[0]) if filled <= _COUNTED_SIDE else None
    for start in state.start:
        start["cells"] = count
    return state


def _joint_quanta(filled: int, seed: int, slope: float, joint_unevenness: float, box_quanta: int):
    """A joint box's whole quanta (amount): box_quanta, or box_quanta times a normalised uneven field."""
    if joint_unevenness <= 0.0:
        return int(box_quanta)
    raw = np.exp(uneven_field(filled, (seed, 1), slope, joint_unevenness))
    return np.maximum(np.rint(box_quanta * raw / raw.mean()), 1).astype(np.int32)


def _cell_count(quanta: np.ndarray) -> int:
    """How many cells (energy volumes and joints) the start's signs hold."""
    return int(cells({"quanta": quanta, "clock": np.zeros(quanta.shape, dtype=np.float32)})["kind"].size)
