"""New universe: energy volumes wrapped in a web of joints, blocks of Planck boxes of uneven whole quanta, in an empty margin.

Contract: `new_universe(cells_per_side=7, boxes_per_cell_side=10, seed=1, unevenness=0.3,
slope=-3, margin=None, box_quanta=1000, universes=1, device="cuda")` returns a State
(state.py) at rest, phase 0. The filled cube is cells_per_side**3 blocks of k**3 boxes
(k = boxes_per_cell_side); block (cx, cy, cz) is an energy volume when cx, cy and cz are
all odd and none is n - 1 (the outer layer), else joint. So every energy volume is
wrapped on all 26 sides by joint boxes and the joints form one web (bible "The
foundation"); energy volumes start about 10 boxes across (owner, 2026-11-16). Each box
holds a whole number: box_quanta * exp(g) / mean(exp(g)) rounded to the nearest whole,
at least 1, plus in an energy volume's block and minus in a joint's, g the uneven field
over the filled cube (uneven_field.py: seed, slope, rms unevenness; ripples at every
size). The empty margin (0, no space) is `margin` boxes on every side (default: the
filled cube's own width), rounded up so the side is a multiple of 8 (whole ladder
blocks). With universes > 1 the batch holds that many universes side by side, universe
u built from seed + u (its own uneven field) and stepped with seed + u. Each
universe's start reference counts its cells: the energy volumes and one joint web.
Raises ValueError for cells_per_side < 3, k < 2, a negative margin, box_quanta < 1 and
the uneven field's errors (filled side under 8).
"""
from __future__ import annotations

import numpy as np

from physics.engine.state import State
from physics.engine.uneven_field import uneven_field


def new_universe(cells_per_side: int = 7, boxes_per_cell_side: int = 10, seed: int = 1, unevenness: float = 0.3,
                 slope: float = -3.0, margin: int | None = None, box_quanta: int = 1000, universes: int = 1,
                 device: str = "cuda") -> State:
    """Seeded wrapped energy volumes in a joint web, uneven whole quanta, at rest, inside an empty margin."""
    if cells_per_side < 3 or boxes_per_cell_side < 2:
        raise ValueError(f"need at least 3 cells per side of at least 2 boxes, got {cells_per_side}, "
                         f"{boxes_per_cell_side}")
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
    sign = np.where(volume_boxes, 1, -1)
    quanta = np.zeros((universes, side, side, side), dtype=np.int32)
    inside = (slice(low, low + filled),) * 3
    for index in range(universes):
        raw = np.exp(uneven_field(filled, seed + index, slope, unevenness))
        whole = np.maximum(np.rint(box_quanta * raw / raw.mean()), 1).astype(np.int32)
        quanta[index][inside] = sign * whole
    phase = np.zeros(quanta.shape, dtype=np.float32)
    state = State.from_host(quanta, phase, seed=seed, device=device)
    for start in state.start:
        start["cells"] = int(volume_cells.sum()) + 1
    return state
