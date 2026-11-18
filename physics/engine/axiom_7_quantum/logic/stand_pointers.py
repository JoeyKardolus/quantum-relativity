"""Stand pointers: a pointer whose cell holds no half of its split any more is drawn again by share.

Contract: `stand_pointers(state, number)` writes only `pointer` of the store. A pointer
stands on one of its split's halves' cells. When that half is gone (it went on with its
cell's fluid, or an event moved it), the pointer jumps to one of the cells the halves
lie in now, with that cell's share of the energy as its chance (`pointer_law`, draw
`SPLIT_POINTER` numbered `number`: `AFTER_CROSSING` 2 ** 32 - 2 from `carry_on`,
`AFTER_EVENTS` 2 ** 32 - 1 from `carry_events`). Every other pointer is left alone.
"""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform_rows
from physics.engine.axiom_1_energy.definitions.passes import SPLIT_POINTER
from physics.engine.axiom_7_quantum.definitions.pointer_law import pointer_law

AFTER_CROSSING = 2 ** 32 - 2
AFTER_EVENTS = 2 ** 32 - 1
_SOURCE_BITS = 32


def stand_pointers(state, number: int) -> None:
    """Draw again, by share, every pointer whose cell holds no half of its split."""
    store = state.quantum
    key = (store["part_source"] << _SOURCE_BITS) | store["part_cell"]
    lost = ~np.isin((store["source"] << _SOURCE_BITS) | store["pointer"], key)
    sources = store["source"][lost]
    uniform = draw_uniform_rows(state.seed, state.step, SPLIT_POINTER, (sources << _SOURCE_BITS) | number)
    store["pointer"][lost] = pointer_law(store["part_source"], store["part_cell"], store["part_quanta"], sources,
                                         uniform)
