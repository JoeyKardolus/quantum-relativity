"""Swirl arrays: the state axiom 2 owns, each cell's swirl.

Contract: `ARRAYS = {name: (dtype, shape, each_step)}`, all per cell id; there is no host
value. Every cell, energy volume and joint, is a swirl (axiom 2): its fluid goes round a
line through the middle of its block.
  swirl_axis   int32: the axis that line lies along, 0, 1 or 2
  swirl_sense  int32: +1 or -1, the way round; 0 only for an id that is no cell
The swirl itself is in the fluid: `start_swirls` writes it into the phase of every box
(swirl_phase.py) and from then on the fluid carries it. These two only remember how
the cell was made, so the going round can be read against them and a tear knows which
way a volume's layers lie. `start_swirls` gives them to the start's cells and a tear
hands the torn volume's to its two new cells; nothing else writes them.
"""
from __future__ import annotations

ARRAYS = {
    "swirl_axis": ("int32", "cell", False),
    "swirl_sense": ("int32", "cell", False),
}

HOST = {}
