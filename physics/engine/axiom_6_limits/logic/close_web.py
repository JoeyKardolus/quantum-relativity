"""Close web: when a joint goes, the joints that lay on opposite sides of it touch each other.

Contract: `close_web(state, held, entries)`. `entries` are the neighbour entries the gone
joint had, `(cell, code)` each. For every living joint it listed at a code and every
other living joint it listed at the opposite code, the two become neighbours: the first
lists the second in the direction the gone joint lay in, and the second lists the first
back. What the gone joint kept apart on two opposite sides is now side by side: its two
volumes merge (`collapse`), and its two joints touch, so minus goes on forming one web
(axiom 3) where a joint has gone. An entry that is there already is not added twice.
It moves no energy, makes no way through a new joint and changes no volume's relations.
"""
from __future__ import annotations

from physics.engine.axiom_3_structure.definitions.direction_codes import opposite_host
from physics.engine.axiom_3_structure.definitions.kinds import JOINT
from physics.engine.axiom_6_limits.logic.cell_value import cell_value
from physics.engine.axiom_6_limits.logic.hold_cell import hold_cell


def close_web(state, held, entries) -> None:
    """Make the living joints on opposite sides of a gone joint neighbours of each other."""
    at_code: dict = {}
    for one, code in entries:
        if int(cell_value(state, held, "kind", one)) == JOINT:
            at_code.setdefault(code, []).append(one)
    for code in sorted(at_code):
        facing = opposite_host(code)
        for one in at_code[code]:
            hold_cell(state, held, one)
            for other in at_code.get(facing, ()):
                if other != one and (other, facing) not in held.entries[one]:
                    held.entries[one].append((other, facing))
