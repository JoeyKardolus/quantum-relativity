"""The engine: `step` steps a store. The starts (`axiom_3_structure/logic/`):
`calm_cells` lays cells as rules, `one_cell` one cell with its own boxes, `three_cells` a
volume, a joint and a volume, `first_volume` the one volume a universe grows from by its leak."""
from physics.engine.axiom_1_energy.logic.flow_inside import flow_inside
from physics.engine.axiom_2_swirls.logic.turn import turn
from physics.engine.axiom_3_structure.definitions.store import Store
from physics.engine.axiom_3_structure.logic.first_volume import first_volume  # noqa: F401
from physics.engine.axiom_3_structure.logic.starts import (  # noqa: F401, the engine's door
    calm_cells, line_cells, one_cell, three_cells)
from physics.engine.axiom_4_crossing.logic.cross import cross
from physics.engine.axiom_4_crossing.logic.leak import leak
from physics.engine.axiom_5_packing.logic.press import press
from physics.engine.axiom_5_packing.logic.spread import spread
from physics.engine.axiom_5_packing.logic.squeeze import squeeze
from physics.engine.axiom_6_limits.logic.collapse import collapse
from physics.engine.axiom_6_limits.logic.make import make
from physics.engine.axiom_6_limits.logic.tear import tear


def step(store: Store, before_pass=None, before_cross=None) -> None:
    """One step: gathered leak tears off; joints past their limit collapse; the fluid moves,
    each whirl turns by its quanta, energy crosses and leaks; the pull is handed on; squeezed joints make quanta.

    `before_pass(axis, parity)` lets a watcher look before each of the six passes inside,
    `before_cross(axis)` before each direction's crossing. A calm cell's step is the count.
    """
    tear(store)
    collapse(store)
    flow_inside(store, before_pass)
    turn(store)
    cross(store, before_cross)
    leak(store)
    spread(store)
    squeeze(store)
    make(store)
    press(store)
    store.step += 1
