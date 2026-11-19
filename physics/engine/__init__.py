"""The engine: `step` steps a store. The starts (`axiom_3_structure/logic/starts.py`):
`calm_cells` lays cells as rules, `one_cell` one cell with its own boxes, `three_cells` a
volume, a joint and a volume, `first_volume` the one volume a universe grows from by tears."""
from physics.engine.axiom_1_energy.logic.flow_inside import flow_inside
from physics.engine.axiom_3_structure.definitions.store import Store
from physics.engine.axiom_3_structure.logic.starts import (  # noqa: F401, the engine's door
    calm_cells, first_volume, one_cell, three_cells)
from physics.engine.axiom_4_crossing.logic.cross import cross
from physics.engine.axiom_5_packing.logic.press import press
from physics.engine.axiom_5_packing.logic.squeeze import squeeze
from physics.engine.axiom_6_limits.logic.make import make
from physics.engine.axiom_6_limits.logic.tear import tear


def step(store: Store, before_pass=None, before_cross=None) -> None:
    """One step: overstretched volumes tear; the fluid moves and crosses; squeezed joints make quanta.

    `before_pass(axis, parity)` lets a watcher look before each of the six passes inside,
    `before_cross(axis)` before each direction's crossing. A calm cell's step is the count.
    """
    tear(store)
    flow_inside(store, before_pass)
    cross(store, before_cross)
    squeeze(store)
    make(store)
    press(store)
    store.step += 1
