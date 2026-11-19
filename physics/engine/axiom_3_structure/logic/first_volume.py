"""First volume: the one energy volume a universe grows from, with room for the layers its leak tears off."""
from physics.engine.axiom_3_structure.definitions.kinds import ACROSS
from physics.engine.axiom_3_structure.definitions.store import Store, put
from physics.engine.axiom_3_structure.logic.starts import calm_cells
from physics.engine.axiom_3_structure.logic.take_own import take_own

# The first volume's quanta, the plainest: half of what a cell's whole number can hold, so
# any two cells' totals can be added without running over.
FIRST_QUANTA = 2 ** 30


def first_volume(across: int = ACROSS, device: str = "cpu", layers: int = 8,
                 quanta: int = FIRST_QUANTA) -> Store:
    """One energy volume and no joint, with room for `layers` layers of cells round it."""
    side, wide = 4 * layers + 1, 2 * layers + 1
    store = calm_cells(side ** 3, wide ** 3 + 3 * wide * wide * (wide - 1), across, device=device)
    store.side, store.leaks = side, True
    store.cell_quanta.zero_()
    put(store.cell_quanta, side ** 3 // 2, quanta)
    take_own(store, side ** 3 // 2)
    return store
