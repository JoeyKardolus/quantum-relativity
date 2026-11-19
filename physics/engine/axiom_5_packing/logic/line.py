"""Line: a bench of calm volumes in a line with a calm joint between each two, every rule on.

Laid by hand, as the crossing bench is, to stage a check where no first volume stands: no
universe, and nothing leaks out of it (`leaks` is off). Some volumes are made fuller, and
the first of those may be set going along the line. `gathering` and `coasting` are the
two scenes the window shows. In a line the pull is handed on to two sides of six, so it
fades to a tenth with every volume: this bench shows neighbours, never a pull from afar.
"""
from physics.engine.axiom_3_structure.definitions.kinds import ACROSS, QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.store import Store, put
from physics.engine.axiom_3_structure.logic.starts import calm_cells
from physics.engine.axiom_3_structure.logic.take_own import take_own
from physics.engine.axiom_5_packing.logic.lumps import FULLER, moving

VOLUMES = 11  # the line's length, chosen before any run: room for a lump to cross several cells


def line_of(store: Store) -> list:
    """The line's cells in order: volume, joint, volume and so on."""
    return [steps * store.side for steps in range(store.side)]


def calm_line(across: int = ACROSS, device: str = "cpu", fuller: tuple = (), going: float = 0.0) -> Store:
    """Eleven calm volumes and their joints; those at `fuller` hold 4 times as much, and the
    first of them goes along the line at `going` calm lengths a turn of a calm swirl."""
    side, calm = 2 * VOLUMES - 1, QUANTA_A_BOX * across ** 3
    store = calm_cells(side ** 3, side, across, device=device)
    store.side = side
    store.cell_quanta.zero_()
    for steps, cell in enumerate(line_of(store)):
        put(store.cell_quanta, cell, -calm if steps % 2 else calm * (FULLER if steps // 2 in fuller else 1))
        take_own(store, cell)
    if going:
        moving(store, [line_of(store)[2 * fuller[0]]], 1, going)
    return store


def gathering(across: int = ACROSS, device: str = "cpu") -> Store:
    """The middle volume 4 times fuller: its neighbours' fluid falls into it."""
    return calm_line(across, device, (VOLUMES // 2,))


def coasting(across: int = ACROSS, device: str = "cpu") -> Store:
    """The second volume 4 times fuller and going a calm length a turn: it travels down the line."""
    return calm_line(across, device, (1,), 1.0)
