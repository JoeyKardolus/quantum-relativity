"""Line: benches of calm volumes with a calm joint wherever two share a side, every rule on.

Laid by hand, as the crossing bench is, to stage a check where no first volume stands: no
universe, and nothing leaks out of it (`leaks` is off). `calm_block` lays a line, a sheet or
a block; some volumes are made fuller, and `calm_line`'s first fuller one may be set going
along the line. `gathering`, `coasting`, `falling` and `dense` are the scenes the window shows. A bench
has ends, and every volume pulls every other, so a bench draws together toward its own
middle: only its middle is calm.
"""
import numpy as np

from physics.engine.axiom_3_structure.definitions.kinds import ACROSS, QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.store import Store, put
from physics.engine.axiom_3_structure.logic.starts import calm_cells
from physics.engine.axiom_3_structure.logic.take_own import take_own
from physics.engine.axiom_5_packing.logic.lumps import FULLER, moving

VOLUMES = 11  # the line's length, chosen before any run: room for a lump to cross several cells


def volume_at(store: Store, out: tuple) -> int:
    """The cell of the volume `out` volumes along the three directions from the bench's first."""
    return int(np.ravel_multi_index(tuple(2 * one for one in out), (store.side,) * 3))


def calm_block(wide: tuple, fuller: dict = {}, across: int = ACROSS, device: str = "cpu") -> Store:
    """Calm volumes, `wide` of them along each direction, and their joints; `fuller` says how
    many times a calm volume's quanta the volume at each named place holds."""
    side, calm = 2 * max(wide) - 1, QUANTA_A_BOX * across ** 3
    store = calm_cells(side ** 3, side ** 3, across, device=device)
    store.side = side
    store.cell_quanta.zero_()
    for steps in np.ndindex(*(2 * one - 1 for one in wide)):
        odd = sum(one % 2 for one in steps)
        if odd > 1:
            continue  # no cell stands where two or three steps are odd
        cell = int(np.ravel_multi_index(steps, (side,) * 3))
        put(store.cell_quanta, cell, -calm if odd else int(calm * fuller.get(tuple(one // 2 for one in steps), 1)))
        take_own(store, cell)
    return store


def line_of(store: Store) -> list:
    """The line's cells in order: volume, joint, volume and so on."""
    return [steps * store.side for steps in range(store.side)]


def calm_line(across: int = ACROSS, device: str = "cpu", fuller: tuple = (), going: float = 0.0) -> Store:
    """Eleven calm volumes and their joints; those at `fuller` hold 4 times as much, and the
    first of them goes along the line at `going` calm lengths a turn of a calm swirl."""
    store = calm_block((1, VOLUMES, 1), {(0, one, 0): FULLER for one in fuller}, across, device)
    if going:
        moving(store, [line_of(store)[2 * fuller[0]]], 1, going)
    return store


def gathering(across: int = ACROSS, device: str = "cpu") -> Store:
    """The middle volume 4 times fuller: its neighbours' fluid falls into it."""
    return calm_line(across, device, (VOLUMES // 2,))


def coasting(across: int = ACROSS, device: str = "cpu") -> Store:
    """The second volume 4 times fuller and going a calm length a turn: it travels down the line."""
    return calm_line(across, device, (1,), 1.0)


def falling(across: int = ACROSS, device: str = "cpu") -> Store:
    """Two volumes 4 times fuller, four volumes apart, alike either side of the middle: they draw together."""
    return calm_line(across, device, (3, 7))


def dense(across: int = ACROSS, device: str = "cpu") -> Store:
    """A sheet of 7 by 7 calm volumes, the middle nine 4 times fuller: they gather on their middle."""
    return calm_block((1, 7, 7), {(0, row, column): FULLER for row in (2, 3, 4) for column in (2, 3, 4)}, across, device)
