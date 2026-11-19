"""Starts: the stores a scene begins from. Nothing here lays a grid: joints come by the leak's tears.

`calm_cells` lays cells as rules, `one_cell` one cell with its own boxes, `three_cells` the
crossing bench's volume, joint and volume (nothing leaks there); `first_volume.py` has the
one volume a universe grows from.
"""
import numpy as np
import warp as wp

from physics.engine.axiom_2_swirls.definitions.standing_mode import standing_mode
from physics.engine.axiom_2_swirls.definitions.swirl_pattern import laid_at, swirl_patterns
from physics.engine.axiom_3_structure.definitions.kinds import ACROSS, QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.merged import ALONE
from physics.engine.axiom_3_structure.definitions.places import steps_of
from physics.engine.axiom_3_structure.definitions.store import TILE, Store, put
from physics.engine.axiom_3_structure.logic.take_own import take_own


def calm_cells(cells: int, own_cap: int, across: int = ACROSS, laid: int = 0, seed: int = 1,
               device: str = "cpu", tile_cap: int | None = None) -> Store:
    """A store of calm cells, each 3,000 quanta a box, its ball whirl laid `laid`.

    Room for `own_cap` own cells in `tile_cap` tile rows (left out: a row each).
    """
    tiles = (cells + TILE - 1) // TILE
    rows = min(tiles, own_cap) if tile_cap is None else tile_cap
    boxes = across ** 3
    with wp.ScopedDevice(device):
        return Store(
            across=across, seed=seed, turn_per_step=2 * standing_mode(across)[1],
            cell_quanta=wp.full(cells, QUANTA_A_BOX * boxes, dtype=wp.int32),
            cell_laid=wp.full(cells, laid, dtype=wp.uint8),
            cell_made=wp.zeros(cells, dtype=wp.int32),
            cell_pull=wp.zeros(cells, dtype=wp.float32),
            tile_home=wp.full(tiles, -1, dtype=wp.int32),
            tile_slots=wp.full(rows * TILE, -1, dtype=wp.int32),
            own_cell=wp.full(own_cap, -1, dtype=wp.int32),
            own_squeeze=wp.zeros(own_cap, dtype=wp.vec4d),
            own_whole=wp.array(np.arange(own_cap), dtype=wp.int32),
            own_ring=wp.array(np.arange(own_cap), dtype=wp.int32),
            own_part=wp.array(np.tile(ALONE, (own_cap, 1)), dtype=wp.vec4i),
            box_quanta=wp.zeros((own_cap, boxes), dtype=wp.int32),
            box_phase=wp.zeros((own_cap, boxes), dtype=wp.float32),
            pattern=wp.array(swirl_patterns(across), dtype=wp.vec2d))


def one_cell(across: int = ACROSS, seed: int = 1, device: str = "cpu") -> Store:
    """One energy volume with its own boxes, one ball whirl laid in them."""
    store = calm_cells(1, 1, across, seed=seed, device=device)
    take_own(store, 0)
    return store


# The crossing scene's start, chosen before it was ever run and not moved: the fresh joint
# holds a calm volume's quanta in minus (no new number), and the two volumes share two calm
# volumes' quanta 3 to 1, so A clearly holds more and they are calm volumes once even.
LINE_PARTS = (3, -2, 1)  # halves of a calm volume's quanta: volume A, the joint, volume B


def line_cells(along: int = 1) -> tuple:
    """The numbers of `three_cells`' volume A, joint and volume B: 0, 1 and 2 steps along `along`."""
    return tuple(steps * 3 ** (2 - along) for steps in range(3))


def three_cells(across: int = ACROSS, seed: int = 1, device: str = "cpu",
                parts: tuple = LINE_PARTS, along: int = 1) -> Store:
    """A volume, a joint and a volume in a line along direction `along`, each with its own boxes."""
    store = calm_cells(27, 3, across, seed=seed, device=device)
    store.side = 3
    store.cell_quanta.zero_()
    for cell, halves in zip(line_cells(along), parts):
        put(store.cell_quanta, cell, halves * QUANTA_A_BOX * across ** 3 // 2)
        put(store.cell_laid, cell, laid_at(steps_of(cell, 3)))  # B: A half round about the line
        take_own(store, cell)
    return store
