"""Starts: the stores a scene begins from. Nothing here lays a grid: joints come by tears.

`calm_cells` lays cells as rules, `one_cell` one cell with its own boxes, `three_cells` the
crossing scene's volume, joint and volume, `first_volume` one overstretched volume alone.
"""
import warp as wp

from physics.engine.axiom_2_swirls.definitions.standing_mode import standing_mode
from physics.engine.axiom_2_swirls.definitions.swirl_pattern import swirl_patterns
from physics.engine.axiom_3_structure.definitions.kinds import ACROSS, QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.relations import join
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
            across=across, seed=seed, turn_per_step=2 * standing_mode(across)[1], cells=cells,
            cell_quanta=wp.full(cells, QUANTA_A_BOX * boxes, dtype=wp.int32),
            cell_laid=wp.full(cells, laid, dtype=wp.uint8),
            cell_made=wp.zeros(cells, dtype=wp.int32),
            cell_born=wp.zeros(cells, dtype=wp.int32),
            joint_ends=wp.full(cells, wp.vec3i(-1, -1, -1), dtype=wp.vec3i),
            joint_part=wp.zeros((cells, 2), dtype=wp.vec4),
            tile_home=wp.full(tiles, -1, dtype=wp.int32),
            tile_slots=wp.full(rows * TILE, -1, dtype=wp.int32),
            own_cell=wp.full(own_cap, -1, dtype=wp.int32),
            own_squeeze=wp.zeros(own_cap, dtype=wp.vec4d),
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


def three_cells(across: int = ACROSS, seed: int = 1, device: str = "cpu",
                parts: tuple = LINE_PARTS, along: int = 1) -> Store:
    """A volume, a joint and a volume in a line along direction `along`, each with its own boxes."""
    store = calm_cells(3, 3, across, seed=seed, device=device)
    for cell, halves in enumerate(parts):
        put(store.cell_quanta, cell, halves * QUANTA_A_BOX * across ** 3 // 2)
        put(store.cell_laid, cell, cell // 2 * (along + 1))  # B: A turned half round about the line
        take_own(store, cell)
    join(store, 1, 0, 2, along)
    return store


def first_volume(across: int = ACROSS, seed: int = 1, device: str = "cpu", room: int = 2048) -> Store:
    """One energy volume and no joint: the least overstretched there is, half a calm volume's
    quanta less one (`TEAR_SIZE` is 2). Room for `room` cells; the rest come by tears."""
    store = calm_cells(room, room, across, seed=seed, device=device)
    store.cell_quanta.zero_()
    put(store.cell_quanta, 0, QUANTA_A_BOX * across ** 3 // 2 - 1)
    store.cells = 1
    take_own(store, 0)
    return store
