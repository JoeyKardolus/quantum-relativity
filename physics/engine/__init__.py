"""The engine, layer 2: energy volumes whose fluid turns as whole swirls, and joints it crosses.

`calm_cells` lays cells as rules, `one_cell` is one cell with its own boxes, `three_cells` a
volume, a joint and a volume in a line; `step` moves the fluid and lets it cross.
"""
import warp as wp

from physics.engine.axiom_1_energy.logic.flow_inside import flow_inside
from physics.engine.axiom_2_swirls.definitions.standing_mode import standing_mode
from physics.engine.axiom_2_swirls.definitions.swirl_pattern import swirl_patterns
from physics.engine.axiom_3_structure.definitions.kinds import ACROSS, QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.store import TILE, Store, put
from physics.engine.axiom_3_structure.logic.take_own import take_own
from physics.engine.axiom_4_crossing.logic.cross import cross
from physics.engine.axiom_5_packing.logic.press import press


def calm_cells(cells: int, own_cap: int, across: int = ACROSS, laid: int = 4, seed: int = 1,
               device: str = "cpu", tile_cap: int | None = None) -> Store:
    """A store of calm cells, each 3,000 quanta a box with the swirl `laid`.

    It has room for `own_cap` own cells in `tile_cap` tile rows (left out: a row for each).
    """
    tiles = (cells + TILE - 1) // TILE
    rows = min(tiles, own_cap) if tile_cap is None else tile_cap
    boxes = across ** 3
    with wp.ScopedDevice(device):
        return Store(
            across=across, seed=seed, turn_per_step=standing_mode(across)[1],
            cell_quanta=wp.full(cells, QUANTA_A_BOX * boxes, dtype=wp.int32),
            cell_laid=wp.full(cells, laid, dtype=wp.uint8),
            cell_heat=wp.zeros(cells, dtype=wp.int32),
            tile_home=wp.full(tiles, -1, dtype=wp.int32),
            tile_slots=wp.full(rows * TILE, -1, dtype=wp.int32),
            own_cell=wp.full(own_cap, -1, dtype=wp.int32),
            box_quanta=wp.zeros((own_cap, boxes), dtype=wp.int32),
            box_phase=wp.zeros((own_cap, boxes), dtype=wp.float32),
            pattern=wp.array(swirl_patterns(across), dtype=wp.vec2d))


def one_cell(across: int = ACROSS, seed: int = 1, device: str = "cpu") -> Store:
    """One energy volume with its own boxes, one whole swirl laid in them."""
    store = calm_cells(1, 1, across, seed=seed, device=device)
    take_own(store, 0)
    return store


# The crossing scene's start, chosen before it was ever run and not moved: the joint holds a
# calm volume's quanta in minus (no new number), and the two volumes share two calm volumes'
# quanta 3 to 1, so A clearly holds more and they are calm volumes once they are even.
LINE_PARTS = (3, -2, 1)  # halves of a calm volume's quanta: volume A, the joint, volume B


def three_cells(across: int = ACROSS, seed: int = 1, device: str = "cpu",
                parts: tuple = LINE_PARTS) -> Store:
    """A volume, a joint and a volume in a line, each with its own boxes and one whole swirl."""
    store = calm_cells(3, 3, across, seed=seed, device=device)
    for cell, halves in enumerate(parts):
        put(store.cell_quanta, cell, halves * QUANTA_A_BOX * across ** 3 // 2)
        put(store.cell_laid, cell, 4 + cell // 2 % 2)  # each volume the mirror of the one before
        take_own(store, cell)
    return store


def step(store: Store, before_pass=None, before_cross=None) -> None:
    """One step: the fluid moves inside every own cell, crosses the joints, and presses on them.

    `before_pass(axis, parity)` lets a watcher look before each of the six passes inside,
    `before_cross()` before the crossing. A calm cell's step is the count.
    """
    flow_inside(store, before_pass)
    cross(store, before_cross)
    press(store)
    store.step += 1
