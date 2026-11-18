"""The engine, layer 1: one energy volume whose fluid turns as one whole swirl.

`calm_cells` lays cells as rules, `one_cell` is one cell with its own boxes, `step` moves the fluid.
"""
import warp as wp

from physics.engine.axiom_1_energy.logic.flow_inside import flow_inside
from physics.engine.axiom_2_swirls.definitions.standing_mode import standing_mode
from physics.engine.axiom_2_swirls.definitions.swirl_pattern import swirl_patterns
from physics.engine.axiom_3_structure.definitions.kinds import ACROSS, QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.store import TILE, Store
from physics.engine.axiom_3_structure.logic.take_own import take_own


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


def step(store: Store) -> None:
    """One step: the fluid moves inside every own cell; a calm cell's step is the count."""
    flow_inside(store)
    store.step += 1
