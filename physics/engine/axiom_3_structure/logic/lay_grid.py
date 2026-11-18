"""Lay grid: a small full grid of energy volumes that wraps round in all three directions.

Every volume has a joint toward each of its six neighbours and every joint lies between
two volumes; there is no edge. Every cell is the same ball whirl. Along each direction a
volume is its neighbour turned half round about the line between them, so the fluid of two
calm volumes that faces across a joint agrees, in all three directions alike. A joint is
laid as the volume before it along each direction.
"""
import numpy as np

from physics.engine import calm_cells
from physics.engine.axiom_3_structure.definitions.kinds import ACROSS, QUANTA_A_BOX
from physics.engine.axiom_3_structure.definitions.store import TILE, Store
from physics.engine.axiom_3_structure.logic.take_own import lay


def take_all(store: Store) -> None:
    """Every cell of a fresh store that holds quanta gets its own boxes, in one deal."""
    cells = np.flatnonzero(store.cell_quanta.numpy()).astype(np.int32)
    tiles = np.unique(cells // TILE)
    home = np.full(store.tile_home.shape[0], -1, dtype=np.int32)
    home[tiles] = np.arange(len(tiles))
    slots = np.full(store.tile_slots.shape[0], -1, dtype=np.int32)
    slots[home[cells // TILE] * TILE + cells % TILE] = np.arange(len(cells))
    store.tile_home.assign(home)
    store.tile_slots.assign(slots)
    store.own_cell.assign(cells)
    store.count, store.open_tiles = len(cells), len(tiles)
    lay(store, store.own_cell, 0, len(cells), store.box_quanta, store.box_phase)


def laid_by_place(place: np.ndarray) -> np.ndarray:
    """How each cell is laid from its place (3, cells): half round for every second one along."""
    turned = place // 2 % 2  # along each direction: is the cell turned half round across it
    odd_one = np.argmax(turned == (turned.sum(axis=0) == 1), axis=0)  # two turns make the third
    return np.where(turned.sum(axis=0) % 3 == 0, 0, 1 + odd_one)


def calm_grid(across: int = ACROSS, seed: int = 1, device: str = "cpu", reds: tuple = (4, 4, 4),
              uneven: float = 0.0, turned: tuple = (0, 1, 2)) -> Store:
    """`reds` volumes along each direction, each a calm volume's quanta, and their fresh joints.

    A fresh joint holds a calm volume's quanta in minus. `uneven` is the share the volumes'
    quanta differ by at most, each drawn from the seed; `turned` deals the same draws with
    the directions in another order. All cells hold their own boxes.
    """
    if any(count % 2 for count in reds):
        raise ValueError("half turns only close round an even number of volumes each direction")
    grid = tuple(2 * count for count in reds)
    place = np.indices(grid).reshape(3, -1)  # every cell's place along the three directions
    odd = place % 2
    volume, joint = odd.sum(axis=0) == 0, odd.sum(axis=0) == 1
    calm = QUANTA_A_BOX * across ** 3
    differ = uneven * (2 * np.random.default_rng(seed).random(reds[0] * reds[1] * reds[2]) - 1)
    drawn = np.ravel_multi_index(place[list(turned)] // 2, [reds[way] for way in turned])
    quanta = np.where(joint, -calm, np.where(volume, np.rint(calm * (1 + differ[drawn])), 0))
    store = calm_cells(quanta.size, int(volume.sum() + joint.sum()), across, seed=seed,
                       device=device, grid=grid)
    store.cell_quanta.assign(quanta.astype(np.int32))
    store.cell_laid.assign(laid_by_place(place).astype(np.uint8))
    take_all(store)
    return store
