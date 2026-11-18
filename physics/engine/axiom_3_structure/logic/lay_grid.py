"""Lay grid: a small full grid of energy volumes that wraps round in all three directions.

Every volume has a joint toward each of its six neighbours and every joint lies between
two volumes; there is no edge. Every swirl turns about one direction, `swirl`. Across it a
volume is its neighbour's mirror (the other sense, and half a turn on along the direction
after the swirl's, which `take_own` lays by place); along it neighbours are laid alike. So
facing fluid agrees in all three directions, but the swirls' own direction is not like the
other two: no laying of single swirls makes it so. A joint's sense follows from its axis.
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


def calm_grid(across: int = ACROSS, seed: int = 1, device: str = "cpu", reds: tuple = (4, 4, 4),
              uneven: float = 0.0, swirl: int = 2) -> Store:
    """`reds` volumes along each direction, each a calm volume's quanta, and their fresh joints.

    A fresh joint holds a calm volume's quanta in minus. `uneven` is the share the volumes'
    quanta differ by at most, each drawn from the seed. All cells hold their own boxes.
    """
    if reds[(swirl + 1) % 3] % 2 or reds[(swirl + 2) % 3] % 2:
        raise ValueError("mirrors only close round an even number of volumes across the swirl")
    grid = tuple(2 * count for count in reds)
    place = np.indices(grid).reshape(3, -1)  # every cell's place along the three directions
    odd = place % 2
    volume, joint = odd.sum(axis=0) == 0, odd.sum(axis=0) == 1
    calm = QUANTA_A_BOX * across ** 3
    differ = uneven * (2 * np.random.default_rng(seed).random(reds[0] * reds[1] * reds[2]) - 1)
    turned = [(swirl + 1) % 3, (swirl + 2) % 3, swirl]  # the draws are dealt as the swirl lies
    drawn = np.ravel_multi_index(place[turned] // 2, [reds[way] for way in turned])
    quanta = np.where(joint, -calm, np.where(volume, np.rint(calm * (1 + differ[drawn])), 0))
    mirror = (place[(swirl + 1) % 3] // 2 + place[(swirl + 2) % 3] // 2) % 2
    sense = np.where(volume, mirror, odd[(swirl + 2) % 3])
    store = calm_cells(quanta.size, int(volume.sum() + joint.sum()), across, seed=seed,
                       device=device, grid=grid)
    store.cell_quanta.assign(quanta.astype(np.int32))
    store.cell_laid.assign((2 * swirl + sense).astype(np.uint8))
    take_all(store)
    return store
