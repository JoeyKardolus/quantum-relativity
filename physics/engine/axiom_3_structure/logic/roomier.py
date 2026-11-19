"""Roomier: the same universe moved into a store with room for more layers, every number kept.

Warp arrays cannot grow, and a place's number depends on the room's width; so a universe
whose outermost layer has reached the room's end is laid into a bigger store: each cell at
the same steps from the first volume, its slot, its boxes, its flow and its pull as they
were. The pull's past moves with the volumes; a group's past is its volumes' summed, so
the groups of the new room (other ones: a group is bookkeeping, not physics) start with
theirs. Nothing is stepped here and no quantum moves.
"""
import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.places import steps_of
from physics.engine.axiom_3_structure.definitions.store import TILE, Store
from physics.engine.axiom_3_structure.logic.first_volume import first_volume
from physics.engine.axiom_5_packing.definitions.groups import top_of, wide_of


def _past(past: np.ndarray, wide: int, new: Store) -> np.ndarray:
    """The volumes' past in the new room's block, and each level of groups summed from the one below."""
    room, out = wide_of(new.side), (wide_of(new.side) - wide) // 2
    level = np.zeros((len(past), room, room, room), dtype=np.float32)
    level[:, out:out + wide, out:out + wide, out:out + wide] = past[:, :wide ** 3].reshape(-1, wide, wide, wide)
    levels = [level]
    for _ in range(top_of(new.side, new.near)):
        odd = levels[-1].shape[1] % 2
        below = np.pad(levels[-1], ((0, 0),) + ((0, odd),) * 3)
        half = below.shape[1] // 2
        levels.append(below.reshape(-1, half, 2, half, 2, half, 2).sum(axis=(2, 4, 6)))
    return np.concatenate([one.reshape(len(past), -1) for one in levels], axis=1)


def roomier(store: Store, layers: int) -> Store:
    """The universe of `store` in a fresh store with room for `layers` layers, on the same device."""
    new = first_volume(store.across, str(store.box_quanta.device), layers, near=store.near)
    out, count = (new.side - store.side) // 2, store.count
    there = np.ravel_multi_index(tuple((steps_of(np.arange(store.side ** 3), store.side) + out).T), (new.side,) * 3)
    for name in ("cell_quanta", "cell_made", "cell_pull", "cell_flow"):
        moved = np.zeros_like(getattr(new, name).numpy())
        moved[there] = getattr(store, name).numpy()
        getattr(new, name).assign(moved)
    for name in ("own_squeeze", "own_turned", "box_quanta", "box_phase"):
        moved = getattr(new, name).numpy()
        moved[:count] = getattr(store, name).numpy()[:count]
        getattr(new, name).assign(moved)
    own = np.full(new.own_cell.shape[0], -1, dtype=np.int32)
    own[:count] = there[store.own_cell.numpy()[:count]]
    home, slots = np.full(new.tile_home.shape[0], -1, dtype=np.int32), np.full(new.tile_slots.shape[0], -1, dtype=np.int32)
    tiles = np.unique(own[:count] // TILE)
    home[tiles] = np.arange(len(tiles))
    slots[home[own[:count] // TILE] * TILE + own[:count] % TILE] = np.arange(count)
    for kept, now in ((new.own_cell, own), (new.tile_home, home), (new.tile_slots, slots)):
        kept.assign(now)
    for name in ("leaks", "step", "ledger", "collapses", "flipped"):
        setattr(new, name, getattr(store, name))
    new.count, new.open_tiles = count, len(tiles)
    if store.cell_past is not None:
        new.cell_past = wp.array(_past(store.cell_past.numpy(), wide_of(store.side), new), dtype=wp.float32,
                                 device=new.box_quanta.device)
    return new
