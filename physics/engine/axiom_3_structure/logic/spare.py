"""Spare: arrays a step works in, kept with the store on its device so no step asks for memory.

They hold nothing between steps (the pull's past, which does, is the store's own
`cell_past`), so a copy of a store or a kept file starts without them. `copies` gives the
totals, the flow and the boxes in use as they are now, for a launch that must read what
was there before it wrote: only the slots in use are copied.
"""
import warp as wp

from physics.engine.axiom_3_structure.definitions.store import Store


def spare(store: Store, name: str, shape, dtype) -> wp.array:
    """The store's spare array `name`, made on its device the first time it is asked for."""
    kept = store.__dict__.setdefault("_spare", {})
    if name not in kept:
        kept[name] = wp.zeros(shape, dtype=dtype, device=store.box_quanta.device)
    return kept[name]


def copies(store: Store) -> tuple:
    """The cells' totals, their flow and the boxes of the slots in use, copied as they are now."""
    held = spare(store, "held", store.cell_quanta.shape, wp.int64)
    flowed = spare(store, "flowed", store.cell_flow.shape, wp.int64)
    had = spare(store, "had", store.box_quanta.shape, wp.int32)
    wp.copy(held, store.cell_quanta)
    wp.copy(flowed, store.cell_flow)
    if store.count:
        wp.copy(had, store.box_quanta, count=store.count * store.box_quanta.shape[1])
    return held, flowed, had
