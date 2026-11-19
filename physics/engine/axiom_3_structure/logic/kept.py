"""Kept: a store written to a file and read back, every number as it was.

A ball takes a minute to grow, so it is grown once and kept in the tmp folder (`ball.py`
names the file by the engine's own source, so a changed engine grows a fresh one).
"""
from dataclasses import fields
from pathlib import Path

import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.store import Store


def keep(store: Store, path: Path) -> None:
    """Write every array and number of the store to `path`."""
    held = {one.name: getattr(store, one.name) for one in fields(store)}
    np.savez_compressed(path, **{name: value.numpy() if isinstance(value, wp.array) else value
                                 for name, value in held.items() if value is not None})


def kept(path: Path, like: Store) -> Store:
    """The store kept at `path`, laid into `like`: a fresh store of the same room, on its device."""
    with np.load(path) as held:
        for name in held.files:
            now = getattr(like, name, None)
            if isinstance(now, wp.array):
                now.assign(held[name])
            elif name == "cell_past":
                like.cell_past = wp.array(held[name], dtype=wp.float32, device=like.box_quanta.device)
            else:
                setattr(like, name, type(now)(held[name]))
    return like
