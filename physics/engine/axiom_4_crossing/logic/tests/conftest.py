"""Shared by the crossing logic's tests: a small real universe on the CPU and a few meters.

`universe(**changed)` is a real 4-a-side start (8 energy volumes wrapped in 56 joints)
made by `new_universe`, nothing staged. The pull is a million times weaker than the
default unless a test asks otherwise, so the start's joints stand. `face_way` finds the
way from one volume straight through a face joint. `ramp` gives one volume's fluid a
motion along box axis 0, its +x side one step ahead of a fluid of phase 0 beyond it.
`still` wipes every phase, `held_by` counts a cell's boxes.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics import engine
from physics.engine.axiom_3_structure.definitions.direction_codes import code_of_host

wp.config.quiet = True

PLUS_X = code_of_host(1, 0, 0)


def universe(side: int = 4, seed: int = 1, **changed):
    """A fresh real universe on the CPU and its settings."""
    settings = engine.Settings(**{"pull_strength": 1e-13, **changed})
    return engine.new_universe(settings, cells_per_side=side, seed=seed, device="cpu"), settings


def face_way(state, volume: int, mouth: int = PLUS_X) -> dict:
    """The way that leaves `volume` through its mouth `mouth`: slot, the far volume, the joint, which end it is."""
    mouths = {"a": state.host("way_mouth_a"), "b": state.host("way_mouth_b")}
    for slot, way_a, way_b, joint in engine.read_ways(state).tolist():
        for end, cell, other in (("a", way_a, way_b), ("b", way_b, way_a)):
            if cell == volume and mouths[end][slot] == mouth:
                return {"slot": slot, "far": other, "joint": joint, "end": end}
    raise LookupError(f"volume {volume} has no way through mouth {mouth}")


def ramp(state, volume: int, step: float) -> None:
    """Set one volume's phases falling by `step` a box along box axis 0, to `step` at its +x side: it moves toward +x."""
    along = (engine.BLOCK - np.arange(engine.BLOCK, dtype=np.float64))[:, None, None]
    state.set_block("phase", volume, np.broadcast_to(step * along, (engine.BLOCK,) * 3).copy())


def still(state) -> None:
    """Give every box of the universe the phase 0 and every cell's boxes one fill: an even fluid of one phase."""
    state.phase.zero_()
    quanta = state.quanta.numpy()
    even = np.abs(quanta).reshape(len(quanta), -1).sum(axis=1) // quanta[0].size
    state.quanta.assign((np.sign(quanta.reshape(len(quanta), -1)[:, :1]) * even[:, None] * np.ones((1, quanta[0].size),
                         dtype=np.int64)).reshape(quanta.shape).astype(np.int32))


def held_by(state, cell: int) -> int:
    """The quanta in one cell's boxes, ignoring sign."""
    return int(np.abs(state.block("quanta", cell)).sum())
