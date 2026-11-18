"""Cell turns: how many whole turns one cell's fluid makes round an axis of its block, layer by layer.

Contract: `cell_turns(quanta, phase, axis)` takes one cell's block as the engine holds
it (`state.block("quanta", cell)`, whole signed quanta, and `state.block("phase", cell)`,
radians; both of one 3D shape) and a block axis (0, 1 or 2). It returns a list of ints,
one per layer along `axis`: the windings (phase_winding.py) of all the unit squares of
that layer added up, counted right-handed about +axis. Where every box of the layer
holds energy this is the whole turns of phase going once round the layer's rim. A box
holding 0 quanta has no phase, and the squares touching it count 0.
A cell that is one whole swirl round `axis` reads definitions/expected_turns.py in
every layer; any other list says the cell is no longer that swirl. Only the cell's own
boxes are read: a swirl lives inside its cell. Raises ValueError for arrays of unequal
or non-3D shape, or another axis.
"""
from __future__ import annotations

import numpy as np

from physics.measure.winding.definitions.in_plane import in_plane
from physics.measure.winding.logic.phase_winding import phase_winding


def cell_turns(quanta: np.ndarray, phase: np.ndarray, axis: int) -> list:
    """The whole turns of phase in each layer of a block across `axis`."""
    quanta = np.asarray(quanta)
    phase = np.asarray(phase, dtype=np.float64)
    if quanta.shape != phase.shape or quanta.ndim != 3:
        raise ValueError(f"quanta and phase must be one 3D shape, got {quanta.shape} and {phase.shape}")
    layer_axes = in_plane(axis)
    field = np.sqrt(np.abs(quanta).astype(np.float64)) * np.exp(1j * phase)
    return [int(turns) for turns in phase_winding(field)[axis].sum(axis=layer_axes)]
