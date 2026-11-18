"""Tear map: where the fluid of a torn volume goes, box by box.

Contract: `tear_map(block_quanta, block_phase, axis)` (host) takes a volume's block
((6, 6, 6) quanta, all 0 or more, and phase) and the axis it tears along, and returns
three `(quanta, phase)` blocks: the low volume, the new joint, the high volume. Along
`axis`, layers 0 and 1 stay plus (the low volume), layers 2 and 3 flip from plus to
minus (the place it tears is the new joint), layers 4 and 5 are the high volume. Each
piece fills its own whole block: its new layer `t` takes from old layer `base + t // 3`
(base 0, 2, 4), so each old layer is stretched over three. A source box's quanta go over
its three new boxes in whole numbers, as `whole_shares` with equal weights deals them:
a third each, the rest one each from the lowest layer up. Each new box copies its
source box's phase. Exact: the three pieces' quanta, ignoring sign, sum to the volume's.
Inputs are not changed.
"""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_3_structure.definitions.kinds import BLOCK

_PIECE_SIGNS = (1, -1, 1)
_STRETCH = BLOCK // 2


def _piece(quanta: np.ndarray, phase: np.ndarray, base: int, sign: int) -> tuple:
    """One piece, torn along axis 0: old layers `base` and `base + 1` stretched over the whole block."""
    new_quanta = np.zeros((BLOCK, BLOCK, BLOCK), dtype=np.int32)
    new_phase = np.zeros((BLOCK, BLOCK, BLOCK), dtype=np.float64)
    for layer in range(BLOCK):
        source = quanta[base + layer // _STRETCH].astype(np.int64)
        rest_goes_here = (source % _STRETCH) > (layer % _STRETCH)
        new_quanta[layer] = sign * (source // _STRETCH + rest_goes_here)
        new_phase[layer] = phase[base + layer // _STRETCH]
    return new_quanta, new_phase


def tear_map(block_quanta: np.ndarray, block_phase: np.ndarray, axis: int) -> tuple:
    """The low volume, the new joint and the high volume of a block torn along `axis`."""
    quanta = np.asarray(block_quanta)
    phase = np.asarray(block_phase)
    if quanta.shape != (BLOCK, BLOCK, BLOCK) or phase.shape != (BLOCK, BLOCK, BLOCK):
        raise ValueError(f"a block is {(BLOCK, BLOCK, BLOCK)}")
    if axis not in (0, 1, 2) or (quanta < 0).any():
        raise ValueError("only an energy volume tears, along axis 0, 1 or 2")
    turned_quanta, turned_phase = np.moveaxis(quanta, axis, 0), np.moveaxis(phase, axis, 0)
    pieces = []
    for number, sign in enumerate(_PIECE_SIGNS):
        new_quanta, new_phase = _piece(turned_quanta, turned_phase, number * 2, sign)
        pieces.append((np.ascontiguousarray(np.moveaxis(new_quanta, 0, axis)),
                       np.ascontiguousarray(np.moveaxis(new_phase, 0, axis))))
    return tuple(pieces)
