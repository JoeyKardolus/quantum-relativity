"""Check step: axiom 3 checked on the state as it stands after a step.

Contract: `check_step(state, settings) -> dict` of plain numbers; it writes nothing. It
holds everything `measure_grid` returns (`size`, `warp`, the cells by kind, `endless`,
shortest and longest by kind) and these, all 0 while the axiom holds:
  wrong_lengths  living cells whose `length` is not, bit for bit, `size_constant` over
                 their quanta (endless for a cell holding 0): a size set by anything
                 but its energy
  dead_lengths   ids that are no cell with a length
  dead_boxes     boxes that are not 0 in the block of an id that is no cell: a block
                 belongs to its cell for good and is never used for anything else
and `dead_ids`, the ids handed out that are no cell now (cells gone in a collapse).
The pattern itself (touching, bare, web pieces, flickered and so on) is `check_pattern`,
which the run calls beside this.
"""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_3_structure.definitions.kinds import NO_CELL
from physics.engine.axiom_3_structure.logic.measure_grid import measure_grid


def _lawful_lengths(cell_quanta: np.ndarray, size_constant: float) -> np.ndarray:
    """The size law on the host for every cell at once, float64 inside and float32 out."""
    held = np.maximum(cell_quanta, 1).astype(np.float64)
    return np.where(cell_quanta > 0, (np.float64(size_constant) / held).astype(np.float32), np.float32(np.inf))


def check_step(state, settings) -> dict:
    """Axiom 3's counts and measures for the state as it stands."""
    kind = state.host("kind")
    length = state.host("length")
    living = kind != NO_CELL
    lawful = _lawful_lengths(state.host("cell_quanta"), settings.size_constant)
    boxes = state.quanta.numpy()[:state.count]
    return {
        **measure_grid(state),
        "wrong_lengths": int(np.count_nonzero(living & (length != lawful))),
        "dead_lengths": int(np.count_nonzero(~living & (length != 0.0))),
        "dead_boxes": int(np.count_nonzero(boxes[~living])),
        "dead_ids": int(np.count_nonzero(~living)),
    }
