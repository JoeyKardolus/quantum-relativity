"""Quantum move: whole quanta taken from the boxes of one energy volume and given to the boxes of another, in one go.

Contract: `quantum_move(state, moves)` applies every (giver, taker, amount) of `moves`
in order: `amount` whole plus quanta leave the boxes of energy volume `giver` and
arrive in the boxes of energy volume `taker`, at once, however far apart the two are
(axiom 7: the direct pull does not pass through neighbouring cells, so distance does
not slow it; a handover between neighbours uses the same move). The giver's boxes give
in proportion to what each holds above 1 quantum (no box drops below 1), the taker's
boxes take in proportion to what each holds, both in whole quanta by largest remainders
(ties by box order), so the shape of the fluid in both cells is kept and nothing is
rounded away. Energy moves as numbers between the two cells' own blocks; no box changes
cell, no phase or sign is touched and the total is exact; state.cell_quanta follows.
Raises ValueError when a cell is not an energy volume, the giver holds fewer than
`amount` quanta above 1 a box, or a box would reach 2**30. An empty list does nothing.
Returns None.
"""
from __future__ import annotations

import numpy as np

from physics.engine.state import State


def quantum_move(state: State, moves: list) -> None:
    """Move whole plus quanta between energy volumes' boxes, exactly, any distance, at once."""
    if not moves:
        return
    held = state.cell_quanta.numpy()
    for giver, taker, amount in moves:
        for cell in (giver, taker):
            if state.kind_h[cell] != 1:
                raise ValueError(f"cell {cell} is not an energy volume")
        giving, _ = state.block(giver)
        taking, _ = state.block(taker)
        spare = giving.astype(np.int64).reshape(-1) - 1
        if amount > int(spare.sum()):
            raise ValueError(f"cell {giver} holds {int(spare.sum())} quanta above 1 a box, fewer than {amount}")
        grown = taking.astype(np.int64).reshape(-1)
        grown += _shared(amount, grown)
        if grown.max() >= 2 ** 30:
            raise ValueError("a box may hold fewer than 2**30 quanta")
        state.set_block(giver, (spare + 1 - _shared(amount, spare)).reshape(giving.shape))
        state.set_block(taker, grown.reshape(taking.shape))
        held[giver] -= amount
        held[taker] += amount
    state.cell_quanta.assign(held)


def _shared(amount: int, weights: np.ndarray) -> np.ndarray:
    """`amount` cut into whole numbers in proportion to `weights` (int64), largest remainders first."""
    total = int(weights.sum())
    shares = amount * weights // total
    left = amount - int(shares.sum())
    if left:
        remainders = amount * weights % total
        shares[np.argsort(-remainders, kind="stable")[:left]] += 1
    return shares
