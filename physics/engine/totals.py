"""Totals: the exact books of a state, counted from its boxes.

Contract: `totals(state)` sums every box on the GPU in whole numbers and returns a dict
of exact Python ints:
  plus, minus   the plus quanta (energy volumes) and the minus quanta (joints, counted
                positive); amount = plus + minus, the total energy
  made          minus quanta squeezed joints made out of themselves since the start
  taken_joint   quanta of collapsed joints, gone with them; taken_volumes the same
                amounts taken off their volumes' plus quanta
  torn          plus quanta tears turned minus
  crossed       quanta that crossed joints (they change no total)
  wrong_boxes   boxes that break their cell: a box of a living cell that is 0 or has the
                other kind's sign, a box of a dead id that is not 0
  wrong_cells   living cells whose boxes do not sum to their cell_quanta
The total is not fixed (axiom 1) but its books are exact:
  amount = start amount + made - taken_joint - taken_volumes
  plus   = start plus - taken_volumes - torn
  minus  = start minus + made - taken_joint + torn
"""
from __future__ import annotations

import warp as wp

from physics.engine.state import BLOCK, State


@wp.kernel
def _count(kind: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64),
           quanta: wp.array4d(dtype=wp.int32), across: int, sums: wp.array(dtype=wp.int64)):
    cell = wp.tid()
    sort = kind[cell]
    total = wp.int64(0)
    wrong = wp.int64(0)
    for one in range(across):
        for two in range(across):
            for three in range(across):
                amount = quanta[cell, one, two, three]
                total += wp.int64(wp.abs(amount))
                if amount * sort <= 0 and not (sort == 0 and amount == 0):
                    wrong += wp.int64(1)
    if sort == 1:
        wp.atomic_add(sums, 0, total)
    elif sort == -1:
        wp.atomic_add(sums, 1, total)
    wp.atomic_add(sums, 2, wrong)
    if sort != 0 and total != cell_quanta[cell]:
        wp.atomic_add(sums, 3, wp.int64(1))


def totals(state: State) -> dict:
    """Count the state's plus and minus quanta from its boxes, with the books of what was made and taken."""
    sums = wp.zeros(4, dtype=wp.int64, device=state.device)
    wp.launch(_count, dim=state.count, device=state.device,
              inputs=[state.kind, state.cell_quanta, state.quanta, BLOCK, sums])
    plus, minus, wrong_boxes, wrong_cells = (int(value) for value in sums.numpy())
    books = state.books.numpy()
    return {"plus": plus, "minus": minus, "amount": plus + minus, "made": int(books[0]), "crossed": int(books[1]),
            "taken_joint": int(state.taken_joint), "taken_volumes": int(state.taken_volumes),
            "torn": int(state.torn), "wrong_boxes": wrong_boxes, "wrong_cells": wrong_cells}
