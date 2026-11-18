"""Meet fork: axiom 7 straight after the crossing, on what the fluid just did.

Contract: `meet_fork(state, settings)` starts this step's counts and settlings afresh
(`store["step"]`, `store["settled"]`), reads this step's crossings once (`read_sent`)
and calls, in this order:
  `turn_splits`  every older split's own oscillation turns; at a whole turn its pointer jumps
  `carry_on`     every half goes on with what its cell sent, and friction takes direction off its split
  `open_splits`  outflow that is no open split's and met a fork is a new split, or feeds its volume's open one
It needs `quanta_before` and the way table the crossing used. Nothing settles here:
`follow` settles at the end of the step, after the step's collapses and tears.
"""
from __future__ import annotations

from physics.engine.axiom_7_quantum.definitions.split_store import SETTLED_COLUMNS, STEP_COUNTS, no_rows
from physics.engine.axiom_7_quantum.logic.carry_on import carry_on
from physics.engine.axiom_7_quantum.logic.open_splits import open_splits
from physics.engine.axiom_7_quantum.logic.read_sent import read_sent
from physics.engine.axiom_7_quantum.logic.turn_splits import turn_splits


def meet_fork(state, settings) -> None:
    """Turn the open splits, carry their halves on with this step's crossing, and open the new ones."""
    store = state.quantum
    store["step"] = dict.fromkeys(STEP_COUNTS, 0)
    store["settled"] = {name: no_rows() for name in SETTLED_COLUMNS}
    turn_splits(state, settings)
    rows = read_sent(state)
    open_splits(state, rows, carry_on(state, rows))
