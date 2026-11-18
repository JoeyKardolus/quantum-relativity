# Axiom 7: quantum, logic

What happens in a step under this axiom. Each file's docstring is its contract.

- `start_store.py`: the empty store.
- `meet_fork.py`: straight after `cross`. Calls `turn_splits`, `carry_on`, `open_splits`.
- `follow.py`: after the step's events. Calls `carry_events`, `settle`: the step's one settling.
- `read_sent.py`: what each volume sent into each other volume this step.
- `carry_on.py`: halves go on with their cell's fluid; friction lowers `directed`.
- `set_splits.py`: writes the tables merged and in order. `stand_pointers.py`: a pointer whose half is gone jumps again by share.
- `draw_whole.py`: the direct pull, two kernels over every cell.
- `react.py`: a caller's detector.

## Must not get wrong

- **`meet_fork` runs before the events, `follow` after.** `read_sent` reads the way table the crossing used; `relink` changes it.
- **Carry on before opening.** What a cell sent that was a half going on is not new moving energy; `open_splits` opens only `sent - claimed`. The other way round every quantum is claimed again at every crossing (claims reached 269% of all energy).
- **Friction in a half's cell is no reaction.** That switch settled every split one step after it opened. Friction takes momentum off the split whose energy crossed, and nothing else.
- **`carry_on` is exact**: the quanta a cell sent are picked evenly from the row of what it held, so its halves never give more than it sent or keep more than it keeps, and no crossing carries more claimed quanta than it sent. Rounding each half by its own draw let claims creep past what cells held. It needs `quanta_before`; `settle` needs this step's `cell_quanta`.
- **All settlings of a step are drawn together**: `take` first, then `give`, one thread per cell. Heat moves with the quanta, in whole numbers that add up per cell.
- **`draw_whole` writes no phase.** What settles comes to rest in the fluid it joins. Laying one wave over a whole cell wiped the cell's swirl.
- **Draw numbers of `SPLIT_POINTER`** for one split: 0 at the split, 1 and up at a turn, `2 ** 31` and up at a tear, `2 ** 32 - 1` for a pointer that lost its cell.
- Logic imports definitions of any axiom and never another axiom's logic.

## Tests

`tests/conftest.py` stages a small lattice and writes what `cross` would have written (`send`, `next_step`).
