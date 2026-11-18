# Axiom 1: energy

The owner's words (`docs/bible.md#physicsquantum-relativity`, "The axioms", 1):

> Energy: an always-moving quantum fluid, one whole signed number of quanta per Planck box at each moment of local time; the total is not fixed: a squeezed joint makes more minus energy out of itself, and a collapse takes the joint's energy off the rest; zero is empty space, which does nothing.

They win over this file and the code.

## Clause by clause

- "always-moving quantum fluid": two boxes trade what their waves differ by (`pair_rotation`). Inside a cell that is `logic/flow_inside.py`; from cell to cell it is the same step, taken by two boxes that face each other across a joint (`cross`, axiom 4).
- "one whole signed number per box": `whole_move.py`, `count_cells.py`, `read_books.py`.
- "local time": every kernel's `tau = step_time * rate`.
- "the total is not fixed": `book_law.py`; a collapse takes off the rest what its joint held, less only what was not there (`short_quanta`).
- "zero is empty space": an empty cell has no clock, makes and carries no pull, and no rule reads its old pull.

## Assumed

- Two equal waves stand still. Nothing drives the fluid.

## Measured (fix round 3, seed 1)

- A universe at rest (8 a side, one phase) with ONE quantum moved, fluid and crossing only: quanta changing box a step 3, 1,392, 39,402, 128,965, 183,144 at steps 1, 10, 50, 400, 800. That is the rounding of whole quanta spreading to all 64 volumes (as the square root of the steps; a closed cell has the same noise). The old side law reached 22,495,048.
- Gentle run, 400 steps: 3.5 to 8% of the plus energy crosses a joint every step.
- Default run: 1,928 collapses leave 4 volumes and about 102 of 333 million plus quanta by step 9. Books exact in every run.

## Not done

- **Minus energy never changes cell**, and plus energy is never inside the joint it crosses: it is in the facing box of the next volume at the end of the same step, whatever the joint's length (300 long: 1,240 times `light_speed`). 18.9% of all energy only goes round inside closed blocks. Axiom 6 lets a joint's energy rise only while squeezed, so nothing may enter it.
- **Default run after the collapse**: most energy sits in one volume whose clock all but stands; `check_step` cannot judge a cell that moves under one quantum a step (`unjudged_share`).
- **The total changes in one burst** on the defaults; on held settings joints make energy step by step from step 1,280 and collapse from 11,590 (8 a side, axiom 6).
- **A cell does not coast by its own tilt**; a settling moves energy on no way (axiom 4).

## Open

- **An emptied joint is a wall for ever** (nothing crosses it, it can gain nothing, its length is endless); **an emptied volume that takes two quanta tears at once** and starts the tear chain.
