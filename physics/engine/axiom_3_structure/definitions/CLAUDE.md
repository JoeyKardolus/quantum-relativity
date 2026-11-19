# Axiom 3, definitions

`store.py`: the arrays, `slot_of`, `slot_at`. `size_law.py`. `relations.py`: `joint_ends`, `joint_part`, `box_along` (the box a joint's row faces), `join`, `joints_of`. `kinds.py`: boxes across a cell, quanta a box.

- A box belongs to its cell and has no place in space.
- Warp arrays cannot grow: they are made once at their full size. A full own set, full tile rows or no free cell stop with a plain message.
- Cells `0 .. cells - 1` are in use; a tear takes the next.
- A part thinner than a box's width may hold no box's middle: nothing crosses there.
- `put` and `peek` are for single cells, not loops.
