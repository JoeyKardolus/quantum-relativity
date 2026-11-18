# Axiom 3, definitions

`store.py`: the plain arrays every cell lives in, `slot_of`, and `slot_at` (the same on the card). `size_law.py`: a cell's length from its quanta. `grid.py`: the cell beside a cell and a joint's axis, from place, round the grid. `kinds.py`: boxes across a cell, quanta a box (a chosen scene number, not moved).

- A box belongs to its cell and has no place in space.
- Warp arrays cannot grow: they are made once at their full size. A full own set or full tile rows (`tile_cap`) stop with a plain message.
- A tile is 512 cells that follow each other. Its row, once opened, stays open.
- `put` and `peek` touch one number from the host. They are for single cells, not for loops over many.
