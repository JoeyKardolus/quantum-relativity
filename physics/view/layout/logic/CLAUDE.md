# physics/view/layout/logic/

What happens when cells are laid out, in this order:

1. `torn_lines.py`: from the engine's `torn_from`, `made_kind` and `swirl_axis`, which
   cells share a slot after tears and in what order.
2. `typical_lengths.py`: the median volume and joint, which make the perfect grid.
3. `cell_slots.py`: for a box of places, every living cell's slot and the widths.
4. `cell_layout.py`: the perfect grid's corners, warped round slots of another size.
5. `cell_pieces.py`: one layer, or the wanted cells, as blocks (`Pieces`).

## Must not get wrong

- **Tears come in pairs.** The engine hands out a joint id and then a volume id for
  each tear. `torn_lines` raises when an id past the lattice is not half of such a
  pair. Never guess a place for a cell.
- **The warp is a sum of fixed weights**, mirrored at the lattice's ends, so the same
  sizes give the same corners and the frame never moves along its own axis.
- **A cut box must equal the whole.** `cell_slots` reads `reach` more places than it
  gives cells for, and `cell_layout` trims them after the sum; `tests/` checks a slab
  against the same layers of the whole lattice, bit for bit.
- **Nothing here walks the whole lattice for one layer**: index the box, never
  `np.flatnonzero` over every cell.
- `tests/` uses staged arrays: one tear, later tears, gone cells, an endless cell, a
  lump whose reach is known, sizes thousands of times apart.
