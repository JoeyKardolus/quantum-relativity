# physics/view/layout/logic/

What happens when cells are laid out, in this order:

1. `torn_lines.py`: from the engine's `torn_from`, `made_kind` and `swirl_axis`, which
   cells share a slot after tears and in what order.
2. `cell_slots.py`: every living cell's slot, and every slot's widths.
3. `cell_layout.py`: the grid's corner points, a best fit to those widths.
4. `cell_pieces.py`: every living cell as its block (`Pieces`).

## Must not get wrong

- **Tears come in pairs.** The engine hands out a joint id and then a volume id for
  each tear. `torn_lines` raises when an id past the lattice is not half of such a
  pair. Never guess a place for a cell.
- **The fit weighs a miss by one over the length wanted**, so short cells keep their
  size and long ones take up the mismatch. The same widths always give the same
  corners.
- **The fit always returns.** When the quick solve has not settled it solves directly,
  which is slow (see "Not done" one level up).
- `tests/` uses staged arrays: one tear, later tears, gone cells, an endless cell, and
  widths whose corners are known exactly.
