# physics/view/layout/definitions/

What things are when cells are laid out. Records, constants and formulas only.

- `kinds.py`: the engine's three kind numbers, written again so drawing loads without
  the engine. `source/logic/tests/` checks them against the engine's.
- `pieces.py`: `Pieces`, one row per living cell: its block's 8 corners, id, kind,
  packing, lattice place.
- `fit_marks.py`: the fit's fixed numbers. None is a physics setting.
- `lattice_place.py`: slot number to (i, j, k). All even: an energy volume. One odd
  index: a face joint; two: an edge joint; three: a corner joint.
- `drawn_length.py`: a cell's length as drawn (gone: 0; endless: the longest finite).
- `slot_widths.py`: how wide each slot is along x, y and z.

## Must not get wrong

- Nothing here imports `logic/`, OpenGL or the engine.
- Corner number c of a block is on the high side along x when `c & 1`, y when
  `c & 2`, z when `c & 4`. The triangle patterns in `drawing/` rely on it.
- `tests/` checks every formula on numbers worked out by hand.
