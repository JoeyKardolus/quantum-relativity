"""Layout, definitions: what things are when the cells are laid out.

  kinds.py          ENERGY_VOLUME, JOINT, NO_CELL: the engine's kind numbers
  pieces.py         Pieces: every living cell as the block it is drawn in
  fit_marks.py      RELATIVE, MOST_TURNS, SHEAR, FLOOR: the fit's fixed numbers
  lattice_place.py  lattice_place: a slot number as its (i, j, k)
  drawn_length.py   drawn_length: the length a cell is drawn with
  slot_widths.py    slot_widths: how wide every place of the pattern is

Nothing here imports logic, OpenGL or the engine.
"""
