"""Layout, definitions: what things are when the cells are laid out.

  kinds.py           ENERGY_VOLUME, JOINT, NO_CELL: the engine's kind numbers
  pieces.py          Pieces: laid-out cells as the blocks they are drawn in
  warp_marks.py      LIMIT, REACH, ACROSS, SAMPLE, EASE_TURNS, LAID_OUT_MOST
  lattice_place.py   lattice_place: a slot number as its (i, j, k)
  drawn_length.py    drawn_length: the length a cell is drawn with
  slot_widths.py     slot_widths: how wide every place of the pattern is
  regular_planes.py  regular_planes: the corner planes of the perfect grid
  pull_weights.py    pull_weights: how far a cell's size pulls the planes of its row

Nothing here imports logic, OpenGL or the engine.
"""
