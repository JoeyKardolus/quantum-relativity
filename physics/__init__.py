"""Quantum-relativity theory simulation (bible: physics/quantum-relativity).

Module map:
  engine/  the engine: the cells are the grid, a perfect A B A B lattice on the GPU (Warp);
           a cell has only its neighbours and a length set by its quanta, and owns a fixed
           block of 216 compute boxes (whole signed quanta and a phase) for its fluid
  view/    the viewer: the grid of cells, every cell at its own size, in a pyglet GPU window
           or as pictures, `python -m physics.view`
  scenes/  the fit's scorecard: scenes that build a start, run the engine and return
           measured numbers (expansion, structure, orbit, bound lump, black hole, Hubble
           gap, timescape, dimension)
  fit/     the fit search: settings scored on expansion shape and structure only, by
           random search, `python -m physics.fit OUT_DIR`
  bench/   the engine's speed and the largest universe at 1 step a second
  measure/ the measures scenes share (dimension, contacts, whirl winding), CPU only
"""
