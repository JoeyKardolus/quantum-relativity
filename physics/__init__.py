"""Quantum-relativity theory simulation (bible: physics/quantum-relativity).

Module map:
  engine/  the engine: a fixed GPU grid of Planck boxes (Warp), each one signed whole
           number of quanta (plus: an energy volume's box, minus: a joint's, zero: no
           space) and a fluid phase; every rule neighbour-only; universes side by side in
           one job; the detail ladder kept beside the boxes once attached
  ladder/  the detail ladder: 8^3 blocks at FULL, COARSE, REUSED or EMPTY, totals exact
  bench/   the harness: side-by-side throughput and the largest universe at 1 step a second
  view/    the viewer: a pyglet GPU window of the occupied Planck boxes, `python -m physics.view`
  scenes/  the fit's scorecard: scenes that build a start, run the engine and return
           measured numbers (expansion, structure, orbit, bound lump, black hole, Hubble gap)
  fit/     the fit search: settings scored on expansion shape and structure only, by
           random search or by Warp gradients through a smooth stand-in,
           `python -m physics.fit OUT_DIR [--compare]`
  measure/ the measures scenes share (dimension, contacts, whirl winding), CPU only
"""
