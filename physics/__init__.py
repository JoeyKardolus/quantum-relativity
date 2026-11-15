"""Quantum-relativity theory simulation (bible: physics/quantum-relativity).

Module map:
  potts/  the engine: a fixed GPU grid of Planck boxes (Warp) carrying the energy as a
          fluid; energy volumes (A) and joints (B) are cells, groups of boxes in a 3D
          checkerboard, sized by how many boxes they hold (cellular Potts)
  view/   the viewer: a pyglet GPU window of the occupied Planck boxes as additive points,
          `python -m physics.view`
  scenes/ the fit's scorecard: scenes that build a start, run the engine and return
          measured numbers (expansion, structure, orbit, bound lump, black hole, Hubble gap)
"""
