"""Quantum-relativity theory simulation (bible: physics/quantum-relativity).

Module map:
  fluid/  the engine: the foam's energy as a fluid on a fixed GPU grid of boxes (Warp);
          every box holds pulling (A) and pushing (B) energy and how each flows
  view/   the viewer: a pyglet GPU window of the grid's boxes as additive points,
          `python -m physics.view`
  scenes/ the fit's scorecard: scenes that build a start, run the engine and return
          measured numbers (expansion, structure, orbit, bound lump, black hole, Hubble gap)
"""
