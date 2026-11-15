"""Quantum-relativity theory simulation (bible: physics/quantum-relativity).

Module map:
  fluid/  the engine: the foam's energy as a fluid on a fixed GPU grid of boxes (Warp);
          every box holds pulling (A) and pushing (B) energy and how each flows
  view/   the viewer: a pyglet GPU window of the grid's boxes as additive points,
          `python -m physics.view`
"""
