"""Live app: the cell engine running on the owner's PC, drawn in the browser. Display only.

Module map:
  setting_ranges.py    setting_ranges: allowed range of every control
  simulation.py        Simulation: the engine, its settings, the step count
  shapes_message.py    shapes_message: the engine's own cell and joint shapes plus its stats
  settings_message.py  settings_message: every control's value and range
  runner.py            Runner: the stepping loop and the open pages
  server.py            create_app: FastAPI, GET / and the /ws connection
  static/index.html    the page (three.js): draws exactly the shapes it is sent
  __main__.py          python -m physics.app [--port 8765] [--cells N] [--seed N]

The engine (`physics.cells`) owns every shape; the app computes no geometry of its own.
"""
from physics.app.simulation import Simulation
from physics.app.server import create_app

__all__ = ["Simulation", "create_app"]
