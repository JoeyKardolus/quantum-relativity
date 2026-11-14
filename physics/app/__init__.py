"""Live app: the simulation running on the owner's PC, drawn in the browser. Display only.

Module map:
  start_network.py     start_network: an even 3D ball and its slightly uneven copy, no lumps placed
  setting_ranges.py    setting_ranges: allowed range of every control
  simulation.py        Simulation: network, settings, seeded generators, display positions
  lump_count.py        lump_count: connected groups of joints well above the mean energy
  energy_shades.py     energy_shades: log10 energy per connection scaled to 0..1
  frame_message.py     frame_message: positions, shades and stats for one frame
  mesh_message.py      mesh_message: the connection list
  settings_message.py  settings_message: every control's value and range
  runner.py            Runner: the stepping loop and the open pages
  server.py            create_app: FastAPI, GET / and the /ws connection
  static/index.html    the page (three.js)
  __main__.py          python -m physics.app [--port 8765] [--joints N] [--seed N]

Display positions live only in Simulation and frame messages; no network holds a coordinate.
"""
from physics.app.simulation import Simulation
from physics.app.server import create_app

__all__ = ["Simulation", "create_app"]
