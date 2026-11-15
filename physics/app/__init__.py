"""Live app: the ABAB engine running on the owner's PC, drawn in the browser. Display only.

Module map:
  setting_ranges.py    setting_ranges: allowed range of every control
  volume_radius.py     volume_radius: the radius of a ball holding a volume's size
  fitted_positions.py  fitted_positions: the building scatter scaled to the drawing
  settle_positions.py  settle_positions: layout, touching volumes at the sum of their radii
  volume_alignment.py  volume_alignment: which new volume continues which old one
  carry_positions.py   carry_positions: drawing positions across a collapse or a split
  colour_ranks.py      colour_ranks: rank of pull (energy volume) or squeeze (joint), uint8
  mesh_key.py          mesh_key: kinds and contacts as one comparable digest
  simulation.py        Simulation: the state, its settings, its drawing positions, the step count
  page_mesh.py         page_mesh: "mesh" JSON, kind per volume and the contacts
  page_frame.py        page_frame: binary frame, float32 positions and sizes, uint8 colour value
  stats_message.py     stats_message: the engine's stats, step and run state
  settings_message.py  settings_message: every control's value and range
  runner.py            Runner: the stepping loop, at most 10 frames/s, the open pages
  server.py            create_app: FastAPI, GET / and the /ws connection
  static/index.html    the page (three.js): red wire balls (energy volumes), blue (joints)
  __main__.py          python -m physics.app [--port 8765] [--volumes N] [--seed N]

The engine (`physics.abab`) keeps only kind, energy, size and who touches whom; the app
alone keeps positions, for drawing, and sends the topology only when it changes.
"""
from physics.app.simulation import Simulation
from physics.app.server import create_app

__all__ = ["Simulation", "create_app"]
