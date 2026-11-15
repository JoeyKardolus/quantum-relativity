"""Live app: the cell engine running on the owner's PC, drawn in the browser. Display only.

Module map:
  setting_ranges.py    setting_ranges: allowed range of every control
  simulation.py        Simulation: the engine, its settings, the step count
  mesh_key.py          mesh_key: the mesh topology as one comparable digest
  page_mesh.py         page_mesh: "mesh" JSON, faces by corner index, cells and joint per face
  page_frame.py        page_frame: binary frame, float32 corners, uint8 pull and push
  stats_message.py     stats_message: the engine's stats, step and run state
  settings_message.py  settings_message: every control's value and range
  runner.py            Runner: the stepping loop, at most 10 frames/s, the open pages
  server.py            create_app: FastAPI, GET / and the /ws connection
  static/index.html    the page (three.js): solid red cells, blue joint walls in the gaps
  __main__.py          python -m physics.app [--port 8765] [--cells N] [--seed N]

The engine (`physics.cells`) owns the mesh (shared corners, faces, joints); the app sends
its topology only when it changes and its corner positions every frame.
"""
from physics.app.simulation import Simulation
from physics.app.server import create_app

__all__ = ["Simulation", "create_app"]
