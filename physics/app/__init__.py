"""Live app: the cell grid running on the owner's PC, drawn in the browser. Display only.

Module map:
  setting_ranges.py    setting_ranges: allowed range of every control
  cell_layout.py       cell_layout: display positions where neighbouring volumes touch
  simulation.py        Simulation: grid, settings, display positions
  energy_shades.py     energy_shades: log10 energy per volume, scaled to 0..1
  squeeze_shades.py    squeeze_shades: each joint's nearness to joint collapse, 0..1
  joint_volume_shares.py  joint_volume_shares: each joint's volume over its neighbours' mean
  frame_message.py     frame_message: positions, radii, shades, charge and stats for one frame
  mesh_message.py      mesh_message: the joint list
  power_cells.py       power_cells: 2D power diagram (foam cells) of weighted points in a box
  joint_index.py       joint_index: the joint between each pair of energy volumes, or -1
  power_faces.py       power_faces: shared faces of the 3D power diagram of weighted points
  walls_message.py     walls_message: the 3D glass foam's joint faces as joint layers
  slice_message.py     slice_message: a plane cut through the layout as foam cells and walls
  settings_message.py  settings_message: every control's value and range
  runner.py            Runner: the stepping loop and the open pages
  server.py            create_app: FastAPI, GET / and the /ws connection
  static/index.html    the page (three.js)
  __main__.py          python -m physics.app [--port 8765] [--cells N] [--seed N]

Display positions live only in Simulation and frame messages; the Grid holds no coordinate.
"""
from physics.app.simulation import Simulation
from physics.app.server import create_app

__all__ = ["Simulation", "create_app"]
