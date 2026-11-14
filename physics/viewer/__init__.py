"""Viewer layer: record gravity scenes and play them in a browser. Display only.

Module map:
  scene.py            Scene: starting network, rules, followed joints, what to watch for
  two_lumps.py        two_lumps: two lumps of extra energy, attraction and joint push on
  collapse.py         collapse: one heavy lump whose squeeze passes the push limit
  reference_ball.py   reference_ball: a ball of a 3D reference network (no wrap-around)
  region.py           carve_ball: the joints nearest a centre, as a network of their own
  rule_settings.py    rule_settings: RuleSettings from the strengths it knows
  record.py           record: step a scene, one Frame per step
  frame.py            Frame: display positions, joint energies, measurements for one step
  recording.py        Recording: the frames and why the run stopped, if it did
  initial_layout.py   initial_layout: landmark MDS of network distances into 3D
  relax_layout.py     relax_layout: sparse stress majorization toward current lengths
  carry_positions.py  carry_positions: positions across joint collapse, energy-weighted
  joint_energy.py     joint_energy: half the energy of each joint's connections
  what_happened.py    what_happened: plain sentences from a recording's numbers
  build_page.py       build_page: the self-contained three.js page
  __main__.py         python -m physics.viewer [--scene NAME|all] [--out DIR]

Display positions exist only in frames; no network ever holds a coordinate.
"""
from physics.viewer.build_page import build_page
from physics.viewer.collapse import collapse
from physics.viewer.record import record
from physics.viewer.two_lumps import two_lumps
from physics.viewer.what_happened import what_happened

__all__ = ["build_page", "collapse", "record", "two_lumps", "what_happened"]
