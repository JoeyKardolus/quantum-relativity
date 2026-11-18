"""Key actions: what every key of the window does.

Contract: `KEY_ACTIONS` maps a key's name (the name of its constant in
`pyglet.window.key`) to `(action, argument)`:
  "pause"   space: start or pause the stepping
  "whole"   C: the whole grid, or the slice through the middle
  "show"    1, 2, 3 (and the number pad's): energy volumes, joints, both
  "move"    arrows and [ ]: the slice one lattice layer up (+1) or down (-1)
  "fit"     F: fit the view to the grid
  "reset"   R: a fresh start
  "close"   Esc: close the window
The window looks the names up; this file loads without pyglet.
"""
from __future__ import annotations

from physics.view.drawing.definitions.shown_kinds import SHOW_BOTH, SHOW_ENERGY_VOLUMES, SHOW_JOINTS

KEY_ACTIONS = {
    "SPACE": ("pause", None),
    "C": ("whole", None),
    "_1": ("show", SHOW_ENERGY_VOLUMES), "NUM_1": ("show", SHOW_ENERGY_VOLUMES),
    "_2": ("show", SHOW_JOINTS), "NUM_2": ("show", SHOW_JOINTS),
    "_3": ("show", SHOW_BOTH), "NUM_3": ("show", SHOW_BOTH),
    "UP": ("move", 1), "RIGHT": ("move", 1), "BRACKETRIGHT": ("move", 1),
    "DOWN": ("move", -1), "LEFT": ("move", -1), "BRACKETLEFT": ("move", -1),
    "F": ("fit", None),
    "R": ("reset", None),
    "ESCAPE": ("close", None),
}
