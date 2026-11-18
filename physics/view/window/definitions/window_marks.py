"""Window marks: the fixed numbers of the window and its stepping thread.

Contract: constants only.
  WIDTH, HEIGHT    the window's first size
  TURN_PER_PIXEL   radians the view turns per pixel dragged
  PITCH_LIMIT      the view never tips past this, in radians, so up stays up
  ZOOM_PER_NOTCH   the camera's distance is multiplied by this per wheel notch
  TITLE_EVERY      seconds between title refreshes
  STATS_EVERY      seconds between stats reads while stepping
  IDLE_WAIT        seconds the stepping thread sleeps while paused
"""
from __future__ import annotations

WIDTH, HEIGHT = 1280, 800
TURN_PER_PIXEL = 0.01
PITCH_LIMIT = 1.55
ZOOM_PER_NOTCH = 0.9
TITLE_EVERY = 0.25
STATS_EVERY = 0.5
IDLE_WAIT = 0.05
