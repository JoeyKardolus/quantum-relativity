"""Pictures of the network itself, laid out for drawing only.

Module map:
  layout.py           display_layout: 2D positions for some joints from network distances
  network_picture.py  network_picture: before and after side by side, connections coloured by energy

Layouts are computed for one drawing and thrown away; nothing here writes to a Network.
"""
from physics.experiments.picture.layout import display_layout
from physics.experiments.picture.network_picture import network_picture

__all__ = ["display_layout", "network_picture"]
