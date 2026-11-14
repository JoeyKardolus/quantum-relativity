"""Network layer: joints and connections with energy and length; no coordinates.

Module map:
  energy_length.py   the energy-length rule (length = LENGTH_SCALE / energy)
  network.py         Network, the frozen data type every layer passes around
  spatial.py         spatial_reference: k-nearest network on a flat d-torus
  causal.py          causal_reference: a causal set sprinkled in a flat diamond
  random_network.py  random_network: no geometry at all
"""
from physics.network.energy_length import LENGTH_SCALE, energy_length
from physics.network.network import Network
from physics.network.spatial import spatial_reference
from physics.network.causal import causal_reference
from physics.network.random_network import random_network

__all__ = ["LENGTH_SCALE", "energy_length", "Network", "spatial_reference",
           "causal_reference", "random_network"]
