"""Lump factor: how much energy makes an energy volume a lump.

Contract: `LUMP_FACTOR` is 2.0: a volume is a lump when it holds that many times an even
start volume's quanta or more. Written before any run; no scene changes it.
"""
from __future__ import annotations

LUMP_FACTOR = 2.0
