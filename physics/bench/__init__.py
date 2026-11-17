"""Bench: how fast the engine steps, and the largest universe that still runs a step a second.

Two measures that know nothing of the engine's rules (they time any `advance`
callable); `python -m physics.bench` points them at physics.engine.

Module map:
  throughput.py        throughput: universe-steps and cell-steps a second of any advance callable
  largest_universe.py  largest_universe: bisect sizes for the biggest at >= 1 step a second
  __main__.py          python -m physics.bench throughput|size
"""
from physics.bench.largest_universe import largest_universe
from physics.bench.throughput import throughput

__all__ = ["largest_universe", "throughput"]
