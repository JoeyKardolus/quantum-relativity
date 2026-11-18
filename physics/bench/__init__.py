"""Bench: how fast the engine steps, and the largest universe that still runs a step a second.

One directory per purpose, each with `definitions/` (what things are), `logic/` (what
happens) and `tests/`:
  throughput/        steps and cell-steps a second of a universe
  largest_universe/  the biggest size that fits the GPU and runs at one step a second
`__main__.py` is the command line: python -m physics.bench throughput|size.
"""
from physics.bench.largest_universe import Probe, Search, fitting_run, largest_universe
from physics.bench.throughput import Rates, Run, engine_run, throughput

__all__ = ["Probe", "Rates", "Run", "Search", "engine_run", "fitting_run", "largest_universe", "throughput"]
