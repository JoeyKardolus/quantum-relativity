"""Largest universe: the biggest universe the engine still runs at one step a second.

  definitions/  the pass mark, the memory law, the size ladder, the search records, the limits
  logic/        the bisection, one probe, and the engine build that refuses what will not fit
"""
from physics.bench.largest_universe.definitions.search import Probe, Search
from physics.bench.largest_universe.logic.fitting_run import fitting_run
from physics.bench.largest_universe.logic.largest_universe import largest_universe
from physics.bench.largest_universe.logic.probe_size import probe_size

__all__ = ["Probe", "Search", "fitting_run", "largest_universe", "probe_size"]
