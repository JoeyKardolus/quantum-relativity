"""Fit: search the engine's settings on the fit set (expansion shape, structure) only.

Bible `physics/quantum-relativity`, "How results are trusted": a setting is fitted on the
expansion shape and on structure, then tested unfitted on the scenes (orbit, bound lump, black
hole, Hubble gap). This package does the first half; it never changes a default.

Module map:
  fit_run.py           fit_run: one universe stepped and scored (0.5 expansion dip + 0.5 structure)
  fit_search.py        fit_search: defaults, Latin hypercube, refine, recheck on seeds; resumable JSON lines
  summarize_search.py  summarize_search: plain-text summary of a search's records
  __main__.py          python -m physics.fit OUT_DIR: the whole search, one command
"""
from physics.fit.fit_run import fit_run
from physics.fit.fit_search import fit_search
from physics.fit.summarize_search import summarize_search

__all__ = ["fit_run", "fit_search", "summarize_search"]
