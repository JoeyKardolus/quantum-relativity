"""Fit: search the engine's settings on the fit set (expansion shape, structure) only.

Bible `physics/quantum-relativity`, "How results are trusted": a setting is fitted on the
expansion shape (the universe's size and the space between its lumps, measured inside a
wrapping universe and timed by the clock where the energy sits) and on structure (lumps
over regions that hold and seal, their size spread), then tested unfitted on the scenes (orbit, bound lump, black hole, Hubble gap,
timescape). This package does the first half; it never changes a default: what it chose
is kept in fitted/settings.json and read with load_fitted. No dark-energy or base-push
setting is searched: bible rule 8 has none.

Module map:
  fit_measure.py       fit_measure: what the fit reads off a universe's cells (size, observer clock, what is alive and spent, lumps, seals)
  fit_snapshot.py      fit_snapshot: the raw cells of one moment, to follow the end's lumps back
  fit_gap.py           fit_gap: the distance between the lumps through the cells between them, and the lumps that held
  fit_score.py         fit_score: the score of a run's samples, 0 when nothing happens: hold x (expansion shape + structure) / 2
  fit_run.py           fit_run: one wrapping universe stepped, measured and scored
  fit_search.py        fit_search: defaults, Latin hypercube, refine rounds, the hold phase, one-setting sweep; a process per universe, resumable JSON lines
  fit_holders.py       fit_holders: which searched points still stand when run four times longer
  summarize_search.py  summarize_search: plain-text summary of a search's records, with what the fit cannot decide
  load_fitted.py       load_fitted: the fitted settings and start from fitted/settings.json, apart from the defaults
  __main__.py          python -m physics.fit OUT_DIR: the whole search, one command
"""
from physics.fit.fit_gap import fit_gap
from physics.fit.fit_holders import fit_holders
from physics.fit.fit_measure import fit_measure
from physics.fit.fit_snapshot import fit_snapshot
from physics.fit.fit_score import fit_score
from physics.fit.fit_run import fit_run
from physics.fit.load_fitted import load_fitted
from physics.fit.fit_search import fit_search
from physics.fit.summarize_search import summarize_search

__all__ = ["fit_gap", "fit_holders", "fit_measure", "fit_run", "fit_score", "fit_search", "fit_snapshot", "load_fitted",
           "summarize_search"]
