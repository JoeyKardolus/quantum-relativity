"""Fit: search the engine's settings on the fit set (expansion shape, structure) only.

Bible `physics/quantum-relativity`, "How results are trusted": a setting is fitted on the
expansion shape (space between lumps, measured inside a wrapping universe and timed by the
clock where the energy sits) and on structure (lumps over regions, sealing, their size
spread), then tested unfitted on the scenes (orbit, bound lump, black hole, Hubble gap,
timescape). This package does the first half; it never changes a default: what it chose
is kept in fitted/settings.json and read with load_fitted. No dark-energy or base-push
setting is searched: bible rule 8 has none.

Module map:
  fit_measure.py       fit_measure: what the fit reads off a wrapping universe (room between lumps, observer clock, lumps, seals)
  fit_score.py         fit_score: the score of a run's samples, 0 when nothing happens (0.5 expansion shape + 0.5 structure)
  fit_run.py           fit_run: one wrapping universe stepped, measured and scored
  fit_search.py        fit_search: defaults, Latin hypercube, refine rounds, recheck on seeds, one-setting sweep; a process per universe, resumable JSON lines
  summarize_search.py  summarize_search: plain-text summary of a search's records, with what the fit cannot decide
  load_fitted.py       load_fitted: the fitted settings and start from fitted/settings.json, apart from the defaults
  stand_in_score.py    stand_in_score: the fit score on a smooth stand-in (no whole numbers, moving cells or collapse), per-universe settings, differentiable by Warp's tape
  stand_in_search.py   stand_in_search: random search on the stand-in, many settings side by side in one GPU job
  gradient_fit.py      gradient_fit: Adam ascent of the stand-in score by tape gradients, several starts side by side
  compare_fits.py      compare_fits: gradient fit against random search, both checked on the real score
  __main__.py          python -m physics.fit OUT_DIR: the whole search, one command
"""
from physics.fit.fit_measure import fit_measure
from physics.fit.fit_score import fit_score
from physics.fit.fit_run import fit_run
from physics.fit.load_fitted import load_fitted
from physics.fit.fit_search import fit_search
from physics.fit.summarize_search import summarize_search
from physics.fit.stand_in_score import stand_in_score
from physics.fit.stand_in_search import stand_in_search
from physics.fit.gradient_fit import gradient_fit
from physics.fit.compare_fits import compare_fits

__all__ = ["fit_measure", "fit_score", "fit_run", "fit_search", "summarize_search", "load_fitted", "stand_in_score",
           "stand_in_search", "gradient_fit", "compare_fits"]
