"""Fit: search the engine's settings on the fit set (expansion shape, structure) only.

Bible `physics/quantum-relativity`, "How results are trusted": a setting is fitted on the
expansion shape (the universe's size and the space between its lumps, measured inside a
wrapping universe and timed by the clock where the energy sits) and on structure (lumps
over regions that hold, their size spread), then tested unfitted on the scenes. This
package does the first half; it never changes a default. One directory per purpose, each
with `definitions/` (what things are), `logic/` (what happens) and `tests/`:

  measures/  what the fit reads off a universe: size, observer clock, lumps, the gap between them
  score/     the score of a run's samples: hold x (expansion shape + structure) / 2, 0 when nothing happens
  search/    one universe run and scored, the search's phases, the holders, the summary
  fitted/    what the fit chose, apart from the defaults, and loading it
  __main__.py  `python -m physics.fit OUT_DIR`: the whole search, one command
"""
from physics.fit.fitted.logic.load_fitted import load_fitted
from physics.fit.measures.logic.gap_between_lumps import gap_between_lumps
from physics.fit.measures.logic.held_share import held_share
from physics.fit.measures.logic.measure_cells import measure_cells
from physics.fit.measures.logic.read_universe import read_universe
from physics.fit.measures.logic.take_snapshot import take_snapshot
from physics.fit.score.logic.score_samples import score_samples
from physics.fit.search.definitions.ranges import SEARCHED
from physics.fit.search.definitions.search_plan import SearchPlan
from physics.fit.search.logic.find_holders import find_holders
from physics.fit.search.logic.run_search import run_search
from physics.fit.search.logic.run_universe import run_universe
from physics.fit.search.logic.summarize_search import summarize_search

__all__ = ["SEARCHED", "SearchPlan", "find_holders", "gap_between_lumps", "held_share", "load_fitted",
           "measure_cells", "read_universe", "run_search", "run_universe", "score_samples", "summarize_search",
           "take_snapshot"]
