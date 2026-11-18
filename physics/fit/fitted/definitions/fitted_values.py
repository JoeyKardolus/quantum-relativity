"""Fitted values: the settings the fit chose, kept apart from the engine's defaults.

Contract: `FITTED` is a dict.
  settings   {name: value}: the fitted values of `physics.engine.Settings` fields; a
             name left out keeps its default. Only a point that holds (`find_holders`)
             is ever written here
  fitted_on  what the search that chose them found, in words and numbers
Nothing is fitted (2026-11-18): the engine was rebuilt on the seven axioms and no search
has run on it, so `settings` is empty and the fitted universe is the default one. The
2026-11-17 search ran on the engine before the rebuild; four of its fifteen settings
(crossing_rate, seal_thin, fluid_pressure, joint_length) no longer exist, so neither its
candidates nor its counts say anything about this engine and they are not kept here
(they are in git history, `physics/fit/fitted/settings.json`).
A fitted value is a candidate until the unfitted scenes pass with it.
"""
from __future__ import annotations

FITTED = {
    "settings": {},
    "fitted_on": {
        "date": "2026-11-18",
        "status": "none: no search has run on the engine rebuilt on the seven axioms, so nothing is fitted and "
                  "the fitted universe is the default one",
        "search": "none yet; the ranges are physics/fit/search/definitions/ranges.py and the score is "
                  "physics/fit/score/logic/score_samples.py, both written down before any search",
    },
}
