"""Fitted values: the settings the fit chose, kept apart from the engine's defaults.

Contract: `FITTED` is a dict.
  settings   {name: value}: values of `physics.engine.Settings` fields; a name left out
             keeps its default
  fitted_on  what the search found, in words and numbers: `phase` (the phase of the fit
             these belong to), `names` (the settings that phase searched), `status`,
             `date`, `search` and the measured `parts`
Phase 1: gravity (2026-11-18). Phase 1 does NOT hold: no setting inside the ranges
(`search/definitions/gravity_ranges.py`) makes lumps fall, coast and go round as bodies.
What is kept here is the best candidate, chosen by hand inside the ranges after the
random search's points scored 0.006 at most; it scores 0.10 of 1 (`fitted_on["parts"]`). It is what
the viewer's gravity scenes run with, so what does and does not work can be watched. A
value here is a candidate until its phase holds; the defaults never change.
"""
from __future__ import annotations

FITTED = {
    "settings": {"pull_strength": 1.5e-6, "friction": 0.003, "light_speed": 20.0, "clock_packing": 6.48e15,
                 "collapse_thin": 300.0, "tear_length": 2e4},
    "fitted_on": {
        "phase": "phase 1: gravity",
        "date": "2026-11-18",
        "names": ["pull_strength", "push_strength", "friction", "light_speed", "clock_packing", "collapse_thin",
                  "tear_length"],
        "status": "candidate: phase 1 does not hold (score 0.10 of 1). The universe stands and two lumps close a "
                  "fifth of their distance keeping their energy, but they do not speed up, a kicked lump stops "
                  "after 0.3 volumes and a small lump falls into the big one after 0.09 of a turn",
        "search": "physics/fit/search/logic/gravity_search.py over gravity_ranges.py, scored by "
                  "physics/scenes/gravity/definitions/pass_mark.py (score and ranges written before any run); 24 "
                  "cells a side, 600 steps, seed 1, the layers below axiom 7. The defaults and the 6 random points "
                  "that were run (a candidate takes 3 to 6 minutes; the search was stopped there) scored 0.006 at most: a weak "
                  "pull lets a lump spread away (it keeps 4 to 43% of its energy), a strong one packs it into "
                  "cells it does not leave, or collapses its joints",
        "parts": {"stands": 1.0, "falls": 0.31, "coasts": 0.0, "fades": 0.0, "orbits": 0.09, "score": 0.102,
                  "stand_steps": 600, "fall_closed": 0.20, "fall_kept": 1.31, "fall_sped_up": 0, "lone_drift": 0.05,
                  "coast_travelled": 0.29, "coast_kept": 1.62, "pull_power": -4.29, "law_power": -2.88,
                  "orbit_turns": 0.09, "orbit_kept": 8.34, "joint_over_volume_length": 0.90},
    },
}
