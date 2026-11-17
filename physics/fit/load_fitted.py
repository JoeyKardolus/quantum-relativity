"""load_fitted: the settings the fit chose, read from their own file, apart from the defaults.

Contract: load_fitted(path=None) -> dict with
  settings   physics.engine Settings with the fitted values in place of the defaults
  start      the fitted start unevennesses, keyword arguments of new_universe
             (unevenness, joint_unevenness)
  fitted_on  what the file says about the search that chose them (sizes, seeds, scores)
`path` defaults to physics/fit/fitted/settings.json, written after a search from its best
rechecked point. The defaults themselves never change (Settings() stays the default run):
a scene that wants the fitted universe asks for it by name,
  fitted = load_fitted(); new_universe(..., **fitted["start"]); step(universe, fitted["settings"])
A fitted value is a candidate until the unfitted scenes pass with it (fit/CLAUDE.md).
Raises FileNotFoundError when the file is missing, TypeError for a name Settings does not
have and ValueError for a start name new_universe does not take.
"""
from __future__ import annotations

import json
import pathlib

from physics.engine import Settings

_START_KEYS = ("unevenness", "joint_unevenness")
_FITTED = pathlib.Path(__file__).parent / "fitted" / "settings.json"


def load_fitted(path=None) -> dict:
    """The fitted Settings, the fitted start unevennesses and the search's own description."""
    stored = json.loads(pathlib.Path(_FITTED if path is None else path).read_text())
    unknown = set(stored["start"]) - set(_START_KEYS)
    if unknown:
        raise ValueError(f"not a start setting of new_universe: {sorted(unknown)}")
    return {"settings": Settings(**stored["settings"]), "start": dict(stored["start"]),
            "fitted_on": stored.get("fitted_on", {})}
