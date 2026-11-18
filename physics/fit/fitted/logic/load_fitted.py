"""Load fitted: the settings the fit chose, as engine Settings, apart from the defaults.

Contract: `load_fitted(path=None)` returns a dict with
  settings   `physics.engine.Settings` with the fitted values in place of the defaults
  fitted_on  what the search that chose them found
With no `path` it reads the shipped `FITTED`; with one it reads a JSON file of the same
shape, `{"settings": {...}, "fitted_on": {...}}`, such as a search wrote outside the repo.
While nothing is fitted the settings are the defaults. The defaults themselves never
change (`Settings()` stays the default run): a scene that wants the fitted universe asks
for it by name,
  fitted = load_fitted(); universe = new_universe(fitted["settings"]); step(universe, fitted["settings"])
Raises FileNotFoundError when the file is missing, KeyError when "settings" is missing,
ValueError for any other top-level name than these two, TypeError for a name Settings
does not have, and the engine's ValueError for a value Settings refuses.
"""
from __future__ import annotations

import json
import pathlib

from physics.engine import Settings
from physics.fit.fitted.definitions.fitted_values import FITTED

_NAMES = {"settings", "fitted_on"}


def load_fitted(path=None) -> dict:
    """The fitted Settings and the search's own description."""
    stored = FITTED if path is None else json.loads(pathlib.Path(path).read_text())
    unknown = set(stored) - _NAMES
    if unknown:
        raise ValueError(f"not a part of the fitted values: {sorted(unknown)}")
    return {"settings": Settings(**stored["settings"]), "fitted_on": dict(stored.get("fitted_on", {}))}
