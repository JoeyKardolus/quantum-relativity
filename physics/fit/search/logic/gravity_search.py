"""Gravity search: phase 1 of the fit, a random search of the gravity ranges scored by the gravity scene.

Contract: `gravity_search(out_path, points, seed=20261118, cells=24, steps=600, every=20,
scene=gravity)` runs the default settings (id 0) and `points` points of a latin
hypercube over `GRAVITY_RANGES` (ids 1 up), each through `scene(cells, steps=steps,
every=every, settings=Settings(**overrides))`, scores it with `gravity_score` and
appends one JSON line `{"id", "overrides", "parts", "numbers"}` to `out_path` (lists left
out of the numbers). An id already in the file is not run again, so a search that was
cut off goes on. A run that raises is recorded with `"error"` and parts of 0. Returns
the records, best score first. Nothing is written into the repo and no default changes:
what the search chose goes into `fitted/` by hand, with its numbers.
"""
from __future__ import annotations

import json
import pathlib

from physics.engine import Settings
from physics.fit.search.definitions.gravity_ranges import GRAVITY_SEARCHED, gravity_point
from physics.fit.search.definitions.latin_hypercube import latin_hypercube
from physics.scenes.gravity import gravity, gravity_score


def _done(path: pathlib.Path) -> list:
    """The records already in the results file."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _one(identity: int, overrides: dict, cells: int, steps: int, every: int, scene) -> dict:
    """One candidate run and scored."""
    try:
        numbers = scene(cells, steps=steps, every=every, settings=Settings(**overrides))
    except Exception as raised:  # a candidate the engine cannot run is a result, not a crash of the search
        return {"id": identity, "overrides": overrides, "parts": gravity_score({}), "numbers": {},
                "error": f"{type(raised).__name__}: {raised}"}
    plain = {name: value for name, value in numbers.items() if not isinstance(value, list)}
    return {"id": identity, "overrides": overrides, "parts": gravity_score(numbers), "numbers": plain}


def gravity_search(out_path, points: int, seed: int = 20261118, cells: int = 24, steps: int = 600, every: int = 20,
                   scene=gravity) -> list:
    """Run and score the defaults and `points` random candidates; the records, best first."""
    path = pathlib.Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = _done(path)
    seen = {record["id"] for record in records}
    candidates = [{}] + [gravity_point(unit) for unit in latin_hypercube(points, len(GRAVITY_SEARCHED), seed)]
    for identity, overrides in enumerate(candidates):
        if identity in seen:
            continue
        record = _one(identity, overrides, cells, steps, every, scene)
        with path.open("a") as results:
            results.write(json.dumps(record) + "\n")
        records.append(record)
    return sorted(records, key=lambda record: -record["parts"]["score"])
