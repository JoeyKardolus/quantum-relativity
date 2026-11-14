# physics/

Quantum-relativity theory simulation. Rules: `docs/bible.md#physicsquantum-relativity`; words: `CONTEXT.md`
(energy volume, joint, connection, joint collapse, drift, reference spacetime; never node, edge,
vertex, or merge for joint collapse).

The core is `cells/`: a woven grid of energy volumes with a joint on every side between two
neighbours. `app/` is the live simulator on it (`python -m physics.app`). `network/`, `rules/`,
`measure/` and `experiments/` are the earlier joint-network milestones, where a joint was still a point.

## Must not get wrong

- **The physics stores no shapes and no coordinates.** `cells.Grid` holds the energy volume count,
  the joints (pairs of neighbouring energy volumes), energy and volume per energy volume, squeeze
  per joint and the `merged_into` map, nothing else. `Network` likewise holds no positions.
  The tetrahedra in `cells/build_grid.py` are scaffolding only: they decide the starting
  neighbours and give display positions, which are returned beside the Grid, never stored on it.
- **Positions are drawing only.** `app/cell_layout.py` relaxes them per frame so neighbours sit at
  the sum of their radii and carries them through joint collapse; no rule reads them. Joints across
  the periodic box's wrap-around are left out of the layout and the drawing.
- **Every measurement reads only the grid or network**: neighbour lists, joint counts, energy per
  volume, length-weighted paths, random walks. If a measure needs a coordinate, it is measuring
  the wrong thing.
- **Energy sets volume**: `cells/rest_volume.py` (`(1/E)^3`), restored each step; in `network/`
  `energy * length = LENGTH_SCALE`. Rules that push or pull change size; the rule moves energy to match.
- **All rules are always on, nothing placed by hand.** Energy starts slightly uneven everywhere;
  lumps must form by themselves. Drift comes from `balancing_drift`, not a hand-picked number.
- **Honesty.** Never tune a setting, seed or fit window until a result passes. Report every number
  with seed and settings; a miss is reported as a miss. Today the grid shrinks as a whole and
  every energy volume falls into one rather than forming lasting lumps; say so until it changes.
- **Speed.** Vectorised numpy and scipy.sparse; a step at 2500 energy volumes runs in about 16 ms
  and an experiment in under ~3 minutes. Only `rules/rewire.py` loops in Python.
- **Layout.** One public function (or type) per file, helpers `_prefixed`, each subpackage
  `__init__.py` holds the module map and `__all__`.
- **Port 8765** is the owner's own running app; test on another port (`--port 8766`).

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics                 # all tests, seconds
PYTHONPATH=. .venv/bin/python -m physics.app             # live cell grid page, localhost:8765
PYTHONPATH=. .venv/bin/python -m physics                 # network milestones, results page, minutes
PYTHONPATH=. .venv/bin/python -m physics.viewer          # network gravity scenes as a 3D page
```

`physics/` is exempt from the test register (owner, 2026-11-14; `UNREGISTERED` in
`tools/test_register/_modules.py`): write plain pytest files under `<module>/tests/`, no rows.
