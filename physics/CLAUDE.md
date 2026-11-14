# physics/

Quantum-relativity simulation. Rules: `docs/bible.md#physicsquantum-relativity`; words: `CONTEXT.md`
(energy volume, joint, connection, joint collapse, drift (the leak), reference spacetime; never node, edge,
vertex, or merge for joint collapse).

The core is `cells/`: the engine. Energy volumes in a periodic box, their exact cell shapes
from a power (Laguerre) tessellation (pyvoro2, pinned), and a joint on every shared face. `app/`
is the live simulator on it (`python -m physics.app`); it draws `cells.shapes(engine)` as sent
and has no geometry code. `network/`, `rules/`, `measure/` and `experiments/` are the earlier
joint-network milestones.

## Must not get wrong

- **The engine owns the shapes; positions are labels.** `cells.Engine` = `Grid` (energy, volume,
  charge per energy volume, joints with their volume) plus a seed position and power weight per
  cell, used only to tessellate. Rules and measurements read only the Grid, never a position.
  `step(engine, settings) -> Engine` is pure. Joints are exactly the shared faces, volumes
  carried by cell pair; a new face starts at rest, a vanished one returns its volume.
- **`tessellate.py` calls pyvoro2's private native function** and does the input checks itself;
  keep `pyvoro2==0.8.0` pinned or re-check that call on an upgrade.
- **Joints hold space.** Total space = energy volumes + joint volumes (`Grid.total_space()`). Every
  rule keeps it except pull (destroys squeezed-out space) and the leak; the shape steps keep it
  exactly. Energy is not conserved: the energy-volume rule raises it under push (open). Volumes
  join only through a joint below `push_limit` times its rest volume.
- **Energy sets volume**: `cells/rest_volume.py` (`(1/E)^3`), restored each step; in `network/`
  `energy * length = LENGTH_SCALE`. Push and pull change size; energy moves to match.
- **All rules are always on, nothing placed by hand.** Energy starts slightly uneven everywhere;
  lumps must form by themselves. Drift is the leak (`cells/leak.py`): a volume's `leak` share of
  energy goes to its neighbours by pull (`p_b / sum p_k`), never all to the hardest.
- **Honesty.** Never tune a setting, seed or fit window until a result passes. Report every number
  with seed and settings; report a miss. Today (seed 1, 1000 cells, defaults) the grid falls to 2
  volumes by step 110, space to 0.01 of its start; about 15 joints per cell. No lasting lumps.
- **Speed.** Vectorised numpy and scipy.sparse; a step is about 21 ms at 500 cells, 126 ms at 2500,
  nearly all tessellation; `shapes()` is a 6 MB message at 1000 cells.
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
