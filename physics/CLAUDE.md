# physics/

Rules: `docs/bible.md#physicsquantum-relativity` ("The foundation" first); words: `CONTEXT.md`
(never node, edge, vertex, or merge for joint collapse).

The core is `abab/`: the engine on the owner's irregular ABAB structure, open (not wrapped).
A = energy volume (pulls), B = joint (pushes); every A touches only B's, every B only A's.
`app/` is the live simulator on it. The earlier milestone code was deleted (owner, 2026-11-15); it is in git history.

## Must not get wrong

- **Physics keeps only kind, energy, size, rate and contacts** (`State`, contacts as (A, B) rows).
  Positions from `new_state` and the app's layout are for drawing; no step reads them.
- **ABAB always.** Collapse: a joint past `push_limit` x its rest size vanishes, its energy volumes
  fall into one holding all their energy plus the joint's and keeping all their other joints.
  Split: an energy volume past `split_size` x the start mean splits around a new joint priced at
  3% of its size (`joint_energy.py`); its other joints alternate between the halves by their own
  energy. A parent too dilute to pay stays whole (`stats()["too_big"]`).
- **Energy moves only A -> B -> A** (`passages.py`, `leak.py`), by pull; the more squeezed the
  joint the less it passes, nothing past `seal_point` (`joint_openness.py`), so lumps can seal.
  A joint keeps none of it. Total energy never changes (test: 1e-12 over 300 steps).
- **Joints swing, undamped**: size plus rate (`swing_joints.py`), push a hard wall
  `k R ((R/t)^push_power - 1) / push_power` (`joint_push.py`), rest from the joint's own energy
  (`joint_rest_size.py`). Velocity Verlet in substeps from each swing's thinnest turning point;
  never a step width that changes along a swing. The pull reads only energy-volume sizes: a
  joint's own size there pumps a lone swing.
- **Size follows energy**: size x (old / new energy)^3 for both kinds.
- **App contract**: the State keeps the order of the volumes it keeps and appends new ones
  (`volume_alignment.py` relies on it). JSON "mesh" (kinds, contacts) only on reset, collapse or
  split; frames binary (`page_frame.py`), at most 10/s.
- **Honesty.** Never tune a setting, seed or fit window until a result passes; report every number,
  seed, settings and miss. Today (seed 1, 2000 volumes, unevenness 0.5, defaults, 300 steps):
  2903 energy volumes and joints, 0 collapses, 1903 splits, 79 too big, size 4.02 x start,
  11 lumps, 0 sealed. Engine 13.6 ms per step; live app about 8 steps/s, frames about 80 kB.
- **Layout.** One public function (or type) per file, helpers `_prefixed`, each subpackage
  `__init__.py` holds the module map and `__all__`.
- **Port 8765** is the owner's app; test on `--port 8766`.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics                 # all tests
PYTHONPATH=. .venv/bin/python -m physics.app             # live page, localhost:8765
```

Exempt from the test register (owner, 2026-11-14): plain pytest under `<module>/tests/`, no rows.
