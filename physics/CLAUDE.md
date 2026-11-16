# physics/

Rules: `docs/bible.md#physicsquantum-relativity` ("The foundation" and "How the engine holds
the foam" first); words: `CONTEXT.md` (Planck box, energy volume, joint; never node, edge,
vertex, or merge for joint collapse).

`engine/` is the engine: a fixed grid of Planck boxes on the GPU (Warp,
CUDA), each holding one signed whole number of quanta (plus: an energy volume's box, minus:
a joint's, 0: no space) and the fluid's phase; every rule reads only neighbours;
universes run side by side; each box belongs to a cell of a cell graph (a joint knows its
two volumes; `engine/cells.py`, `engine/seams.py`). `ladder/` is the detail ladder the engine keeps
beside its boxes once attached; `bench/` the side-by-side and size harness. `view/` is the
only viewer, a pyglet window on the GPU. `scenes/` is the scorecard: each scene runs the
engine and returns measured numbers, no verdict. `fit/` searches settings on the fit set
only (expansion shape, structure), by random search or by Warp gradients through a smooth
stand-in of the engine, and never changes a default. `measure/` holds the measures
scenes share (dimension, whirl winding), CPU only.

## Must not get wrong

- **The rules are the bible's**: A pulls, fading with distance, at light speed; B pushes,
  steeply when squeezed, and collapses only when packed past `collapse_packing`, into one cell (the only way minus
  turns plus); flow frictionless inside a cell; energy crosses A to A down the pull (plus
  diffusion) through their one joint, the only friction, making heat, less when squeezed, none past the seal; every A
  wrapped on all 26 sides, joints thin mortar in one web; only a too-big A tears; no size
  at rest, no drift setting; a box changes cell only once emptied or split, sign only by
  collapse or a split's tear. The amount ignoring sign is exact (bible rule 4).
- **Open space**: `new_universe` defaults to margin 0 (wraps round when its side is even).
  Open-space scenes start the block inside an empty margin as wide as the filled cube
  (`scenes/open_margin.py`); space grows there as outer joints take empty boxes.
  Cells reaching the closed walls (`stats()["edge_share"]`) end open space; report it.
- **Honesty.** Never tune a setting, seed or threshold until a result passes; report every
  number, seed, settings and miss.
- **Expansion check before any fit** (seed 1, defaults): the numbers are in
  `scenes/CLAUDE.md`.
- **GPU**: RTX 5080, WSL2: no CUDA-OpenGL sharing, copy GPU to host once per frame. Tests that need CUDA skip cleanly without it.
- **A machine keeps no datasets**: long runs and their results go to object storage or tmp,
  never into the repo.
- **Layout.** One public function (or type) per file, helpers `_prefixed`, docstring
  contracts, no single-letter names outside comprehensions and lambdas; each package's
  `__init__.py` holds the module map and `__all__`.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics                 # all tests
PYTHONPATH=. .venv/bin/python -m physics.view            # the window, starts paused, nearest look
```

Exempt from the test register (owner, 2026-11-14): plain pytest under `<package>/tests/`, no rows.
