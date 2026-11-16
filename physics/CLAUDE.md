# physics/

Rules: `docs/bible.md#physicsquantum-relativity` ("The foundation" and "How the engine holds
the foam" first); words: `CONTEXT.md` (Planck box, energy volume, joint; never node, edge,
vertex, or merge for joint collapse).

Seven packages. `engine/` is the engine: a fixed grid of Planck boxes on the GPU (Warp,
CUDA), each holding one signed whole number of quanta (plus: an energy volume's box, minus:
a joint's, 0: no space) and the quantum fluid's phase; every rule reads only neighbours;
universes run side by side in one batch; energy volumes and joints are read off the signs
(`engine/cells.py`), never stored as labels. `ladder/` is the detail ladder the engine keeps
beside its boxes once attached; `bench/` the side-by-side and size harness. `view/` is the
only viewer, a pyglet window on the GPU. `scenes/` is the scorecard: each scene runs the
engine and returns measured numbers, no verdict. `fit/` searches settings on the fit set
only (expansion shape, structure), by random search or by Warp gradients through a smooth
stand-in of the engine, and never changes a default. `measure/` holds the measures
scenes share (dimension, contacts, whirl winding), CPU only. The cell engine (`potts/`,
cellular Potts with float energy) was replaced by `engine/` (issue #3, 2026-11-16); it and
earlier engines are in git history.

## Must not get wrong

- **The rules are the bible's**: A pulls, fading with distance, at light speed; B pushes,
  steeply when squeezed, and collapses past the limit (its minus quanta turn plus); flow
  frictionless inside one sign, friction only crossing plus-minus faces, nothing past the
  seal; every A wrapped in joints on all 26 sides, joints one web. The amount ignoring sign
  is exact to the quantum (bible rule 4). `engine/CLAUDE.md` says how.
- **Open space**: the start is a block of wrapped cells with uneven energy (ripples at every size,
  `engine/uneven_field.py`) inside an empty margin; space grows as joints claim empty boxes.
  The grid's walls are closed, so a run whose cells reach them (`stats()["edge_share"]`) is
  no longer open space. Report it.
- **Honesty.** Never tune a setting, seed or threshold until a result passes; report every
  number, seed, settings and miss.
- **Expansion check before any fit** (seed 1, defaults): the numbers are in
  `scenes/CLAUDE.md`. The margin is as wide as the filled cube.
- **GPU**: RTX 5080 under WSL2. CUDA-OpenGL sharing does not work on WSL: copy GPU to host
  once per frame. Tests that need CUDA skip cleanly without it.
- **A machine keeps no datasets**: long runs and their results go to object storage or tmp,
  never into the repo.
- **Layout.** One public function (or type) per file, helpers `_prefixed`, docstring
  contracts, no single-letter names outside comprehensions and lambdas; each package's
  `__init__.py` holds the module map and `__all__`.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics                 # all tests
PYTHONPATH=. .venv/bin/python -m physics.view            # the window, 8^3 cells of 6^3 boxes, starts paused, nearest look
```

Exempt from the test register (owner, 2026-11-14): plain pytest under `<package>/tests/`, no rows.
