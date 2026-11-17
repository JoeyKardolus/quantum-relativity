# physics/

Rules: `docs/bible.md#physicsquantum-relativity` ("The foundation", "The axioms" and "How the engine holds
the grid" first); words: `CONTEXT.md` (Planck box, energy volume, joint; never node, edge,
vertex, or merge for joint collapse).

`engine/` is the engine (rebuilt 2026-11-17): the cells are the grid, a perfect
A B A B lattice on the GPU (Warp, CUDA); a cell has only its neighbours and a length set by
its quanta, and owns a fixed block of 216 compute boxes (whole signed quanta and a
phase) for its fluid; boxes are not space. `bench/`, `ladder/`, `view/`, `scenes/` and
`fit/` were built on the earlier engine (boxes as space) and do not import until each
is rebuilt on `cells`, `stats` and `State`. `ladder/` is the detail ladder the engine keeps
beside its boxes once attached; `bench/` the side-by-side and size harness. `view/` is the
only viewer, a pyglet window on the GPU. `scenes/` is the scorecard: each scene runs the
engine and returns measured numbers, no verdict. `fit/` searches settings on the fit set
only (expansion shape, structure), by random search or by Warp gradients through a smooth
stand-in of the engine, and never changes a default. `measure/` holds the measures
scenes share (dimension, whirl winding), CPU only.

## Must not get wrong

- **The rules are the bible's axioms**: plus pulls, minus pushes; the pattern never
  breaks; size is energy alone; a squeezed joint makes minus energy out of itself and
  collapses when thinned too far (its energy taken off the rest, its volumes merged);
  only a too-big volume tears; energy crosses A to A through the joint between, bent by
  the pull, the only friction, making heat. The total is not fixed; its books are exact.
- **No open space**: the grid wraps round; the universe's size is the sum of its cells'
  lengths (`stats()["size"]`).
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
