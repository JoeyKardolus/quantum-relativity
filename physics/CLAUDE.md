# physics/

Rules: `docs/bible.md#physicsquantum-relativity` ("The axioms" first; the seven axioms win
over every older sentence and over the code); words: `CONTEXT.md` (Planck box, energy
volume, joint, swirl; never node, edge, vertex, or merge for joint collapse).

`engine/` is the engine (rebuilt 2026-11-18): one directory per axiom, on the GPU (Warp,
CUDA). The cells are the grid, a perfect A B A B pattern; a cell has only its neighbours
and a length set by its quanta, and owns a fixed block of 216 Planck boxes (whole signed
quanta and a phase) for its fluid; boxes are not space. The rest is built on the
engine's public names:
`view/` is the only viewer, a pyglet window (or pictures without one) of the whole grid
in 3D. `scenes/` is the scorecard: each scene runs the engine and returns measured
numbers, no verdict. `fit/` searches settings on the fit set only (expansion shape,
structure) and never changes a default. `bench/` times the engine and finds the largest
universe at a step a second. `measure/` holds the measures scenes share (dimension, a
cell's whole turns), CPU only.

## Must not get wrong

- **One layout everywhere.** A package holds one directory per purpose (in the engine,
  per axiom). Each holds `CLAUDE.md`, `__init__.py`, `tests/`, `definitions/` (what
  things are: state, settings, laws as formulas) and `logic/` (what happens: code that
  changes state); those two each hold their own `CLAUDE.md`, `__init__.py` and `tests/`.
  Nothing else is loose, except `__main__.py` where `python -m` needs it. Definitions
  never import logic.
- **Only `physics.engine`'s public names** are imported outside the engine
  (`scenes/placed_start` still breaks this: there is no public door for placing energy).
- **The rules are the bible's axioms.** Each axiom's `CLAUDE.md` says what is "Assumed",
  "Not done" and "Open". A failing engine test is a clause the engine does not keep;
  never bend one to pass.
- **No open space**: the grid wraps round; the universe's size is the sum of its cells'
  lengths (`measure_grid(state)["size"]`).
- **Honesty.** Never tune a setting, seed or threshold until a result passes; report every
  number, seed, settings and miss.
- **Nothing is measured at full size yet** on the rebuilt engine: no scorecard, no fit,
  no trusted speed. The default universe collapses in its first steps
  (`engine/CLAUDE.md`); package tests run on the engine's "held" settings.
- **GPU**: RTX 5080, WSL2: no CUDA-OpenGL sharing, copy GPU to host once per frame. Tests that need CUDA skip cleanly without it.
- **A machine keeps no datasets**: long runs and their results go to tmp or object
  storage, never into the repo.
- **Code.** One public function (or type) per file, helpers `_prefixed`, docstring
  contracts, no single-letter names outside comprehensions and lambdas.

## Run

```
PYTHONPATH=. .venv/bin/pytest -q physics                 # all tests
PYTHONPATH=. .venv/bin/python -m physics.view            # the window, starts paused on the whole grid
```

Exempt from the test register (owner, 2026-11-14): plain pytest under `<package>/tests/`, no rows.
