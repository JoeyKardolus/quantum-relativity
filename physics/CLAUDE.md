# physics/

Rules: `docs/bible.md#physicsquantum-relativity` ("The foundation" and "How the engine holds
the foam" first); words: `CONTEXT.md` (never node, edge, vertex, or merge for joint collapse).

Three packages. `fluid/` is the engine: energy as a fluid on a fixed cube of boxes on the GPU
(Warp, CUDA), every box holding pulling (A) and pushing (B) energy and a momentum per kind.
Energy volumes and joints are regions of the fluid; the ABAB pattern lives below the box size.
Nothing is a ball or a link. `view/` is the only viewer, a pyglet window on the GPU. `scenes/`
is the scorecard: each scene runs the engine and returns measured numbers, no verdict. The
earlier ABAB engine (`abab/`), the browser app (`app/`) and the milestone code were removed
(owner, 2026-11-15); they are in git history.

## Must not get wrong

- **The rules are the bible's**: A pulls (spreads and fades, gravity-like), B pushes, steeply
  when compressed, and gives way past the collapse limit; the flow is frictionless with
  momentum; friction only where A passes through B, and past the seal point A cannot pass.
  Total energy exact (whole quanta, drift 0). Each package's overlay says how.
- **Open space**: the start fills the grid with uneven energy (ripples at every size,
  `fluid/uneven_field.py`) except an empty margin of side / 16; the walls are closed, so a
  run whose energy reaches them (`stats()["edge_share"]`) is no longer open space. Report it.
- **Honesty.** Never tune a setting, seed or threshold until a result passes; report every
  number, seed, settings and miss. Today the whole filled cube still falls into one core
  within about 10 steps (free fall 7.7 time units at any size); see `fluid/CLAUDE.md`.
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
PYTHONPATH=. .venv/bin/python -m physics.view            # the window, side 128, starts paused
```

Exempt from the test register (owner, 2026-11-14): plain pytest under `<package>/tests/`, no rows.
