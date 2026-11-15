# physics/

Rules: `docs/bible.md#physicsquantum-relativity` ("The foundation" and "How the engine holds
the foam" first); words: `CONTEXT.md` (Planck box, energy volume, joint; never node, edge,
vertex, or merge for joint collapse).

Four packages. `potts/` is the engine: a fixed grid of Planck boxes on the GPU (Warp, CUDA)
carrying the energy as a quantum fluid, one complex number per box; energy volumes (A) and
joints (B) are cells, groups of boxes, every A wrapped in joints, and a cell's size is how many
boxes it holds (cellular Potts). `view/` is the only viewer, a pyglet window on the GPU.
`scenes/` is the scorecard: each scene runs the engine and returns measured numbers, no
verdict. `measure/` holds the measures scenes share (dimension, whirl winding), CPU only. The fluid engine without cells (`fluid/`),
the earlier ABAB engine (`abab/`), the browser app and the milestone code were removed (owner,
2026-11-15); they are in git history. Its kept kernels (multigrid pull, uneven field) live in
`potts/`.

## Must not get wrong

- **The rules are the bible's**: A pulls through joints, fading with distance; B pushes,
  steeply when squeezed, and collapses past the limit (the joint vanishes, its energy volumes
  fall together); energy sets size; flow frictionless inside a cell, friction only crossing
  into a joint (at its seams), nothing past the seal; a joint is a full cell (rule 8); every A
  wrapped in joints on all 26 sides, joints one web. Total energy (sum |psi|^2) holds to
  float64 rounding. `potts/CLAUDE.md` says how.
- **Open space**: the start is a block of wrapped cells with uneven energy (ripples at every size,
  `potts/uneven_field.py`) inside an empty margin; space grows as cells claim empty boxes.
  The grid's walls are closed, so a run whose cells reach them (`stats()["edge_share"]`) is
  no longer open space. Report it.
- **Honesty.** Never tune a setting, seed or threshold until a result passes; report every
  number, seed, settings and miss.
- **Expansion check before any fit** (seed 1, defaults): the numbers are in
  `potts/CLAUDE.md`. The margin is as wide as the filled cube, so walls come late.
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
