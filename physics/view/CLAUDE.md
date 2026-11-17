# physics/view/

The only viewer: the grid of cells of `physics.engine`, every cell at its own size, in a
pyglet GPU window or as pictures without one. Module map in `__init__.py`.

## Must not get wrong

- **The cells are the grid; boxes are never drawn.** Red = energy volume, blue = joint,
  brightness by packing on one fixed log scale (`cell_colours.py`), never ranked or
  stretched per frame.
- **A cell never changes place** (`cell_slots.py`, the one rule). Every lattice place is a
  slot: a lattice cell is drawn in its own while it lives; a tear's joint and volume are
  stripes inside the torn volume's slot (`state.torn_from`); a cell that is gone leaves its
  slot empty and the grid closes up.
- **Sizes warp the grid, nothing else** (`cell_layout.py`). A joint is as thick as its
  length across its odd axes and as long as the volumes beside it. Slots share corners:
  no gap, no overlap. Rows of unequal cells cannot all be exact in one gapless grid, so
  the corners are a least-squares fit, stiffness one over length (thin joints stay thin).
  Default start: drawn size follows a volume's length to the power 0.46,
  5% off on average, 27% at most; a region of small cells is drawn small, one small cell
  among big ones only partly. Colour carries the exact packing. Never call the drawn
  sizes exact.
- **Events are the weak spot**: with sizes thousands of times apart (tear chains) the fit
  falls back to a direct solve and cells fold over each other (16 places, 825 tears, 513
  collapses: up to 386 of 4,222 cells).
- **Whole grid** (key C): blocks, faces shaded by axis; with both kinds only the outer
  slots are drawn. The opened grid ends in a joint layer at the high index, so those
  faces are all blue; the camera starts at the index-0 corner.
- **GPU under WSL2**: `use_gpu_driver()` must run before pyglet loads, else Mesa draws on
  the CPU. The layout runs on the host from three per-cell arrays.
- **Source interface** (`cells_source.py`): reset(), step(), steps, stats(), view() (a
  `Pieces` record). Stepping and layout run on the stepper thread; an error stops stepping
  and shows in the title. View changes republish while paused.
- **Headless tests**: `__init__.py` imports neither OpenGL nor the engine.

## Measured (2026-11-17, RTX 5080, WSL2, machine under load)

32 places: layout 0.09 to 0.14 s, window 108 to 116 fps, 44 steps/s at one
step a frame. Default universe, 5,000 steps: a perfect A B A B slice in all 21 pictures.

## Run

```
PYTHONPATH=. .venv/bin/python -m physics.view [--cells N] [--seed S] [--unevenness U] [--set name=value ...]
PYTHONPATH=. .venv/bin/python -m physics.view --pictures DIR --steps 5000 --every 250   # no window
PYTHONPATH=. .venv/bin/pytest -q physics/view
```
Starts paused on the middle slice: space starts / pauses, C slice / whole, arrows or [ ]
move the slice, 1 / 2 / 3 kinds, F fits, drag rotates, wheel zooms, R resets, Esc quits.
