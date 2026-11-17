# physics/view/

The only viewer: the grid of cells of `physics.engine`, every cell at its own size.
Module map in `__init__.py`.

## Must not get wrong

- **The cells are the grid; boxes are never drawn.** Red = energy volume, blue = joint;
  the hue never changes. Opacity carries the packing on one fixed scale
  (`cell_colours.py`), never ranked or stretched per frame. Slice: packing / 4 (the
  start's mean 0.25, four times it solid). Whole grid: that to the power 3.8 per face
  (mean 0.005, three times 0.33): thin cells are haze, packed ones masses.
- **Depth is by order, not a depth buffer** (`cell_mesh.py`, `far_to_near.py`): cells
  are sorted far to near on every change of cells or view, then blended. A depth test
  would let a clear cell hide what is behind it.
- **A cell never changes place** (`cell_slots.py`). A lattice cell is drawn in its own
  slot; a tear's joint and volume are stripes inside the torn volume's slot
  (`state.torn_from`); a gone cell leaves its slot empty and the grid closes up.
- **Sizes warp the grid, nothing else** (`cell_layout.py`). Slots share corners: no gap,
  no overlap. The corners are a least-squares fit (thin joints stay thin). Default
  start: drawn size follows a volume's length to the power 0.46, 5% off on average, 27%
  at most. Never call the drawn sizes exact.
- **Events are the weak spot**: with sizes thousands of times apart (tear chains) cells
  fold over each other (16 places, 825 tears: up to 386 of 4,222 cells).
- **Whole grid** (the start; C goes to the slice): every cell a block, faces shaded by
  axis, camera at the index-0 corner. Squeezed joints (pull x10) turn it solid blue:
  key 1 shows the energy volumes alone.
- **Pictures**: the whole grid is drawn by the GPU in a hidden window
  (`whole_picture.py`, needs a display); `--slice` needs neither.
- **GPU under WSL2**: `use_gpu_driver()` must run before pyglet loads, else Mesa draws on
  the CPU.
- **Source interface** (`cells_source.py`): reset(), step(), steps, stats(), view().
  Stepping and layout run on the stepper thread; an error stops it and shows in the title.
- **Headless tests**: `__init__.py` imports neither OpenGL nor the engine.

## Measured (2026-11-17, RTX 5080, WSL2)

Whole grid, 1280 x 800, off screen, other jobs holding the GPU at 100% (a floor); fps
still / turning / new cells every frame. 32 places: layout 0.09 to 0.14 s, 77 to 168 /
77 to 136 / 54 to 66. 64 places: layout 1.5 to 1.9 s, 36 to 73 / 9 to 29 / 6 to 11.

## Run

```
PYTHONPATH=. .venv/bin/python -m physics.view [--cells N] [--seed S] [--unevenness U] [--set name=value ...]
PYTHONPATH=. .venv/bin/python -m physics.view --pictures DIR --steps 5000 --every 250 [--slice]
PYTHONPATH=. .venv/bin/pytest -q physics/view
```
Starts paused on the whole grid: space starts / pauses, drag rotates, wheel zooms, C
whole / slice, arrows or [ ] move the slice, 1 / 2 / 3 kinds, F fits, R resets.
