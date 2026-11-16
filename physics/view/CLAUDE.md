# physics/view/

The only viewer: a pyglet GPU window of the whole-quanta engine (`physics.engine`) as small
squares. Module map in `__init__.py`.

## Must not get wrong

- **Draw the simulation as it is.** One hard square per occupied Planck box at its centre
  (owner: never round blurred dots); empty boxes never drawn. Red = energy volume (A), blue =
  joint (B). Keys 1 / 2 / 3: energy volumes / joints / both. No link lines.
- **Two looks, G switches.** Nearest (the start): depth-tested, the nearest box hides those
  behind, colour by kind, density only a fixed log tint (`NEAREST_DIMMEST` to 1 over 1/16x to
  16x); squares `SQUARE_PIXELS` (3) or a box's true size if larger, so a near face closes.
  Glow: additive, see-through, so it averages the kinds away along the line of sight.
- **Glow light is a fixed contrast scale per kind** (`opacity.py`, the only copy): log-linear in
  density over its own kind's start mean (`start_reference.py`, held from start or reset), 1 at
  16x, `AVERAGE_GLOW` / boxes deep at 1x; borders `BORDER_GAIN` (2x). Never rank-based or
  re-normalised per frame. Summed in a half-float buffer through 1 - exp(-sum); alpha carries
  light as a log byte (`light_code.py`). Past `MOST_POINTS` (3M) a uniform share is kept.
- **Slice (S)**: one box layer along z through the middle of the occupied bounds; arrows move
  it, held inside the occupied layers; S turns the camera to face it. Never thinned.
- **Camera fit** (`view_fit.py`): on start, reset, F, and when the occupied extent outgrows
  the last fit by `REFIT_GROWTH` (1.5x); turning kept, zoom reset.
- **GPU under WSL2**: `use_gpu_driver()` sets `GALLIUM_DRIVER=d3d12` and
  `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` before pyglet loads, else Mesa draws on the CPU.
  CUDA-GL sharing fails on WSL (error 304): shown boxes are gathered on the GPU, copied to the
  host once per frame, uploaded.
- **Source interface** (`cells_source.py` docstring): reset(), step(), steps, stats(), view()
  (positions, density, kind 0 / 1, border; Warp arrays on one device). Stepping and gathers
  run on the stepper thread; a source or gather error stops stepping and shows in the title.
  Look, kind and slice changes republish while paused.
- **A CPU Warp array's `.numpy()` shares its buffer**: `box_points.py` copies.
- **Headless tests**: `__init__.py` imports neither OpenGL nor the engine; tests use a fake
  source on the Warp CPU device. Never a fake source in the shipped command.

## Measured

2026-11-15, RTX 5080, WSL2: gather of 884,736 boxes 5.9 ms; 8^3 cells of 6^3, 113 to 118 fps,
7 to 14 steps/s. 2026-11-16, fake 24^3 checkerboard of 6^3 cells, density 1: nearest 25,115
red / 25,441 blue lit pixels; glow brightest 2/255; slice 50/50.

## Run

```
PYTHONPATH=. .venv/bin/python -m physics.view [--cells N] [--boxes-per-cell K] [--seed S] [--unevenness U] [--joint-unevenness J] [--slope S] [--set name=value ...] [--steps-per-frame F]
PYTHONPATH=. .venv/bin/pytest -q physics/view
```
Starts paused: space starts / pauses, 1 / 2 / 3 kinds, G look, S slice, up / down move it,
F fits, drag rotates, wheel zooms, R resets, Esc quits.
