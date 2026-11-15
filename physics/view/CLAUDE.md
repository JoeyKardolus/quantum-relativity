# physics/view/

The only viewer: a pyglet GPU window of the Planck-box engine (`physics.potts`) as soft
additive points. Module map in `__init__.py`.

## Must not get wrong

- **Draw the simulation as it is.** A point per occupied Planck box at its centre in box units;
  empty boxes are never drawn. Red = the box's cell is an energy volume (A), blue = a joint (B);
  border boxes (touching another cell) get `BORDER_GAIN` (2x) light so the checkerboard of
  cells shows. Past `MOST_POINTS` (3M) one uniform share is kept (light / share), never by
  order or rank. No layout, no link lines (`box_points.py`).
- **Light is a fixed contrast scale** (`opacity.py`, a Warp function, the only copy): log-linear
  in a box's density over the start's mean density (`start_reference.py`, taken at start and on
  reset, then held, with boxes deep = cube root of the start's occupied boxes), 1 at 16x, and
  at 1x only `AVERAGE_GLOW` / boxes deep, so a line of sight of
  average boxes sums to 0.02: an even universe is black, thinner than average is nothing. Never
  rank-based or re-normalised per frame, or growth of structure would never show. Tests pin it.
- **Light is summed in a half-float buffer, shown through 1 - exp(-sum)** (`point_cloud.py`).
  An 8-bit screen rounds 1e-4 light to 0; the uint8 alpha carries light as a log byte
  (`light_code.py`). Additive, depth write off: order-free.
- **GPU under WSL2**: `use_gpu_driver()` sets `GALLIUM_DRIVER=d3d12` and
  `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` before pyglet loads; without them Mesa draws on the
  CPU. CUDA-GL sharing fails on WSL (error 304), so the shown boxes are gathered on the GPU and
  copied to the host once per frame, then uploaded (`point_cloud.py`, orphaned stream buffers).
- **Source interface** (`cells_source.py` docstring): reset(), step(), steps, stats(), view()
  (occupied boxes: positions (n, 3) or (n,) 3-vectors, density, kind 0 = energy volume /
  1 = joint, border; Warp arrays on one device). Stepping and every gather run on the stepper
  thread (`stepper.py`), so a slow step never freezes the camera; a source or gather error (a
  non-finite box, an unknown kind) stops stepping and shows in the title. The view fits the
  occupied boxes' bounds on start and reset, filling about half the window.
- **A CPU Warp array's `.numpy()` shares its buffer**: `box_points.py` copies on CPU.
- **Headless tests**: `__init__.py` imports neither OpenGL nor the engine; tests use a fake
  source on the Warp CPU device. Never a fake source in the shipped command.

## Measured (2026-11-15, RTX 5080, WSL2)

Gather of 884,736 boxes: 5.9 ms. Real engine, 20 s, seed 1: default 8^3 cells of 6^3, 113 to
118 fps, 7 to 14 steps/s; 10^3 cells, 117 fps but 0.5 steps/s from step 15. Boxes whose light
codes to 0 are not drawn.

## Run

```
PYTHONPATH=. .venv/bin/python -m physics.view [--cells N] [--boxes-per-cell K] [--seed S] [--unevenness U] [--slope S] [--set name=value ...] [--steps-per-frame F]
PYTHONPATH=. .venv/bin/pytest -q physics/view
```
Starts paused: space starts and pauses, drag rotates, wheel zooms, R resets, Esc quits.
