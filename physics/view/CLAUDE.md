# physics/view/

The only viewer: a pyglet GPU window of the fluid engine (`physics.fluid`) as soft additive
points. Module map in `__init__.py`.

## Must not get wrong

- **Draw the simulation as it is.** A point per box at its centre in box units: every box at or
  above `CONTRAST_SHOWN` (1.5x the start's mean density), plus a fixed random 1% of the rest
  standing in for them (light x 100); past `MOST_POINTS` (3M) the dense ones are thinned by one
  uniform share (light / share), never by order or rank. No layout, no link lines. Red = pulling
  (A), blue = pushing (B), mixed in one point (`box_points.py`).
- **Light is a fixed contrast scale** (`opacity.py`, a Warp function, the only copy): log-linear
  in a box's density over the start's mean density (`start_reference.py`, taken at start and on
  reset, then held), 1 at 16x, and at 1x only `AVERAGE_GLOW` / side, so a line of sight of
  average boxes sums to 0.02: an even universe is black, thinner than average is nothing. Never
  rank-based or re-normalised per frame, or growth of structure would never show. Tests pin it.
- **Light is summed in a half-float buffer, shown through 1 - exp(-sum)** (`point_cloud.py`).
  An 8-bit screen rounds 1e-4 light to 0; the uint8 alpha carries light as a log byte
  (`light_code.py`). Additive, depth write off: order-free.
- **GPU under WSL2**: `use_gpu_driver()` sets `GALLIUM_DRIVER=d3d12` and
  `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` before pyglet loads; without them Mesa draws on the
  CPU. CUDA-GL sharing fails on WSL (error 304), so the shown boxes are gathered on the GPU and
  copied to the host once per frame, then uploaded (`point_cloud.py`, orphaned stream buffers).
- **Source interface** (`fluid_source.py` docstring): reset(), step(), steps, side, a_energy,
  b_energy (Warp arrays, side^3 boxes, index (x * side + y) * side + z), stats(). Stepping and
  every gather run on the stepper thread (`stepper.py`), so a slow step never freezes the
  camera; a source or gather error (a non-finite box) stops stepping and shows in the title.
- **A CPU Warp array's `.numpy()` shares its buffer**: `box_points.py` copies on CPU.
- **Headless tests**: `__init__.py` imports neither OpenGL nor physics.fluid; tests use a fake
  source on the Warp CPU device. Never a fake source in the shipped command.

## Measured (2026-11-15, RTX 5080, WSL2, d3d12)

Real engine, side 128, seed 1, unevenness 0.1: about 118 fps (the 120 Hz draw interval);
3.6 to 4.3 steps/s in the first steps, about 1.1 once the ball has fallen into its core.
Stand-in source at 2,003,424 boxes: 75 to 104 fps.

## Run

```
PYTHONPATH=. .venv/bin/python -m physics.view [--side N] [--seed S] [--unevenness U] [--set name=value ...] [--steps-per-frame K]
PYTHONPATH=. .venv/bin/pytest -q physics/view
```
Starts paused: space starts and pauses, drag rotates, wheel zooms, R resets, Esc quits.
