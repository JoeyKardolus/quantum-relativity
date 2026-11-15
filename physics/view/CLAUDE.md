# physics/view/

The only viewer: a pyglet GPU window of the fluid engine (`physics.fluid`) as soft additive
points. Module map in `__init__.py`.

## Must not get wrong

- **Draw the simulation as it is.** One point per box above the `SHOWN` line (1% of the start's
  mean of either kind), at the box centre in box units; no layout, no correction, no link lines.
  Red = pulling (A) energy, blue = pushing (B), both in one point (`box_points.py`).
- **Opacity is a fixed scale** (`opacity.py`, a Warp function, the only copy): log of a box's
  energy over the start's mean of that kind (`start_reference.py`, taken at start and on reset,
  then held). Never rank-based or re-normalised per frame, or a mass packing would never show.
  The test pins it.
- **Additive blending, depth write off**: order-free, faint dots add up into clouds.
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
