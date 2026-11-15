# physics/view/

The only viewer: a pyglet GPU window of plain GL points. Module map in `__init__.py`.

## Must not get wrong

- **Draw the simulation as it is.** Positions go to the GPU untouched; no layout, no
  correction, no link lines. Red = energy volume, blue = joint.
- **Opacity is a fixed scale** (`opacity.py`): log of density over the start's mean of that kind
  (`density_reference.py`, taken at start and on reset, then held). Never rank-based or
  re-normalised per frame, or a mass growing packed would never show. The test pins it.
- **Additive blending, depth write off**: order-free, faint dots add up.
- **GPU under WSL2**: `use_gpu_driver()` sets `GALLIUM_DRIVER=d3d12` and
  `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` before pyglet loads; without them Mesa draws on the
  CPU. CUDA-GL sharing fails on WSL (error 304), so each snapshot is copied to the host and
  uploaded (`point_cloud.py`, orphaned stream buffers).
- **Source interface** (`abab_source.py` docstring): reset(), step(), steps, positions, kind,
  density, stats(). The GPU engine plugs in by providing the same; Warp arrays are read with
  `.numpy()`. Stepping runs on its own thread (`stepper.py`) so a slow step never freezes the
  camera; a source error stops stepping and shows in the title, R restarts.
- **Headless tests**: `__init__.py` imports only modules that need no OpenGL.

## Measured (2026-11-15, RTX 5080, WSL2, d3d12)

Random points re-uploaded every frame: 200,000 at 397 fps, 1,000,000 at 201 fps; 200,000
static: 520 fps. Colours for 200,000 volumes on the CPU: 6 ms.

## Run

```
PYTHONPATH=. .venv/bin/python -m physics.view [--volumes N] [--seed S] [--set name=value ...] [--steps-per-frame K]
PYTHONPATH=. .venv/bin/pytest -q physics/view
```
Drag rotates, wheel zooms, space pauses, R resets, Esc quits.
