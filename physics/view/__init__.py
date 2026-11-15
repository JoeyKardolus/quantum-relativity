"""The viewer: the fluid engine in a GPU window, drawn as soft additive points (the only viewer).

`python -m physics.view [--side N] [--seed S] [--unevenness U] [--set name=value ...] [--steps-per-frame K]`.
Every box of the grid holding energy above a small line is one soft point at its
centre: red for pulling (A) energy, blue for pushing (B), both in the same point;
how solid each kind is reads its energy on a fixed log scale against the start's
mean of that kind, so thin energy is a faint haze and packed masses glow. No
link lines. A source is anything with reset(), step(), steps, side, a_energy,
b_energy and stats() (`fluid_source.py` has the contract). The shown boxes are
gathered on the GPU and copied to the host once per frame (CUDA-GL sharing fails
under WSL2 with error 304), then uploaded to GL.

Module map:
  opacity.py           how solid a kind is in a box: fixed log scale (a Warp function)
  start_reference.py   the start's mean A and B energy per occupied box, the fixed unit
  box_points.py        BoxPoints: gathers shown boxes' centres and colours on the GPU
  orbit_camera.py      view-projection matrix of a camera circling a centre
  title_line.py        the window title from a snapshot's stats
  gpu_driver.py        Mesa d3d12 variables so OpenGL draws on the NVIDIA card
  fluid_source.py      FluidSource: physics.fluid's grid behind the source interface
  stepper.py           Stepper, Snapshot: steps a source on a thread, hands out snapshots
  point_cloud.py       PointCloud: GL points, round, additive, no depth write (needs GL)
  point_window.py      PointWindow: the pyglet window, controls and title (needs GL)
  __main__.py          the command line
Only modules that need no OpenGL (nor physics.fluid) are imported here, so tests run headless.
"""
from physics.view.opacity import opacity
from physics.view.start_reference import start_reference
from physics.view.box_points import BoxPoints
from physics.view.orbit_camera import orbit_camera
from physics.view.title_line import title_line
from physics.view.gpu_driver import use_gpu_driver

__all__ = ["opacity", "start_reference", "BoxPoints", "orbit_camera", "title_line", "use_gpu_driver"]
