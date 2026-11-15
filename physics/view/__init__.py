"""The viewer: the fluid engine in a GPU window, drawn as soft additive points (the only viewer).

`python -m physics.view [--side N] [--seed S] [--unevenness U] [--set name=value ...] [--steps-per-frame K]`.
Boxes denser than the start's mean (plus a fixed 1% sample of the rest) are soft
points of light at their centres: red for pulling (A) energy, blue for pushing
(B), both in the same point; the light reads the box's density contrast on a
fixed log scale normalised by the line of sight, so an even universe is nearly
black and only denser zones glow; light is summed in a float buffer. No link
lines. A source is anything with reset(), step(), steps, side, a_energy,
b_energy and stats() (`fluid_source.py` has the contract). The shown boxes are
gathered on the GPU and copied to the host once per frame (CUDA-GL sharing fails
under WSL2 with error 304), then uploaded to GL.

Module map:
  opacity.py           the light a box adds from its density contrast: fixed log scale (Warp)
  light_code.py        a point's light as one log byte, decoded by the window (Warp)
  start_reference.py   the start's mean A and B energy per occupied box, the fixed unit
  box_points.py        BoxPoints: selects and gathers drawn boxes on the GPU
  orbit_camera.py      view-projection matrix of a camera circling a centre
  title_line.py        the window title from a snapshot's stats
  gpu_driver.py        Mesa d3d12 variables so OpenGL draws on the NVIDIA card
  fluid_source.py      FluidSource: physics.fluid's grid behind the source interface
  stepper.py           Stepper, Snapshot: steps a source on a thread, hands out snapshots
  point_cloud.py       PointCloud: GL points summed in a float buffer, fixed curve (needs GL)
  point_window.py      PointWindow: the pyglet window, controls and title (needs GL)
  __main__.py          the command line
Only modules that need no OpenGL (nor physics.fluid) are imported here, so tests run headless.
"""
from physics.view.opacity import opacity
from physics.view.light_code import light_code
from physics.view.start_reference import start_reference
from physics.view.box_points import BoxPoints
from physics.view.orbit_camera import orbit_camera
from physics.view.title_line import title_line
from physics.view.gpu_driver import use_gpu_driver

__all__ = ["opacity", "light_code", "start_reference", "BoxPoints", "orbit_camera", "title_line", "use_gpu_driver"]
