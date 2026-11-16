"""The viewer: the Planck-box engine in a GPU window, drawn as small squares (the only viewer).

`python -m physics.view [--cells N] [--boxes-per-cell K] [--seed S] [--unevenness U] [--joint-unevenness J] [--slope S] [--set name=value ...] [--steps-per-frame F]`.
Every occupied Planck box is a small square at its centre: red in an energy
volume (A), blue in a joint (B); empty boxes are not drawn; keys 1, 2 and 3 show
energy volumes, joints or both. Two looks (key G): nearest, the start, draws the
squares depth-tested so the nearest box hides those behind it, coloured by kind,
density only a brightness tint, so the structure reads; glow sums light along
the line of sight, each box lit by its density over the start's mean of its own
kind on a fixed log scale (an even universe nearly black, only denser zones glow,
cell borders brighter), summed in a float buffer. Key S draws only a one-box
slice through the middle; key C draws only a slab a few boxes thick through the
middle in brighter colours (red energy volumes, blue joints, lighter cell borders,
packed cells brighter), to see inside; arrows or [ ] move either; the view refits as space grows (F
fits by hand). No link lines. A source is anything with reset(), step(), steps,
view() and stats() (`cells_source.py` has the contract). The shown boxes are
gathered on the GPU and copied to the host once per frame (CUDA-GL sharing fails
under WSL2 with error 304), then uploaded to GL.

Module map:
  opacity.py           the light a box adds from its density contrast: fixed log scale (Warp)
  light_code.py        a point's light as one log byte, decoded by the window (Warp)
  start_reference.py   the start's mean density of each kind and boxes deep, the fixed unit
  box_points.py        BoxPoints: selects (kind, slice, slab) and lights (nearest, glow, slab) drawn boxes on the GPU
  orbit_camera.py      view-projection matrix of a camera circling a centre
  view_fit.py          the camera fit to the occupied bounds, and when space has outgrown it
  title_line.py        the window title from a snapshot's stats
  gpu_driver.py        Mesa d3d12 variables so OpenGL draws on the NVIDIA card
  cells_source.py      CellsSource: physics.engine's state behind the source interface
  stepper.py           Stepper, Snapshot: steps a source on a thread, hands out snapshots
  point_cloud.py       PointCloud: GL points, depth-tested or summed in a float buffer (needs GL)
  point_window.py      PointWindow: the pyglet window, controls and title (needs GL)
  __main__.py          the command line
Only modules that need no OpenGL (nor the engine) are imported here, so tests run headless.
"""
from physics.view.opacity import opacity
from physics.view.light_code import light_code
from physics.view.start_reference import start_reference
from physics.view.box_points import BoxPoints
from physics.view.orbit_camera import orbit_camera
from physics.view.view_fit import view_fit
from physics.view.title_line import title_line
from physics.view.gpu_driver import use_gpu_driver

__all__ = ["opacity", "light_code", "start_reference", "BoxPoints", "orbit_camera", "view_fit", "title_line", "use_gpu_driver"]
