"""The viewer: physics in a GPU window, drawn as plain round points (the only viewer).

`python -m physics.view [--volumes N] [--seed S] [--set name=value ...] [--steps-per-frame K]`.
Red dots are energy volumes, blue dots joints; a dot's opacity is its energy
density on a fixed log scale against the start's mean of its kind. No link lines.
A source is anything with reset(), step(), steps, positions, kind, density and
stats() (`abab_source.py` has the contract); physics.abab's State is one, the GPU
engine another. Under WSL2 the frame is copied GPU -> CPU -> GL each frame (CUDA-GL
sharing fails there with error 304).

Module map:
  opacity.py            how solid a dot is: fixed log scale of density over the start's mean
  density_reference.py  the start's mean density per kind, the fixed unit of opacity
  point_colours.py      RGBA bytes per volume: red or blue, alpha from opacity
  orbit_camera.py       view-projection matrix of a camera circling a centre
  title_line.py         the window title from a snapshot's stats
  gpu_driver.py         Mesa d3d12 variables so OpenGL draws on the NVIDIA card
  abab_source.py        AbabSource: physics.abab's State behind the source interface
  stepper.py            Stepper, Snapshot: steps a source on a thread, hands out snapshots
  point_cloud.py        PointCloud: GL points, round, additive, no depth write (needs GL)
  point_window.py       PointWindow: the pyglet window, controls and title (needs GL)
  __main__.py           the command line
Only the modules that need no OpenGL are imported here, so tests run headless.
"""
from physics.view.opacity import opacity
from physics.view.density_reference import density_reference
from physics.view.point_colours import point_colours
from physics.view.orbit_camera import orbit_camera
from physics.view.title_line import title_line
from physics.view.gpu_driver import use_gpu_driver

__all__ = ["opacity", "density_reference", "point_colours", "orbit_camera", "title_line", "use_gpu_driver"]
