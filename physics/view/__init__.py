"""The viewer: the grid of cells in a GPU window, every cell at its own size (the only viewer).

`python -m physics.view [--cells N] [--seed S] [--unevenness U] [--slope S] [--set name=value ...] [--steps-per-frame F]`
opens the window; with `--pictures DIR --steps N --every K` it saves slice pictures
instead, without a window. The cells are the grid (bible `physics/quantum-relativity`,
axiom 3): a perfect A B A B pattern, energy volumes red, joints blue, brighter the more
packed on one fixed scale. A cell has no place of its own, so the drawing lays the
pattern out from the cells' lengths alone (`cell_slots`, `cell_layout`): every cell
keeps its place in the pattern from frame to frame and only sizes warp it. Planck boxes
are not space and are never drawn. The window starts on a slice through the middle; key
C switches between the slice and the whole grid, arrows or [ ] move the slice, 1 / 2 / 3
show energy volumes, joints or both.

Module map:
  cell_slots.py      cell_slots: the place in the pattern every cell is drawn in, and each place's widths
  cell_layout.py     cell_layout: the warped grid's corner points from those widths
  cell_pieces.py     cell_pieces, Pieces: every living cell as its block of the warped grid
  cell_colours.py    cell_colours: red volumes, blue joints, brightness by packing, fixed scale
  grid_triangles.py  grid_triangles: the triangles of a slice or of the whole grid
  middle_layer.py    middle_layer: the even layer the slice starts at
  slice_picture.py   slice_picture: a slice as an image, no window
  write_png.py       write_png: an image to a PNG file
  slice_frames.py    slice_frames: step a source, save a slice picture every few steps
  orbit_camera.py    view-projection matrix of a camera circling a centre
  view_fit.py        the camera fit to the grid's bounds, and when a fit is due
  title_line.py      the window title from a snapshot's stats
  gpu_driver.py      Mesa variables for the NVIDIA GPU under WSL2
  cells_source.py    the engine as a source                              (needs the engine)
  stepper.py         background stepping thread and frame snapshots
  cell_mesh.py       GL triangles, depth-tested                          (needs OpenGL)
  cell_window.py     pyglet window, controls                             (needs OpenGL)
  __main__.py        command line

This package imports no OpenGL and no engine, so its tests run headless; import
`cell_mesh`, `cell_window` and `cells_source` by their module paths.
"""
from physics.view.cell_colours import cell_colours
from physics.view.cell_layout import cell_layout
from physics.view.cell_pieces import Pieces, cell_pieces
from physics.view.cell_slots import cell_slots
from physics.view.gpu_driver import use_gpu_driver
from physics.view.grid_triangles import grid_triangles
from physics.view.middle_layer import middle_layer
from physics.view.orbit_camera import orbit_camera
from physics.view.slice_frames import slice_frames
from physics.view.slice_picture import slice_picture
from physics.view.title_line import title_line
from physics.view.view_fit import view_fit
from physics.view.write_png import write_png

__all__ = ["Pieces", "cell_colours", "cell_layout", "cell_pieces", "cell_slots", "grid_triangles", "middle_layer",
           "orbit_camera", "slice_frames", "slice_picture", "title_line", "use_gpu_driver", "view_fit", "write_png"]
