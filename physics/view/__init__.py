"""The viewer: the grid of cells in a GPU window, a perfect grid that only lumps warp (the only viewer).

`python -m physics.view [--cells N] [--seed S] [--set name=value ...] [--steps-per-frame F]`
opens the window; with `--pictures DIR --steps N --every K [--whole]` it saves pictures
instead. The cells are the grid (bible `physics/quantum-relativity`, axiom 3): a perfect
A B A B pattern, energy volumes red, joints blue, every cell outlined, its fill clear at
the calm background's packing and solid where packed, so a lump stands like a mass in
empty-looking space and the smaller cells of a lump bend the lines round it. Planck
boxes are not space and are never drawn.

One directory per purpose, each with `definitions/` (what things are), `logic/` (what
happens) and `tests/`:
  layout/    every cell in its place of the pattern, the perfect grid warped round lumps
  colours/   red and blue, fills following packing on one fixed scale, outlines always
  drawing/   the slice (the default) and the whole grid in three dimensions (key C)
  source/    the engine read as something to draw                  (needs the engine)
  window/    the pyglet window, its keys and the stepping thread
  pictures/  PNG files saved while a source steps

This file imports no OpenGL and no engine, so the names below load headless. Import
`CellMesh`, `CellWindow`, `WholePicture` and `CellsSource` by their module paths.
"""
from physics.view.colours.logic.cell_colours import cell_colours
from physics.view.colours.logic.relative_packing import relative_packing
from physics.view.colours.logic.start_marks import start_marks
from physics.view.drawing.definitions.middle_layer import middle_layer
from physics.view.drawing.definitions.orbit_camera import orbit_camera
from physics.view.drawing.logic.far_to_near import far_to_near
from physics.view.drawing.logic.grid_triangles import grid_triangles
from physics.view.drawing.logic.view_fit import view_fit
from physics.view.layout.definitions.pieces import Pieces
from physics.view.layout.logic.cell_layout import cell_layout
from physics.view.layout.logic.cell_pieces import cell_pieces
from physics.view.layout.logic.cell_slots import cell_slots
from physics.view.layout.logic.torn_lines import torn_lines
from physics.view.pictures.logic.saved_frames import saved_frames
from physics.view.pictures.logic.slice_of_the_middle import slice_of_the_middle
from physics.view.pictures.logic.slice_picture import slice_picture
from physics.view.pictures.logic.write_png import write_png
from physics.view.window.logic.stepper import Stepper
from physics.view.window.logic.title_line import title_line
from physics.view.window.logic.use_gpu_driver import use_gpu_driver

__all__ = ["Pieces", "Stepper", "cell_colours", "cell_layout", "cell_pieces", "cell_slots", "far_to_near",
           "grid_triangles", "middle_layer", "orbit_camera", "relative_packing", "saved_frames",
           "slice_of_the_middle", "slice_picture", "start_marks", "title_line", "torn_lines", "use_gpu_driver",
           "view_fit", "write_png"]
