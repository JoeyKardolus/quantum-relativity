"""Drawing, logic: what happens when a view is drawn.

  grid_triangles.py  grid_triangles: the triangles of the whole grid or of a slice
  far_to_near.py     far_to_near: the order to draw see-through cells in
  view_fit.py        view_fit: the camera fit to the grid's frame
  link_program.py    link_program: GLSL compiled and linked        (needs OpenGL)
  cell_mesh.py       CellMesh: the cells on the GPU, blended        (needs OpenGL)

Import `cell_mesh` and `link_program` by their module paths; the rest runs headless.
"""
