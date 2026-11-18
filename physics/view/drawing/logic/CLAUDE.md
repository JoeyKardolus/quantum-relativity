# physics/view/drawing/logic/

What happens when a view is drawn:

- `grid_triangles.py`: `Pieces` to corner points, colours and a triangle pattern, as
  blocks (`layer=None`) or as one slice's faces.
- `far_to_near.py`: the order to draw see-through cells in, farthest first.
- `view_fit.py`: where the camera stands so the grid's frame fills the view. Drawers
  fit every frame, so the grid stays still while the universe's size changes.
- `link_program.py`: compiles and links the shaders. Needs OpenGL.
- `cell_mesh.py`: `CellMesh`, the cells on the GPU. Needs an OpenGL 3.3 context.

## Must not get wrong

- **One instance per cell.** Corners and colours sit in two buffer textures; only the
  order (four bytes a cell) is sent again when the view turns.
- **Outlines are a second pass of the same mesh** (`draw(outline=...)`): lines in the
  cell's plain hue, set in toward its middle, whatever its fill.
- `CellMesh.upload` raises on shapes that disagree or cells that do not fit a buffer
  texture. A shader that fails raises with the driver's log.
- `tests/` here runs headless. The GPU path is tested in `pictures/tests/` (a solid
  cube's pixel, a clear cell, near over far) and `window/tests/`.
