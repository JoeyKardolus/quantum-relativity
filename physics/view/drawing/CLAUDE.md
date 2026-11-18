# physics/view/drawing/

The two views: the slice through the grid (the default) and the whole grid in three
dimensions (key C in the window).

- `definitions/`: triangle and line patterns, shown kinds, the camera and its fixed
  numbers, the shader text.
- `logic/`: pieces to triangles, far-to-near order, the camera fit, the GPU mesh.
- `tests/`: both views come from the same pieces; the fitted camera sees the whole grid.

## Must not get wrong

- **The cells are the grid; boxes are never drawn.**
- **Slice**: every cell of one lattice layer along z as one flat face, filled by its
  packing, and **always its outline** in its kind's hue, set in so a volume's red and
  a joint's blue lie side by side. The A B A B pattern must show where every fill is
  clear. The outline fades with the cell's size on screen (`outline_opacity`), so
  millions of cells seen from far never become a haze.
- **Whole grid**: the slice, plus only the packed cells as blocks, faces shaded by the
  axis they face (`FACE_SHADE`), no light. The calm background is not drawn in three
  dimensions at all: lumps stand as masses in empty-looking space.
- **Depth is by order, not a depth buffer.** Cells are sorted far to near whenever the
  cells or the view change, then blended. A depth test would let a clear cell hide
  what is behind it.
- **The slice starts on an even layer** (`middle_layer`): odd layers hold only joints.
- Only `logic/cell_mesh.py` and `logic/link_program.py` need OpenGL. Import them by
  module path; everything else runs headless.

## Not done

- Faces within one cell are not sorted; they share a colour, so only their shade can
  come out slightly off.
- The slice's lines are drawn before the blocks, so a block hides the lines in front
  of it too.
