# physics/view/drawing/

The two views: the whole grid in three dimensions (the default) and a slice through it
(key C in the window).

- `definitions/`: triangle patterns, shown kinds, the camera and its fixed numbers,
  the shader text.
- `logic/`: pieces to triangles, far-to-near order, the camera fit, the GPU mesh.
- `tests/`: both views come from the same pieces; the fitted camera sees the whole grid.

## Must not get wrong

- **The cells are the grid; boxes are never drawn.**
- **Whole grid**: every cell is its block, faces shaded by the axis they face
  (`FACE_SHADE`), no light, no lines. **Slice**: every cell of one lattice layer along
  z as one flat face through the middle of its block.
- **Depth is by order, not a depth buffer.** Cells are sorted far to near whenever the
  cells or the view change, then blended. A depth test would let a clear cell hide
  what is behind it.
- **The slice starts on an even layer** (`middle_layer`): odd layers hold only joints.
- Only `logic/cell_mesh.py` and `logic/link_program.py` need OpenGL. Import them by
  module path; everything else runs headless.

## Not done

- Faces within one cell are not sorted; they share a colour, so only their shade can
  come out slightly off.
- With the engine's sizes the whole grid is mostly fat blue joints with energy volumes
  as small red specks between them. Key 1 shows the volumes alone.
