# physics/view/drawing/definitions/

What the two views are made of. Constants and formulas.

- `shown_kinds.py`: energy volumes, joints, or both.
- `block_faces.py`: a block's 12 triangles, a slice face's 2, the face shades.
- `camera_marks.py`: field of view, the corner and facing angles, fit and refit numbers.
- `middle_layer.py`: the even lattice layer nearest the middle.
- `orbit_camera.py`: the world-to-clip matrix of a camera circling a centre.
- `shader_text.py`: the GLSL text. It is only text here; nothing compiles it.

## Must not get wrong

- Nothing here imports `logic/`, OpenGL or the engine.
- The shader's `pattern[36]` and `MAX_PATTERN` must agree.
- `orbit_camera` returns a row-major matrix; the mesh uploads it transposed.
- `CORNER` looks at the grid from the corner at lattice index 0. `tests/` checks that
  corner is the nearest, and that right is right and up is up.
