"""Drawing, definitions: what the two views are made of.

  shown_kinds.py   SHOW_ENERGY_VOLUMES, SHOW_JOINTS, SHOW_BOTH, SHOWN
  block_faces.py   BLOCK_PATTERN, FACE_PATTERN, MAX_PATTERN, FACE_SHADE, FACE_LINES, BLOCK_LINES
  camera_marks.py  FOV_DEGREES, CORNER, FACING, FILL, FACE_FILL, LEVELS
  middle_layer.py  middle_layer: the even layer the slice starts at
  orbit_camera.py  orbit_camera: the matrix of a camera circling a centre
  place_pixels.py  place_pixels: how wide a lattice place is on screen
  shader_text.py   VERTEX, FRAGMENT: the GLSL the mesh draws with

Nothing here imports logic, OpenGL or the engine.
"""
