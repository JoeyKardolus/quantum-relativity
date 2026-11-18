"""Block faces: the triangles of a cell's block and of its slice face.

Contract: constants only. A block has 8 corners, corner c on the high side along x when
c & 1, along y when c & 2, along z when c & 4 (`Pieces`).
  BLOCK_PATTERN  int32 (36,): the block's six faces as twelve triangles, corner numbers
  FACE_PATTERN   int32 (6,): a four-cornered slice face as two triangles
  MAX_PATTERN    the longest pattern the mesh takes (36)
  FACE_SHADE     how bright a face is drawn by the axis it faces (x, y, z), so the
                 whole grid's blocks read as blocks without any light
"""
from __future__ import annotations

import numpy as np

_FACES = ((0, 2, 6, 4), (1, 3, 7, 5), (0, 1, 5, 4), (2, 3, 7, 6), (0, 1, 3, 2), (4, 5, 7, 6))
BLOCK_PATTERN = np.array([face[corner] for face in _FACES for corner in (0, 1, 2, 0, 2, 3)], dtype=np.int32)
FACE_PATTERN = np.array((0, 1, 2, 0, 2, 3), dtype=np.int32)
MAX_PATTERN = 36
FACE_SHADE = (0.8, 1.0, 0.9)
