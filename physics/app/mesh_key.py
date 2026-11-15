"""What the drawn mesh's topology is, as one comparable value.

Contract: `mesh_key(engine)` returns bytes that are equal for two engines exactly when
they have the same corner count, the same faces (corner index lists), the same cells
on each face and the same joint on each face. Corner positions, pulls and pushes do
not enter it. The runner sends a new "mesh" message only when this changes (start,
joint collapse, volume split).
"""
from __future__ import annotations

import hashlib

import numpy as np


def mesh_key(engine) -> bytes:
    """A digest of the engine's corner count, faces, face cells and face joints."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.int64(len(engine.corners)).tobytes())
    digest.update(np.int64(len(engine.faces)).tobytes())
    for face in engine.faces:
        corners = np.asarray(face, dtype=np.int64)
        digest.update(np.int64(len(corners)).tobytes())
        digest.update(corners.tobytes())
    digest.update(np.asarray(engine.face_cells, dtype=np.int64).tobytes())
    digest.update(np.asarray(engine.face_joint, dtype=np.int64).tobytes())
    return digest.digest()
