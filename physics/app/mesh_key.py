"""What the structure's topology is, as one comparable value.

Contract: `mesh_key(state)` returns bytes that are equal for two states exactly when they
have the same volume count, the same kind per volume and the same contacts (the A-B pair
array, in order). Energies, sizes and rates do not enter it. The runner sends a new "mesh"
message only when this changes (start, joint collapse, volume split).
"""
from __future__ import annotations

import hashlib

import numpy as np


def mesh_key(state) -> bytes:
    """A digest of the state's kinds and contacts."""
    digest = hashlib.blake2b(digest_size=16)
    kinds = np.ascontiguousarray(state.kind, dtype=np.uint8).ravel()
    pairs = np.ascontiguousarray(state.contacts, dtype=np.int64).reshape(-1, 2)
    digest.update(np.int64(len(kinds)).tobytes())
    digest.update(kinds.tobytes())
    digest.update(np.int64(len(pairs)).tobytes())
    digest.update(pairs.tobytes())
    return digest.digest()
