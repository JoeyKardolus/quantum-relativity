"""Block adjacency: per place of a 3 x 3 x 3 block the bit mask of its neighbours there, on the device.

Contract: `block_adjacency(device, faces_only=False)` returns an int32 Warp array of 27
masks: for block place i the bits of the places that touch it (26-neighbours, or only
across a face), the centre 13 left out. Cached per device. one_group.py reads it.
"""
from __future__ import annotations

import numpy as np
import warp as wp

_STORE = {}


def block_adjacency(device, faces_only: bool = False) -> wp.array:
    """The 27 neighbour masks of a 3 x 3 x 3 block, centre left out."""
    key = (str(device), faces_only)
    if key not in _STORE:
        places = np.array([(index // 9, index // 3 % 3, index % 3) for index in range(27)])
        masks = np.zeros(27, dtype=np.int64)
        for index in range(27):
            for other in range(27):
                apart = np.abs(places[index] - places[other])
                touching = apart.sum() == 1 if faces_only else apart.max() == 1
                if other not in (index, 13) and touching:
                    masks[index] |= 1 << other
        _STORE[key] = wp.array(masks.astype(np.int32), dtype=wp.int32, device=device)
    return _STORE[key]
