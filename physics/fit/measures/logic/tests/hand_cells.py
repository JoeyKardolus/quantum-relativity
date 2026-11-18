"""Hand-made cells for the measures' tests: small grids whose every number is known.

`lattice_cells()` is a `read_universe` dict of a start lattice 4 a side: 64 ids, the 8
energy volumes at the even places (ids 0, 2, 8, 10, 32, 34, 40, 42) holding 100 quanta
and 10 long, every other id a joint holding 10 and 30 long, every clock at full rate.
Eight ways join the volumes in one ring, 0 - 2 - 10 - 8 - 40 - 42 - 34 - 32 - 0.
`ring_snapshot(lengths, lumps, held)` is a `take_snapshot` dict of six volumes (ids
0 .. 5) in a ring with a joint (ids 6 .. 11) between each pair of neighbours.
"""
import numpy as np

from physics.engine import ENERGY_VOLUME, JOINT

VOLUMES = [0, 2, 8, 10, 32, 34, 40, 42]
_RING = [(1, 0, 2), (6, 2, 10), (9, 10, 8), (24, 8, 40), (41, 40, 42), (38, 42, 34), (33, 34, 32), (16, 32, 0)]


def lattice_cells() -> dict:
    kind = np.full(64, JOINT, dtype=np.int32)
    kind[VOLUMES] = ENERGY_VOLUME
    ways = tuple(np.array(column, dtype=np.int64) for column in zip(*_RING))
    return {"side": 4, "count": 64, "kind": kind, "held": np.where(kind == ENERGY_VOLUME, 100, 10).astype(np.int64),
            "length": np.where(kind == ENERGY_VOLUME, 10.0, 30.0), "rate": np.ones(64), "ways": ways,
            "fell_into": np.full(64, -1, dtype=np.int32), "start_volume_quanta": 100.0, "version": 0,
            "collapses": 0, "merges": 0, "tears": 0, "squeeze_max": 0.0, "packing_max": 0.1}


def ring_snapshot(lengths, lumps, held=None) -> dict:
    volumes = np.arange(6)
    held = np.array([100] * 6 + [10] * 6 if held is None else held, dtype=np.int64)
    return {"held": held, "length": np.asarray(lengths, dtype=np.float64), "count": 12, "plus": int(held[:6].sum()),
            "lumps": np.asarray(lumps, dtype=np.int64), "ways": (volumes + 6, volumes, (volumes + 1) % 6), "version": 0}
