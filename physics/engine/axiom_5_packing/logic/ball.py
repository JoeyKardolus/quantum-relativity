"""Ball: a universe the engine grew by its leak, kept, and the scenes that show gravity in it.

Nothing here lays a grid: `ball` steps a first volume until a volume stands three layers out
(a minute on the CPU; kept in the tmp folder under a name made of the engine's source, so a
changed engine grows its own). A scene then stages one thing in that ball, as a check does: a region
(a volume and the six at its sides, those that stand) is made 4 times fuller, and for
`coasting` set going. Chosen before any run: the two falling regions lie 4 volumes apart, as
on the bench, and one volume off the middle so the first volume is not between them; the
dense and the coasting region lie two volumes out, in the thin outer part.
"""
import hashlib
import tempfile
from pathlib import Path

import numpy as np

from physics.engine.axiom_3_structure.definitions.kinds import ACROSS
from physics.engine.axiom_3_structure.definitions.store import Store
from physics.engine.axiom_3_structure.logic.first_volume import first_volume
from physics.engine.axiom_3_structure.logic.kept import keep, kept
from physics.engine.axiom_5_packing.logic.lumps import fuller, grown, moving, region

LAYERS, ROOM = 3, 7  # the ball's layers and the store's room for more
PAIR = ((1, -2, 0), (1, 2, 0))  # the two falling regions' middles, in volumes from the first
OUT = (0, 0, -2)  # the dense and the coasting region's middle
GOING = (1, 1.0)  # coasting: along the second direction, a calm length a turn of a calm swirl


def _file(across: int) -> Path:
    """The kept ball's file: named by every line of the engine and the boxes across."""
    source = sorted(Path(__file__).parents[2].rglob("*.py"))
    mark = hashlib.sha1(b"".join(one.read_bytes() for one in source if "tests" not in one.parts))
    return Path(tempfile.gettempdir()) / f"physics-ball-{mark.hexdigest()[:12]}-{across}.npz"


def ball(across: int = ACROSS, device: str = "cpu") -> Store:
    """The grown ball on `device`: grown on the CPU and kept, unless this engine has grown it before."""
    path = _file(across)
    if not path.exists():
        for old in path.parent.glob(f"physics-ball-*-{across}.npz"):
            old.unlink()  # a ball of an engine that is no more
        keep(grown(LAYERS, ROOM, across, "cpu"), path)
    return kept(path, first_volume(across, device, ROOM))


def standing(store: Store, out: tuple) -> list:
    """The volumes of the region round `out` that stand in the ball."""
    own = store.own_cell.numpy()[:store.count]
    return [cell for cell in region(store, out) if cell in own]


def staged(spots: tuple = (), going: bool = False, across: int = ACROSS, device: str = "cpu") -> Store:
    """The ball with the region at each of `spots` made fuller, the first set going if `going`."""
    store = ball(across, device)
    for spot in spots:
        fuller(store, standing(store, spot))
    if going:
        moving(store, standing(store, spots[0]), *GOING)
    return store


def falling(across: int = ACROSS, device: str = "cpu") -> Store:
    """Two fuller regions in the ball: do they draw together."""
    return staged(PAIR, False, across, device)


def dense(across: int = ACROSS, device: str = "cpu") -> Store:
    """One fuller region in the ball's outer part: does it gather on its own middle and hold."""
    return staged((OUT,), False, across, device)


def coasting(across: int = ACROSS, device: str = "cpu") -> Store:
    """One fuller region in the ball's outer part set going: how far does it get."""
    return staged((OUT,), True, across, device)
