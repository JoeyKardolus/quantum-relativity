"""Cell layout: the corners of the drawn grid: the perfect grid, warped only round cells of another size.

Contract: `cell_layout(widths, typical, first=(0, 0, 0), places=None, flat=False,
trim=((0, 0),) * 3)` takes (3, nx, ny, nz) widths of a box of slots whose first slot is
lattice place `first` in a lattice of `places` places a side (None: the box is the
lattice), and `typical`, the (volume, joint) lengths of `typical_lengths`. It returns
`(corners, limited)`: float64 (nx + 1, ny + 1, nz + 1, 3) corner points, less the `trim`
(low, high) slots cut off each axis's ends, and whether any size was further from the
typical than is drawn. The slot at (i, j, k) is the block between corners [i, j, k] and
[i + 1, j + 1, k + 1]: slots share corners, so there is never a gap.

The method, and why. The drawing starts from the perfect grid (`regular_planes`): every
even place as wide as the typical volume, every odd one as the typical joint. Each
slot's width is then compared with the typical one, and the difference moves the corner
planes of its own row by fixed weights (`pull_weights`): its own two planes by half the
difference each, the planes further along by less and less, none beyond REACH places;
a line between rows takes the ACROSS-weighted mean of the rows beside it. That is a
plain weighted sum, with no fit and no memory, so:
  - where nothing differs within REACH, corners are exactly the perfect grid's, in
    every frame; a change moves only the corners within REACH of it
  - the weights of mirrored places cancel, so the outer frame never moves along its
    own axis; it changes only when the typical lengths do, and then evenly
  - a slot is drawn at least 1 / LIMIT of its typical width less whatever its
    neighbours need (at most 1 / (2 LIMIT) of the smaller typical width), so every
    edge keeps at least 1 / (2 LIMIT) of its regular length: corners keep their order
    along every line whatever the sizes
What it costs: a cell is drawn smaller only against its neighbourhood. A lone small cell
shows part of its size (its lines are shared with the rows beside it); a region wider
than REACH is drawn regular inside; a cell larger than typical is drawn hardly larger.
The colour carries each cell's own packing exactly.
Should the bends between rows still turn a block inside out, the whole warp is drawn at
the largest share that folds none (found by halving, EASE_TURNS times) and `limited`
says so. With `flat` the z planes stay regular (a slice is a flat cut) and what must
not fold is each slot's face through its middle, the thing a slice draws.
"""
from __future__ import annotations

import numpy as np

from physics.view.layout.definitions.pull_weights import pull_weights
from physics.view.layout.definitions.regular_planes import regular_planes
from physics.view.layout.definitions.warp_marks import ACROSS, EASE_TURNS, LIMIT, REACH


def cell_layout(widths: np.ndarray, typical: tuple, first: tuple = (0, 0, 0), places: int | None = None,
                flat: bool = False, trim: tuple = ((0, 0),) * 3) -> tuple:
    """Corner points of the perfect grid warped round the slots that differ, and whether sizes were limited."""
    widths = np.asarray(widths, dtype=np.float64)
    count = widths.shape[1:]
    places = count[0] if places is None else places
    volume, joint = typical
    room = min(volume, joint) / (2.0 * LIMIT)
    regular = np.zeros((count[0] + 1, count[1] + 1, count[2] + 1, 3))
    shift = np.zeros_like(regular)
    limited = False
    for axis in range(3):
        along = [-1 if other == axis else 1 for other in range(3)]
        regular[..., axis] = regular_planes(count[axis], first[axis], volume, joint, places).reshape(along)
        if flat and axis == 2:
            continue
        usual = np.where((first[axis] + np.arange(count[axis])) % 2 == 0, volume, joint).reshape(along)
        limited = limited or bool(((widths[axis] < usual / LIMIT) | (widths[axis] > usual * LIMIT)).any())
        shift[..., axis] = _spread(np.clip(widths[axis], usual / LIMIT, usual + room) - usual, axis)
    kept = tuple(slice(low, regular.shape[axis] - high) for axis, (low, high) in enumerate(trim))
    regular, shift = regular[kept], shift[kept]
    share = _share_without_folds(regular, shift, flat)
    return regular + share * shift, limited or share < 1.0


def _spread(change: np.ndarray, axis: int) -> np.ndarray:
    """How far every corner moves along `axis` for these changes of the slots' widths along it."""
    for other in range(3):
        if other != axis:
            change = _onto_planes(change, other, np.asarray(ACROSS))
    return _onto_planes(change, axis, pull_weights(REACH))


def _onto_planes(field: np.ndarray, axis: int, weights: np.ndarray) -> np.ndarray:
    """A weighted sum of the slots on either side of every plane across `axis`, the slots mirrored at both ends."""
    half = len(weights) // 2
    padded = np.pad(field, [(half, half) if other == axis else (0, 0) for other in range(3)], mode="symmetric")
    planes = field.shape[axis] + 1
    summed = np.zeros([planes if other == axis else field.shape[other] for other in range(3)])
    for tap, weight in enumerate(weights):
        summed += weight * padded[tuple(slice(tap, tap + planes) if other == axis else slice(None) for other in range(3))]
    return summed


def _share_without_folds(regular: np.ndarray, shift: np.ndarray, flat: bool) -> float:
    """The largest share of the warp, 1 when all of it, at which nothing drawn is turned inside out."""
    folded = _face_folded if flat else _block_folded
    if not folded(regular + shift):
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(EASE_TURNS):
        middle = 0.5 * (low + high)
        low, high = (low, middle) if folded(regular + middle * shift) else (middle, high)
    return low


def _face_folded(corners: np.ndarray) -> bool:
    """Whether any slot's face through its middle along z has a corner that does not turn the way a face's does."""
    face = 0.5 * (corners[:, :, 1:, :2] + corners[:, :, :-1, :2])
    quad = np.stack([face[:-1, :-1], face[1:, :-1], face[1:, 1:], face[:-1, 1:]], axis=0)
    ahead, back = np.roll(quad, -1, axis=0) - quad, quad - np.roll(quad, 1, axis=0)
    return bool((back[..., 0] * ahead[..., 1] - back[..., 1] * ahead[..., 0] <= 0.0).any())


def _block_folded(corners: np.ndarray) -> bool:
    """Whether any block has a corner whose three edges do not turn the way a block's do."""
    edges = [np.diff(corners, axis=axis) for axis in range(3)]
    for corner in range(8):
        one, two, three = [edges[axis][tuple(slice(None) if other == axis
                                             else (slice(1, None) if corner >> other & 1 else slice(None, -1))
                                             for other in range(3))] for axis in range(3)]
        turn = (one[..., 0] * (two[..., 1] * three[..., 2] - two[..., 2] * three[..., 1])
                - one[..., 1] * (two[..., 0] * three[..., 2] - two[..., 2] * three[..., 0])
                + one[..., 2] * (two[..., 0] * three[..., 1] - two[..., 1] * three[..., 0]))
        if (turn <= 0.0).any():
            return True
    return False
