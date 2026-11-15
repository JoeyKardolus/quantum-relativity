"""Volume split: an energy volume grown too big tears in two, and the tear becomes a new joint (rule 8).

Contract: `volume_split(universe, settings)` finds every living energy volume holding
more than settings.split_boxes * base_boxes boxes (the trigger is size, not density: a
split keeps density, so a density trigger would split for ever) and tears each across
its longest extent: the box layer at the median coordinate along that axis (or, when
that breaks a rule, the nearest of the layers up to 2 boxes off it that does not)
becomes a new joint, the boxes beyond it a new energy volume, the volume keeps the
boxes before it. The tear is one box thick, so the two halves are two boxes apart
along the axis and never touch, not even at an edge or corner: each half stays wrapped
in joints (bible foundation 1), and the tear meets the joints round the volume in the
web (joints may touch joints; the seams are where they meet). The torn volume's labels
are shared by box count among the halves and the tear, so each piece keeps the
volume's density and its wanted size is the boxes it holds: a split never changes the
space a region fills (rule 4). No energy moves (boxes are relabelled, psi untouched),
so total energy stays exact. When no flat layer keeps the rules (a volume wrapped round
a joint, so a layer falls in pieces), the tear closes on itself: the volume's core (the
boxes deeper than some depth, as near half of it as the depths allow) becomes the new
energy volume, the shell of boxes round the core (every box of the volume touching the
core by a face, edge or corner) the new joint. A tear is worked out and checked in the
volume's bounding box widened by CROP (3) boxes: no two energy volumes touching (face,
edge or corner, touch_pairs.py) and the halves and the tear each one piece; a tear
that fails is not made and counted in universe.scratch["split_refused"]; that volume
is tried again only once it holds RETRY (9/8) times the boxes it held then. New cells
get new ids (add_cells), universe.splits grows by the tears made, and that number is
returned. Call cell_sums after a split.
"""
from __future__ import annotations

import numpy as np
import warp as wp
from scipy import ndimage

from physics.potts.region_host import region_host
from physics.potts.settings import Settings
from physics.potts.touch_pairs import touch_pairs
from physics.potts.universe import ENERGY_VOLUME, JOINT, Universe

CROP = 3
CUT_SHIFTS = (0, -1, 1, -2, 2)
RETRY = 1.125


@wp.kernel
def _scatter(region: wp.array3d(dtype=wp.int32), low: wp.vec3i, owner: wp.array3d(dtype=wp.int32)):
    i, j, k = wp.tid()
    owner[i + low[0], j + low[1], k + low[2]] = region[i, j, k]


def volume_split(universe: Universe, settings: Settings) -> int:
    """Tear every energy volume past the split size in two around a new joint; return how many tore."""
    kind = universe.kind.numpy()
    alive = universe.alive.numpy()
    boxes = universe.cell_boxes.numpy()
    torn = np.flatnonzero((kind == ENERGY_VOLUME) & (alive == 1) & (boxes > settings.split_boxes * universe.base_boxes))
    tried = universe.scratch.setdefault("split_tried", {})
    torn = [volume for volume in torn if boxes[volume] > RETRY * tried.get(int(volume), 0)]
    if not torn:
        return 0
    owner = region_host(universe, universe.owner)
    kinds = list(kind)
    made = []
    refused = 0
    boxes_of = _boxes_of(owner, torn)
    for volume in torn:
        upper, tear = len(kinds), len(kinds) + 1
        places = boxes_of[int(volume)]
        crop = tuple(slice(max(0, int(bottom) - CROP), int(top) + 1 + CROP)
                     for bottom, top in zip(places.min(axis=0), places.max(axis=0)))
        trial = _tear(owner[crop], int(volume), upper, tear, np.array(kinds + [ENERGY_VOLUME, JOINT]))
        if trial is None:
            refused += 1
            tried[int(volume)] = int(boxes[volume])
            continue
        owner[crop] = trial
        kinds += [ENERGY_VOLUME, JOINT]
        made.append((int(volume), upper, tear))
    universe.scratch["split_refused"] = universe.scratch.get("split_refused", 0) + refused
    if not made:
        return 0
    universe.add_cells(kinds[universe.cell_count:])
    device = universe.owner.device
    wp.launch(_scatter, dim=owner.shape, device=device,
              inputs=[wp.array(owner, dtype=wp.int32, device=device), wp.vec3i(*universe.low), universe.owner])
    counts = np.bincount(owner[owner >= 0], minlength=universe.cell_count)
    labels = universe.labels.numpy()
    previous = universe.previous_boxes.numpy()
    for volume, upper, tear in made:
        pieces = counts[volume] + counts[upper] + counts[tear]
        whole = labels[volume]
        for piece in (volume, upper, tear):
            labels[piece] = whole * counts[piece] / pieces
            previous[piece] = counts[piece]
    universe.labels.assign(labels.astype(np.float32))
    universe.previous_boxes.assign(previous.astype(np.float32))
    universe.splits += len(made)
    return len(made)


def _boxes_of(owner: np.ndarray, cells: list) -> dict:
    """Each listed cell's box indices, found in one pass over the owners."""
    places = np.argwhere(np.isin(owner, cells))
    held = owner[tuple(places.T)]
    order = np.argsort(held, kind="stable")
    places, held = places[order], held[order]
    starts = np.searchsorted(held, cells)
    ends = np.searchsorted(held, cells, side="right")
    return {int(cell): places[start:end] for cell, start, end in zip(cells, starts, ends)}


def _tear(owner: np.ndarray, volume: int, upper: int, tear: int, kinds: np.ndarray) -> np.ndarray | None:
    """The owners after tearing `volume` at the first cut that keeps every rule, or None."""
    places = np.argwhere(owner == volume)
    extent = places.max(axis=0) - places.min(axis=0)
    for axis in np.argsort(-extent, kind="stable")[:1]:
        middle = int(np.median(places[:, axis]))
        for shift in CUT_SHIFTS:
            trial = _cut(owner, places, int(axis), middle + shift, volume, upper, tear, kinds)
            if trial is not None:
                return trial
    return _enclose(owner, volume, upper, tear, kinds)


def _enclose(owner: np.ndarray, volume: int, upper: int, tear: int, kinds: np.ndarray) -> np.ndarray | None:
    """The owners after a closed tear: the volume's core becomes `upper`, a shell round it the tear."""
    inside = owner == volume
    depth = ndimage.distance_transform_cdt(np.pad(inside, 1), metric="taxicab")[1:-1, 1:-1, 1:-1]
    whole = int(inside.sum())
    best, best_gap = None, None
    for level in range(2, int(depth.max())):
        cores, count = ndimage.label(depth > level)
        if count == 0:
            break
        sizes = np.bincount(cores.ravel())[1:]
        core = cores == 1 + int(np.argmax(sizes))
        shell = inside & ~core & ndimage.binary_dilation(core, structure=np.ones((3, 3, 3), dtype=bool))
        trial = owner.copy()
        trial[core] = upper
        trial[shell] = tear
        gap = abs(int(core.sum()) - 0.5 * whole)
        if (best_gap is None or gap < best_gap) and _keeps_rules(trial, (volume, upper, tear), kinds):
            best, best_gap = trial, gap
    return best


def _keeps_rules(trial: np.ndarray, cells: tuple, kinds: np.ndarray) -> bool:
    """No two energy volumes touching (face, edge or corner) and every listed cell one piece."""
    low, high = touch_pairs(trial)
    real = low >= 0
    if np.any((kinds[low[real]] == ENERGY_VOLUME) & (kinds[high[real]] == ENERGY_VOLUME)):
        return False
    return all(ndimage.label(trial == cell)[1] == 1 for cell in cells)


def _cut(owner: np.ndarray, places: np.ndarray, axis: int, cut: int, volume: int, upper: int, tear: int,
         kinds: np.ndarray) -> np.ndarray | None:
    """The owners after tearing `volume` across `axis` at `cut`, or None when it breaks a rule."""
    if cut <= places[:, axis].min() or cut >= places[:, axis].max():
        return None
    trial = owner.copy()
    beyond = places[places[:, axis] > cut]
    layer = places[places[:, axis] == cut]
    trial[tuple(beyond.T)] = upper
    trial[tuple(layer.T)] = tear
    if not _keeps_rules(trial, (volume, upper, tear), kinds):
        return None
    return trial
