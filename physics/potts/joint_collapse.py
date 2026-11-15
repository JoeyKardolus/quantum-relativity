"""Joint collapse: a joint squeezed past its limit vanishes and its energy volumes fall together.

Contract: `joint_collapse(universe, settings)` finds every living joint whose
compression (want / boxes, cell_sums.py) is at or past settings.collapse_compression
(bible rule 8: the joint gives way past its limit). Each such joint vanishes: its
boxes and every energy volume touching it (across a face, an edge or a corner,
touch_pairs.py) become one energy volume, the lowest id among
those volumes; joints that collapse onto shared volumes join one group. The merged
volume keeps every box and every bit of energy (box owners are relabelled; the energy
fluid's psi is untouched) and stands for the sum of its members' labels, so its wanted
size is theirs together; with the joint's energy added it is denser than they were,
and it packs from there by its box moves. Every energy volume near the joint fell in,
and each member was wrapped in joints, so the merged volume is wrapped in joints too:
no other energy volume and no empty box lies among its boxes' 26 neighbours. A joint
touching empty space (on the open edge) does not collapse: its boxes would leave the
merged volume touching empty space, and with nothing outside pushing back it can grow
into the empty boxes instead (box_moves.py). Nor does a joint whose group would not be
one face-connected piece (a joint touching its energy volumes only at edges or
corners: they could not hand energy to each other through faces); it collapses once
its moves have brought them to its faces. Joints are taken in id order, each added
to the groups only if its group stays one piece. The vanished cells are marked dead. A joint touching no energy volume
is left as it is. universe.collapses grows by the joints that collapsed. Returns that
number; with none it changes nothing and copies only per-cell arrays.
"""
from __future__ import annotations

import numpy as np
import warp as wp
from scipy import ndimage

from physics.potts.region_host import region_host
from physics.potts.settings import Settings
from physics.potts.touch_pairs import touch_pairs
from physics.potts.universe import ENERGY_VOLUME, JOINT, Universe


@wp.kernel
def _scatter(region: wp.array3d(dtype=wp.int32), low: wp.vec3i, owner: wp.array3d(dtype=wp.int32)):
    i, j, k = wp.tid()
    owner[i + low[0], j + low[1], k + low[2]] = region[i, j, k]


def joint_collapse(universe: Universe, settings: Settings) -> int:
    """Collapse every joint past its limit; its energy volumes fall together; return how many collapsed."""
    kind = universe.kind.numpy()
    alive = universe.alive.numpy()
    boxes = universe.cell_boxes.numpy()
    compression = universe.cell_compression.numpy()
    failing = np.flatnonzero((kind == JOINT) & (alive == 1) & (boxes > 0)
                             & (compression >= settings.collapse_compression))
    if failing.size == 0:
        return 0
    owner = region_host(universe, universe.owner)
    low, high = touch_pairs(owner)
    open_edge = set(high[low < 0].tolist())
    real = low >= 0
    low, high = low[real], high[real]
    root = np.arange(universe.cell_count)
    for joint in failing:
        if int(joint) in open_edge:
            continue
        volumes = np.concatenate([high[low == joint], low[high == joint]])
        volumes = volumes[kind[volumes] == ENERGY_VOLUME]
        if volumes.size == 0:
            continue
        trial = root.copy()
        members = [_find(trial, int(cell)) for cell in volumes] + [_find(trial, int(joint))]
        lowest = min(_find(trial, int(cell)) for cell in volumes)
        for member in members:
            trial[member] = lowest
        if _one_piece(owner, trial, lowest):
            root = trial
    remap = np.array([_find(root, cell) for cell in range(universe.cell_count)], dtype=np.int32)
    labels = universe.labels.numpy()
    merged_labels = np.bincount(remap, weights=np.where(kind == ENERGY_VOLUME, labels, 0.0),
                                minlength=universe.cell_count)
    changed = remap != np.arange(universe.cell_count)
    collapsed = int(np.sum(changed[failing]))
    if collapsed == 0:
        return 0
    merged_into = np.zeros(universe.cell_count, dtype=bool)
    merged_into[remap[changed]] = True
    new_alive = np.where(changed, 0, alive).astype(np.int32)
    new_labels = np.where(merged_into, merged_labels, labels).astype(np.float32)
    relabelled = np.where(owner >= 0, remap[np.maximum(owner, 0)], -1).astype(np.int32)
    device = universe.owner.device
    wp.launch(_scatter, dim=relabelled.shape, device=device,
              inputs=[wp.array(relabelled, dtype=wp.int32, device=device), wp.vec3i(*universe.low), universe.owner])
    universe.alive.assign(new_alive)
    universe.labels.assign(new_labels)
    universe.collapses += collapsed
    return collapsed


def _one_piece(owner: np.ndarray, root: np.ndarray, group: int) -> bool:
    """Whether every cell whose root is `group` together makes one face-connected piece of boxes."""
    members = [cell for cell in range(root.size) if _find(root, cell) == group]
    inside = np.isin(owner, members)
    places = np.argwhere(inside)
    crop = tuple(slice(int(bottom), int(top) + 1) for bottom, top in zip(places.min(axis=0), places.max(axis=0)))
    return ndimage.label(inside[crop])[1] == 1


def _find(root: np.ndarray, cell: int) -> int:
    while root[cell] != cell:
        root[cell] = root[root[cell]]
        cell = int(root[cell])
    return cell
