"""Joint collapse: a joint squeezed past its limit vanishes and its energy volumes fall together.

Contract: `joint_collapse(universe, settings)` finds every living joint whose
compression (want / boxes, cell_sums.py) is at or past settings.collapse_compression
(bible rule 8: the joint gives way past its limit). Each such joint vanishes: its
boxes and every energy volume touching it become one energy volume, the lowest id
among those volumes; joints that collapse onto shared volumes join one group. The
merged volume keeps every box and every quantum of energy (box owners are relabelled;
no energy moves) and stands for the sum of its members' labels, so its wanted size
is theirs together before it packs. Since a joint touched only energy volumes and
they touched only joints, the merged volume touches only joints: ABAB holds. The
vanished cells are marked dead and the contact table is rebuilt from the boxes
(contact_table.py, its rows widened when a merged volume needs it). A joint touching no energy volume is left as it is.
universe.collapses grows by the joints that collapsed. Returns that number; with
none it changes nothing and copies only the per-cell arrays.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.potts.contact_pairs import contact_pairs
from physics.potts.contact_table import contact_table
from physics.potts.settings import Settings
from physics.potts.universe import ENERGY_VOLUME, JOINT, Universe


@wp.kernel
def _relabel(owner: wp.array3d(dtype=wp.int32), remap: wp.array(dtype=wp.int32)):
    i, j, k = wp.tid()
    cell = owner[i, j, k]
    if cell >= 0:
        owner[i, j, k] = remap[cell]


def joint_collapse(universe: Universe, settings: Settings) -> int:
    """Collapse every joint past its limit; merge its energy volumes; return how many collapsed."""
    kind = universe.kind.numpy()
    alive = universe.alive.numpy()
    boxes = universe.cell_boxes.numpy()
    compression = universe.cell_compression.numpy()
    failing = np.flatnonzero((kind == JOINT) & (alive == 1) & (boxes > 0)
                             & (compression >= settings.collapse_compression))
    if failing.size == 0:
        return 0
    owner = universe.owner.numpy()
    low, high, _ = contact_pairs(owner)
    root = np.arange(universe.cell_count)
    for joint in failing:
        volumes = np.concatenate([high[low == joint], low[high == joint]])
        volumes = volumes[kind[volumes] == ENERGY_VOLUME]
        if volumes.size == 0:
            continue
        members = [_find(root, int(cell)) for cell in volumes] + [_find(root, int(joint))]
        lowest = min(_find(root, int(cell)) for cell in volumes)
        for member in members:
            root[member] = lowest
    remap = np.array([_find(root, cell) for cell in range(universe.cell_count)], dtype=np.int32)
    labels = universe.labels.numpy()
    merged_labels = np.bincount(remap, weights=np.where(kind == ENERGY_VOLUME, labels, 0),
                                minlength=universe.cell_count).astype(np.int32)
    changed = remap != np.arange(universe.cell_count)
    new_alive = np.where(changed, 0, alive).astype(np.int32)
    new_labels = np.where(changed, labels, np.where(merged_labels > 0, merged_labels, labels)).astype(np.int32)
    wp.launch(_relabel, dim=(universe.side,) * 3, device=universe.owner.device,
              inputs=[universe.owner, wp.array(remap, dtype=wp.int32, device=universe.owner.device)])
    neighbour, faces, slots = contact_table(np.where(owner >= 0, remap[np.maximum(owner, 0)], -1),
                                            universe.cell_count, universe.slots)
    if slots == universe.slots:
        universe.neighbour.assign(neighbour)
        universe.faces.assign(faces)
    else:
        device = universe.owner.device
        universe.neighbour = wp.array(neighbour, dtype=wp.int32, device=device)
        universe.faces = wp.array(faces, dtype=wp.int32, device=device)
        universe.slots = slots
    universe.alive.assign(new_alive)
    universe.labels.assign(new_labels)
    collapsed = int(np.sum(changed[failing]))
    universe.collapses += collapsed
    return collapsed


def _find(root: np.ndarray, cell: int) -> int:
    while root[cell] != cell:
        root[cell] = root[root[cell]]
        cell = int(root[cell])
    return cell
