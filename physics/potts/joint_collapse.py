"""Joint collapse: a joint squeezed past its limit vanishes and its energy volumes fall together.

Contract: `joint_collapse(universe, settings)` finds every living joint whose
compression (want / boxes, cell_sums.py) is at or past settings.collapse_compression
(bible rule 8: the joint gives way past its limit). Each such joint vanishes: its
boxes and every energy volume touching it become one energy volume, the lowest id
among those volumes; joints that collapse onto shared volumes join one group. The
merged volume keeps every box and every quantum of energy (box owners are relabelled;
no energy moves) and stands for the sum of its members' labels, so its wanted size is
theirs together; with the joint's energy added it is denser than they were, and it
packs from there by its box moves. Since a joint touched only energy volumes and they
touched only joints, the merged volume touches only joints: ABAB holds. (In the
checkerboard a joint touches up to six volumes, so all of them fall in: keeping any
out would leave it touching the merged volume, A on A.) The vanished cells are marked
dead and the contact and link tables are rebuilt from the old table's pairs, relabelled
and added up (table_pairs.py, merged_pairs.py, rebuild_tables.py): the boxes are never
recounted. A joint touching no
energy volume is left as it is. universe.collapses grows by the joints that collapsed.
Returns that number; with none it changes nothing and copies only per-cell arrays.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.potts.merged_pairs import merged_pairs
from physics.potts.rebuild_tables import rebuild_tables
from physics.potts.table_pairs import table_pairs
from physics.potts.settings import Settings
from physics.potts.universe import ENERGY_VOLUME, JOINT, Universe


@wp.kernel
def _relabel(owner: wp.array3d(dtype=wp.int32), remap: wp.array(dtype=wp.int32), low: wp.vec3i):
    ti, tj, tk = wp.tid()
    i = ti + low[0]
    j = tj + low[1]
    k = tk + low[2]
    cell = owner[i, j, k]
    if cell >= 0:
        owner[i, j, k] = remap[cell]


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
    low, high, faces = table_pairs(universe)
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
    merged_labels = np.bincount(remap, weights=np.where(kind == ENERGY_VOLUME, labels, 0.0),
                                minlength=universe.cell_count)
    changed = remap != np.arange(universe.cell_count)
    merged_into = np.zeros(universe.cell_count, dtype=bool)
    merged_into[remap[changed]] = True
    new_alive = np.where(changed, 0, alive).astype(np.int32)
    new_labels = np.where(merged_into, merged_labels, labels).astype(np.float32)
    device = universe.owner.device
    dims = tuple(top - bottom for bottom, top in zip(universe.low, universe.high))
    wp.launch(_relabel, dim=dims, device=device,
              inputs=[universe.owner, wp.array(remap, dtype=wp.int32, device=device), wp.vec3i(*universe.low)])
    remapped = remap[low], remap[high], faces
    joined = remapped[0] != remapped[1]
    rebuild_tables(universe, merged_pairs([tuple(part[joined] for part in remapped)]))
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
