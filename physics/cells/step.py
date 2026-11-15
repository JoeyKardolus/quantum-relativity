"""Step: one engine step; the rules run on the grid, the topology changes only by its two events, the corners follow.

Contract: `step(engine, settings)` is pure: it returns a new Engine and leaves the
old one as it was. Deterministic: the same engine and settings give the same result.
  1. pull      p = pull_potential(grid); each energy volume exerts attraction * p / 2
               (Grid.pull); joint (a, b) feels squeeze s = pull_a + pull_b (rule 7)
  2. swing     every joint swings one step under push and squeeze (swing_joints.py,
               the push steep as a joint thins, joint_push.py; rest thickness from
               the energy volumes' measured sizes); no damping (rule 5). The thickness
               is the joint cell's own: its size aims at thickness times its area
  3. leak      energy leaks along seams (the seam cells where joint cells meet) as
               open as the joints at the start of the step let them be
               (seam_openness.py, leak.py); quanta do not move
  4. collapse  joints thinner than push_limit times their rest thickness give way,
               the thinnest (by thickness / rest) first: the joint cell vanishes and
               the skeleton face it sits on gives way along its shortest edge, drawn
               into one corner (collapse_edge.py), the next shortest when that is
               refused (it would invert a tetrahedron, break the mesh, or move the
               outer surface), none when all three are; the energy volumes on that
               edge, the joint's two among them, fall in and hand their energy, quanta
               and charge to the energy volumes across their other faces, which fall
               into each other's touch, exactly. A giving joint whose face an earlier
               collapse this step removed is skipped.
  5. split     every energy volume whose size (at the start of the step) is past
               split_size times the start's mean energy volume (grown too big, rule 8)
               is cut into four by the 1-4 move on its skeleton tetrahedron
               (split_cell.py); energy, quanta and charge shared by volume (the last
               child takes the rest, so the sum is exact). The six inner triangles of
               the tear become new joint cells at rest thickness (from the children's
               measured sizes), rate 0
  6. corners   the skeleton first: RELAX_PASSES (5) rounds of relax_skeleton() with each
               tetrahedron aiming at its volume times rest / size of its energy volume,
               rest = rest_volume(energy, quanta) (rule 4); the cells' corners ride on
               that move (each where place_corners.py puts it for the moved skeleton and
               the new thicknesses, plus how far it stood from where it was put before),
               a corner whose cell would shrink below a fifth of its placed size placed
               afresh (woven_corners.py); then FINE_PASSES (3) rounds of relax_corners()
               toward each cell's target (cell_targets.py): energy volumes held at the
               size they rode to, joint cells thickness times area, seam and corner
               cells the size the joints around them give. Every corner moves; no
               neighbour changes. The open cube's outside corners have nothing pushing
               back, so the grid grows or shrinks there. Grid.volume is then each
               energy volume's size.
After 4 or 5 the cells are cut again from the skeleton (joint_cells.py), each joint
keeping the thickness and rate of the joint on its face before (joint_carry.py), and
corners no tetrahedron uses are dropped. A corner keeps its place unless a collapse or
split touched its skeleton corner (every corner of a tetrahedron the event changed);
those are placed afresh (woven_corners.py, place_corners.py). A step with neither event
compares the energy volumes each joint cell touches with the grid's joints and adds any
difference to neighbour_changes (0 by construction). Total energy is changed by no
rule: the leak moves it, collapse hands it on, split shares it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from physics.cells.cell_complex import KIND_JOINT
from physics.cells.cell_sizes import cell_sizes
from physics.cells.cell_targets import cell_targets
from physics.cells.cell_volumes import cell_volumes
from physics.cells.collapse_edge import collapse_edge
from physics.cells.drop_loose_corners import drop_loose_corners
from physics.cells.engine import Engine
from physics.cells.engine_settings import EngineSettings
from physics.cells.grid import Grid
from physics.cells.joint_carry import joint_carry
from physics.cells.joint_cells import joint_cells
from physics.cells.joint_offsets import joint_offsets
from physics.cells.joint_rest import joint_rest
from physics.cells.joint_squeeze import joint_squeeze
from physics.cells.keyed_corners import keyed_corners
from physics.cells.leak import leak
from physics.cells.mesh_seams import mesh_seams
from physics.cells.place_corners import place_corners
from physics.cells.pull_potential import pull_potential
from physics.cells.relax_corners import relax_corners
from physics.cells.relax_skeleton import relax_skeleton
from physics.cells.rest_volume import rest_volume
from physics.cells.seam_openness import seam_openness
from physics.cells.split_cell import split_cell
from physics.cells.swing_joints import swing_joints
from physics.cells.woven_corners import woven_corners

RELAX_PASSES = 5
FINE_PASSES = 3


@dataclass
class _Work:
    """The mutable working copy of one step's skeleton and per-cell values."""

    corners: np.ndarray
    tets: np.ndarray
    values: np.ndarray
    origin: np.ndarray
    corner_map: np.ndarray
    touched: list = field(default_factory=list)


def step(engine: Engine, settings: EngineSettings) -> Engine:
    """Pull, swing, leak, then collapse and split, then the corners follow the new targets."""
    grid = engine.grid
    exerted = settings.attraction * pull_potential(grid) / 2
    heads, tails = grid.joints[:, 0], grid.joints[:, 1]
    rest = joint_rest(grid.volume, grid.joints, settings.joint_rest)
    thickness, rate = swing_joints(grid.thickness, grid.rate, rest, exerted[heads] + exerted[tails],
                                   settings.joint_push, settings.push_power)
    energy = leak(grid.joints, grid.seams, grid.energy, exerted,
                  seam_openness(grid, settings), settings.leak)
    work = _Work(corners=engine.skeleton_corners, tets=engine.skeleton_tets,
                 values=np.column_stack([energy, grid.quanta, grid.charge]),
                 origin=np.arange(grid.cell_count), corner_map=np.arange(len(engine.skeleton_corners)))
    giving = np.flatnonzero(thickness < settings.push_limit * rest)
    collapses = _collapse(work, engine, giving[np.argsort(thickness[giving] / rest[giving], kind="stable")])
    splits = _split(work, grid.volume, settings.split_size * engine.start_mean_volume)
    if not (collapses or splits):
        return _moved(engine, exerted, energy, thickness, rate, settings)
    return _rewoven(engine, work, exerted, thickness, rate, settings, collapses, splits)


def _collapse(work: _Work, engine: Engine, giving: np.ndarray) -> int:
    """Collapse each giving joint's skeleton face along its shortest workable edge; returns how many collapsed."""
    woven = engine.woven
    face_of_joint = np.full(len(woven.joints), -1, dtype=np.int64)
    inside = woven.skeleton_face_joint >= 0
    face_of_joint[woven.skeleton_face_joint[inside]] = np.flatnonzero(inside)
    done = 0
    for joint in giving:
        face = work.corner_map[woven.skeleton_faces[face_of_joint[joint]]]
        if len(set(face.tolist())) < 3 or (np.isin(work.tets, face).sum(axis=1) == 3).sum() != 2:
            continue
        edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
        lengths = [float(np.linalg.norm(work.corners[one] - work.corners[two])) for one, two in edges]
        for index in np.argsort(lengths, kind="stable"):
            result = collapse_edge(work.corners, work.tets, work.values, edges[index])
            if result is None:
                continue
            work.corners, work.tets, work.values, kept, gone = result
            kept_corner = edges[index][0] if edges[index][1] == gone else edges[index][1]
            work.corner_map[work.corner_map == gone] = kept_corner
            work.touched = [kept_corner if corner == gone else corner for corner in work.touched]
            work.touched += np.unique(work.tets[(work.tets == kept_corner).any(axis=1)]).tolist()
            work.origin = work.origin[kept]
            done += 1
            break
    return done


def _split(work: _Work, volume: np.ndarray, trigger: float) -> int:
    """Split every energy volume measured past the trigger by the 1-4 move on its skeleton tetrahedron."""
    big = np.flatnonzero(volume[work.origin] > trigger)
    for cell in big:
        work.touched += work.tets[cell].tolist() + [len(work.corners)]
        work.corners, work.tets, children, shares = split_cell(work.corners, work.tets, int(cell))
        parent = work.values[cell].copy()
        work.values = np.vstack([work.values, np.zeros((3, work.values.shape[1]))])
        work.values[children[:3]] = shares[:3, None] * parent
        work.values[children[3]] = parent - work.values[children[:3]].sum(axis=0)
        work.origin = np.append(work.origin, np.repeat(work.origin[cell], 3))
    return len(big)


def _moved(engine: Engine, exerted: np.ndarray, energy: np.ndarray, thickness: np.ndarray,
           rate: np.ndarray, settings: EngineSettings) -> Engine:
    """The next engine when the topology did not change: only the corners move."""
    grid, woven = engine.grid, engine.woven
    skeleton, corners = _relaxed(woven, engine.skeleton_corners, engine.corners,
                                 rest_volume(energy, grid.quanta), grid.thickness, thickness, settings)
    sizes = cell_sizes(corners, woven.faces, woven.face_cells, len(woven.cell_kind))
    after = _finished(replace(grid, energy=energy, volume=sizes[:grid.cell_count], thickness=thickness,
                              rate=rate, pull=exerted), settings)
    changes = _pair_changes(grid.joints, _touching_pairs(woven, grid.cell_count))
    return replace(engine, grid=after, corners=corners,
                   skeleton_corners=skeleton,
                   neighbour_changes=engine.neighbour_changes + changes)


def _rewoven(engine: Engine, work: _Work, exerted: np.ndarray, thickness: np.ndarray, rate: np.ndarray,
             settings: EngineSettings, collapses: int, splits: int) -> Engine:
    """The next engine after a collapse or split: the cells cut again from the skeleton, corners kept where untouched."""
    skeleton, renumber = drop_loose_corners(work.corners, work.tets)
    woven = joint_cells(renumber[work.tets])
    old = engine.woven
    joint_thickness, joint_rate, fresh = joint_carry(
        renumber[work.corner_map[old.skeleton_faces]], old.skeleton_face_joint, thickness, rate,
        woven.skeleton_faces, woven.skeleton_face_joint, len(woven.joints))
    count = len(woven.skeleton_tets)
    joint_thickness[fresh] = joint_rest(cell_volumes(skeleton, woven.skeleton_tets), woven.joints[fresh],
                                        settings.joint_rest)
    touched = np.zeros(len(skeleton), dtype=bool)
    touched[renumber[np.array(work.touched, dtype=np.int64)]] = True
    placed = woven_corners(woven, _built(woven, skeleton, joint_thickness, settings),
                           keyed_corners(woven.keys, _renumbered_keys(old.keys, renumber[work.corner_map]),
                                         engine.corners), touched)
    volume = cell_sizes(placed, woven.faces, woven.face_cells, len(woven.cell_kind))[:count]
    joint_thickness[fresh] = joint_rest(volume, woven.joints[fresh], settings.joint_rest)
    energy, quanta, charge = (work.values[:, column].copy() for column in range(3))
    skeleton, corners = _relaxed(woven, skeleton, placed, rest_volume(energy, quanta), joint_thickness,
                                 joint_thickness, settings)
    seams = mesh_seams(woven.face_cells, woven.face_joint, woven.cell_kind)
    after = _finished(Grid(cell_count=count, joints=woven.joints, energy=energy, quanta=quanta,
                           volume=cell_sizes(corners, woven.faces, woven.face_cells,
                                             len(woven.cell_kind))[:count],
                           thickness=joint_thickness, rate=joint_rate, squeeze=np.zeros(len(woven.joints)),
                           pull=exerted[work.origin], charge=charge, seams=seams,
                           seam_open=np.ones(int(seams[-1, 0]) + 1 if len(seams) else 0)), settings)
    return Engine(grid=after, corners=corners, woven=woven,
                  skeleton_corners=skeleton,
                  collapses=engine.collapses + collapses, splits=engine.splits + splits,
                  start_space=engine.start_space, start_mean_volume=engine.start_mean_volume,
                  neighbour_changes=engine.neighbour_changes)


def _relaxed(woven, skeleton: np.ndarray, corners: np.ndarray, rest: np.ndarray, before: np.ndarray,
             thickness: np.ndarray, settings: EngineSettings) -> tuple[np.ndarray, np.ndarray]:
    """Skeleton and corners: the skeleton's RELAX_PASSES, the corners riding on it, then FINE_PASSES of the cells."""
    count = len(woven.cell_kind)
    tets = woven.skeleton_tets
    sizes = cell_sizes(corners, woven.faces, woven.face_cells, count)
    moved = relax_skeleton(skeleton, tets, cell_volumes(skeleton, tets) * rest / sizes[:len(tets)], RELAX_PASSES)
    was = _built(woven, skeleton, before, settings)
    now = _built(woven, moved, thickness, settings)
    ridden = woven_corners(woven, now, now + (corners - was), np.zeros(len(moved), dtype=bool))
    built = cell_sizes(now, woven.faces, woven.face_cells, count)
    held = cell_sizes(ridden, woven.faces, woven.face_cells, count)[:len(tets)]
    fine = cell_targets(held, thickness, ridden, woven.faces, woven.face_cells, woven.cell_kind, built)
    return moved, relax_corners(ridden, woven.faces, woven.face_cells, fine, FINE_PASSES)


def _built(woven, skeleton: np.ndarray, thickness: np.ndarray, settings: EngineSettings) -> np.ndarray:
    """Every corner where place_corners.py puts it for this skeleton and these joint thicknesses."""
    offsets = joint_offsets(skeleton, woven.skeleton_tets, woven.slot_joint, thickness, settings.joint_rest)
    return place_corners(woven.keys, skeleton, woven.skeleton_tets, offsets)


def _renumbered_keys(keys: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """Old corner keys in the new skeleton's corner numbers, their sorted columns sorted again."""
    renamed = keys.copy()
    present = renamed[:, 1:] >= 0
    renamed[:, 1:][present] = mapping[renamed[:, 1:][present]]
    copies = renamed[:, 0] == 0
    renamed[copies, 2:5] = np.sort(renamed[copies, 2:5], axis=1)
    on_face = renamed[:, 0] == 1
    renamed[on_face, 2:4] = np.sort(renamed[on_face, 2:4], axis=1)
    return renamed


def _touching_pairs(woven, count: int) -> np.ndarray:
    """Per joint cell the two energy volumes it touches by its big sides, sorted."""
    first, second = woven.face_cells[:, 0], woven.face_cells[:, 1]
    big_side = (first < count) & (second >= 0) & (woven.cell_kind[np.maximum(second, 0)] == KIND_JOINT)
    pairs = np.unique(np.column_stack([second[big_side] - count, first[big_side]]), axis=0).reshape(-1, 2)
    joint_count = int(pairs[:, 0].max()) + 1 if len(pairs) else 0
    low = np.full(joint_count, np.iinfo(np.int64).max)
    high = np.full(joint_count, -1)
    np.minimum.at(low, pairs[:, 0], pairs[:, 1])
    np.maximum.at(high, pairs[:, 0], pairs[:, 1])
    two = np.bincount(pairs[:, 0], minlength=joint_count) == 2
    return np.column_stack([np.where(two, low, -1), np.where(two, high, -1)])


def _finished(grid: Grid, settings: EngineSettings) -> Grid:
    """The grid with its derived squeeze and seam openness set."""
    with_squeeze = replace(grid, squeeze=joint_squeeze(grid, settings))
    return replace(with_squeeze, seam_open=seam_openness(with_squeeze, settings))


def _pair_changes(before: np.ndarray, after: np.ndarray) -> int:
    """Neighbour pairs in one joint list and not the other."""
    first = {tuple(pair) for pair in before.tolist()}
    second = {tuple(pair) for pair in after.tolist()}
    return len(first ^ second)
