"""Step: one engine step; the rules run on the grid, the topology changes only by its two events, the corners follow.

Contract: `step(engine, settings)` is pure: it returns a new Engine and leaves the
old one as it was. Deterministic: the same engine and settings give the same result.
  1. pull      p = pull_potential(grid); each energy volume exerts attraction * p / 2
               (Grid.pull); joint (a, b) feels squeeze s = pull_a + pull_b (rule 7)
  2. swing     every joint swings one step under push and squeeze (swing_joints.py,
               the push steep as a joint thins, joint_push.py; rest thickness from
               the cells' measured volumes at the start); no damping (rule 5)
  3. leak      energy leaks along seams as open as the joints at the start of the step
               let them be (seam_openness.py, leak.py); quanta do not move
  4. collapse  joints thinner than push_limit times their rest thickness give way,
               the thinnest (by thickness / rest) first: the joint's face's shortest
               edge is drawn into one corner (collapse_edge.py), the next shortest
               when that is refused (it would invert a tetrahedron, break the mesh,
               or move the outer surface), none when all three are; the tetrahedra
               on that edge fall in and hand their energy, quanta and charge to the
               tetrahedra across their other faces, exactly. A giving joint whose
               face an earlier collapse this step removed is skipped.
  5. split     every energy volume whose measured volume is past split_size times
               the start's mean volume (grown too big, rule 8) is cut into four by
               the 1-4 move (split_cell.py); energy, quanta and charge shared by
               volume (the last child takes the rest, so the sum is exact), so each
               child keeps the density and has a quarter of the size and of the
               rest volume: below the trigger unless the parent was past four times
               it. The six inner triangles of the tear are new joints at rest
               thickness (from the children's measured volumes), rate 0
  6. corners   RELAX_PASSES (5) rounds of relax_corners() toward each tetrahedron's
               target, rest_volume(energy, quanta) plus half its joint walls
               (cell_targets.py). Corners only move: no neighbour changes. The open
               cube's outside corners have nothing pushing back, so the grid grows
               or shrinks there. Grid.volume is then each measured volume.
After 4 or 5 the faces, joints and seams are rebuilt from the tetrahedra
(tet_faces.py, face_joints.py, mesh_seams.py), each joint keeping the thickness and
rate of the joint on its face before (joint_carry.py), and corners no tetrahedron
uses are dropped. A step with neither event compares the joints of its tetrahedra
with the grid's and adds any difference to neighbour_changes (0 by construction).
Total energy is changed by no rule: the leak moves it, collapse hands it on, split
shares it.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from physics.cells.cell_targets import cell_targets
from physics.cells.cell_volumes import cell_volumes
from physics.cells.collapse_edge import collapse_edge
from physics.cells.drop_loose_corners import drop_loose_corners
from physics.cells.engine import Engine
from physics.cells.engine_settings import EngineSettings
from physics.cells.face_joints import face_joints
from physics.cells.grid import Grid
from physics.cells.joint_carry import joint_carry
from physics.cells.joint_rest import joint_rest
from physics.cells.joint_squeeze import joint_squeeze
from physics.cells.leak import leak
from physics.cells.mesh_seams import mesh_seams
from physics.cells.pull_potential import pull_potential
from physics.cells.relax_corners import relax_corners
from physics.cells.rest_volume import rest_volume
from physics.cells.seam_openness import seam_openness
from physics.cells.split_cell import split_cell
from physics.cells.swing_joints import swing_joints
from physics.cells.tet_faces import tet_faces

RELAX_PASSES = 5


@dataclass
class _Work:
    """The mutable working copy of one step's tetrahedra and per-cell values."""

    corners: np.ndarray
    tets: np.ndarray
    values: np.ndarray
    origin: np.ndarray
    corner_map: np.ndarray


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
    work = _Work(corners=engine.corners, tets=engine.tets,
                 values=np.column_stack([energy, grid.quanta, grid.charge]),
                 origin=np.arange(grid.cell_count), corner_map=np.arange(len(engine.corners)))
    giving = np.flatnonzero(thickness < settings.push_limit * rest)
    collapses = _collapse(work, engine, giving[np.argsort(thickness[giving] / rest[giving], kind="stable")])
    splits = _split(work, settings.split_size * engine.start_mean_volume)
    if not (collapses or splits):
        return _moved(engine, exerted, energy, thickness, rate, settings)
    return _rewoven(engine, work, exerted, thickness, rate, settings, collapses, splits)


def _collapse(work: _Work, engine: Engine, giving: np.ndarray) -> int:
    """Collapse each giving joint's face along its shortest workable edge; returns how many collapsed."""
    face_of_joint = np.full(len(engine.grid.joints), -1, dtype=np.int64)
    inside = engine.face_joint >= 0
    face_of_joint[engine.face_joint[inside]] = np.flatnonzero(inside)
    done = 0
    for joint in giving:
        face = work.corner_map[engine.faces[face_of_joint[joint]]]
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
            work.origin = work.origin[kept]
            done += 1
            break
    return done


def _split(work: _Work, trigger: float) -> int:
    """Split every energy volume measured past the trigger volume by the 1-4 move; returns how many split."""
    big = np.flatnonzero(cell_volumes(work.corners, work.tets) > trigger)
    for cell in big:
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
    grid = engine.grid
    targets = cell_targets(rest_volume(energy, grid.quanta), thickness, engine.faces, engine.face_cells,
                           engine.face_joint, engine.corners)
    corners = relax_corners(engine.corners, engine.tets, targets, RELAX_PASSES)
    after = _finished(replace(grid, energy=energy, volume=cell_volumes(corners, engine.tets),
                              thickness=thickness, rate=rate, pull=exerted), settings)
    rejoined = face_joints(tet_faces(engine.tets)[1])[0]
    return replace(engine, grid=after, corners=corners,
                   neighbour_changes=engine.neighbour_changes + _pair_changes(grid.joints, rejoined))


def _rewoven(engine: Engine, work: _Work, exerted: np.ndarray, thickness: np.ndarray, rate: np.ndarray,
             settings: EngineSettings, collapses: int, splits: int) -> Engine:
    """The next engine after a collapse or split: faces, joints and seams rebuilt from the tetrahedra."""
    faces, face_cells = tet_faces(work.tets)
    joints, face_joint = face_joints(face_cells)
    joint_thickness, joint_rate, fresh = joint_carry(work.corner_map[engine.faces], engine.face_joint,
                                                     thickness, rate, faces, face_joint, len(joints))
    joint_thickness[fresh] = joint_rest(cell_volumes(work.corners, work.tets), joints[fresh],
                                        settings.joint_rest)
    corners, renumber = drop_loose_corners(work.corners, work.tets)
    tets, faces = renumber[work.tets], renumber[faces]
    energy, quanta, charge = (work.values[:, column].copy() for column in range(3))
    targets = cell_targets(rest_volume(energy, quanta), joint_thickness, faces, face_cells, face_joint,
                           corners)
    corners = relax_corners(corners, tets, targets, RELAX_PASSES)
    seams = mesh_seams(faces, face_joint)
    after = _finished(Grid(cell_count=len(tets), joints=joints, energy=energy, quanta=quanta,
                           volume=cell_volumes(corners, tets), thickness=joint_thickness,
                           rate=joint_rate, squeeze=np.zeros(len(joints)), pull=exerted[work.origin],
                           charge=charge, seams=seams,
                           seam_open=np.ones(int(seams[-1, 0]) + 1 if len(seams) else 0)), settings)
    return Engine(grid=after, corners=corners, tets=tets, faces=faces, face_cells=face_cells,
                  face_joint=face_joint, collapses=engine.collapses + collapses,
                  splits=engine.splits + splits, start_space=engine.start_space,
                  start_mean_volume=engine.start_mean_volume, neighbour_changes=engine.neighbour_changes)


def _finished(grid: Grid, settings: EngineSettings) -> Grid:
    """The grid with its derived squeeze and seam openness set."""
    with_squeeze = replace(grid, squeeze=joint_squeeze(grid, settings))
    return replace(with_squeeze, seam_open=seam_openness(with_squeeze, settings))


def _pair_changes(before: np.ndarray, after: np.ndarray) -> int:
    """Neighbour pairs in one joint list and not the other."""
    first = {tuple(pair) for pair in before.tolist()}
    second = {tuple(pair) for pair in after.tolist()}
    return len(first ^ second)
