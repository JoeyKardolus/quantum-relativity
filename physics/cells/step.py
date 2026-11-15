"""Step: one engine step; the rules run on the grid, the topology changes only by its two events, the corners follow.

Contract: `step(engine, settings)` is pure: it returns a new Engine and leaves the
old one as it was. Deterministic: the same engine and settings give the same result.
  1. pull      p = pull_potential(grid); each energy volume exerts attraction * p / 2
               (Grid.pull); joint (a, b) feels squeeze s = pull_a + pull_b (rule 7)
  2. swing     every joint swings one step under push and squeeze (swing_joints.py,
               the push steep as a joint thins, joint_push.py;
               rest thickness from the cells' measured volumes at the start); no
               damping (rule 5)
  3. leak      energy leaks along seams as open as the joints at the start of the step
               let them be (seam_openness.py, leak.py)
  4. collapse  joints thinner than push_limit times their rest thickness give way: their cells become one (collapse_cells.py), energy
               and charge added, joints of a neighbour with both become one
  5. split     every cell whose energy per measured volume is below split_threshold
               (and whose measured volume is positive) is cut in two across its
               longest direction (split_cell.py); energy and charge shared by volume;
               the tear is a new joint at rest thickness (from the halves'
               measured volumes), rate 0. A cut that is not
               clean is skipped (the cell tries again next step).
  6. corners   RELAX_PASSES (5) rounds of relax_corners() toward each cell's target,
               rest_volume(energy) plus half its joint walls (cell_targets.py). Corners
               only move: no neighbour changes. The open cube's outside corners have
               nothing pushing back, so the grid grows or shrinks there.
               Grid.volume is then each cell's measured volume (cell_volumes.py).
After 4 or 5 the joints, seams and fan are rebuilt from the faces (face_joints.py,
mesh_seams.py, build_fan.py). A step with neither event compares its joints before
and after and adds any difference to neighbour_changes (0 by construction).
Total energy is changed by no rule: the leak moves it, collapse adds it, split shares it.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from physics.cells.build_fan import build_fan
from physics.cells.cell_targets import cell_targets
from physics.cells.cell_volumes import cell_volumes
from physics.cells.collapse_cells import collapse_cells
from physics.cells.drop_loose_corners import drop_loose_corners
from physics.cells.engine import Engine
from physics.cells.engine_settings import EngineSettings
from physics.cells.face_areas import face_areas
from physics.cells.face_joints import face_joints
from physics.cells.grid import Grid
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

RELAX_PASSES = 5


@dataclass
class _Mesh:
    """The mutable working copy of one step's topology and per-cell values."""

    corners: np.ndarray
    faces: list
    face_cells: np.ndarray
    face_origin: np.ndarray
    energy: np.ndarray
    charge: np.ndarray
    pull: np.ndarray


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
    mesh = _Mesh(corners=engine.corners, faces=list(engine.faces), face_cells=engine.face_cells,
                 face_origin=engine.face_joint, energy=energy, charge=grid.charge, pull=exerted)
    giving = thickness < settings.push_limit * rest
    collapses = _collapse(mesh, grid, giving) if giving.any() else 0
    fan = build_fan(mesh.faces, mesh.face_cells, len(mesh.energy)) if collapses else engine.fan
    splits = _split(mesh, cell_volumes(mesh.corners, fan), settings.split_threshold)
    if not (collapses or splits):
        return _moved(engine, grid, mesh, thickness, rate, settings)
    return _rewoven(engine, mesh, thickness, rate, settings, collapses, splits)


def _collapse(mesh: _Mesh, grid: Grid, giving: np.ndarray) -> int:
    """Collapse the giving-way joints in the working mesh; returns the energy volumes that fell in."""
    faces, face_cells, origin, merged_into = collapse_cells(
        mesh.faces, mesh.face_cells, mesh.face_origin, grid.joints, giving, grid.cell_count)
    count = int(merged_into.max()) + 1
    lowest = np.full(count, grid.cell_count)
    np.minimum.at(lowest, merged_into, np.arange(grid.cell_count))
    mesh.faces, mesh.face_cells, mesh.face_origin = faces, face_cells, origin
    mesh.energy = np.bincount(merged_into, mesh.energy, count)
    mesh.charge = np.bincount(merged_into, mesh.charge, count)
    mesh.pull = mesh.pull[lowest]
    return grid.cell_count - count


def _split(mesh: _Mesh, measured: np.ndarray, threshold: float) -> int:
    """Split every cell stretched below the threshold in the working mesh; returns how many split."""
    thin = np.flatnonzero((measured > 0) & (mesh.energy < threshold * measured))
    done = 0
    for cell in thin:
        new_cell = len(mesh.energy)
        cut = split_cell(mesh.corners, mesh.faces, mesh.face_cells, mesh.face_origin, int(cell),
                         new_cell)
        if cut is None:
            continue
        mesh.corners, mesh.faces, mesh.face_cells, mesh.face_origin, share = cut
        kept_energy = mesh.energy[cell] * share
        kept_charge = mesh.charge[cell] * share
        mesh.energy = np.append(mesh.energy, mesh.energy[cell] - kept_energy)
        mesh.charge = np.append(mesh.charge, mesh.charge[cell] - kept_charge)
        mesh.energy[cell], mesh.charge[cell] = kept_energy, kept_charge
        mesh.pull = np.append(mesh.pull, mesh.pull[cell])
        done += 1
    return done


def _moved(engine: Engine, grid: Grid, mesh: _Mesh, thickness: np.ndarray, rate: np.ndarray,
           settings: EngineSettings) -> Engine:
    """The next engine when the topology did not change: only the corners move."""
    corners = relax_corners(engine.corners, engine.fan,
                            cell_targets(rest_volume(mesh.energy), thickness, engine.face_joint,
                                         engine.corners, engine.fan), RELAX_PASSES)
    after = _finished(Grid(cell_count=grid.cell_count, joints=grid.joints, energy=mesh.energy,
                           volume=cell_volumes(corners, engine.fan), thickness=thickness, rate=rate, squeeze=grid.squeeze,
                           pull=mesh.pull, charge=grid.charge, seams=grid.seams,
                           seam_open=grid.seam_open), settings)
    rejoined = face_joints(engine.face_cells, engine.face_joint, np.ones(len(engine.faces)),
                           thickness, rate)[0]
    return Engine(grid=after, corners=corners, faces=engine.faces, face_cells=engine.face_cells,
                  face_joint=engine.face_joint, fan=engine.fan, collapses=engine.collapses,
                  splits=engine.splits, start_space=engine.start_space,
                  neighbour_changes=engine.neighbour_changes + _pair_changes(grid.joints, rejoined))


def _rewoven(engine: Engine, mesh: _Mesh, thickness: np.ndarray, rate: np.ndarray,
             settings: EngineSettings, collapses: int, splits: int) -> Engine:
    """The next engine after a collapse or split: joints, seams and fan rebuilt from the faces."""
    corners, faces = drop_loose_corners(mesh.corners, mesh.faces)
    count = len(mesh.energy)
    fan = build_fan(faces, mesh.face_cells, count)
    measured = cell_volumes(corners, fan)
    joints, face_joint, joint_thickness, joint_rate, fresh = face_joints(
        mesh.face_cells, mesh.face_origin, face_areas(corners, fan), thickness, rate)
    joint_thickness[fresh] = joint_rest(measured, joints[fresh], settings.joint_rest)
    seams = mesh_seams(faces, mesh.face_cells)
    corners = relax_corners(corners, fan, cell_targets(rest_volume(mesh.energy), joint_thickness,
                                                       face_joint, corners, fan), RELAX_PASSES)
    after = _finished(Grid(cell_count=count, joints=joints, energy=mesh.energy,
                           volume=cell_volumes(corners, fan),
                           thickness=joint_thickness, rate=joint_rate, squeeze=np.zeros(len(joints)),
                           pull=mesh.pull, charge=mesh.charge, seams=seams,
                           seam_open=np.ones(len(seams))), settings)
    return Engine(grid=after, corners=corners, faces=tuple(faces), face_cells=mesh.face_cells,
                  face_joint=face_joint, fan=fan, collapses=engine.collapses + collapses,
                  splits=engine.splits + splits, start_space=engine.start_space,
                  neighbour_changes=engine.neighbour_changes)


def _finished(grid: Grid, settings: EngineSettings) -> Grid:
    """The grid with its derived squeeze and seam openness set."""
    squeeze = joint_squeeze(grid, settings)
    with_squeeze = replace(grid, squeeze=squeeze)
    return replace(with_squeeze, seam_open=seam_openness(with_squeeze, settings))


def _pair_changes(before: np.ndarray, after: np.ndarray) -> int:
    """Neighbour pairs in one joint list and not the other."""
    first = {tuple(pair) for pair in before.tolist()}
    second = {tuple(pair) for pair in after.tolist()}
    return len(first ^ second)
