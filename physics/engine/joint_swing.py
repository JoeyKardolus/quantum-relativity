"""Joint swing: every joint's size moves like a frictionless spring with inertia, pushed by its own push against its neighbours'.

Contract: `joint_swing(state, settings, pull)` advances every joint's size and speed
(state.size, state.speed) by one step_time. A joint's neighbours are the joints it
shares an energy volume with (the cell graph, state.between). Its load is its push
(cell_laws.py) plus `pull`: step.py passes the mean pull over its boxes (cell_count.py;
the pull is at most 0 where energy is, so it squeezes the joint: bible rule 8, "pull
squeezes it and its push swells it back") less the volumes' fluid pressing on its faces
(wall_press.py). Each pair of neighbouring joints i, j is a spring running on the
two joints' clocks: joint i's size accelerates by rate_i * rate_j * (load_i - load_j) /
joint_inertia per unit of counter time, so what one joint gains its neighbour loses and
the joints' summed size is kept (bible rule 5: the joints are the springs of the grid,
they overshoot and swing round their balance). The swing is frictionless (axiom 4: the
only friction is energy crossing a joint), so the step must neither gain nor lose swing
energy: it is the discrete-gradient (Itoh-Abe) step of the Hamiltonian
  E = 1/2 v^T M^+ v + sum_i U_i(s_i),  U_i(s) = push * wanted_i^6 / (5 s^5) - pull_i * s
(M the clock-weighted graph Laplacian, -dU_i/ds = load_i), which keeps E exactly at any
step length; it is implicit and solved by Newton's method (numerics, not a rule: an
explicit step gained energy at the steep wall and crushed joints to nothing). The step
is cut into as many equal substeps as the stiffest joint needs for accuracy
(omega * substep <= 1/2), at most MOST_SUBSTEPS. A joint whose size falls to the
collapse packing (its minus quanta over its size reach collapse_packing, size <= quanta /
collapse_packing; owner, 2026-11-16: only too high a concentration collapses a joint,
never too little energy) stops there, at rest, and is returned for collapse in the same
step (axiom 6); an outer joint, with no second volume, cannot collapse: it is held at the
limit size for the rest of the step. A joint already past the limit when the swing starts
does not swing: the collapse that opens the next step takes it (step.py). state.speed is the
size change per unit of counter time (the clocks enter through the springs, so the
summed size is kept exactly). Energy volumes are not moved (their size
follows their energy). Returns a bool array (universes, capacity), True for each joint
pressed to the collapse limit.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

from physics.engine.settings import Settings
from physics.engine.state import State

MOST_SUBSTEPS = 64
NEWTON_ROUNDS = 100
HALVINGS = 60
ROUNDOFF = 1e-7
TOLERANCE = 1e-10


def joint_swing(state: State, settings: Settings, pull: np.ndarray) -> np.ndarray:
    """Swing every joint's size one step, energy kept; returns the joints pressed to the collapse limit."""
    pressed = np.zeros(state.kind.shape, dtype=bool)
    for universe in range(state.universes):
        joints = np.flatnonzero((state.kind[universe] == -1) & (state.wanted[universe] > 0.0))
        if joints.size < 2:
            continue
        pressed[universe, joints] = _swing_universe(state, settings, universe, joints, pull[universe, joints])
    state.cell_push = np.where(state.kind == -1, _push(state, settings), 0.0)
    return pressed


def _swing_universe(state: State, settings: Settings, universe: int, joints: np.ndarray,
                    load_pull: np.ndarray) -> np.ndarray:
    """One universe's joints swung one step; returns which of them reached the collapse limit."""
    rate = state.rate[universe, joints]
    wanted = state.wanted[universe, joints]
    laplacian = _laplacian(state, universe, joints, rate) / settings.joint_inertia
    hardness = settings.push * wanted ** 6 / 5.0
    floor = state.cell_quanta[universe, joints].astype(np.float64) / settings.collapse_packing
    size = state.size[universe, joints].astype(np.float64)
    speed = state.speed[universe, joints].astype(np.float64)
    stiffness = 30.0 * hardness / np.maximum(size, floor) ** 7
    largest = float(sparse_linalg.norm(laplacian, ord=1)) if laplacian.nnz else 0.0
    omega = math.sqrt(largest * float(stiffness.max()))
    count = min(MOST_SUBSTEPS, max(1, math.ceil(settings.step_time * omega / 0.5)))
    span = settings.step_time / count
    paired = (state.between[universe, joints] >= 0).all(axis=1)
    born = size < floor
    held = born.copy()
    speed = np.where(born, 0.0, speed)
    for _ in range(count):
        size, speed = _substep(size, speed, span, laplacian, hardness, load_pull, held)
        held |= (size <= floor) & paired
        size = np.where(held & ~born, np.maximum(size, floor), size)
        speed = np.where(held, 0.0, speed)
    state.size[universe, joints] = size
    state.speed[universe, joints] = speed
    return held & ~born & paired


def _substep(size: np.ndarray, speed: np.ndarray, span: float, laplacian, hardness: np.ndarray,
             load_pull: np.ndarray, held: np.ndarray) -> tuple:
    """One discrete-gradient substep: (size, speed) after `span`, swing energy kept exactly."""
    moving = ~held
    scale = 0.5 * span * span
    start = size
    target = start + span * speed
    guess = start.copy()
    residual = _residual(start, guess, target, scale, laplacian, hardness, load_pull, moving)
    for _ in range(NEWTON_ROUNDS):
        if np.abs(residual).max() <= TOLERANCE * max(1.0, float(np.abs(start).max())):
            break
        _, bend = _secant(start, guess, hardness, load_pull)
        jacobian = sparse.identity(size.size, format="csr") + scale * (laplacian @ sparse.diags(np.where(moving, bend, 0.0)))
        change = sparse_linalg.spsolve(jacobian.tocsc(), -residual)
        found = _backtrack(start, guess, change, residual, target, scale, laplacian, hardness, load_pull, moving)
        if found is None:
            if np.abs(residual).max() <= ROUNDOFF * max(1.0, float(np.abs(start).max())):
                break
            raise RuntimeError(f"joint swing: no Newton step shrinks the residual {np.abs(residual).max():.3e}")
        guess, residual = found
    else:
        raise RuntimeError(f"joint swing: Newton did not converge in {NEWTON_ROUNDS} rounds "
                           f"(residual {np.abs(residual).max():.3e})")
    slope, _ = _secant(start, guess, hardness, load_pull)
    slope = np.where(moving, slope, 0.0)
    after = speed - span * (laplacian @ slope)
    return np.where(moving, guess, start), np.where(moving, after, speed)


def _residual(start: np.ndarray, guess: np.ndarray, target: np.ndarray, scale: float, laplacian, hardness: np.ndarray,
              load_pull: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """The implicit step's equation, guess - (start + span * speed) + span^2 / 2 * M * secant; 0 for held joints."""
    slope, _ = _secant(start, guess, hardness, load_pull)
    return np.where(moving, guess - target + scale * (laplacian @ np.where(moving, slope, 0.0)), 0.0)


def _backtrack(start: np.ndarray, guess: np.ndarray, change: np.ndarray, residual: np.ndarray, target: np.ndarray,
               scale: float, laplacian, hardness: np.ndarray, load_pull: np.ndarray, moving: np.ndarray):
    """The Newton step cut in half until every size stays positive and the residual shrinks (at most HALVINGS times); None when none does."""
    norm = float(np.linalg.norm(residual))
    fraction = 1.0
    for _ in range(HALVINGS):
        trial = guess + fraction * change
        if np.all(trial > 0.0):
            after = _residual(start, trial, target, scale, laplacian, hardness, load_pull, moving)
            if float(np.linalg.norm(after)) <= (1.0 - 1e-4 * fraction) * norm:
                return trial, after
        fraction *= 0.5
    return None


def _secant(before: np.ndarray, after: np.ndarray, hardness: np.ndarray, load_pull: np.ndarray) -> tuple:
    """(U(after) - U(before)) / (after - before) per joint and its derivative in `after`, written without cancellation."""
    slope = -load_pull.astype(np.float64)
    bend = np.zeros(before.shape)
    for power in range(1, 6):
        term = 1.0 / (before ** power * after ** (6 - power))
        slope = slope - hardness * term
        bend = bend + hardness * (6 - power) * term / after
    return slope, bend


def _laplacian(state: State, universe: int, joints: np.ndarray, rate: np.ndarray):
    """Graph Laplacian over the joints of one universe, edge i-j weighted rate_i * rate_j (two joints sharing a volume)."""
    index = np.full(state.kind.shape[1], -1, dtype=np.int64)
    index[joints] = np.arange(joints.size)
    sides = state.between[universe, joints]
    member = np.concatenate([np.arange(joints.size), np.arange(joints.size)])
    volume = np.concatenate([sides[:, 0], sides[:, 1]])
    keep = volume >= 0
    member, volume = member[keep], volume[keep]
    order = np.argsort(volume, kind="stable")
    member, volume = member[order], volume[order]
    starts = np.flatnonzero(np.r_[True, volume[1:] != volume[:-1]])
    ends = np.r_[starts[1:], volume.size]
    rows, columns = [], []
    for first, last in zip(starts, ends):
        group = member[first:last]
        if group.size > 1:
            one, two = np.meshgrid(group, group, indexing="ij")
            off = one != two
            rows.append(one[off])
            columns.append(two[off])
    if not rows:
        return sparse.csr_matrix((joints.size, joints.size))
    row, column = np.concatenate(rows), np.concatenate(columns)
    weight = rate[row] * rate[column]
    adjacency = sparse.csr_matrix((weight, (row, column)), shape=(joints.size, joints.size))
    adjacency.sum_duplicates()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return (sparse.diags(degree) - adjacency).tocsr()


def _push(state: State, settings: Settings) -> np.ndarray:
    """push * (wanted / size) ** steepness per cell."""
    return settings.push * (state.wanted / np.maximum(state.size, 1e-6)) ** settings.push_steepness
