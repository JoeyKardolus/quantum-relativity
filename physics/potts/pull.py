"""Pull: the A-energy's gravity-like potential over the Planck boxes, solved on the GPU.

Contract: `pull(universe, settings)` improves universe.potential in place towards
laplacian(potential) = 4 pi G a, a the A-energy per box (boxes whose owner is an
energy volume) and G = settings.attraction, with settings.potential_cycles multigrid
V-cycles (multigrid.py) warm-started from the potential it holds. The solve runs on a
cube around the active region (active_region.py), its side a multiple of 32 (or the
grid's side) at least the region's widest extent plus 8, placed once and moved only when
the region outgrows it; its faces get the open-space value -G M / r, M the total
A-energy and r from its energy-weighted centre, both summed on the GPU in a fixed
order (no float atomics). Outside a ball of A-energy the potential falls as -G M / r
(the 1/r test): only A is a source, the pull fades with distance. Boxes outside the
cube are empty and keep their old potential, which nothing reads. Returns None.
Nothing is copied to the host: on CUDA the solve is captured as a CUDA graph and
replayed (re-captured when the cube, the arrays, G or cycles change).
"""
from __future__ import annotations

import math

import warp as wp

from physics.potts.multigrid import Multigrid
from physics.potts.settings import Settings
from physics.potts.universe import QUANTUM, Universe

_QUANTUM = wp.constant(QUANTUM)
_SOLVERS: dict[tuple[int, str], Multigrid] = {}
_WORK: dict[tuple[int, str], tuple] = {}
_GRAPHS: dict[tuple, tuple[tuple, object]] = {}


@wp.kernel
def _gather(owner: wp.array3d(dtype=wp.int32), energy: wp.array3d(dtype=wp.int64), kind: wp.array(dtype=wp.int32),
            potential: wp.array3d(dtype=wp.float32), corner: wp.vec3i, scale: float,
            a_quanta: wp.array3d(dtype=wp.int64), rhs: wp.array3d(dtype=float), phi: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    gi = i + corner[0]
    gj = j + corner[1]
    gk = k + corner[2]
    cell = owner[gi, gj, gk]
    held = wp.int64(0)
    if cell >= 0:
        if kind[cell] == 0:
            held = wp.max(energy[gi, gj, gk], wp.int64(0))
    a_quanta[i, j, k] = held
    rhs[i, j, k] = scale * float(held)
    phi[i, j, k] = potential[gi, gj, gk]


@wp.kernel
def _scatter(phi: wp.array3d(dtype=float), corner: wp.vec3i, potential: wp.array3d(dtype=wp.float32)):
    i, j, k = wp.tid()
    potential[i + corner[0], j + corner[1], k + corner[2]] = phi[i, j, k]


@wp.kernel
def _row_moments(a_quanta: wp.array3d(dtype=wp.int64), side: int, rows: wp.array3d(dtype=wp.float64)):
    """Each thread sums one row along z: its A quanta and their x, y, z moments."""
    i, j = wp.tid()
    mass = wp.float64(0.0)
    along_z = wp.float64(0.0)
    for k in range(side):
        quanta = wp.float64(a_quanta[i, j, k])
        mass += quanta
        along_z += quanta * wp.float64(k)
    rows[i, j, 0] = mass
    rows[i, j, 1] = mass * wp.float64(i)
    rows[i, j, 2] = mass * wp.float64(j)
    rows[i, j, 3] = along_z


@wp.kernel
def _plane_moments(rows: wp.array3d(dtype=wp.float64), side: int, planes: wp.array2d(dtype=wp.float64)):
    """Each thread adds the rows of one x plane, in order of y."""
    i = wp.tid()
    for moment in range(4):
        total = wp.float64(0.0)
        for j in range(side):
            total += rows[i, j, moment]
        planes[i, moment] = total


@wp.kernel
def _boundary(planes: wp.array2d(dtype=wp.float64), attraction: float, side: int,
              boundary: wp.array(dtype=float)):
    """One thread adds the planes in order of x (no atomics: the same sums every run)."""
    totals = wp.vector(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    for i in range(side):
        for moment in range(4):
            totals[moment] = totals[moment] + planes[i, moment]
    mass = totals[0]
    boundary[0] = attraction * float(mass) * _QUANTUM
    middle = 0.5 * float(side - 1)
    for axis in range(3):
        boundary[axis + 1] = middle
        if mass > wp.float64(0.0):
            boundary[axis + 1] = float(totals[axis + 1] / mass)


def pull(universe: Universe, settings: Settings) -> None:
    """Improve the pull potential of the A-energy by settings.potential_cycles V-cycles on the active cube."""
    device = wp.get_device(universe.owner.device)
    cube, corner = _cube(universe)
    key = (cube, str(device))
    cycles = int(settings.potential_cycles)
    if key not in _SOLVERS:
        _SOLVERS[key] = Multigrid(cube, str(device))
        _WORK[key] = (wp.zeros((cube,) * 3, dtype=wp.int64, device=device),
                      wp.zeros((cube,) * 3, dtype=float, device=device),
                      wp.zeros((cube,) * 3, dtype=float, device=device),
                      wp.zeros((cube, cube, 4), dtype=wp.float64, device=device),
                      wp.zeros((cube, 4), dtype=wp.float64, device=device),
                      wp.zeros(4, dtype=float, device=device))
    run = lambda: _solve(universe, settings.attraction, cycles, key, corner)
    if not device.is_cuda:
        run()
        return
    recorded = (universe.owner.ptr, universe.energy.ptr, universe.potential.ptr, universe.kind.ptr, corner,
                settings.attraction, cycles)
    if key not in _GRAPHS or _GRAPHS[key][0] != recorded:
        run()
        with wp.ScopedCapture(device=device) as capture:
            run()
        _GRAPHS[key] = (recorded, capture.graph)
    else:
        wp.capture_launch(_GRAPHS[key][1])


def _cube(universe: Universe) -> tuple[int, tuple]:
    """The solve cube's side and corner: kept while it still holds the active region."""
    kept = universe.scratch.get("pull_cube")
    if kept is not None:
        cube, corner = kept
        if all(corner[axis] <= universe.low[axis] and universe.high[axis] <= corner[axis] + cube for axis in range(3)):
            return kept
    extent = max(top - bottom for bottom, top in zip(universe.low, universe.high)) + 8
    cube = min(universe.side, -(-extent // 32) * 32)
    corner = []
    for axis in range(3):
        middle = (universe.low[axis] + universe.high[axis]) // 2
        corner.append(int(min(max(0, middle - cube // 2), universe.side - cube)))
    universe.scratch["pull_cube"] = (cube, tuple(corner))
    return universe.scratch["pull_cube"]


def _solve(universe: Universe, attraction: float, cycles: int, key: tuple, corner: tuple) -> None:
    a_quanta, rhs, phi, rows, planes, boundary = _WORK[key]
    cube = key[0]
    device = universe.owner.device
    offset = wp.vec3i(*corner)
    wp.launch(_gather, dim=(cube,) * 3, device=device,
              inputs=[universe.owner, universe.energy, universe.kind, universe.potential, offset,
                      4.0 * math.pi * attraction * QUANTUM, a_quanta, rhs, phi])
    wp.launch(_row_moments, dim=(cube, cube), device=device, inputs=[a_quanta, cube, rows])
    wp.launch(_plane_moments, dim=cube, device=device, inputs=[rows, cube, planes])
    wp.launch(_boundary, dim=1, device=device, inputs=[planes, attraction, cube, boundary])
    _SOLVERS[key].solve(rhs, phi, boundary, cycles)
    wp.launch(_scatter, dim=(cube,) * 3, device=device, inputs=[phi, offset, universe.potential])
