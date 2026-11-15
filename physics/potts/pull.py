"""Pull: the A-energy's gravity-like potential over the Planck boxes, solved on the GPU.

Contract: `pull(universe, settings)` improves universe.potential in place towards
laplacian(potential) = 4 pi G a, a the A-energy per box (universe.a_quanta, set by
cell_sums.py) and G = settings.attraction, with settings.potential_cycles multigrid
V-cycles (multigrid.py) warm-started from the potential it holds. The open-space
boundary value -G M / r on the grid's faces takes M, the total A-energy, and r from its
energy-weighted centre, both summed on the GPU in a fixed order (no float atomics).
Outside a ball of A-energy the potential falls as -G M / r (the 1/r test): only A is a
source, the pull fades with distance. Returns None. Nothing is copied to the host: on
CUDA the whole solve is captured as a CUDA graph and replayed (one kept per side and
device, re-captured when the arrays, G or cycles change), since a V-cycle is many
small launches. Moved here from the removed fluid engine, unchanged in what it solves.
"""
from __future__ import annotations

import math

import warp as wp

from physics.potts.multigrid import Multigrid
from physics.potts.settings import Settings
from physics.potts.universe import QUANTUM, Universe

_QUANTUM = wp.constant(QUANTUM)
_SOLVERS: dict[tuple[int, str], Multigrid] = {}
_WORK: dict[tuple[int, str], tuple[wp.array, wp.array, wp.array, wp.array]] = {}
_GRAPHS: dict[tuple[int, str], tuple[tuple, object]] = {}


@wp.kernel
def _source(a_quanta: wp.array3d(dtype=wp.int64), scale: float, rhs: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    rhs[i, j, k] = scale * float(wp.max(a_quanta[i, j, k], wp.int64(0)))


@wp.kernel
def _row_moments(a_quanta: wp.array3d(dtype=wp.int64), side: int, rows: wp.array3d(dtype=wp.float64)):
    """Each thread sums one row along z: its A quanta and their x, y, z moments."""
    i, j = wp.tid()
    mass = wp.float64(0.0)
    along_z = wp.float64(0.0)
    for k in range(side):
        quanta = wp.float64(wp.max(a_quanta[i, j, k], wp.int64(0)))
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
    """Improve the pull potential of the A-energy by settings.potential_cycles V-cycles."""
    device = wp.get_device(universe.a_quanta.device)
    key = (universe.side, str(device))
    cycles = int(settings.potential_cycles)
    if key not in _SOLVERS:
        _SOLVERS[key] = Multigrid(universe.side, str(device))
        _WORK[key] = (wp.zeros((universe.side,) * 3, dtype=float, device=device),
                      wp.zeros((universe.side, universe.side, 4), dtype=wp.float64, device=device),
                      wp.zeros((universe.side, 4), dtype=wp.float64, device=device),
                      wp.zeros(4, dtype=float, device=device))
    run = lambda: _solve(universe, settings.attraction, cycles, key)
    if not device.is_cuda:
        run()
        return
    recorded = (universe.a_quanta.ptr, universe.potential.ptr, settings.attraction, cycles)
    if key not in _GRAPHS or _GRAPHS[key][0] != recorded:
        run()
        with wp.ScopedCapture(device=device) as capture:
            run()
        _GRAPHS[key] = (recorded, capture.graph)
    else:
        wp.capture_launch(_GRAPHS[key][1])


def _solve(universe: Universe, attraction: float, cycles: int, key: tuple) -> None:
    rhs, rows, planes, boundary = _WORK[key]
    side = universe.side
    device = universe.a_quanta.device
    wp.launch(_row_moments, dim=(side, side), device=device, inputs=[universe.a_quanta, side, rows])
    wp.launch(_plane_moments, dim=side, device=device, inputs=[rows, side, planes])
    wp.launch(_boundary, dim=1, device=device, inputs=[planes, attraction, side, boundary])
    wp.launch(_source, dim=(side,) * 3, device=device,
              inputs=[universe.a_quanta, 4.0 * math.pi * attraction * QUANTUM, rhs])
    _SOLVERS[key].solve(rhs, universe.potential, boundary, cycles)
