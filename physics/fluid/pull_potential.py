"""Pull potential: the gravity-like potential of the A-energy, solved on the GPU.

Contract: `pull_potential(grid, settings, cycles)` updates grid.potential in place
towards laplacian(potential) = 4 pi G a (a the A-energy per box, G =
settings.attraction, box spacing 1) with `cycles` multigrid V-cycles (multigrid.py),
warm-started from the potential it holds. The open-space boundary value -G M / r on
the grid's faces takes M, the total A-energy, and r from its energy-weighted centre,
both summed on the GPU in a fixed order (no float atomics), so a run repeats
exactly (measured: two runs of seed 1, side 64, 20 steps, identical boxes). Outside a ball of A-energy the potential falls as -G M / r
(the 1/r test). Only A-energy is a source: A pulls, B does not. Returns the device
array (G * M, centre x, y, z). Nothing is copied to the host: on CUDA the whole solve
is captured as a CUDA graph and replayed (one kept per side and device, re-captured
when the grid, G or cycles change), since a V-cycle is many small launches.
"""
from __future__ import annotations

import math

import warp as wp

from physics.fluid.grid import QUANTUM, Grid
from physics.fluid.multigrid import Multigrid
from physics.fluid.settings import Settings

_QUANTUM = wp.constant(QUANTUM)
_SOLVERS: dict[tuple[int, str], Multigrid] = {}
_WORK: dict[tuple[int, str], tuple[wp.array, wp.array, wp.array, wp.array]] = {}
_GRAPHS: dict[tuple[int, str], tuple[tuple, object]] = {}


@wp.kernel
def _source(a_energy: wp.array3d(dtype=wp.int64), scale: float, rhs: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    rhs[i, j, k] = scale * float(wp.max(a_energy[i, j, k], wp.int64(0)))


@wp.kernel
def _row_moments(a_energy: wp.array3d(dtype=wp.int64), side: int, rows: wp.array3d(dtype=wp.float64)):
    """Each thread sums one row along z: its A quanta and their x, y, z moments."""
    i, j = wp.tid()
    mass = wp.float64(0.0)
    along_z = wp.float64(0.0)
    for k in range(side):
        quanta = wp.float64(wp.max(a_energy[i, j, k], wp.int64(0)))
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


def pull_potential(grid: Grid, settings: Settings, cycles: int) -> wp.array:
    """Improve grid.potential by `cycles` V-cycles; return the device (G * M, centre)."""
    device = wp.get_device(grid.a_energy.device)
    key = (grid.side, str(device))
    if key not in _SOLVERS:
        _SOLVERS[key] = Multigrid(grid.side, str(device))
        _WORK[key] = (wp.zeros((grid.side,) * 3, dtype=float, device=device),
                      wp.zeros((grid.side, grid.side, 4), dtype=wp.float64, device=device),
                      wp.zeros((grid.side, 4), dtype=wp.float64, device=device),
                      wp.zeros(4, dtype=float, device=device))
    run = lambda: _solve(grid, settings.attraction, int(cycles), key)
    if not device.is_cuda:
        run()
        return _WORK[key][3]
    recorded = (grid.a_energy.ptr, grid.potential.ptr, settings.attraction, int(cycles))
    if key not in _GRAPHS or _GRAPHS[key][0] != recorded:
        run()
        with wp.ScopedCapture(device=device) as capture:
            run()
        _GRAPHS[key] = (recorded, capture.graph)
    else:
        wp.capture_launch(_GRAPHS[key][1])
    return _WORK[key][3]


def _solve(grid: Grid, attraction: float, cycles: int, key: tuple) -> None:
    rhs, rows, planes, boundary = _WORK[key]
    device = grid.a_energy.device
    wp.launch(_row_moments, dim=(grid.side, grid.side), device=device, inputs=[grid.a_energy, grid.side, rows])
    wp.launch(_plane_moments, dim=grid.side, device=device, inputs=[rows, grid.side, planes])
    wp.launch(_boundary, dim=1, device=device, inputs=[planes, attraction, grid.side, boundary])
    wp.launch(_source, dim=(grid.side,) * 3, device=device,
              inputs=[grid.a_energy, 4.0 * math.pi * attraction * QUANTUM, rhs])
    _SOLVERS[key].solve(rhs, grid.potential, boundary, cycles)
