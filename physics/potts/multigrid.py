"""Multigrid: a GPU Poisson solver for the pull, open space by a monopole boundary.

Contract: `Multigrid(side, device)` holds the levels for a side**3 cube of unit boxes.
`solve(rhs, phi, boundary, cycles)` improves phi in place towards laplacian(phi) = rhs
(7-point, box spacing 1) with `cycles` V-cycles, starting from the phi it is given
(warm start). `boundary` is a device float32 array (G * M, centre x, y, z), read on
the GPU, so a solve needs no host copy and can be captured in a CUDA graph. Boxes are
cell-centred; the value on the grid's outer faces is -G M / |face - centre|, the
potential of the source's total mass at its centre (pull.py): open space,
nothing wrapped. The higher multipoles at the faces are dropped, which on a grid much
larger than the source is a small error that the 1/r test measures.
`residual(rhs, phi, boundary)` returns the max |rhs - laplacian(phi)| (host).
Numerics: red-black Gauss-Seidel (2 sweeps before, 2 after), full-weighting restriction
by 8-box averages, trilinear prolongation, levels halved while the side is even and
above 2; the coarsest level (2 for a power of two, 3 for 192) gets 10 sweeps. Ghost
boxes outside a face take 2 * boundary - phi (the boundary value sits on the face);
coarse levels solve for the error, so their boundary is 0.
"""
from __future__ import annotations

import numpy as np
import warp as wp

PRE_SWEEPS = 2
POST_SWEEPS = 2
COARSEST_SWEEPS = 10


@wp.func
def _face_value(x: float, y: float, z: float, boundary: wp.array(dtype=float), fine: int) -> float:
    if fine == 0 or boundary[0] == 0.0:
        return 0.0
    centre = wp.vec3(boundary[1], boundary[2], boundary[3])
    return -boundary[0] / wp.max(wp.length(wp.vec3(x, y, z) - centre), 0.5)


@wp.func
def _neighbour(phi: wp.array3d(dtype=float), i: int, j: int, k: int, di: int, dj: int, dk: int,
               side: int, spacing: float, boundary: wp.array(dtype=float), fine: int) -> wp.vec2:
    """(contribution to the neighbour sum, 1 when the neighbour is outside the grid)."""
    ni = i + di
    nj = j + dj
    nk = k + dk
    if ni >= 0 and ni < side and nj >= 0 and nj < side and nk >= 0 and nk < side:
        return wp.vec2(phi[ni, nj, nk], 0.0)
    half = 0.5 * spacing
    x = (float(i) + 0.5) * spacing - 0.5 + float(di) * half
    y = (float(j) + 0.5) * spacing - 0.5 + float(dj) * half
    z = (float(k) + 0.5) * spacing - 0.5 + float(dk) * half
    return wp.vec2(2.0 * _face_value(x, y, z, boundary, fine), 1.0)


@wp.func
def _neighbours(phi: wp.array3d(dtype=float), i: int, j: int, k: int, side: int, spacing: float,
                boundary: wp.array(dtype=float), fine: int) -> wp.vec2:
    total = _neighbour(phi, i, j, k, 1, 0, 0, side, spacing, boundary, fine)
    total += _neighbour(phi, i, j, k, -1, 0, 0, side, spacing, boundary, fine)
    total += _neighbour(phi, i, j, k, 0, 1, 0, side, spacing, boundary, fine)
    total += _neighbour(phi, i, j, k, 0, -1, 0, side, spacing, boundary, fine)
    total += _neighbour(phi, i, j, k, 0, 0, 1, side, spacing, boundary, fine)
    total += _neighbour(phi, i, j, k, 0, 0, -1, side, spacing, boundary, fine)
    return total


@wp.kernel
def _smooth(phi: wp.array3d(dtype=float), rhs: wp.array3d(dtype=float), colour: int, side: int,
            spacing: float, boundary: wp.array(dtype=float), fine: int):
    i, j, k = wp.tid()
    if (i + j + k) % 2 != colour:
        return
    around = _neighbours(phi, i, j, k, side, spacing, boundary, fine)
    phi[i, j, k] = (around[0] - spacing * spacing * rhs[i, j, k]) / (6.0 + around[1])


@wp.kernel
def _residual(phi: wp.array3d(dtype=float), rhs: wp.array3d(dtype=float), out: wp.array3d(dtype=float),
              side: int, spacing: float, boundary: wp.array(dtype=float), fine: int):
    i, j, k = wp.tid()
    around = _neighbours(phi, i, j, k, side, spacing, boundary, fine)
    laplacian = (around[0] - (6.0 + around[1]) * phi[i, j, k]) / (spacing * spacing)
    out[i, j, k] = rhs[i, j, k] - laplacian


@wp.kernel
def _restrict(fine: wp.array3d(dtype=float), coarse_rhs: wp.array3d(dtype=float),
              coarse_phi: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    total = float(0.0)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                total += fine[2 * i + di, 2 * j + dj, 2 * k + dk]
    coarse_rhs[i, j, k] = total / 8.0
    coarse_phi[i, j, k] = 0.0


@wp.func
def _coarse(coarse: wp.array3d(dtype=float), ci: int, cj: int, ck: int, ni: int, nj: int, nk: int,
            side: int) -> float:
    """Coarse value at (ni, nj, nk); outside, the homogeneous ghost -value at (ci, cj, ck) per axis."""
    sign = float(1.0)
    if ni < 0 or ni >= side:
        ni = ci
        sign = -sign
    if nj < 0 or nj >= side:
        nj = cj
        sign = -sign
    if nk < 0 or nk >= side:
        nk = ck
        sign = -sign
    return sign * coarse[ni, nj, nk]


@wp.kernel
def _prolong(coarse: wp.array3d(dtype=float), fine: wp.array3d(dtype=float), coarse_side: int):
    i, j, k = wp.tid()
    ci = i // 2
    cj = j // 2
    ck = k // 2
    si = 2 * (i % 2) - 1
    sj = 2 * (j % 2) - 1
    sk = 2 * (k % 2) - 1
    total = float(0.0)
    for oi in range(2):
        for oj in range(2):
            for ok in range(2):
                weight = (0.75 - 0.5 * float(oi)) * (0.75 - 0.5 * float(oj)) * (0.75 - 0.5 * float(ok))
                total += weight * _coarse(coarse, ci, cj, ck, ci + oi * si, cj + oj * sj, ck + ok * sk,
                                          coarse_side)
    fine[i, j, k] = fine[i, j, k] + total


class Multigrid:
    """V-cycles on a cube of unit boxes with an open-space monopole boundary."""

    def __init__(self, side: int, device: str = "cuda") -> None:
        self.device = device
        self.sides = [side]
        while self.sides[-1] % 2 == 0 and self.sides[-1] > 2:
            self.sides.append(self.sides[-1] // 2)
        shape = lambda level_side: (level_side, level_side, level_side)
        self.work = [wp.zeros(shape(level_side), dtype=float, device=device) for level_side in self.sides]
        self.coarse_rhs = [None] + [wp.zeros(shape(level_side), dtype=float, device=device)
                                    for level_side in self.sides[1:]]
        self.coarse_phi = [None] + [wp.zeros(shape(level_side), dtype=float, device=device)
                                    for level_side in self.sides[1:]]

    def solve(self, rhs: wp.array, phi: wp.array, boundary: wp.array, cycles: int) -> None:
        """`cycles` V-cycles on phi in place, towards laplacian(phi) = rhs."""
        for _ in range(int(cycles)):
            self._cycle(0, rhs, phi, boundary)

    def residual(self, rhs: wp.array, phi: wp.array, boundary: wp.array) -> float:
        """Max |rhs - laplacian(phi)| over the boxes."""
        side = self.sides[0]
        wp.launch(_residual, dim=(side, side, side), device=self.device,
                  inputs=[phi, rhs, self.work[0], side, 1.0, boundary, 1])
        return float(np.abs(self.work[0].numpy()).max())

    def _sweeps(self, level: int, rhs, phi, boundary, count: int) -> None:
        side = self.sides[level]
        spacing = float(2 ** level)
        for _ in range(count):
            for colour in (0, 1):
                wp.launch(_smooth, dim=(side, side, side), device=self.device,
                          inputs=[phi, rhs, colour, side, spacing, boundary, int(level == 0)])

    def _cycle(self, level: int, rhs, phi, boundary) -> None:
        side = self.sides[level]
        spacing = float(2 ** level)
        if level == len(self.sides) - 1:
            self._sweeps(level, rhs, phi, boundary, COARSEST_SWEEPS)
            return
        self._sweeps(level, rhs, phi, boundary, PRE_SWEEPS)
        wp.launch(_residual, dim=(side, side, side), device=self.device,
                  inputs=[phi, rhs, self.work[level], side, spacing, boundary, int(level == 0)])
        coarse_side = self.sides[level + 1]
        wp.launch(_restrict, dim=(coarse_side,) * 3, device=self.device,
                  inputs=[self.work[level], self.coarse_rhs[level + 1], self.coarse_phi[level + 1]])
        self._cycle(level + 1, self.coarse_rhs[level + 1], self.coarse_phi[level + 1], boundary)
        wp.launch(_prolong, dim=(side, side, side), device=self.device,
                  inputs=[self.coarse_phi[level + 1], phi, coarse_side])
        self._sweeps(level, rhs, phi, boundary, POST_SWEEPS)
