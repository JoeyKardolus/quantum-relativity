"""Spread heat: the energy friction took from directed flow spreads out into the boxes around the seam.

Contract: `spread_heat(universe, settings)` applies the spreading friction planned this
substep in universe.spread (hop.py: per box the energy to take and to give) and clears
it, on the GPU, without waiting. Bible rule 8, drift from imperfect transfer: energy
crossing a seam rubs against the joint's push, so the transfer is imperfect and the
energy spreads a little; spread-out energy is less concentrated, so by the size rule its
cells grow. Every box gives up what was planned, at most what it holds now (several
pairs may have planned to take from one box), and the energy taken, R, is given back in
the planned proportions, each box receiving give * R / G (G all the planned gifts), so
the sum of |psi|**2 is kept to float64 rounding and no box goes negative. A box's phase
is kept and only its size changes: the spreading is undirected (a box that held nothing
gets phase 0). R is added to universe.spread_moved[0]; the change it makes to the
clocked fluid energy (fluid_energy.py), measured just before and after, to
universe.spread_work[0], so the books (stats.py energy_balance) keep closing. Does
nothing when settings.friction is 0. Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.potts.fluid_energy import fluid_energy
from physics.potts.settings import Settings
from physics.potts.universe import Universe


@wp.kernel
def _take(owner: wp.array3d(dtype=wp.int32), psi: wp.array3d(dtype=wp.vec2d), spread: wp.array3d(dtype=wp.vec2d),
          low: wp.vec3i, sums: wp.array(dtype=wp.float64)):
    ti, tj, tk = wp.tid()
    i = ti + low[0]
    j = tj + low[1]
    k = tk + low[2]
    planned = spread[i, j, k]
    if planned[0] == wp.float64(0.0) and planned[1] == wp.float64(0.0):
        return
    if owner[i, j, k] < 0:
        spread[i, j, k] = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
        return
    taken = wp.min(planned[0], wp.dot(psi[i, j, k], psi[i, j, k]))
    spread[i, j, k] = wp.vec2d(taken, planned[1])
    wp.atomic_add(sums, 0, taken)
    wp.atomic_add(sums, 1, planned[1])


@wp.kernel
def _give(psi: wp.array3d(dtype=wp.vec2d), spread: wp.array3d(dtype=wp.vec2d), low: wp.vec3i,
          sums: wp.array(dtype=wp.float64)):
    ti, tj, tk = wp.tid()
    i = ti + low[0]
    j = tj + low[1]
    k = tk + low[2]
    planned = spread[i, j, k]
    if planned[0] == wp.float64(0.0) and planned[1] == wp.float64(0.0):
        return
    spread[i, j, k] = wp.vec2d(wp.float64(0.0), wp.float64(0.0))
    if sums[1] <= wp.float64(0.0):
        return
    held = psi[i, j, k]
    energy = wp.dot(held, held)
    after = wp.max(energy - planned[0] + planned[1] * sums[0] / sums[1], wp.float64(0.0))
    if energy > wp.float64(0.0):
        psi[i, j, k] = held * wp.sqrt(after / energy)
    else:
        psi[i, j, k] = wp.vec2d(wp.sqrt(after), wp.float64(0.0))


@wp.kernel
def _book(sums: wp.array(dtype=wp.float64), moved: wp.array(dtype=wp.float64)):
    moved[0] = moved[0] + sums[0]


def spread_heat(universe: Universe, settings: Settings) -> None:
    """Spread the energy friction took this substep into the boxes it planned, exactly; book it."""
    if settings.friction <= 0.0:
        return
    device = universe.owner.device
    dims = tuple(top - bottom for bottom, top in zip(universe.low, universe.high))
    sums = wp.zeros(2, dtype=wp.float64, device=device)
    fluid_energy(universe, settings, clocked=True, into=universe.spread_work, sign=-1.0)
    wp.launch(_take, dim=dims, device=device,
              inputs=[universe.owner, universe.psi, universe.spread, wp.vec3i(*universe.low), sums])
    wp.launch(_give, dim=dims, device=device, inputs=[universe.psi, universe.spread, wp.vec3i(*universe.low), sums])
    wp.launch(_book, dim=1, device=device, inputs=[sums, universe.spread_moved])
    fluid_energy(universe, settings, clocked=True, into=universe.spread_work, sign=1.0)
