"""Fluid energy: the energy fluid's own energy in its equation, summed on the GPU.

Contract: `fluid_energy(universe, settings)` returns, as a Python float, the energy the
fluid's equation (hop.py, on_site.py) would keep if nothing else changed:
  sum over box faces of t/2 |psi_b - psi_a|**2     (flow, t the face's openness)
  + sum over boxes of V |psi|**2                   (pull)
  + sum over boxes of g_box / 2 |psi|**4           (the fluid's push, a joint's stiffer)
with V, g_box as on_site.py has them and t from border_openness.py with this step's
seam flags. The rotations keep it while V, the openness and g_box hold still; the
pull's re-solve, box moves, the joints' changing compression and friction (which turns
its flow part into heat) change it. It is reported beside total_energy, which is the
rule (the sum of |psi|**2). Float64 atomics: the last bits depend on thread order.
"""
from __future__ import annotations

import warp as wp

from physics.potts.border_openness import border_openness
from physics.potts.direction import direction
from physics.potts.settings import Settings
from physics.potts.universe import Universe


@wp.kernel
def _sum(owner: wp.array3d(dtype=wp.int32), kind: wp.array(dtype=wp.int32), compression: wp.array(dtype=wp.float32),
         seam: wp.array3d(dtype=wp.int32), potential: wp.array3d(dtype=wp.float32), psi: wp.array3d(dtype=wp.vec2d),
         fluid_g: float, push: float, push_power: float, seal: float, low: wp.vec3i, side: int,
         total: wp.array(dtype=wp.float64)):
    ti, tj, tk = wp.tid()
    here = wp.vec3i(ti + low[0], tj + low[1], tk + low[2])
    cell = owner[here[0], here[1], here[2]]
    if cell < 0:
        return
    held = psi[here[0], here[1], here[2]]
    density = wp.dot(held, held)
    stiffness = fluid_g
    if kind[cell] == 1:
        stiffness = fluid_g * (1.0 + push * wp.pow(wp.max(compression[cell], 0.0), push_power))
    energy = (wp.float64(potential[here[0], here[1], here[2]]) + wp.float64(0.5 * stiffness) * density) * density
    for axis in range(3):
        there = here + direction(2 * axis)
        if there[axis] < side:
            openness = border_openness(owner, kind, compression, seam, here, there, seal, side)
            if openness > 0.0:
                gap = psi[there[0], there[1], there[2]] - held
                energy += wp.float64(0.5 * openness) * wp.dot(gap, gap)
    wp.atomic_add(total, 0, energy)


def fluid_energy(universe: Universe, settings: Settings) -> float:
    """The energy fluid's flow, pull and push energy in its equation."""
    device = universe.owner.device
    total = wp.zeros(1, dtype=wp.float64, device=device)
    dims = tuple(top - bottom for bottom, top in zip(universe.low, universe.high))
    wp.launch(_sum, dim=dims, device=device,
              inputs=[universe.owner, universe.kind, universe.cell_compression, universe.seam, universe.potential,
                      universe.psi, settings.sound * settings.sound / universe.rest, settings.push,
                      settings.push_power, settings.seal_compression, wp.vec3i(*universe.low), universe.side, total])
    return float(total.numpy()[0])
