"""Fluid energy: the energy fluid's own energy in its equation, summed on the GPU.

Contract: `fluid_energy(universe, settings, clocked=False, into=None, sign=1.0)` marks the
seam boxes of the boxes' present owners (seam_flags.py) and returns, as a Python float,
the energy the fluid's equation (hop.py, on_site.py) would keep if nothing else changed
(given `into`, a one-entry float64 Warp array, it instead adds sign times that energy
to into[0] on the GPU, without waiting for it, and returns None):
  sum over box faces of w_f t/2 |psi_b - psi_a|**2     (flow, t the face's openness)
  + sum over boxes of w (V |psi|**2 / 2               (pull: A boxes only; half, the
                                                        self-energy of a field the
                                                        A-energy exerts on itself)
  + g_box / 2 |psi|**4)                                (the fluid's push, a joint's stiffer)
with V, g_box as on_site.py has them and t from border_openness.py. Unclocked (the
default) every weight is 1: the energy in proper time, the units friction's heat is
booked in (hop.py). clocked=True weighs each box by its cell's clock (w = clock) and
each face by sqrt(clock_a clock_b), as the hops and turns advance them: that is the
quantity the local-clock steps keep (up to the time levels' splitting error) while V,
the openness and g_box hold still. The pull's re-solve, box moves, the joints' changing
compression and friction (which turns its flow part into heat) change both. Reported
beside total_energy, which is the rule (the sum of |psi|**2). Float64 atomics: the
last bits depend on thread order.
"""
from __future__ import annotations

import warp as wp

from physics.potts.border_openness import border_openness
from physics.potts.direction import direction
from physics.potts.seam_flags import seam_flags
from physics.potts.settings import Settings
from physics.potts.universe import Universe


@wp.kernel
def _sum(owner: wp.array3d(dtype=wp.int32), kind: wp.array(dtype=wp.int32), compression: wp.array(dtype=wp.float32),
         seam: wp.array3d(dtype=wp.int32), potential: wp.array3d(dtype=wp.float32), psi: wp.array3d(dtype=wp.vec2d),
         clock: wp.array(dtype=wp.float32), clocked: int, fluid_g: float, push: float, push_power: float,
         seal: float, low: wp.vec3i, side: int, sign: float, total: wp.array(dtype=wp.float64)):
    ti, tj, tk = wp.tid()
    here = wp.vec3i(ti + low[0], tj + low[1], tk + low[2])
    cell = owner[here[0], here[1], here[2]]
    if cell < 0:
        return
    held = psi[here[0], here[1], here[2]]
    density = wp.dot(held, held)
    stiffness = fluid_g
    felt = wp.float64(0.5) * wp.float64(potential[here[0], here[1], here[2]])
    if kind[cell] == 1:
        stiffness = fluid_g * (1.0 + push * wp.pow(wp.max(compression[cell], 0.0), push_power))
        felt = wp.float64(0.0)
    weight = float(1.0)
    if clocked != 0:
        weight = clock[cell]
    energy = wp.float64(weight) * (felt + wp.float64(0.5 * stiffness) * density) * density
    for axis in range(3):
        there = here + direction(2 * axis)
        if there[axis] < side:
            openness = border_openness(owner, kind, compression, seam, here, there, seal, side)
            if openness > 0.0:
                face_weight = float(1.0)
                if clocked != 0:
                    face_weight = wp.sqrt(clock[cell] * clock[owner[there[0], there[1], there[2]]])
                gap = psi[there[0], there[1], there[2]] - held
                energy += wp.float64(0.5 * openness * face_weight) * wp.dot(gap, gap)
    wp.atomic_add(total, 0, wp.float64(sign) * energy)


def fluid_energy(universe: Universe, settings: Settings, clocked: bool = False, into: wp.array | None = None,
                 sign: float = 1.0) -> float | None:
    """The energy fluid's flow, pull and push energy in its equation, in proper time or clock-weighted."""
    seam_flags(universe)
    device = universe.owner.device
    total = wp.zeros(1, dtype=wp.float64, device=device) if into is None else into
    dims = tuple(top - bottom for bottom, top in zip(universe.low, universe.high))
    wp.launch(_sum, dim=dims, device=device,
              inputs=[universe.owner, universe.kind, universe.cell_compression, universe.seam, universe.potential,
                      universe.psi, universe.cell_clock, int(bool(clocked)), settings.sound * settings.sound /
                      universe.rest, settings.push, settings.push_power, settings.seal_compression,
                      wp.vec3i(*universe.low), universe.side, float(sign), total])
    return float(total.numpy()[0]) if into is None else None
