"""On site: the energy fluid's phase turns in each box by the pull and the fluid's own push.

Contract: `on_site(universe, settings, threshold, fraction)` advances the on-site part
of the energy fluid's equation (Gross-Pitaevskii, hbar = 1, box mass 1; the hops are
hop.py) for the boxes due this substep (the first universe.scratch["site_prefix"]
[threshold] boxes of "site_order", time_levels.py), each for its own dt = fraction *
step_time / 2**level times its cell's clock (bible: time is local):
  psi <- psi * exp(-i (V + g_box |psi|**2) dt)
  V      the pull potential (pull.py), in the energy volumes' boxes only: sourced by
         the energy volumes' energy and felt by it alone (bible: A pulls on nearby
         A's), so the pull on all A-energy sums to 0 (Newton's third law); a joint's
         box turns by its push only
  g_box  the fluid's own push (its pressure), g = sound**2 / rest, so its sound speed
         is `sound` at the start's mean density, in both kinds (one fluid); in a
         joint times (1 + push * c**push_power), c its compression: the joint's push
         as the nonlinear term, stiffer the more it is squeezed
This is the exact solution of the on-site equation (|psi| does not change while it
runs), so each box's energy |psi|**2 is kept exactly. Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.potts.settings import Settings
from physics.potts.universe import Universe


@wp.kernel
def _site(order: wp.array(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32), kind: wp.array(dtype=wp.int32),
          potential: wp.array3d(dtype=wp.float32), compression: wp.array(dtype=wp.float32),
          clock: wp.array(dtype=wp.float32), box_level: wp.array3d(dtype=wp.int32), fluid_g: float, push: float,
          push_power: float, step_time: float, fraction: float, side: int, psi: wp.array3d(dtype=wp.vec2d)):
    index = order[wp.tid()]
    i = index // (side * side)
    j = (index // side) % side
    k = index % side
    cell = owner[i, j, k]
    held = psi[i, j, k]
    stiffness = fluid_g
    felt = wp.float64(potential[i, j, k])
    if kind[cell] == 1:
        stiffness = fluid_g * (1.0 + push * wp.pow(wp.max(compression[cell], 0.0), push_power))
        felt = wp.float64(0.0)
    dt = fraction * step_time / wp.pow(2.0, float(box_level[i, j, k])) * clock[cell]
    angle = -(felt + wp.float64(stiffness) * wp.dot(held, held)) * wp.float64(dt)
    turn_re = wp.cos(angle)
    turn_im = wp.sin(angle)
    psi[i, j, k] = wp.vec2d(held[0] * turn_re - held[1] * turn_im, held[0] * turn_im + held[1] * turn_re)


def on_site(universe: Universe, settings: Settings, threshold: int, fraction: float) -> None:
    """Turn the due boxes' phase by the pull and the fluid's push, exactly."""
    count = universe.scratch["site_prefix"][threshold]
    if count == 0:
        return
    wp.launch(_site, dim=count, device=universe.owner.device,
              inputs=[universe.scratch["site_order"], universe.owner, universe.kind, universe.potential,
                      universe.cell_compression, universe.cell_clock, universe.box_level,
                      settings.sound * settings.sound / universe.rest, settings.push, settings.push_power,
                      settings.step_time, float(fraction), universe.side, universe.psi])
