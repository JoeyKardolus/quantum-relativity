"""Hop: the energy fluid moves between face-neighbouring Planck boxes, one exact rotation per box pair.

Contract: `hop(universe, settings, threshold, axis, parity, fraction)` advances the
kinetic part of the energy fluid's equation (Gross-Pitaevskii, hbar = 1, box mass 1:
i dpsi/dt = -1/2 sum over faces t_f (psi_there - psi) + on-site terms, on_site.py)
across the box faces along `axis` whose lower box has that coordinate's `parity`. Those
faces share no box, so each pair (a, b) is advanced on its own by the exact solution
of its two-box equation, H = t/2 [[1, -1], [-1, 1]]: the mean (a + b)/2 stays and
the half difference (a - b)/2 turns by exp(-i t dt). That is a unitary 2x2 rotation,
so |a|**2 + |b|**2, the pair's energy, is kept to rounding (float64): total energy is
conserved however long dt is (the product-formula method, De Raedt 1987; the twelve
calls of a substep, every axis and parity forward then back, make a second-order
Strang splitting, step.py).
  t    the face's openness (border_openness.py): 1 inside a cell, through a joint only
       at its seams, less the more squeezed, 0 past the seal, into empty space or
       between two energy volumes. No hop through a closed face.
  dt   fraction * step_time / 2**L * sqrt(clock_a * clock_b), L the finer of the two
       boxes' time levels (time_levels.py); the pair moves only when L >= threshold
       (it is due at this substep). Over one step every pair moves for step_time of
       the counter, each in its own time (bible: time is local).
Friction, only where the fluid crosses between an energy volume and a joint: after the
rotation the phase difference across the face, delta = arg(b conj(a)) (the flow
there), keeps kept = 1 / (1 + friction * c * dt) of itself, c the joint's compression
(more squeezed, more friction), by turning a and b towards each other by (1 - kept)
delta / 2 each. Their sizes do not change, so no energy is made or lost (bible:
friction turns directed flow into jiggling, heat). The turn changes the flow energy
of every face that touches a turned box, not only the rubbed one, so the drop over all
of them (each face once, every turned pair of this call turned together) is booked as
heat in the rubbed pair's joint box, in the clock-weighted fluid energy the steps keep
(fluid_energy.py clocked: sqrt(clock_a clock_b) t/2 |psi_b - psi_a|**2 per open face,
same t): friction's heat equals the flow energy it took, to rounding. The
rotations run first and work out friction for every rubbed pair (scratch "rub_before",
"rub_after", the rubbed pair's joint box "rub_joint", marked with this call's number in
"rub_mark"); a second launch books and applies it, every box reading its neighbours'
before and after values from the scratch, so no pair reads a neighbour mid-change.
Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.potts.border_openness import border_openness
from physics.potts.direction import direction
from physics.potts.settings import Settings
from physics.potts.universe import Universe


@wp.func
def _times(first: wp.vec2d, second: wp.vec2d) -> wp.vec2d:
    return wp.vec2d(first[0] * second[0] - first[1] * second[1], first[0] * second[1] + first[1] * second[0])


@wp.func
def _turn(angle: wp.float64) -> wp.vec2d:
    return wp.vec2d(wp.cos(angle), wp.sin(angle))


@wp.kernel
def _hop(owner: wp.array3d(dtype=wp.int32), kind: wp.array(dtype=wp.int32), compression: wp.array(dtype=wp.float32),
         seam: wp.array3d(dtype=wp.int32), clock: wp.array(dtype=wp.float32), box_level: wp.array3d(dtype=wp.int32),
         threshold: int, step_time: float, fraction: float, seal: float, friction: float, low: wp.vec3i,
         high: wp.vec3i, axis: int, first: int, side: int, call: int, psi: wp.array3d(dtype=wp.vec2d),
         rub_before: wp.array3d(dtype=wp.vec2d), rub_after: wp.array3d(dtype=wp.vec2d),
         rub_joint: wp.array3d(dtype=wp.int32), rub_mark: wp.array3d(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    here = wp.vec3i(low[0] + ti, low[1] + tj, low[2] + tk)
    if axis == 0:
        here[0] = first + 2 * ti
    elif axis == 1:
        here[1] = first + 2 * tj
    else:
        here[2] = first + 2 * tk
    there = here + direction(2 * axis)
    if here[axis] >= high[axis] or there[axis] >= high[axis]:
        return
    if here[0] >= high[0] or here[1] >= high[1] or here[2] >= high[2]:
        return
    level = wp.max(box_level[here[0], here[1], here[2]], box_level[there[0], there[1], there[2]])
    if level < threshold:
        return
    openness = border_openness(owner, kind, compression, seam, here, there, seal, side)
    if openness <= 0.0:
        return
    near = owner[here[0], here[1], here[2]]
    far = owner[there[0], there[1], there[2]]
    dt = fraction * step_time / wp.pow(2.0, float(level)) * wp.sqrt(clock[near] * clock[far])
    upper = psi[here[0], here[1], here[2]]
    lower = psi[there[0], there[1], there[2]]
    mean = (upper + lower) * wp.float64(0.5)
    half = _times((upper - lower) * wp.float64(0.5), _turn(wp.float64(-openness * dt)))
    upper = mean + half
    lower = mean - half
    psi[here[0], here[1], here[2]] = upper
    psi[there[0], there[1], there[2]] = lower
    if kind[near] != kind[far] and friction > 0.0:
        joint = far
        joint_box = there
        if kind[near] == 1:
            joint = near
            joint_box = here
        kept = 1.0 / (1.0 + friction * wp.max(compression[joint], 0.0) * dt)
        cross = _times(lower, wp.vec2d(upper[0], -upper[1]))
        delta = wp.atan2(cross[1], cross[0])
        shift = wp.float64(0.5 * (1.0 - kept)) * delta
        place = (joint_box[0] * side + joint_box[1]) * side + joint_box[2]
        rub_before[here[0], here[1], here[2]] = upper
        rub_before[there[0], there[1], there[2]] = lower
        rub_after[here[0], here[1], here[2]] = _times(upper, _turn(shift))
        rub_after[there[0], there[1], there[2]] = _times(lower, _turn(-shift))
        rub_joint[here[0], here[1], here[2]] = place
        rub_joint[there[0], there[1], there[2]] = place
        rub_mark[here[0], here[1], here[2]] = call
        rub_mark[there[0], there[1], there[2]] = call


@wp.kernel
def _book(owner: wp.array3d(dtype=wp.int32), kind: wp.array(dtype=wp.int32), compression: wp.array(dtype=wp.float32),
          seam: wp.array3d(dtype=wp.int32), clock: wp.array(dtype=wp.float32), seal: float, low: wp.vec3i,
          side: int, call: int, rub_before: wp.array3d(dtype=wp.vec2d), rub_after: wp.array3d(dtype=wp.vec2d),
          rub_joint: wp.array3d(dtype=wp.int32), rub_mark: wp.array3d(dtype=wp.int32),
          psi: wp.array3d(dtype=wp.vec2d), heat: wp.array3d(dtype=wp.float64)):
    ti, tj, tk = wp.tid()
    here = wp.vec3i(low[0] + ti, low[1] + tj, low[2] + tk)
    if rub_mark[here[0], here[1], here[2]] != call:
        return
    held = rub_before[here[0], here[1], here[2]]
    turned = rub_after[here[0], here[1], here[2]]
    index = (here[0] * side + here[1]) * side + here[2]
    lost = wp.float64(0.0)
    for face in range(6):
        there = here + direction(face)
        if there[0] < 0 or there[1] < 0 or there[2] < 0 or there[0] >= side or there[1] >= side or there[2] >= side:
            continue
        openness = border_openness(owner, kind, compression, seam, here, there, seal, side)
        if openness <= 0.0:
            continue
        other_before = psi[there[0], there[1], there[2]]
        other_after = other_before
        if rub_mark[there[0], there[1], there[2]] == call:
            if (there[0] * side + there[1]) * side + there[2] < index:
                continue
            other_before = rub_before[there[0], there[1], there[2]]
            other_after = rub_after[there[0], there[1], there[2]]
        weight = wp.sqrt(clock[owner[here[0], here[1], here[2]]] * clock[owner[there[0], there[1], there[2]]])
        before = other_before - held
        after = other_after - turned
        lost += wp.float64(0.5 * openness * weight) * (wp.dot(before, before) - wp.dot(after, after))
    place = rub_joint[here[0], here[1], here[2]]
    wp.atomic_add(heat, place // (side * side), (place // side) % side, place % side, lost)
    psi[here[0], here[1], here[2]] = turned


def hop(universe: Universe, settings: Settings, threshold: int, axis: int, parity: int, fraction: float) -> None:
    """Advance the fluid's hops across the due faces along `axis` whose lower box has `parity`, exactly."""
    low = list(universe.low)
    high = list(universe.high)
    first = low[axis] + (parity - low[axis]) % 2
    dims = [top - bottom for bottom, top in zip(low, high)]
    region = tuple(dims)
    dims[axis] = max(0, -(-(high[axis] - first) // 2))
    if min(dims) == 0:
        return
    device = universe.owner.device
    rub_before, rub_after, rub_joint, rub_mark = _rub_scratch(universe)
    call = universe.scratch["rub_call"] = universe.scratch["rub_call"] % (2 ** 30) + 1
    wp.launch(_hop, dim=tuple(dims), device=device,
              inputs=[universe.owner, universe.kind, universe.cell_compression, universe.seam, universe.cell_clock,
                      universe.box_level, int(threshold), settings.step_time, float(fraction),
                      settings.seal_compression, settings.friction, wp.vec3i(*low), wp.vec3i(*high), int(axis),
                      int(first), universe.side, call, universe.psi, rub_before, rub_after, rub_joint, rub_mark])
    if settings.friction <= 0.0:
        return
    wp.launch(_book, dim=region, device=device,
              inputs=[universe.owner, universe.kind, universe.cell_compression, universe.seam, universe.cell_clock,
                      settings.seal_compression, wp.vec3i(*low), universe.side, call, rub_before, rub_after,
                      rub_joint, rub_mark, universe.psi, universe.heat])


def _rub_scratch(universe: Universe) -> tuple:
    """Friction's per-box scratch (before, after, joint box, call mark), made once per universe."""
    if "rub_mark" not in universe.scratch:
        shape = (universe.side,) * 3
        device = universe.owner.device
        universe.scratch["rub_before"] = wp.zeros(shape, dtype=wp.vec2d, device=device)
        universe.scratch["rub_after"] = wp.zeros(shape, dtype=wp.vec2d, device=device)
        universe.scratch["rub_joint"] = wp.zeros(shape, dtype=wp.int32, device=device)
        universe.scratch["rub_mark"] = wp.zeros(shape, dtype=wp.int32, device=device)
        universe.scratch["rub_call"] = 0
    return (universe.scratch["rub_before"], universe.scratch["rub_after"], universe.scratch["rub_joint"],
            universe.scratch["rub_mark"])
