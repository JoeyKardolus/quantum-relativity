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
Friction, only where flow actually crosses between an energy volume and a joint (bible:
friction arises only in passing through a joint): after every due pair of the call has
turned, each A-B pair (a, b) is rubbed on its own (two passes, colour (coordinate
along axis // 2 + the other two) mod 2, so no two pairs rubbed together share a face).
The flow crossing is the phase difference delta = arg(b conj a) as far as the flow on
both sides carries it: the face behind a in a's cell and the face beyond b in b's
cell (along the axis, both open inside their cell) must turn the same way, and the
crossing flow is the smallest of the three, sign kept (0 when either side does not
flow that way, or has no box there). A fluid at rest, whose cells' phases turn at
different on-site rates (on_site.py), has a phase step only at the face and no flow
behind it: it makes no heat. a and b turn towards each other by (1 - kept) * flow /
2 each, kept = 1 / (1 + friction * c * dt), c the joint's compression (more squeezed,
more friction). Their sizes do not change, so no energy is made or lost (bible:
friction turns directed flow into jiggling, heat). The turn changes the flow energy
of all eleven faces the two boxes touch (fluid_energy.py clocked: sqrt(clock_a
clock_b) t/2 |psi_b - psi_a|**2 per open face, same t); that drop is worked out
exactly (no neighbour turns in the same pass) and booked as heat in the pair's joint
box. A turn that would not lower it is halved, up to three times, then not made:
friction never books negative heat, and its heat is the flow energy it took, to
rounding. Returns None.
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
         threshold: int, step_time: float, fraction: float, seal: float, low: wp.vec3i, high: wp.vec3i,
         axis: int, first: int, side: int, psi: wp.array3d(dtype=wp.vec2d)):
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


@wp.func
def _phase_step(first: wp.vec2d, second: wp.vec2d) -> wp.float64:
    """arg(second conj first), the phase turned from first to second."""
    cross = _times(second, wp.vec2d(first[0], -first[1]))
    return wp.atan2(cross[1], cross[0])


@wp.func
def _behind(owner: wp.array3d(dtype=wp.int32), psi: wp.array3d(dtype=wp.vec2d), box: wp.vec3i, step: wp.vec3i,
            side: int) -> wp.float64:
    """The phase step across the face from box - step to box inside one cell, 0 when the two are not one cell's."""
    back = box - step
    if back[0] < 0 or back[1] < 0 or back[2] < 0 or back[0] >= side or back[1] >= side or back[2] >= side:
        return wp.float64(0.0)
    if box[0] < 0 or box[1] < 0 or box[2] < 0 or box[0] >= side or box[1] >= side or box[2] >= side:
        return wp.float64(0.0)
    if owner[back[0], back[1], back[2]] != owner[box[0], box[1], box[2]]:
        return wp.float64(0.0)
    return _phase_step(psi[back[0], back[1], back[2]], psi[box[0], box[1], box[2]])


@wp.func
def _crossing(delta: wp.float64, near_side: wp.float64, far_side: wp.float64) -> wp.float64:
    """The smallest of the three phase steps when all turn the same way, else 0."""
    if delta * near_side <= wp.float64(0.0) or delta * far_side <= wp.float64(0.0):
        return wp.float64(0.0)
    size = wp.min(wp.abs(delta), wp.min(wp.abs(near_side), wp.abs(far_side)))
    return wp.sign(delta) * size


@wp.func
def _pair_flow_energy(owner: wp.array3d(dtype=wp.int32), kind: wp.array(dtype=wp.int32),
                      compression: wp.array(dtype=wp.float32), seam: wp.array3d(dtype=wp.int32),
                      clock: wp.array(dtype=wp.float32), psi: wp.array3d(dtype=wp.vec2d), seal: float, side: int,
                      near: wp.vec3i, far: wp.vec3i, near_psi: wp.vec2d, far_psi: wp.vec2d) -> wp.float64:
    """The clocked flow energy of the eleven faces the pair (near, far) touches, the pair holding these values."""
    energy = wp.float64(0.0)
    for pick in range(2):
        box = near
        held = near_psi
        if pick == 1:
            box = far
            held = far_psi
        for face in range(6):
            there = box + direction(face)
            if there[0] < 0 or there[1] < 0 or there[2] < 0 or there[0] >= side or there[1] >= side or there[2] >= side:
                continue
            is_near = there[0] == near[0] and there[1] == near[1] and there[2] == near[2]
            is_far = there[0] == far[0] and there[1] == far[1] and there[2] == far[2]
            if pick == 1 and is_near:
                continue
            openness = border_openness(owner, kind, compression, seam, box, there, seal, side)
            if openness <= 0.0:
                continue
            other = psi[there[0], there[1], there[2]]
            if is_far:
                other = far_psi
            if is_near:
                other = near_psi
            weight = wp.sqrt(clock[owner[box[0], box[1], box[2]]] * clock[owner[there[0], there[1], there[2]]])
            gap = other - held
            energy += wp.float64(0.5 * openness * weight) * wp.dot(gap, gap)
    return energy


@wp.kernel
def _rub(owner: wp.array3d(dtype=wp.int32), kind: wp.array(dtype=wp.int32), compression: wp.array(dtype=wp.float32),
         seam: wp.array3d(dtype=wp.int32), clock: wp.array(dtype=wp.float32), box_level: wp.array3d(dtype=wp.int32),
         threshold: int, step_time: float, fraction: float, seal: float, friction: float, low: wp.vec3i,
         high: wp.vec3i, axis: int, first: int, side: int, colour: int, psi: wp.array3d(dtype=wp.vec2d),
         heat: wp.array3d(dtype=wp.float64)):
    ti, tj, tk = wp.tid()
    here = wp.vec3i(low[0] + ti, low[1] + tj, low[2] + tk)
    if axis == 0:
        here[0] = first + 2 * ti
    elif axis == 1:
        here[1] = first + 2 * tj
    else:
        here[2] = first + 2 * tk
    step = direction(2 * axis)
    there = here + step
    if here[axis] >= high[axis] or there[axis] >= high[axis]:
        return
    if here[0] >= high[0] or here[1] >= high[1] or here[2] >= high[2]:
        return
    if (here[axis] // 2 + here[(axis + 1) % 3] + here[(axis + 2) % 3]) % 2 != colour:
        return
    near = owner[here[0], here[1], here[2]]
    far = owner[there[0], there[1], there[2]]
    if near < 0 or far < 0 or kind[near] == kind[far]:
        return
    level = wp.max(box_level[here[0], here[1], here[2]], box_level[there[0], there[1], there[2]])
    if level < threshold:
        return
    openness = border_openness(owner, kind, compression, seam, here, there, seal, side)
    if openness <= 0.0:
        return
    joint = far
    joint_box = there
    if kind[near] == 1:
        joint = near
        joint_box = here
    dt = fraction * step_time / wp.pow(2.0, float(level)) * wp.sqrt(clock[near] * clock[far])
    kept = 1.0 / (1.0 + friction * wp.max(compression[joint], 0.0) * dt)
    upper = psi[here[0], here[1], here[2]]
    lower = psi[there[0], there[1], there[2]]
    delta = _phase_step(upper, lower)
    flow = _crossing(delta, _behind(owner, psi, here, step, side), _behind(owner, psi, there + step, step, side))
    if flow == wp.float64(0.0):
        return
    before = _pair_flow_energy(owner, kind, compression, seam, clock, psi, seal, side, here, there, upper, lower)
    shift = wp.float64(0.5 * (1.0 - kept)) * flow
    for attempt in range(4):
        turned_upper = _times(upper, _turn(shift))
        turned_lower = _times(lower, _turn(-shift))
        lost = before - _pair_flow_energy(owner, kind, compression, seam, clock, psi, seal, side, here, there,
                                          turned_upper, turned_lower)
        if lost > wp.float64(0.0):
            psi[here[0], here[1], here[2]] = turned_upper
            psi[there[0], there[1], there[2]] = turned_lower
            wp.atomic_add(heat, joint_box[0], joint_box[1], joint_box[2], lost)
            return
        shift = shift * wp.float64(0.5)


def hop(universe: Universe, settings: Settings, threshold: int, axis: int, parity: int, fraction: float) -> None:
    """Advance the fluid's hops across the due faces along `axis` whose lower box has `parity`, exactly."""
    low = list(universe.low)
    high = list(universe.high)
    first = low[axis] + (parity - low[axis]) % 2
    dims = [top - bottom for bottom, top in zip(low, high)]
    dims[axis] = max(0, -(-(high[axis] - first) // 2))
    if min(dims) == 0:
        return
    device = universe.owner.device
    wp.launch(_hop, dim=tuple(dims), device=device,
              inputs=[universe.owner, universe.kind, universe.cell_compression, universe.seam, universe.cell_clock,
                      universe.box_level, int(threshold), settings.step_time, float(fraction),
                      settings.seal_compression, wp.vec3i(*low), wp.vec3i(*high), int(axis), int(first),
                      universe.side, universe.psi])
    if settings.friction <= 0.0:
        return
    for colour in range(2):
        wp.launch(_rub, dim=tuple(dims), device=device,
                  inputs=[universe.owner, universe.kind, universe.cell_compression, universe.seam,
                          universe.cell_clock, universe.box_level, int(threshold), settings.step_time,
                          float(fraction), settings.seal_compression, settings.friction, wp.vec3i(*low),
                          wp.vec3i(*high), int(axis), int(first), universe.side, colour, universe.psi,
                          universe.heat])
