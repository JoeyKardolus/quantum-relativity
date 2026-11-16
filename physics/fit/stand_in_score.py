"""Stand-in score: the fit score on a smooth stand-in of the engine, so Warp can take its gradient by the settings.

Contract: `stand_in_score(start, units, steps, every, block=10, device="cuda")` returns
(score, parts), two Warp arrays on `device`: score (U,) and parts (U, 5) = E, dip, Sc, St,
ln(R_last / R_first) per universe. `start` is one universe's signed whole quanta (a cube,
numpy, as physics.engine.host_boxes gives it); `units` is a float32 Warp array (U, 6),
one point of the fit's unit cube per universe (physics.fit.fit_search NAMES and ranges;
seal_squeeze = collapse_squeeze * seal_fraction). The U universes run side by side, each
with its own settings, from the same start. Called inside a `wp.Tape`, the tape holds the
whole run and `tape.backward(grads={score: ones})` gives d score / d units in units.grad.

The stand-in is the engine's fluid without its whole numbers: per box the complex
psi = sqrt(n) e^{i phase} in float32, n = |psi|^2 the quanta as a real number, each box
keeping its start's sign. Each step is the engine's (physics.engine.step): the pull's
leapfrog wave sourced by plus quanta, then Strang splitting of on_site (the box
potential on local clocks) and De Raedt pair rotations along x, y, z by parity, forward
then back, with an openness graded by the joint's squeeze and shut past the seal, and
friction and drift spreading, on plus-minus faces. Here it differs from the engine,
which since 2026-11-16 moves no quanta across a plus-minus face and instead passes plus
quanta between two plus boxes across a one-box joint (physics.engine.leak), so quanta
change sign only in collapse. What it leaves out, because
none of it has a gradient: the rounding to whole quanta, the claims (space does not
grow: every box keeps its kind) and the collapse (collapse_squeeze still acts through
the seal). Empty boxes and the grid's walls hold pull 0 and pass nothing, as in the
engine. Each array a pass writes is new, so the tape can run the run backwards. The
potentials, openness, clock rates and pull source read each pass's |psi|^2 from a copy
the tape does not record: the forward run is unchanged, but the gradient leaves out a
box's own quanta feeding back on its phase, which through the joints' steep push grows
the full gradient without bound (gradient_fit.py says by how much).

Score, measured every `every` steps on the occupied boxes, from the sample after step 0
on (as physics.fit.fit_run): 0.5 E + 0.5 S with
  E  tanh(dip), dip = min(r1 - r2, r3 - r2) / mean |r| of the growth rate d ln R / dt in
     thirds of the run (np.array_split), R the rms radius of the plus quanta round the
     grid centre, t the counter (the real score times by the observers' local clocks and
     the occupied radius, which cannot move without claims)
  S  (Sc + St) / 2, Sc = tanh(ln contrast ratio), contrast sqrt(mean n^2 / mean n^2 - 1);
     St = tanh(ln growth of the large-scale variance (block means over blocks of `block`
     boxes from the occupied cube's corner) minus ln growth of the small-scale one (mean
     squared step between face neighbours)); the real score's lump part has no gradient
     and is left out.
Raises ValueError for fewer than 4 samples after the first (steps // every < 5), a start
that is not a cube, or units not of shape (U, 6).
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.engine import Settings
from physics.fit.fit_search import LINEAR_RANGES, LOG_RANGES, NAMES

PI = 3.14159265358979
_SUMS = 8


@wp.kernel
def _values(units: wp.array2d(dtype=wp.float32), low: wp.array(dtype=wp.float32), high: wp.array(dtype=wp.float32),
            logged: wp.array(dtype=wp.int32), values: wp.array2d(dtype=wp.float32)):
    universe, index = wp.tid()
    unit = units[universe, index]
    if logged[index] == 1:
        values[universe, index] = wp.exp(wp.log(low[index]) + unit * (wp.log(high[index]) - wp.log(low[index])))
    else:
        values[universe, index] = low[index] + unit * (high[index] - low[index])


@wp.kernel
def _seal(values: wp.array2d(dtype=wp.float32), seal: wp.array(dtype=wp.float32)):
    universe = wp.tid()
    seal[universe] = values[universe, 2] * values[universe, 5]


@wp.func
def _rate(amount: float, clock_packing: float) -> float:
    """The engine's clock rate for a box of `amount` quanta."""
    return 1.0 / (1.0 + amount / clock_packing)


@wp.func
def _turn(psi: wp.vec2, angle: float) -> wp.vec2:
    """psi times e^{i angle}."""
    return wp.vec2(psi[0] * wp.cos(angle) - psi[1] * wp.sin(angle), psi[0] * wp.sin(angle) + psi[1] * wp.cos(angle))


@wp.func
def _pull_at(pull: wp.array3d(dtype=wp.float32), home: int, side: int, x: int, y: int, z: int) -> float:
    """The pull at a local place of universe `home`; 0 beyond its walls."""
    if x < 0 or y < 0 or z < 0 or x >= side or y >= side or z >= side:
        return 0.0
    return pull[home * side + x, y, z]


@wp.kernel
def _density(psi: wp.array3d(dtype=wp.vec2), density: wp.array3d(dtype=wp.float32)):
    ti, tj, tk = wp.tid()
    density[ti, tj, tk] = wp.length_sq(psi[ti, tj, tk])


@wp.kernel
def _speed_up(sign: wp.array3d(dtype=wp.int32), density: wp.array3d(dtype=wp.float32), pull: wp.array3d(dtype=wp.float32),
              rate_in: wp.array3d(dtype=wp.float32), values: wp.array2d(dtype=wp.float32), side: int, kick: float,
              rate_out: wp.array3d(dtype=wp.float32)):
    ti, tj, tk = wp.tid()
    home = ti // side
    x = ti - home * side
    result = float(0.0)
    if sign[ti, tj, tk] != 0:
        spread = (_pull_at(pull, home, side, x - 1, tj, tk) + _pull_at(pull, home, side, x + 1, tj, tk)
                  + _pull_at(pull, home, side, x, tj - 1, tk) + _pull_at(pull, home, side, x, tj + 1, tk)
                  + _pull_at(pull, home, side, x, tj, tk - 1) + _pull_at(pull, home, side, x, tj, tk + 1)
                  - 6.0 * pull[ti, tj, tk])
        source = float(0.0)
        if sign[ti, tj, tk] > 0:
            source = density[ti, tj, tk]
        result = rate_in[ti, tj, tk] + kick * (spread - 4.0 * PI * values[home, 0] * source)
    rate_out[ti, tj, tk] = result


@wp.kernel
def _move(sign: wp.array3d(dtype=wp.int32), pull_in: wp.array3d(dtype=wp.float32),
          rate: wp.array3d(dtype=wp.float32), step_time: float, pull_out: wp.array3d(dtype=wp.float32)):
    ti, tj, tk = wp.tid()
    result = float(0.0)
    if sign[ti, tj, tk] != 0:
        result = pull_in[ti, tj, tk] + step_time * rate[ti, tj, tk]
    pull_out[ti, tj, tk] = result


@wp.kernel
def _on_site(sign: wp.array3d(dtype=wp.int32), psi_in: wp.array3d(dtype=wp.vec2), density: wp.array3d(dtype=wp.float32),
             pull: wp.array3d(dtype=wp.float32),
             values: wp.array2d(dtype=wp.float32), side: int, duration: float, volume_rest: float,
             volume_stiffness: float, joint_rest: float, clock_packing: float, psi_out: wp.array3d(dtype=wp.vec2)):
    ti, tj, tk = wp.tid()
    home = ti // side
    here = psi_in[ti, tj, tk]
    amount = density[ti, tj, tk]
    energy = float(0.0)
    if sign[ti, tj, tk] > 0:
        energy = pull[ti, tj, tk] + volume_stiffness * amount / volume_rest
    if sign[ti, tj, tk] < 0:
        energy = values[home, 1] * wp.pow(amount / joint_rest, values[home, 4])
    psi_out[ti, tj, tk] = _turn(here, -energy * duration * _rate(amount, clock_packing))


@wp.kernel
def _hop(sign: wp.array3d(dtype=wp.int32), psi_in: wp.array3d(dtype=wp.vec2), density: wp.array3d(dtype=wp.float32),
         values: wp.array2d(dtype=wp.float32),
         seal: wp.array(dtype=wp.float32), side: int, axis: int, parity: int, duration: float, joint_rest: float,
         clock_packing: float, spread_share: float, psi_out: wp.array3d(dtype=wp.vec2)):
    ti, tj, tk = wp.tid()
    home = ti // side
    here = wp.vec3i(ti, tj, tk)
    local = wp.vec3i(ti - home * side, tj, tk)
    lower = local[axis] % 2 == parity
    there = here
    other = local[axis] + 1
    if lower:
        there[axis] = here[axis] + 1
    else:
        there[axis] = here[axis] - 1
        other = local[axis] - 1
    result = psi_in[ti, tj, tk]
    own_sign = sign[ti, tj, tk]
    if other >= 0 and other < side and own_sign != 0:
        their_sign = sign[there[0], there[1], there[2]]
        if their_sign != 0:
            near = psi_in[ti, tj, tk]
            far = psi_in[there[0], there[1], there[2]]
            near_sign = own_sign
            far_sign = their_sign
            if not lower:
                near = psi_in[there[0], there[1], there[2]]
                far = psi_in[ti, tj, tk]
                near_sign = their_sign
                far_sign = own_sign
            near_size = density[ti, tj, tk]
            far_size = density[there[0], there[1], there[2]]
            if not lower:
                near_size = density[there[0], there[1], there[2]]
                far_size = density[ti, tj, tk]
            crossing = (near_sign > 0) != (far_sign > 0)
            pressed = float(0.0)
            if crossing:
                joint_size = far_size
                if near_sign < 0:
                    joint_size = near_size
                pressed = wp.max(joint_size / joint_rest - 1.0, 0.0)
            open_share = 1.0 - pressed / seal[home]
            open_share = wp.max(open_share, 0.0)
            tau = duration * wp.sqrt(_rate(near_size, clock_packing) * _rate(far_size, clock_packing))
            mean = (near + far) * 0.5
            half = (near - far) * 0.5
            turned = _turn(half, -open_share * tau)
            near_after = mean + turned
            far_after = mean - turned
            if crossing:
                kept = 1.0 / (1.0 + values[home, 3] * (1.0 + pressed) * tau)
                apart = wp.atan2(near_after[0] * far_after[1] - near_after[1] * far_after[0],
                                 near_after[0] * far_after[0] + near_after[1] * far_after[1])
                rub = 0.5 * (1.0 - kept) * apart
                near_after = _turn(near_after, rub)
                far_after = _turn(far_after, -rub)
                near_new = wp.length_sq(near_after)
                far_new = wp.length_sq(far_after)
                spread = spread_share * wp.abs(wp.length_sq(near) - near_new)
                near_end = near_new
                far_end = far_new
                if near_new > far_new:
                    spread = wp.min(spread, 0.5 * (near_new - far_new))
                    near_end = near_new - spread
                    far_end = far_new + spread
                else:
                    spread = wp.min(spread, 0.5 * (far_new - near_new))
                    near_end = near_new + spread
                    far_end = far_new - spread
                near_after = near_after * wp.sqrt(near_end / wp.max(near_new, 1.0e-12))
                far_after = far_after * wp.sqrt(far_end / wp.max(far_new, 1.0e-12))
            if lower:
                result = near_after
            else:
                result = far_after
    psi_out[ti, tj, tk] = result


@wp.kernel
def _sample(sign: wp.array3d(dtype=wp.int32), psi: wp.array3d(dtype=wp.vec2), side: int, centre: float,
            corner: int, block: int, blocks: int, slot: int, sums: wp.array3d(dtype=wp.float32),
            block_sums: wp.array3d(dtype=wp.float32)):
    ti, tj, tk = wp.tid()
    home = ti // side
    x = ti - home * side
    if sign[ti, tj, tk] != 0:
        amount = wp.length_sq(psi[ti, tj, tk])
        wp.atomic_add(sums, home, slot, 0, amount)
        wp.atomic_add(sums, home, slot, 1, amount * amount)
        wp.atomic_add(sums, home, slot, 2, 1.0)
        if sign[ti, tj, tk] > 0:
            reach = (float(x) - centre) * (float(x) - centre) + (float(tj) - centre) * (float(tj) - centre) \
                + (float(tk) - centre) * (float(tk) - centre)
            wp.atomic_add(sums, home, slot, 3, amount * reach)
            wp.atomic_add(sums, home, slot, 4, amount)
        bx = (x - corner) // block
        by = (tj - corner) // block
        bz = (tk - corner) // block
        if x >= corner and tj >= corner and tk >= corner and bx < blocks and by < blocks and bz < blocks:
            wp.atomic_add(block_sums, home, slot, (bx * blocks + by) * blocks + bz, amount)
        for axis in range(3):
            step = wp.vec3i(x, tj, tk)
            step[axis] = step[axis] + 1
            if step[axis] < side:
                if sign[home * side + step[0], step[1], step[2]] != 0:
                    gap = amount - wp.length_sq(psi[home * side + step[0], step[1], step[2]])
                    wp.atomic_add(sums, home, slot, 5, gap * gap)
                    wp.atomic_add(sums, home, slot, 6, 1.0)


@wp.kernel
def _block_spread(block_sums: wp.array3d(dtype=wp.float32), boxes_per_block: float, count: int,
                  sums: wp.array3d(dtype=wp.float32)):
    home, slot, index = wp.tid()
    mean = block_sums[home, slot, index] / boxes_per_block
    wp.atomic_add(sums, home, slot, 7, mean * mean / float(count))


@wp.func
def _contrast(sums: wp.array3d(dtype=wp.float32), home: int, slot: int) -> float:
    mean = sums[home, slot, 0] / sums[home, slot, 2]
    return wp.sqrt(wp.max(sums[home, slot, 1] / sums[home, slot, 2] / (mean * mean) - 1.0, 1.0e-12))


@wp.func
def _large(sums: wp.array3d(dtype=wp.float32), home: int, slot: int) -> float:
    mean = sums[home, slot, 0] / sums[home, slot, 2]
    return wp.max(sums[home, slot, 7] - mean * mean, 1.0e-12) / (mean * mean)


@wp.func
def _small(sums: wp.array3d(dtype=wp.float32), home: int, slot: int) -> float:
    mean = sums[home, slot, 0] / sums[home, slot, 2]
    return sums[home, slot, 5] / sums[home, slot, 6] / (mean * mean)


@wp.func
def _radius(sums: wp.array3d(dtype=wp.float32), home: int, slot: int) -> float:
    return wp.sqrt(sums[home, slot, 3] / sums[home, slot, 4])


@wp.func
def _third_mean(sums: wp.array3d(dtype=wp.float32), home: int, first: int, count: int, span: float) -> float:
    """Mean growth rate d ln R / dt over `count` rates starting at rate `first` (rate i: samples 1 + i, 2 + i)."""
    total = float(0.0)
    for index in range(count):
        slot = 1 + first + index
        total = total + (wp.log(_radius(sums, home, slot + 1)) - wp.log(_radius(sums, home, slot))) / span
    return total / float(count)


@wp.kernel
def _score(sums: wp.array3d(dtype=wp.float32), samples: int, span: float, score: wp.array(dtype=wp.float32),
           parts: wp.array2d(dtype=wp.float32)):
    home = wp.tid()
    rates = samples - 2
    first_count = rates // 3 + wp.where(rates % 3 > 0, 1, 0)
    second_count = rates // 3 + wp.where(rates % 3 > 1, 1, 0)
    third_count = rates - first_count - second_count
    r1 = _third_mean(sums, home, 0, first_count, span)
    r2 = _third_mean(sums, home, first_count, second_count, span)
    r3 = _third_mean(sums, home, first_count + second_count, third_count, span)
    scale = (wp.abs(r1) + wp.abs(r2) + wp.abs(r3)) / 3.0 + 1.0e-9
    dip = wp.min(r1 - r2, r3 - r2) / scale
    expansion = wp.tanh(dip)
    last = samples - 1
    contrast_part = wp.tanh(wp.log(_contrast(sums, home, last) / _contrast(sums, home, 1)))
    tilt = wp.log(_large(sums, home, last) / _large(sums, home, 1)) - wp.log(_small(sums, home, last) / _small(sums, home, 1))
    tilt_part = wp.tanh(tilt)
    structure = 0.5 * (contrast_part + tilt_part)
    score[home] = 0.5 * expansion + 0.5 * structure
    parts[home, 0] = expansion
    parts[home, 1] = dip
    parts[home, 2] = contrast_part
    parts[home, 3] = tilt_part
    parts[home, 4] = wp.log(_radius(sums, home, last) / _radius(sums, home, 1))


_ORDER = [(axis, parity) for axis in range(3) for parity in range(2)]


def stand_in_score(start: np.ndarray, units: wp.array, steps: int, every: int, block: int = 10,
                   device: str = "cuda") -> tuple:
    """Run the smooth stand-in side by side, one settings point per universe, and score it like the fit."""
    start = np.asarray(start)
    if start.ndim != 3 or len(set(start.shape)) != 1:
        raise ValueError(f"start must be one cube of quanta, got shape {start.shape}")
    if units.ndim != 2 or units.shape[1] != len(NAMES):
        raise ValueError(f"units must be (universes, {len(NAMES)}), got {units.shape}")
    samples = steps // every + 1
    if samples < 6:
        raise ValueError(f"need at least 4 samples after the first: steps // every >= 5, got {steps} // {every}")
    universes, side = units.shape[0], start.shape[0]
    defaults = Settings()
    shape = (universes * side, side, side)
    values, seal = _settings(units, universes, device)
    sign = wp.array(np.tile(np.sign(start).astype(np.int32), (universes, 1, 1)), dtype=wp.int32, device=device)
    amounts = np.abs(start).astype(np.float32)
    psi = wp.array(np.tile(np.stack([np.sqrt(amounts), np.zeros_like(amounts)], axis=-1), (universes, 1, 1, 1)),
                   dtype=wp.vec2, device=device)
    pull = wp.zeros(shape, dtype=wp.float32, device=device)
    rate = wp.zeros(shape, dtype=wp.float32, device=device)
    occupied = np.argwhere(start != 0)
    corner = int(occupied.min()) if occupied.size else 0
    blocks = max((int(occupied.max()) + 1 - corner) // block, 1) if occupied.size else 1
    sums = wp.zeros((universes, samples, _SUMS), dtype=wp.float32, device=device, requires_grad=True)
    block_sums = wp.zeros((universes, samples, blocks ** 3), dtype=wp.float32, device=device, requires_grad=True)
    sample_inputs = [side, 0.5 * (side - 1), corner, block, blocks]
    wp.launch(_sample, dim=shape, device=device, inputs=[sign, psi, *sample_inputs, 0], outputs=[sums, block_sums])
    half = 0.5 * defaults.step_time
    fixed = [defaults.joint_rest, defaults.clock_packing, defaults.spread_share]
    for step_index in range(1, steps + 1):
        rate = _new(_speed_up, shape, device, [sign, _detached(psi), pull, rate, values, side,
                                               defaults.step_time * defaults.light_speed ** 2])
        pull = _new(_move, shape, device, [sign, pull, rate, defaults.step_time])
        on_site_inputs = [values, side, half, defaults.volume_rest, defaults.volume_stiffness, defaults.joint_rest,
                          defaults.clock_packing]
        psi = _new(_on_site, shape, device, [sign, psi, _detached(psi), pull, *on_site_inputs], dtype=wp.vec2)
        for axis, parity in _ORDER + _ORDER[::-1]:
            psi = _new(_hop, shape, device, [sign, psi, _detached(psi), values, seal, side, axis, parity, half, *fixed],
                       dtype=wp.vec2)
        psi = _new(_on_site, shape, device, [sign, psi, _detached(psi), pull, *on_site_inputs], dtype=wp.vec2)
        if step_index % every == 0 and step_index // every < samples:
            wp.launch(_sample, dim=shape, device=device, inputs=[sign, psi, *sample_inputs, step_index // every],
                      outputs=[sums, block_sums])
    wp.launch(_block_spread, dim=(universes, samples, blocks ** 3), device=device,
              inputs=[block_sums, float(block ** 3), blocks ** 3], outputs=[sums])
    score = wp.zeros(universes, dtype=wp.float32, device=device, requires_grad=True)
    parts = wp.zeros((universes, 5), dtype=wp.float32, device=device, requires_grad=True)
    wp.launch(_score, dim=universes, device=device, inputs=[sums, samples, float(every * defaults.step_time)],
              outputs=[score, parts])
    return score, parts


def _settings(units: wp.array, universes: int, device: str) -> tuple:
    """Each universe's searched settings (pull_strength, push, collapse, friction, steepness, seal fraction) and seal."""
    ranges = [LOG_RANGES.get(name) or LINEAR_RANGES[name] for name in NAMES]
    low = wp.array([lower for lower, _ in ranges], dtype=wp.float32, device=device)
    high = wp.array([upper for _, upper in ranges], dtype=wp.float32, device=device)
    logged = wp.array([int(name in LOG_RANGES) for name in NAMES], dtype=wp.int32, device=device)
    values = wp.zeros((universes, len(NAMES)), dtype=wp.float32, device=device, requires_grad=units.requires_grad)
    wp.launch(_values, dim=(universes, len(NAMES)), device=device, inputs=[units, low, high, logged], outputs=[values])
    seal = wp.zeros(universes, dtype=wp.float32, device=device, requires_grad=units.requires_grad)
    wp.launch(_seal, dim=universes, device=device, inputs=[values], outputs=[seal])
    return values, seal


def _detached(psi: wp.array) -> wp.array:
    """|psi|^2 per box in an array the tape does not see: forward exact, no gradient through it."""
    density = wp.empty(psi.shape, dtype=wp.float32, device=psi.device)
    wp.launch(_density, dim=psi.shape, device=psi.device, inputs=[psi], outputs=[density], record_tape=False)
    return density


def _new(kernel, shape: tuple, device: str, inputs: list, dtype=wp.float32) -> wp.array:
    """Launch a pass into a fresh array (the tape needs every written array kept) and return it."""
    out = wp.zeros(shape, dtype=dtype, device=device, requires_grad=True)
    wp.launch(kernel, dim=shape, device=device, inputs=inputs, outputs=[out])
    return out
