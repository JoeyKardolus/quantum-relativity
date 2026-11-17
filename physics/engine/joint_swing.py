"""Joint swing: every joint's size moves like a frictionless spring with inertia, pushed by its own push against its neighbours'.

Contract: `joint_swing(state, settings, pull)` advances every joint's size and speed
(state.size, state.speed) by one step_time. A joint's neighbours are the joints it
shares an energy volume with (the cell graph, state.between). Its load is its push
(cell_laws.py) plus `pull`: step.py passes the mean pull over its boxes (cell_count.py;
the pull is at most 0 where energy is, so it squeezes the joint: bible rule 8, "pull
squeezes it and its push swells it back") less the volumes' fluid pressing on its faces
(wall_press.py). Each pair of neighbouring joints i, j is a spring running on the
two joints' clocks: joint i's size accelerates by rate_i * rate_j * (load_i - load_j) /
joint_inertia per unit of counter time, so what one joint gains its neighbour loses and
the joints' summed size is kept (bible rule 5: the joints are the springs of the grid,
they overshoot and swing round their balance). The swing is frictionless (axiom 4: the
only friction is energy crossing a joint), so the step must neither gain nor lose swing
energy: it is the discrete-gradient (Itoh-Abe) step of the Hamiltonian
  E = 1/2 v^T M^+ v + sum_i U_i(s_i),  U_i(s) = push * wanted_i^p / ((p - 1) s^(p - 1)) - pull_i * s
(p = push_steepness, above 1; M the clock-weighted graph Laplacian over the joints that
swing, -dU_i/ds = load_i; before 2026-11-17 p was written in as 6 whatever the setting
said), which keeps E exactly at any
step length; it is implicit (numerics, not a rule: an explicit step gained energy at the
steep wall and crushed joints to nothing) and solved on the GPU, every joint at once, by
Newton's method with the Jacobian's diagonal (each joint's own spring; the neighbours'
come in through the next round), in float64, until the step's equation holds to
TOLERANCE in every joint; nothing is solved on the host, so the cost grows only with the
number of joints (before 2026-11-17 a sparse direct solve on the host: 3 s a step at
10,000 joints, and a 32-place universe did not finish its first step). Sums over a
volume's joints run in one fixed order (swing_table.py), so one seed gives one run and a
universe swings the same alone as in a batch. The step
is cut into as many equal substeps as the stiffest joint needs for accuracy
(omega * substep <= 1/2), at most MOST_SUBSTEPS; when the rounds do not settle on a
substep's equation (joints flying inward by more than their own size in one substep),
the whole step is taken again from its start with FINER times as many substeps in that
universe, up to RETRIES times (numerics: the same equation at a shorter substep; before
2026-11-17 the step raised, which ended 37 of the fit's 105 universes). A joint whose size falls to the
collapse packing (its minus quanta over its size reach collapse_packing, size <= quanta /
collapse_packing; owner, 2026-11-16: only too high a concentration collapses a joint,
never too little energy) stops there, at rest, hands its speed to the joints it shared
springs with (speed_handover.py, as does a joint found past the limit) and is returned
for collapse in the same step (axiom 6); an outer joint, with no second volume, cannot collapse: it is held at the
limit size for the rest of the step. A joint already past the limit when the swing starts
does not swing: the collapse that opens the next step takes it (step.py). A joint that
does not swing is in no spring: its neighbours trade size only with each other, so the
swinging joints' summed size and the swing energy are kept (before 2026-11-17 it stood
in its neighbours' springs with load 0, which swelled them without end). state.speed is the
size change per unit of counter time (the clocks enter through the springs, so the
summed size is kept exactly). Energy volumes are not moved (their size
follows their energy). Returns a bool array (universes, capacity), True for each joint
pressed to the collapse limit.
"""
from __future__ import annotations

import math

import numpy as np
import warp as wp

from physics.engine.settings import Settings
from physics.engine.speed_handover import speed_handover
from physics.engine.state import State
from physics.engine.swing_table import swing_table

MOST_SUBSTEPS = 64
ROUNDS = 400
TOLERANCE = 1e-12
FINER = 4
RETRIES = 6
_LOST = 1e150  # numerics: a miss this large will not come back; the step is taken again, finer
_ACTIVE, _PAIRED, _HELD, _BORN = 1, 2, 4, 8


@wp.func
def _secant(before: wp.float64, after: wp.float64, hardness: wp.float64, load: wp.float64,
            power: wp.float64) -> wp.vec2d:
    """(U(after) - U(before)) / (after - before) and its derivative in `after`, written without cancellation.

    U(s) = hardness / s**power - load * s, power = push_steepness - 1. With turn = ln(after / before),
    (after**-power - before**-power) / (after - before) = -(after * before)**(-(power + 1) / 2) *
    sinh(power * turn / 2) / sinh(turn / 2), the ratio going to `power` as the two sizes meet.
    """
    half = wp.float64(0.5)
    turn = wp.log(after / before)
    curve = power * (power * power - wp.float64(1.0)) / wp.float64(24.0)
    ratio = power + curve * turn * turn
    ratio_rate = wp.float64(2.0) * curve * turn
    if wp.abs(turn) > wp.float64(1.0e-6):
        low = wp.sinh(half * turn)
        high = wp.sinh(half * power * turn)
        ratio = high / low
        ratio_rate = half * (power * wp.cosh(half * power * turn) * low - high * wp.cosh(half * turn)) / (low * low)
    scale = hardness * wp.exp(-half * (power + wp.float64(1.0)) * wp.log(after * before))
    slope = -load - scale * ratio
    bend = scale * (half * (power + wp.float64(1.0)) * ratio - ratio_rate) / after
    return wp.vec2d(slope, bend)


@wp.func
def _moving(flag: int) -> bool:
    """True for a joint that swings this substep: it is a live joint and not held at its limit."""
    return (flag & _ACTIVE) != 0 and (flag & _HELD) == 0


@wp.kernel
def _begin(flags: wp.array2d(dtype=wp.int32), counts: wp.array(dtype=wp.int32), spans: wp.array(dtype=wp.float64),
           sub_index: int, size: wp.array2d(dtype=wp.float64), speed: wp.array2d(dtype=wp.float64),
           hardness: wp.array2d(dtype=wp.float64), load: wp.array2d(dtype=wp.float64),
           start: wp.array2d(dtype=wp.float64), target: wp.array2d(dtype=wp.float64),
           slope: wp.array2d(dtype=wp.float64), power: wp.float64):
    home, cell = wp.tid()
    slope[home, cell] = wp.float64(0.0)
    if sub_index >= counts[home] or (flags[home, cell] & _ACTIVE) == 0:
        return
    own = size[home, cell]
    start[home, cell] = own
    target[home, cell] = own + spans[home] * speed[home, cell]
    if _moving(flags[home, cell]):
        slope[home, cell] = _secant(own, own, hardness[home, cell], load[home, cell], power)[0]


@wp.kernel
def _gather(offsets: wp.array2d(dtype=wp.int32), members: wp.array2d(dtype=wp.int32),
            flags: wp.array2d(dtype=wp.int32), rate: wp.array2d(dtype=wp.float64),
            slope: wp.array2d(dtype=wp.float64), pulled: wp.array2d(dtype=wp.float64),
            rates: wp.array2d(dtype=wp.float64)):
    """Per energy volume, over its joints that swing: their summed clock rate and rate times slope."""
    home, volume = wp.tid()
    total = wp.float64(0.0)
    clocks = wp.float64(0.0)
    for index in range(offsets[home, volume], offsets[home, volume + 1]):
        joint = members[home, index]
        if _moving(flags[home, joint]):
            total = total + rate[home, joint] * slope[home, joint]
            clocks = clocks + rate[home, joint]
    pulled[home, volume] = total
    rates[home, volume] = clocks


@wp.func
def _spring(home: int, cell: int, sides: wp.vec2i, own_rate: wp.float64, own_slope: wp.float64,
            rates: wp.array2d(dtype=wp.float64), pulled: wp.array2d(dtype=wp.float64)) -> wp.float64:
    """Row `cell` of the clock-weighted graph Laplacian applied to the slopes (inertia not yet divided out)."""
    total = wp.float64(0.0)
    for pick in range(2):
        volume = sides[pick]
        if volume >= 0:
            total = total + (rates[home, volume] * own_slope - pulled[home, volume])
    return own_rate * total


@wp.kernel
def _miss(flags: wp.array2d(dtype=wp.int32), live: wp.array(dtype=wp.int32), spans: wp.array(dtype=wp.float64),
          between: wp.array2d(dtype=wp.vec2i), rate: wp.array2d(dtype=wp.float64),
          rates: wp.array2d(dtype=wp.float64), pulled: wp.array2d(dtype=wp.float64),
          size: wp.array2d(dtype=wp.float64), target: wp.array2d(dtype=wp.float64),
          slope: wp.array2d(dtype=wp.float64), inertia: wp.float64, missed: wp.array2d(dtype=wp.float64),
          worst: wp.array(dtype=wp.float64)):
    home, cell = wp.tid()
    if live[home] == 0 or not _moving(flags[home, cell]):
        return
    scale = wp.float64(0.5) * spans[home] * spans[home] / inertia
    spring = _spring(home, cell, between[home, cell], rate[home, cell], slope[home, cell], rates, pulled)
    miss = size[home, cell] - target[home, cell] + scale * spring
    missed[home, cell] = miss
    wp.atomic_max(worst, home, wp.abs(miss))


@wp.kernel
def _improve(flags: wp.array2d(dtype=wp.int32), live: wp.array(dtype=wp.int32), spans: wp.array(dtype=wp.float64),
             between: wp.array2d(dtype=wp.vec2i), rate: wp.array2d(dtype=wp.float64),
             rates: wp.array2d(dtype=wp.float64), hardness: wp.array2d(dtype=wp.float64),
             load: wp.array2d(dtype=wp.float64), start: wp.array2d(dtype=wp.float64),
             missed: wp.array2d(dtype=wp.float64), inertia: wp.float64, size: wp.array2d(dtype=wp.float64),
             slope: wp.array2d(dtype=wp.float64), power: wp.float64):
    home, cell = wp.tid()
    if live[home] == 0 or not _moving(flags[home, cell]):
        return
    scale = wp.float64(0.5) * spans[home] * spans[home] / inertia
    own_rate = rate[home, cell]
    sides = between[home, cell]
    others = wp.float64(0.0)
    for pick in range(2):
        if sides[pick] >= 0:
            others = others + (rates[home, sides[pick]] - own_rate)
    guess = size[home, cell]
    bend = _secant(start[home, cell], guess, hardness[home, cell], load[home, cell], power)[1]
    better = guess - missed[home, cell] / (wp.float64(1.0) + scale * own_rate * others * bend)
    if better <= wp.float64(0.0):
        better = wp.float64(0.5) * guess
    size[home, cell] = better
    slope[home, cell] = _secant(start[home, cell], better, hardness[home, cell], load[home, cell], power)[0]


@wp.kernel
def _finish(flags: wp.array2d(dtype=wp.int32), counts: wp.array(dtype=wp.int32), spans: wp.array(dtype=wp.float64),
            sub_index: int, between: wp.array2d(dtype=wp.vec2i), rate: wp.array2d(dtype=wp.float64),
            rates: wp.array2d(dtype=wp.float64), pulled: wp.array2d(dtype=wp.float64),
            slope: wp.array2d(dtype=wp.float64), floor: wp.array2d(dtype=wp.float64), inertia: wp.float64,
            size: wp.array2d(dtype=wp.float64), speed: wp.array2d(dtype=wp.float64)):
    home, cell = wp.tid()
    flag = flags[home, cell]
    if sub_index >= counts[home] or (flag & _ACTIVE) == 0:
        return
    if _moving(flag):
        spring = _spring(home, cell, between[home, cell], rate[home, cell], slope[home, cell], rates, pulled)
        speed[home, cell] = speed[home, cell] - spans[home] * spring / inertia
    if size[home, cell] <= floor[home, cell] and (flag & _PAIRED) != 0:
        flag = flag | _HELD
    if (flag & _HELD) != 0 and (flag & _BORN) == 0:
        size[home, cell] = wp.max(size[home, cell], floor[home, cell])
    flags[home, cell] = flag


def joint_swing(state: State, settings: Settings, pull: np.ndarray) -> np.ndarray:
    """Swing every joint's size one step, energy kept; returns the joints pressed to the collapse limit."""
    if settings.push_steepness <= 1.0:
        raise ValueError("push_steepness must be above 1 (the push's energy push * wanted**p / ((p - 1) size**(p - 1)))")
    table = swing_table(state)
    shape = state.kind.shape
    pressed = np.zeros(shape, dtype=bool)
    rows, columns = table["places"]
    if rows.size == 0:
        state.cell_push = _push(state, settings)
        return pressed
    device = state.quanta.device
    rate = state.rate[rows, columns]
    floor = state.cell_quanta[rows, columns] / settings.collapse_packing
    size = state.size[rows, columns].astype(np.float64)
    born = size < floor
    for universe in np.unique(rows[born]):
        speed_handover(state, settings, int(universe), columns[born & (rows == universe)])
    power = settings.push_steepness - 1.0
    hardness = settings.push * state.wanted[rows, columns] ** settings.push_steepness / power
    counts = _counts(settings, rows, rate, _volume_rates(table, _spread(shape, rows, columns, rate)),
                     state.between[rows, columns], hardness, np.maximum(size, floor), state.universes)
    paired = table["paired"][rows, columns]
    flags = np.zeros(shape, dtype=np.int32)
    flags[rows, columns] = _ACTIVE + paired * _PAIRED + born * (_HELD | _BORN)
    host = {"rate": (rate, 0.0), "hardness": (hardness, 0.0), "floor": (floor, 0.0), "size": (size, 1.0),
            "speed": (state.speed[rows, columns], 0.0), "load": (pull[rows, columns], 0.0)}
    largest = np.zeros(state.universes)
    np.maximum.at(largest, rows, np.abs(size))
    limit = TOLERANCE * np.maximum(1.0, largest)
    for attempt in range(RETRIES + 1):
        on = {name: wp.array(_spread(shape, rows, columns, values, fill), dtype=wp.float64, device=device)
              for name, (values, fill) in host.items()}
        on["spans"] = wp.array(settings.step_time / np.maximum(counts, 1), dtype=wp.float64, device=device)
        for name in ("start", "target", "slope", "pulled", "missed", "rates"):
            on[name] = wp.zeros(shape, dtype=wp.float64, device=device)
        on["flags"] = wp.array(flags, dtype=wp.int32, device=device)
        on["counts"] = wp.array(counts.astype(np.int32), dtype=wp.int32, device=device)
        on["inertia"], on["power"] = wp.float64(settings.joint_inertia), wp.float64(power)
        unsolved = np.zeros(state.universes, dtype=bool)
        for sub_index in range(int(counts.max())):
            unsolved = _substep(state, table, on, sub_index, counts, limit)
            if unsolved.any():
                break
        if not unsolved.any():
            break
        counts = np.where(unsolved, counts * FINER, counts)
    else:
        raise RuntimeError(f"joint swing: not solved in {ROUNDS} rounds with {int(counts.max()) // FINER} substeps")
    held = (on["flags"].numpy()[rows, columns] & _HELD) != 0
    state.size, state.speed = state.size.copy(), state.speed.copy()
    state.size[rows, columns] = on["size"].numpy()[rows, columns]
    state.speed[rows, columns] = on["speed"].numpy()[rows, columns]
    pressed[rows, columns] = paired & held & ~born
    for universe in np.flatnonzero(pressed.any(axis=1)):
        speed_handover(state, settings, int(universe), np.flatnonzero(pressed[universe]))
    state.cell_push = _push(state, settings)
    return pressed


def _spread(shape: tuple, rows: np.ndarray, columns: np.ndarray, values: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """The swinging joints' values at their places in a full (universes, capacity) table, `fill` elsewhere."""
    table = np.full(shape, fill)
    table[rows, columns] = values
    return table


def _substep(state: State, table: dict, on: dict, sub_index: int, counts: np.ndarray, limit: np.ndarray) -> np.ndarray:
    """One discrete-gradient substep of every universe still swinging, solved to TOLERANCE, swing energy kept.

    Returns per universe True when its rounds did not settle (the substep is then not finished in any universe).
    """
    device = state.quanta.device
    shape = state.kind.shape
    inertia = on["inertia"]
    wp.launch(_begin, dim=shape, device=device,
              inputs=[on["flags"], on["counts"], on["spans"], sub_index, on["size"], on["speed"], on["hardness"],
                      on["load"], on["start"], on["target"], on["slope"], on["power"]])
    live = (counts > sub_index).astype(np.int32)
    worst = wp.zeros(state.universes, dtype=wp.float64, device=device)
    for _ in range(ROUNDS):
        live_d = wp.array(live, dtype=wp.int32, device=device)
        wp.launch(_gather, dim=shape, device=device,
                  inputs=[table["offsets"], table["members"], on["flags"], on["rate"], on["slope"], on["pulled"],
                          on["rates"]])
        worst.zero_()
        wp.launch(_miss, dim=shape, device=device,
                  inputs=[on["flags"], live_d, on["spans"], table["between"], on["rate"], on["rates"], on["pulled"],
                          on["size"], on["target"], on["slope"], inertia, on["missed"], worst])
        miss = worst.numpy()
        live = live * ((miss > limit) | ~np.isfinite(miss))
        if not live.any():
            break
        if not np.all(miss < _LOST):
            return live.astype(bool)
        wp.launch(_improve, dim=shape, device=device,
                  inputs=[on["flags"], wp.array(live, dtype=wp.int32, device=device), on["spans"], table["between"],
                          on["rate"], on["rates"], on["hardness"], on["load"], on["start"], on["missed"], inertia,
                          on["size"], on["slope"], on["power"]])
    else:
        return live.astype(bool)
    wp.launch(_finish, dim=shape, device=device,
              inputs=[on["flags"], on["counts"], on["spans"], sub_index, table["between"], on["rate"], on["rates"],
                      on["pulled"], on["slope"], on["floor"], inertia, on["size"], on["speed"]])
    return np.zeros(state.universes, dtype=bool)


def _volume_rates(table: dict, rate: np.ndarray) -> np.ndarray:
    """Per energy volume the summed clock rate of its swinging joints, added in the table's order."""
    rates = np.zeros(rate.shape)
    for universe in range(rate.shape[0]):
        used = table["used"][universe]
        rates[universe] = np.bincount(table["volumes_host"][universe, :used],
                                      weights=rate[universe, table["members_host"][universe, :used]],
                                      minlength=rate.shape[1])
    return rates


def _counts(settings: Settings, rows: np.ndarray, rate: np.ndarray, rates: np.ndarray, sides: np.ndarray,
            hardness: np.ndarray, size: np.ndarray, universes: int) -> np.ndarray:
    """Per universe the substeps its stiffest joint needs (omega * substep <= 1/2), 0 without swinging joints."""
    others = np.zeros(rate.shape)
    for pick in range(2):
        volume = sides[:, pick]
        others += np.where(volume >= 0, rates[rows, np.maximum(volume, 0)] - rate, 0.0)
    largest, stiffness = np.zeros(universes), np.zeros(universes)
    np.maximum.at(largest, rows, 2.0 * rate * others / settings.joint_inertia)
    steepness = settings.push_steepness
    np.maximum.at(stiffness, rows, steepness * (steepness - 1.0) * hardness / size ** (steepness + 1.0))
    counts = np.zeros(universes, dtype=np.int64)
    for universe in np.unique(rows):
        omega = math.sqrt(largest[universe] * stiffness[universe])
        counts[universe] = min(MOST_SUBSTEPS, max(1, math.ceil(settings.step_time * omega / 0.5)))
    return counts


def _push(state: State, settings: Settings) -> np.ndarray:
    """push * (wanted / size) ** steepness per joint, 0 for every other cell."""
    rows, columns = np.nonzero(state.kind == -1)
    push = np.zeros(state.kind.shape)
    push[rows, columns] = settings.push * (state.wanted[rows, columns] / np.maximum(state.size[rows, columns], 1e-6)
                                           ) ** settings.push_steepness
    return push
