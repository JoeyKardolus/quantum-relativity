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
  E = 1/2 v^T M^+ v + sum_i U_i(s_i),  U_i(s) = push * wanted_i^6 / (5 s^5) - pull_i * s
(M the clock-weighted graph Laplacian, -dU_i/ds = load_i), which keeps E exactly at any
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
(omega * substep <= 1/2), at most MOST_SUBSTEPS. A joint whose size falls to the
collapse packing (its minus quanta over its size reach collapse_packing, size <= quanta /
collapse_packing; owner, 2026-11-16: only too high a concentration collapses a joint,
never too little energy) stops there, at rest, and is returned for collapse in the same
step (axiom 6); an outer joint, with no second volume, cannot collapse: it is held at the
limit size for the rest of the step. A joint already past the limit when the swing starts
does not swing: the collapse that opens the next step takes it (step.py). state.speed is the
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
from physics.engine.state import State
from physics.engine.swing_table import swing_table

MOST_SUBSTEPS = 64
ROUNDS = 400
TOLERANCE = 1e-12
_ACTIVE, _PAIRED, _HELD, _BORN = 1, 2, 4, 8


@wp.func
def _secant(before: wp.float64, after: wp.float64, hardness: wp.float64, load: wp.float64) -> wp.vec2d:
    """(U(after) - U(before)) / (after - before) and its derivative in `after`, written without cancellation."""
    slope = -load
    bend = wp.float64(0.0)
    low = wp.float64(1.0)
    for power in range(1, 6):
        low = low * before
        high = wp.float64(1.0)
        for fold in range(6 - power):
            high = high * after
        term = hardness / (low * high)
        slope = slope - term
        bend = bend + wp.float64(float(6 - power)) * term / after
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
           slope: wp.array2d(dtype=wp.float64)):
    home, cell = wp.tid()
    slope[home, cell] = wp.float64(0.0)
    if sub_index >= counts[home] or (flags[home, cell] & _ACTIVE) == 0:
        return
    own = size[home, cell]
    start[home, cell] = own
    target[home, cell] = own + spans[home] * speed[home, cell]
    if _moving(flags[home, cell]):
        slope[home, cell] = _secant(own, own, hardness[home, cell], load[home, cell])[0]


@wp.kernel
def _gather(offsets: wp.array2d(dtype=wp.int32), members: wp.array2d(dtype=wp.int32),
            rate: wp.array2d(dtype=wp.float64), slope: wp.array2d(dtype=wp.float64),
            pulled: wp.array2d(dtype=wp.float64)):
    home, volume = wp.tid()
    total = wp.float64(0.0)
    for index in range(offsets[home, volume], offsets[home, volume + 1]):
        joint = members[home, index]
        total = total + rate[home, joint] * slope[home, joint]
    pulled[home, volume] = total


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
             slope: wp.array2d(dtype=wp.float64)):
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
    bend = _secant(start[home, cell], guess, hardness[home, cell], load[home, cell])[1]
    better = guess - missed[home, cell] / (wp.float64(1.0) + scale * own_rate * others * bend)
    if better <= wp.float64(0.0):
        better = wp.float64(0.5) * guess
    size[home, cell] = better
    slope[home, cell] = _secant(start[home, cell], better, hardness[home, cell], load[home, cell])[0]


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
    if (flag & _HELD) != 0:
        if (flag & _BORN) == 0:
            size[home, cell] = wp.max(size[home, cell], floor[home, cell])
        speed[home, cell] = wp.float64(0.0)
    flags[home, cell] = flag


def joint_swing(state: State, settings: Settings, pull: np.ndarray) -> np.ndarray:
    """Swing every joint's size one step, energy kept; returns the joints pressed to the collapse limit."""
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
    hardness = settings.push * state.wanted[rows, columns] ** 6 / 5.0
    rates = _volume_rates(table, _spread(shape, rows, columns, rate))
    counts = _counts(settings, rows, rate, rates, state.between[rows, columns], hardness, np.maximum(size, floor),
                     state.universes)
    paired = table["paired"][rows, columns]
    flags = np.zeros(shape, dtype=np.int32)
    flags[rows, columns] = _ACTIVE + paired * _PAIRED + born * (_HELD | _BORN)
    on = {name: wp.array(_spread(shape, rows, columns, values, fill), dtype=wp.float64, device=device)
          for name, values, fill in (("rate", rate, 0.0), ("hardness", hardness, 0.0), ("floor", floor, 0.0),
                                     ("size", size, 1.0), ("speed", np.where(born, 0.0, state.speed[rows, columns]), 0.0),
                                     ("load", pull[rows, columns], 0.0))}
    on["rates"] = wp.array(rates, dtype=wp.float64, device=device)
    on["spans"] = wp.array(settings.step_time / np.maximum(counts, 1), dtype=wp.float64, device=device)
    for name in ("start", "target", "slope", "pulled", "missed"):
        on[name] = wp.zeros(shape, dtype=wp.float64, device=device)
    on["flags"] = wp.array(flags, dtype=wp.int32, device=device)
    on["counts"] = wp.array(counts.astype(np.int32), dtype=wp.int32, device=device)
    inertia = wp.float64(settings.joint_inertia)
    largest = np.zeros(state.universes)
    np.maximum.at(largest, rows, np.abs(size))
    limit = TOLERANCE * np.maximum(1.0, largest)
    for sub_index in range(int(counts.max())):
        _substep(state, table, on, inertia, sub_index, counts, limit)
    held = (on["flags"].numpy()[rows, columns] & _HELD) != 0
    state.size, state.speed = state.size.copy(), state.speed.copy()
    state.size[rows, columns] = on["size"].numpy()[rows, columns]
    state.speed[rows, columns] = on["speed"].numpy()[rows, columns]
    pressed[rows, columns] = paired & held & ~born
    state.cell_push = _push(state, settings)
    return pressed


def _spread(shape: tuple, rows: np.ndarray, columns: np.ndarray, values: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """The swinging joints' values at their places in a full (universes, capacity) table, `fill` elsewhere."""
    table = np.full(shape, fill)
    table[rows, columns] = values
    return table


def _substep(state: State, table: dict, on: dict, inertia, sub_index: int, counts: np.ndarray,
             limit: np.ndarray) -> None:
    """One discrete-gradient substep of every universe still swinging, solved to TOLERANCE, swing energy kept."""
    device = state.quanta.device
    shape = state.kind.shape
    wp.launch(_begin, dim=shape, device=device,
              inputs=[on["flags"], on["counts"], on["spans"], sub_index, on["size"], on["speed"], on["hardness"],
                      on["load"], on["start"], on["target"], on["slope"]])
    live = (counts > sub_index).astype(np.int32)
    worst = wp.zeros(state.universes, dtype=wp.float64, device=device)
    for _ in range(ROUNDS):
        live_d = wp.array(live, dtype=wp.int32, device=device)
        wp.launch(_gather, dim=shape, device=device,
                  inputs=[table["offsets"], table["members"], on["rate"], on["slope"], on["pulled"]])
        worst.zero_()
        wp.launch(_miss, dim=shape, device=device,
                  inputs=[on["flags"], live_d, on["spans"], table["between"], on["rate"], on["rates"], on["pulled"],
                          on["size"], on["target"], on["slope"], inertia, on["missed"], worst])
        live = live * (worst.numpy() > limit)
        if not live.any():
            break
        wp.launch(_improve, dim=shape, device=device,
                  inputs=[on["flags"], wp.array(live, dtype=wp.int32, device=device), on["spans"], table["between"],
                          on["rate"], on["rates"], on["hardness"], on["load"], on["start"], on["missed"], inertia,
                          on["size"], on["slope"]])
    else:
        raise RuntimeError(f"joint swing: not solved in {ROUNDS} rounds (largest miss {worst.numpy().max():.3e})")
    wp.launch(_finish, dim=shape, device=device,
              inputs=[on["flags"], on["counts"], on["spans"], sub_index, table["between"], on["rate"], on["rates"],
                      on["pulled"], on["slope"], on["floor"], inertia, on["size"], on["speed"]])


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
    np.maximum.at(stiffness, rows, 30.0 * hardness / size ** 7)
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
