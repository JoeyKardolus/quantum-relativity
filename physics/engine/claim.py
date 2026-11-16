"""Claim: a box changes kind at a border by a neighbour-only move of whole quanta, when that lowers the energy.

Contract: `claim(state, settings, phase_index)` runs one of the 64 claim phases: the
sites are the boxes at offset + 4 * (i, j, k), offset = (phase_index // 16, phase_index
// 4 % 4, phase_index % 4). A site reads its 26 neighbours and writes only itself and its
six face neighbours, so sites 4 apart never touch the same box and run together. Each
site weighs every face neighbour as the giver of one move:
  energy volume claims a joint box  (site minus, giver plus with >= 2 quanta) the site's
      minus quanta are pushed into its minus face neighbours (split as evenly as whole
      numbers allow; none there, no move), the giver hands half its plus quanta (rounded
      down) to the site, which turns plus
  joint claims an energy volume box (site plus, giver minus with >= 2 quanta) the site's
      plus quanta are pushed into its plus face neighbours, the giver hands half its
      minus quanta to the site, which turns minus
  joint claims an empty box (site 0, giver minus with >= 2 quanta) the giver hands half
      its minus quanta to the site: space grows where joints claim empty boxes
and takes the move that lowers the sum of box_energy.py (pull, the volume's own push,
the joint's steep push) over the touched boxes the most, if any lowers it. The claimed
box takes the giver's phase and clock reading. Every quantum keeps its sign and the
amount is exact (the box changes kind, its quanta do not); tallies[u, 2] counts
universe u's claims and the claimed box's marks get bit 2 (step.py zeroes marks).
Wrapping (bible rule 1, owner 2026-11-16), checked in the site's 3x3x3 block before a
move: a box turns plus only when none of its 26 neighbours is empty or off the grid and
the plus boxes among them form one group (26-connected inside the block), so a claim
never lets an energy volume touch empty space or a second energy volume (only collapse
joins volumes, collapse.py); a plus box turns minus only when the plus boxes round it
still form one group, so a claim never tears a volume in two. Returns None.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.engine.box_energy import box_energy
from physics.engine.face_step import face_step
from physics.engine.inside import inside
from physics.engine.settings import Settings
from physics.engine.state import State

_NEIGHBOURS = {}


@wp.func
def _one_group(members: int, adjacency: wp.array(dtype=wp.int32)) -> bool:
    """True when the set bits of `members` (3x3x3 block, centre left out) are one 26-connected group, or none."""
    if members == 0:
        return True
    first = int(0)
    for index in range(27):
        if ((members >> index) & 1) == 1 and first == 0:
            first = 1 << index
    reached = first
    for _ in range(27):
        grown = reached
        for index in range(27):
            if ((reached >> index) & 1) == 1:
                grown = grown | (adjacency[index] & members)
        if grown == reached:
            return reached == members
        reached = grown
    return reached == members


@wp.func
def _half(given: int) -> int:
    """Half of `given` quanta, rounded down in amount, sign kept."""
    if given < 0:
        return -((-given) // 2)
    return given // 2


@wp.func
def _energy(amount: int, pull: float, volume_rest: float, volume_stiffness: float, joint_rest: float, push: float,
            push_steepness: float) -> float:
    return box_energy(amount, pull, volume_rest, volume_stiffness, joint_rest, push, push_steepness)


@wp.func
def _pushed_out(quanta: wp.array3d(dtype=wp.int32), pull: wp.array3d(dtype=wp.float32), site: wp.vec3i, side: int,
                amount: int, sign: int, apply: int, volume_rest: float, volume_stiffness: float, joint_rest: float,
                push: float, push_steepness: float) -> float:
    """Energy change of pushing `amount` quanta of `sign` into the site's face neighbours of that sign (written when
    apply is 1); a huge number when there is none."""
    home = site[0] // side
    count = int(0)
    for face in range(6):
        there = site + face_step(face)
        if inside(there, home, side):
            if quanta[there[0], there[1], there[2]] * sign > 0:
                count += 1
    if count == 0:
        return 1.0e30
    share = amount // count
    left = amount - share * count
    change = float(0.0)
    for face in range(6):
        there = site + face_step(face)
        if inside(there, home, side):
            held = quanta[there[0], there[1], there[2]]
            if held * sign > 0:
                added = share
                if left > 0:
                    added += 1
                    left -= 1
                after = held + sign * added
                field = pull[there[0], there[1], there[2]]
                change += (_energy(after, field, volume_rest, volume_stiffness, joint_rest, push, push_steepness)
                           - _energy(held, field, volume_rest, volume_stiffness, joint_rest, push, push_steepness))
                if apply == 1:
                    quanta[there[0], there[1], there[2]] = after
    return change


@wp.kernel
def _claim(quanta: wp.array3d(dtype=wp.int32), phase: wp.array3d(dtype=wp.float32),
           pull: wp.array3d(dtype=wp.float32), clock: wp.array3d(dtype=wp.float32),
           marks: wp.array3d(dtype=wp.uint8), adjacency: wp.array(dtype=wp.int32), side: int, length: int,
           offset: wp.vec3i, volume_rest: float,
           volume_stiffness: float, joint_rest: float, push: float, push_steepness: float,
           tallies: wp.array2d(dtype=wp.int64)):
    ti, tj, tk = wp.tid()
    site = wp.vec3i(offset[0] + 4 * ti, offset[1] + 4 * tj, offset[2] + 4 * tk)
    if site[0] >= length:
        return
    home = site[0] // side
    if not inside(site, home, side):
        return
    held = quanta[site[0], site[1], site[2]]
    plus_round = int(0)
    empty_round = int(0)
    for index in range(27):
        if index != 13:
            there = site + wp.vec3i(index // 9 - 1, index // 3 % 3 - 1, index % 3 - 1)
            if not inside(there, home, side):
                empty_round = empty_round | (1 << index)
            else:
                value = quanta[there[0], there[1], there[2]]
                if value > 0:
                    plus_round = plus_round | (1 << index)
                elif value == 0:
                    empty_round = empty_round | (1 << index)
    if held < 0 and (empty_round != 0 or not _one_group(plus_round, adjacency)):
        return
    if held > 0 and not _one_group(plus_round, adjacency):
        return
    field = pull[site[0], site[1], site[2]]
    best = float(0.0)
    best_face = int(-1)
    for face in range(6):
        giver = site + face_step(face)
        if not inside(giver, home, side):
            continue
        given = quanta[giver[0], giver[1], giver[2]]
        if wp.abs(given) < 2 or (held != 0 and (held > 0) == (given > 0)) or (held == 0 and given > 0):
            continue
        half = _half(given)
        giver_field = pull[giver[0], giver[1], giver[2]]
        change = (_energy(given - half, giver_field, volume_rest, volume_stiffness, joint_rest, push, push_steepness)
                  + _energy(half, field, volume_rest, volume_stiffness, joint_rest, push, push_steepness)
                  - _energy(given, giver_field, volume_rest, volume_stiffness, joint_rest, push, push_steepness))
        if held != 0:
            sign = int(1)
            if held < 0:
                sign = -1
            change += _pushed_out(quanta, pull, site, side, wp.abs(held), sign, 0, volume_rest, volume_stiffness,
                                  joint_rest, push, push_steepness)
            change -= _energy(held, field, volume_rest, volume_stiffness, joint_rest, push, push_steepness)
        if change < best:
            best = change
            best_face = face
    if best_face < 0:
        return
    giver = site + face_step(best_face)
    given = quanta[giver[0], giver[1], giver[2]]
    half = _half(given)
    if held != 0:
        sign = int(1)
        if held < 0:
            sign = -1
        _pushed_out(quanta, pull, site, side, wp.abs(held), sign, 1, volume_rest, volume_stiffness, joint_rest,
                    push, push_steepness)
    quanta[giver[0], giver[1], giver[2]] = given - half
    quanta[site[0], site[1], site[2]] = half
    phase[site[0], site[1], site[2]] = phase[giver[0], giver[1], giver[2]]
    clock[site[0], site[1], site[2]] = clock[giver[0], giver[1], giver[2]]
    marks[site[0], site[1], site[2]] = marks[site[0], site[1], site[2]] | wp.uint8(2)
    wp.atomic_add(tallies, home, 2, wp.int64(1))


def _adjacency(device) -> wp.array:
    """Per block index 0..26 the bit mask of its 26-neighbours inside the 3x3x3 block (the centre, 13, left out)."""
    key = str(device)
    if key not in _NEIGHBOURS:
        places = np.array([(index // 9, index // 3 % 3, index % 3) for index in range(27)])
        masks = np.zeros(27, dtype=np.int64)
        for index in range(27):
            for other in range(27):
                if other not in (index, 13) and np.abs(places[index] - places[other]).max() == 1:
                    masks[index] |= 1 << other
        _NEIGHBOURS[key] = wp.array(masks.astype(np.int32), dtype=wp.int32, device=device)
    return _NEIGHBOURS[key]


def claim(state: State, settings: Settings, phase_index: int) -> None:
    """Run claim phase `phase_index` (0..63): border boxes change kind where whole-quanta moves lower the energy."""
    offset = (phase_index // 16, phase_index // 4 % 4, phase_index % 4)
    dims = tuple(max(0, -(-(length - start) // 4)) for length, start in zip(state.shape, offset))
    if min(dims) == 0:
        return
    device = state.quanta.device
    wp.launch(_claim, dim=dims, device=device,
              inputs=[state.quanta, state.phase, state.pull, state.clock, state.marks, _adjacency(device), state.side,
                      state.shape[0],
                      wp.vec3i(*offset), settings.volume_rest, settings.volume_stiffness, settings.joint_rest,
                      settings.push, settings.push_steepness, state.tallies])
