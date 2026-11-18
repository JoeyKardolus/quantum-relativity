"""Relink: after relations changed, the way table is made to follow them.

Contract: `relink(state, cells)` is called after `Grid.set_neighbours` changed the
relations of some cells (a tear or a collapse). For the named cells and any cell whose
entries were set since the last call:
  1. their neighbour rows go up to the device;
  2. every joint among them, and every joint they touch, gets exactly the ways its own
     entries make (two different energy volumes on opposite sides of it): a way that
     still exists keeps its slot, one that no longer does dies (`way_joint = -1`, its
     slot free), a new one takes the lowest free slot and starts with 0 in every other
     way array;
  3. each new way gets the lowest `way_pass` no other way holds on either of its two
     mouths, in slot order, so two ways on one mouth never share a pass; `way_passes` is
     set;
  4. the changed slots go up to the device.
It moves no energy and changes no relation: the relations are the truth, the ways follow.
"""
from __future__ import annotations

import heapq
import itertools

import numpy as np

from physics.engine.axiom_3_structure.definitions.direction_codes import CENTRE
from physics.engine.axiom_3_structure.definitions.grid import JOINT_COLUMN, PASS_COLUMN, WAY_COLUMNS
from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT


def _wanted_ways(state, joint: int, kind: np.ndarray) -> set:
    """The ways a joint's own entries make: `(a, b, mouth_a, mouth_b)` each."""
    by_code: dict = {}
    for one, code in state.neighbours(joint):
        if kind[one] == ENERGY_VOLUME:
            by_code.setdefault(code, []).append(one)
    wanted = set()
    for code in sorted(by_code):
        if code > CENTRE:
            continue
        for first in by_code[code]:
            for second in by_code.get(26 - code, ()):
                if first < second:
                    wanted.add((first, second, 26 - code, code))
                elif second < first:
                    wanted.add((second, first, code, 26 - code))
    return wanted


def _taken_passes(state) -> dict:
    """The passes the living ways hold on every mouth they use: `{(volume, mouth): {pass}}`."""
    table = state.way_table[:state.way_count]
    taken: dict = {}
    for way_a, way_b, mouth_a, mouth_b, way_joint, way_pass in table[table[:, PASS_COLUMN] >= 0].tolist():
        taken.setdefault((way_a, mouth_a), set()).add(way_pass)
        taken.setdefault((way_b, mouth_b), set()).add(way_pass)
    return taken


def _touched_joints(state, cells: list, kind: np.ndarray) -> list:
    """The joints among the cells, and the joints the cells touch, in id order."""
    joints = set()
    for cell in cells:
        if kind[cell] == JOINT or state.slots_of(cell):
            joints.add(cell)
        joints.update(entry[0] for entry in state.neighbours(cell) if kind[entry[0]] == JOINT)
    return sorted(joints)


def _rebuild_joint(state, joint: int, kind: np.ndarray) -> tuple:
    """Keep the joint's ways that still exist and kill the rest; returns (dead slots, ways to be born)."""
    wanted = _wanted_ways(state, joint, kind) if kind[joint] == JOINT else set()
    kept, dead = [], []
    for slot in state.slots_of(joint):
        key = tuple(int(value) for value in state.way_table[slot, :JOINT_COLUMN])
        if key in wanted:
            wanted.discard(key)
            kept.append(slot)
        else:
            state.way_table[slot] = -1
            heapq.heappush(state.free_slots, slot)
            dead.append(slot)
    state.joint_slots[joint] = kept
    return dead, [(joint,) + key for key in sorted(wanted)]


def _place_born(state, born: list) -> list:
    """Give every new way the lowest free slot; returns the slots."""
    fresh = []
    for joint, way_a, way_b, mouth_a, mouth_b in born:
        slot = state.free_slot()
        state.way_table[slot] = (way_a, way_b, mouth_a, mouth_b, joint, -1)
        state.joint_slots[joint].append(slot)
        fresh.append(slot)
    return fresh


def _hand_out_passes(state, fresh: list) -> None:
    """Give each new way the lowest pass free on both its mouths, in slot order."""
    taken = _taken_passes(state) if fresh else {}
    for slot in sorted(fresh):
        way_a, way_b, mouth_a, mouth_b = (int(value) for value in state.way_table[slot, :JOINT_COLUMN])
        on_a, on_b = taken.setdefault((way_a, mouth_a), set()), taken.setdefault((way_b, mouth_b), set())
        free = next(number for number in itertools.count() if number not in on_a and number not in on_b)
        state.way_table[slot, PASS_COLUMN] = free
        on_a.add(free)
        on_b.add(free)
    state.way_passes = int(state.way_table[:state.way_count, PASS_COLUMN].max(initial=0)) + 1


def _send_ways(state, changed: list, fresh: list) -> None:
    """Upload the changed way slots; a new way starts with 0 in every other way array."""
    for column, name in enumerate(WAY_COLUMNS):
        state.send_rows(name, state.way_table[:, column], changed)
    for name, (dtype, shape, each_step) in state.arrays.items():
        if shape == "way" and name not in WAY_COLUMNS and fresh:
            values = getattr(state, name).numpy()
            values[fresh] = 0
            state.send_rows(name, values, fresh)


def relink(state, cells) -> None:
    """Upload the named cells' relations and make the ways of the joints they are or touch follow them."""
    cells = sorted({int(cell) for cell in cells} | state.unsent)
    state.unsent = set()
    kind = state.kind.numpy()
    state.send_neighbours(cells)
    changed, born = [], []
    for joint in _touched_joints(state, cells, kind):
        dead, new = _rebuild_joint(state, joint, kind)
        changed.extend(dead)
        born.extend(new)
    fresh = _place_born(state, born)
    _hand_out_passes(state, fresh)
    _send_ways(state, sorted(set(changed) | set(fresh)), sorted(fresh))
