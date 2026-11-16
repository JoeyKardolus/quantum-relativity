"""Quantum: one step of every moving energy's own life: its wave turns, its oscillation turns and the pointer jumps, and it settles when something reacts or its momentum is gone.

Contract: `quantum(state, settings, pull)` runs once a step, right after the crossing
(step.py), and does nothing while no moving energy is tracked (quantum_store.py). `pull`
is each cell's mean pull (cell_count.py). Per moving energy, in this order (axiom 7,
bible physics/quantum-relativity):
  1. a collapse or a tear that changed one of its cells is something reacting with it:
     it settles whole (quantum_settle.py, cause "collapse or tear"). Looked for only
     when the universe's collapses + volume splits changed: a cell's joints change by
     nothing else. A cell that fell into another by collapse is followed there first.
  2. its wave in each cell turns as the fluid does there (on_site.py): each part's
     phase by -pull * step_time * the cell's clock rate.
  3. a split's own oscillation turns by speed * step_time: a shared state of the split,
     counted on neither the counter's time nor any cell's clock but in its own turns.
     At each whole turn completed the pointer jumps to one of the cells, with that
     cell's share of the split energy as its chance (every jump drawn).
  4. friction: the crossing is the only friction and takes all direction off what
     crosses (cross.py). Of a cell's quanta, the share that left it through its joints
     this step (state.crossed, against the count before the crossing) is the share of
     this energy's still directed quanta there that lost their direction, in whole
     quanta, floor(lost + uniform). When none is left directed the energy has lost its
     momentum and settles whole in the cell the pointer is on (cause "momentum").
No setting. Quanta move only in a settle, exactly. Returns the events of this step.
"""
from __future__ import annotations

import math

import numpy as np

from physics.engine.quantum_joints import quantum_joints
from physics.engine.quantum_settle import quantum_settle
from physics.engine.quantum_store import quantum_store
from physics.engine.settings import Settings
from physics.engine.state import State


def quantum(state: State, settings: Settings, pull: np.ndarray) -> list:
    """Advance every moving energy by one step and settle those reacted with or out of momentum."""
    if state.quantum is None or not state.quantum["moving"]:
        return []
    reacted, spent, left = [], [], {}
    for record in quantum_store(state)["moving"]:
        universe = record["universe"]
        if _cells_changed(state, record):
            reacted.append(record)
            continue
        for index, cell in enumerate(record["cells"]):
            record["phases"][index] -= float(pull[universe, cell] * state.rate[universe, cell]) * settings.step_time
        _turn(record, settings)
        if universe not in left:
            left[universe] = _left(state, universe)
        _rub(record, left[universe], state.cell_quanta[universe])
        if record["directed"] == 0:
            spent.append(record)
    return quantum_settle(state, reacted, "collapse or tear") + quantum_settle(state, spent, "momentum")


def _cells_changed(state: State, record: dict) -> bool:
    """Whether a collapse or tear changed one of the record's cells since last looked; vanished cells are followed."""
    universe = record["universe"]
    count = int(state.collapses[universe] + state.splits[universe])
    if record["seen"] == count:
        return False
    home = _successor(state, record, record["cells"][record["pointer"]])
    changed = False
    merged = {}
    for cell, part in zip(record["cells"], record["parts"]):
        now = _successor(state, record, cell)
        changed = changed or now != cell or quantum_joints(state, universe, now) != record["joints"][cell]
        merged[now] = merged.get(now, 0) + part
    if changed:
        record["cells"], record["parts"] = list(merged), list(merged.values())
        record["phases"] = [0.0] * len(merged)
        record["pointer"] = record["cells"].index(home)
    record["seen"] = count
    record["joints"] = {cell: quantum_joints(state, universe, cell) for cell in record["cells"]}
    return changed


def _successor(state: State, record: dict, cell: int) -> int:
    """The volume a cell is part of now: itself, or the one it fell into by collapse.

    A joint the cell had that still exists lies now between the volume it fell into and the joint's other
    volume; with none left, it is the volume across a vanished joint that now neighbours the others.
    """
    universe = record["universe"]
    if state.kind[universe, cell] == 1:
        return cell
    known = record["joints"].get(cell, {})
    for joint, other in known.items():
        if state.kind[universe, joint] == -1:
            sides = [int(side) for side in state.between[universe, joint]]
            if other in sides:
                sides.remove(other)
            if sides[0] >= 0 and state.kind[universe, sides[0]] == 1:
                return sides[0]
    across = [other for joint, other in known.items()
              if state.kind[universe, joint] == 0 and other >= 0 and state.kind[universe, other] == 1]
    for volume in across:
        beside = set(quantum_joints(state, universe, volume).values())
        if all(other in beside for other in across if other != volume):
            return volume
    raise RuntimeError(f"moving energy {record['serial']}: cell {cell} vanished, the volume it fell into not found")


def _turn(record: dict, settings: Settings) -> None:
    """Turn a split's own oscillation by one step; the pointer jumps once per whole turn completed, by share."""
    if record["speed"] is None:
        return
    before = record["turn"]
    record["turn"] = before + record["speed"] * settings.step_time
    jumps = int(math.floor(record["turn"]) - math.floor(before))
    if jumps > 0:
        shares = np.array(record["parts"], dtype=np.float64) / record["amount"]
        record["pointer"] = int(record["rng"].choice(len(shares), size=jumps, p=shares)[-1])
        record["jumps"] += jumps


def _left(state: State, universe: int) -> np.ndarray:
    """Per cell of one universe the quanta that left it through its joints in this step's crossing."""
    left = np.zeros(state.capacity, dtype=np.int64)
    sides = state.between[universe]
    joints = np.flatnonzero((state.kind[universe] == -1) & (sides[:, 0] >= 0) & (sides[:, 1] >= 0))
    crossed = state.crossed[universe, joints]
    np.add.at(left, sides[joints, 0], np.maximum(crossed, 0))
    np.add.at(left, sides[joints, 1], np.maximum(-crossed, 0))
    return left


def _rub(record: dict, left: np.ndarray, cell_quanta: np.ndarray) -> None:
    """Friction: take the direction off the record's share of what crossed out of its cells, in whole quanta."""
    still = record["directed"] / record["amount"]
    lost = sum(part * still * left[cell] / max(int(cell_quanta[cell]), 1)
               for cell, part in zip(record["cells"], record["parts"]))
    record["directed"] = max(0, record["directed"] - int(math.floor(lost + record["rng"].random())))
