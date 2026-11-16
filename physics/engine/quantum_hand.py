"""Quantum hand: a moving energy is handed on between two neighbouring energy volumes without friction, its waves adding and cancelling.

Contract: `quantum_hand(state, giver, taker, turn, universe=None)` acts on every moving
energy (quantum_store.py) with a part in energy volume `giver` or `taker`, in one
universe or in all of the batch. The two parts, each a wave sqrt(part) e^{i phase},
take the fluid's own exact two-box step (pair_rotation.py, De Raedt 1987, here between
two cells): their mean stays, their half difference turns by `turn` radians, and the
flux becomes floor(flux + uniform) whole quanta moved from giver to taker (negative:
back), never more than the giving part holds; the cells' boxes follow by quantum_move.
From one part alone, turn pi hands all of it on, pi / 2 half, 2 asin(sqrt(share)) that
share: a splitter. Two parts meeting add or cancel by their phases: brought back
together in step they end whole in one cell.
A split (axiom 7): when a handover leaves a whole energy in two cells it is split; it
gets its own oscillation, its speed drawn there from the record's own draws (turns per
unit of the counter, log-uniform in SPEEDS), and its pointer. The pointer rides with
the energy it sits on: when quanta leave its cell it goes along with the chance of the
share that left, so its chance of being on a cell stays that cell's share. A handover
that leaves all of it in one cell is the merge back: the oscillation stops, logged as
"merge". Staged: the engine's own crossing (cross.py) takes all direction off what
crosses, so only a scene calls this, where it places a splitter or a route. Raises
ValueError unless both cells are energy volumes sharing a joint. Returns [(serial,
moved), ...].
"""
from __future__ import annotations

import math

import numpy as np

from physics.engine.quantum_joints import quantum_joints
from physics.engine.quantum_move import quantum_move
from physics.engine.quantum_store import quantum_store
from physics.engine.state import State

SPEEDS = (4.0, 256.0)


def quantum_hand(state: State, giver: int, taker: int, turn: float, universe: int | None = None) -> list:
    """Hand moving energy on between two neighbouring volumes by one exact pair rotation of its two parts."""
    store = quantum_store(state)
    met = [record for record in store["moving"] if universe in (None, record["universe"])
           and (giver in record["cells"] or taker in record["cells"])]
    moves, handed = [], []
    for record in met:
        _neighbours(state, record["universe"], giver, taker)
        moved = _rotate(record, giver, taker, turn)
        if moved > 0:
            moves.append((record["universe"], giver, taker, moved))
        elif moved < 0:
            moves.append((record["universe"], taker, giver, -moved))
        record["joints"] = {cell: record["joints"].get(cell) or quantum_joints(state, record["universe"], cell)
                            for cell in record["cells"]}
        handed.append((record["serial"], moved))
        _oscillation(state, store, record)
    quantum_move(state, moves)
    return handed


def _neighbours(state: State, universe: int, giver: int, taker: int) -> None:
    """Raise unless giver and taker are two energy volumes of the universe sharing a joint."""
    kinds, sides = state.kind[universe], state.between[universe]
    pair = sorted((giver, taker))
    if giver == taker or kinds[giver] != 1 or kinds[taker] != 1 or not np.any(
            (kinds == -1) & (sides[:, 0] == pair[0]) & (sides[:, 1] == pair[1])):
        raise ValueError(f"cells {giver} and {taker} of universe {universe} are not energy volumes sharing a joint")


def _rotate(record: dict, giver: int, taker: int, turn: float) -> int:
    """One pair rotation of the record's parts in giver and taker; returns the whole quanta moved giver to taker."""
    near, near_angle = _part(record, giver)
    far, far_angle = _part(record, taker)
    near_psi = math.sqrt(near) * complex(math.cos(near_angle), math.sin(near_angle))
    far_psi = math.sqrt(far) * complex(math.cos(far_angle), math.sin(far_angle))
    mean = (near_psi + far_psi) / 2.0
    turned = (near_psi - far_psi) / 2.0 * complex(math.cos(turn), math.sin(turn))
    near_new, far_new = mean + turned, mean - turned
    flux = near - abs(near_new) ** 2
    moved = min(max(int(math.floor(flux + record["rng"].random())), -far), near)
    home = _ride(record, giver, taker, moved, near, far)
    kept = {cell: (part, angle) for cell, part, angle in zip(record["cells"], record["parts"], record["phases"])
            if cell not in (giver, taker)}
    for cell, part, psi in ((giver, near - moved, near_new), (taker, far + moved, far_new)):
        if part > 0:
            kept[cell] = (part, math.atan2(psi.imag, psi.real))
    record["cells"] = list(kept)
    record["parts"] = [part for part, _ in kept.values()]
    record["phases"] = [angle for _, angle in kept.values()]
    record["pointer"] = record["cells"].index(home)
    return moved


def _part(record: dict, cell: int) -> tuple:
    """The record's whole quanta and phase in `cell`, (0, 0.0) when it has none there."""
    if cell not in record["cells"]:
        return 0, 0.0
    index = record["cells"].index(cell)
    return record["parts"][index], record["phases"][index]


def _ride(record: dict, giver: int, taker: int, moved: int, near: int, far: int) -> int:
    """The cell the pointer is on after `moved` quanta left: it goes along with the chance of the share that left."""
    home = record["cells"][record["pointer"]]
    source, target, held = (giver, taker, near) if moved > 0 else (taker, giver, far)
    if moved != 0 and home == source and record["rng"].random() * held < abs(moved):
        return target
    return home


def _oscillation(state: State, store: dict, record: dict) -> None:
    """Start the split's own oscillation when the energy is newly in two cells; stop it at the merge back."""
    if len(record["cells"]) > 1 and record["speed"] is None:
        low, high = SPEEDS
        record["speed"] = float(low * (high / low) ** record["rng"].random())
        record["turn"] = 0.0
        record["born"] = int(state.step)
    elif len(record["cells"]) == 1 and record["speed"] is not None:
        record["speed"], record["born"], record["turn"] = None, None, 0.0
        store["ended"].append({"serial": record["serial"], "universe": record["universe"], "step": int(state.step),
                               "cause": "merge", "cell": record["cells"][0], "amount": record["amount"], "drawn": 0,
                               "parts": {record["cells"][0]: record["amount"]}})
