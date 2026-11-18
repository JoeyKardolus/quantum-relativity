"""Tear: an overstretched energy volume splits, and the place it tears is the new joint.

Contract: `tear(state, held, volume, new_joint, new_volume)` tears volume V on the cells
in `held` into V (the low piece), J (`new_joint`) and W (`new_volume`), and returns the
event's record. Only V splits.
 1. The axis is V's own swirl axis: a swirl is a stack of layers its energy goes round
    in, so it can only come apart between layers, and every piece is a whole swirl again.
 2. Fluid by `tear_map`: V keeps its two low layers, the two middle layers flip from
    plus to minus and are J, W gets the two high layers; each piece fills its own whole
    block. `torn` rises by J's quanta (the amount does not change); `fresh_quanta[J]` is
    J's quanta. V's heat is shared by quanta. J and W are made with V's swirl axis and
    sense, V's `pull` and `pull_old`, and `torn_from = V`.
 3. Relations, by each old entry `(joint, code)` of V and that code's step along the axis:
      step -1  the joint stays with V;
      step +1  the joint's entry for V now names W at the same code, and W lists it;
      step  0  the joint keeps V and gains W at the same code, W lists it too, and J
               and the joint list each other at that code and its opposite.
    J wraps the whole side of V that faces W and the whole side of W that faces V: V
    lists J at all nine codes that step +1 along the axis, W at all nine that step -1,
    and J lists them back. So V and W are wrapped on all 26 sides, J has nine ways from
    V to W, and the web of joints stays one piece.
 4. `tears` rises by 1. The record: `{"what": "tear", "volume", "joint", "new_volume",
    "axis", "quanta": [V, J, W]}`; `run_events` adds `"lengths"`, which needs a setting.
A joint beside V gains W as a neighbour (a new cell now lies next to it); no other cell's
energy, phase or kind changes.
"""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_1_energy.definitions.whole_shares import whole_shares
from physics.engine.axiom_3_structure.definitions.direction_codes import CENTRE, opposite_host, step_of_host
from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT
from physics.engine.axiom_6_limits.definitions.tear_map import tear_map
from physics.engine.axiom_6_limits.logic.cell_value import cell_value
from physics.engine.axiom_6_limits.logic.hold_cell import hold_cell
from physics.engine.axiom_6_limits.logic.set_value import set_value

_CODES = [code for code in range(27) if code != CENTRE]
_CARRIED = ("pull", "pull_old", "swirl_axis", "swirl_sense")


def _side_codes(axis: int, sense: int) -> list:
    """The nine codes that step `sense` along `axis`."""
    return [code for code in _CODES if step_of_host(code)[axis] == sense]


def _share_fluid(state, held, volume: int, new_joint: int, new_volume: int, axis: int) -> list:
    """Give the three cells their pieces of the volume's fluid; returns their quanta, V, J, W."""
    pieces = tear_map(held.quanta[volume], held.phase[volume], axis)
    for cell, (piece_quanta, piece_phase) in zip((volume, new_joint, new_volume), pieces):
        held.quanta[cell], held.phase[cell] = piece_quanta, piece_phase
    return [int(np.abs(piece_quanta.astype(np.int64)).sum()) for piece_quanta, piece_phase in pieces]


def _make_cells(state, held, volume: int, new_joint: int, new_volume: int, quanta: list) -> None:
    """Make J a joint and W a volume, both pieces of V: its pull, its swirl, its heat by quanta."""
    carried = {name: cell_value(state, held, name, volume) for name in _CARRIED}
    heat = [min(share, piece) for share, piece in
            zip(whole_shares(int(cell_value(state, held, "heat_quanta", volume)), quanta), quanta)]
    for cell, kind, cell_heat in ((volume, ENERGY_VOLUME, heat[0]), (new_joint, JOINT, heat[1]),
                                  (new_volume, ENERGY_VOLUME, heat[2])):
        set_value(state, held, "heat_quanta", cell, cell_heat)
        if cell == volume:
            continue
        set_value(state, held, "kind", cell, kind)
        set_value(state, held, "made_kind", cell, kind)
        set_value(state, held, "torn_from", cell, volume)
        for name, value in carried.items():
            set_value(state, held, name, cell, value)
    set_value(state, held, "fresh_quanta", new_joint, quanta[1])


def _share_relations(state, held, volume: int, new_joint: int, new_volume: int, axis: int) -> None:
    """Deal V's old joints over V, J and W by where each lies along the axis; J wraps the torn sides."""
    stays, goes, round_middle = [], [], []
    for joint, code in held.entries[volume]:
        hold_cell(state, held, joint)
        along, back = step_of_host(code)[axis], opposite_host(code)
        if along < 0:
            stays.append((joint, code))
        elif along > 0:
            held.entries[joint] = [(new_volume, there) if (one, there) == (volume, back) else (one, there)
                                   for one, there in held.entries[joint]]
            goes.append((joint, code))
        else:
            held.entries[joint] = held.entries[joint] + [(new_volume, back), (new_joint, back)]
            stays.append((joint, code))
            goes.append((joint, code))
            round_middle.append((joint, code))
    forward, backward = _side_codes(axis, 1), _side_codes(axis, -1)
    held.entries[volume] = stays + [(new_joint, code) for code in forward]
    held.entries[new_volume] = goes + [(new_joint, code) for code in backward]
    held.entries[new_joint] = ([(volume, code) for code in backward] + [(new_volume, code) for code in forward]
                               + round_middle)


def tear(state, held, volume: int, new_joint: int, new_volume: int) -> dict:
    """Tear one volume into a low volume, a new joint and a high volume; returns the event's record."""
    volume, new_joint, new_volume = int(volume), int(new_joint), int(new_volume)
    hold_cell(state, held, volume, with_block=True)
    hold_cell(state, held, new_joint, with_block=True)
    hold_cell(state, held, new_volume, with_block=True)
    axis = int(cell_value(state, held, "swirl_axis", volume))
    quanta = _share_fluid(state, held, volume, new_joint, new_volume, axis)
    _make_cells(state, held, volume, new_joint, new_volume, quanta)
    _share_relations(state, held, volume, new_joint, new_volume, axis)
    state.torn += quanta[1]
    state.tears += 1
    return {"what": "tear", "volume": volume, "joint": new_joint, "new_volume": new_volume, "axis": axis,
            "quanta": quanta}
