"""Collapse: a joint squeezed too far gives way, its energy is taken off the rest, its two volumes merge.

Contract: `collapse(state, held, joint)` carries out one collapse on the cells in `held`
and returns the event's record.
 1. The two volumes are the two of the joint's hardest way (`squeezed_way`), each as it
    stands now: a volume an earlier collapse of this step merged away is the volume it
    fell into (`fell_into`). When those two are one volume by now, the next hardest of
    the joint's ways whose two volumes still differ (`way_press`, a tie to the lowest
    slot). `kept` is the lower id, `gone` the other. A joint none of whose ways still
    has two volumes only goes (steps 2 and 5): there is nothing left for it to merge.
 2. The joint goes (`remove_joint`): its quanta are booked in `taken_joint`. The joints
    that lay on opposite sides of it now touch each other (`close_web`), so the web of
    joints stays one piece.
 3. The two volumes merge into `kept`: box onto box, amounts add exactly and the phase
    is that of the two fluids joined (`joined_phase_host`, axiom 7: each counts by its
    quanta; kept's own where they cancel); heat adds. `gone`'s
    boxes become 0, its id is no cell, `fell_into[gone] = kept`. `kept` keeps its swirl
    axis and sense.
 4. Relations: every joint that listed `gone` lists `kept` at the same code, and `kept`
    mirrors it. No other relation of a volume changes: the joints round the two stay.
 5. The joint's number of quanta is taken off the merged volume (`take_off`).
`collapses` and `merges` rise. The record: `{"what": "collapse", "joint", "kept", "gone",
"taken_joint", "taken_volume", "joint_type", "volumes" (how many the joint had)}`.
"""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_3_structure.definitions.direction_codes import step_of_host
from physics.engine.axiom_3_structure.definitions.kinds import BLOCK, ENERGY_VOLUME, NO_CELL
from physics.engine.axiom_6_limits.logic.cell_value import cell_value
from physics.engine.axiom_6_limits.logic.close_web import close_web
from physics.engine.axiom_6_limits.logic.hold_cell import hold_cell
from physics.engine.axiom_6_limits.logic.remove_joint import remove_joint
from physics.engine.axiom_6_limits.logic.set_value import set_value
from physics.engine.axiom_6_limits.logic.take_off import take_off
from physics.engine.axiom_7_quantum.definitions.joined_phase import joined_phase_host

_JOINT_TYPES = {1: "face", 2: "edge", 3: "corner"}
_MOST_IN_A_BOX = 2 ** 31 - 1


def _volumes_of(state, held, joint: int) -> list:
    """The joint's entries that name a living energy volume, `(volume, code)` each."""
    return [(one, code) for one, code in held.entries[joint]
            if int(cell_value(state, held, "kind", one)) == ENERGY_VOLUME]


def _stands_now(state, held, volume: int) -> int:
    """The living volume an id stands for now: itself, or the one it fell into, followed to the end."""
    volume = int(volume)
    while int(cell_value(state, held, "kind", volume)) != ENERGY_VOLUME:
        into = int(cell_value(state, held, "fell_into", volume))
        if into < 0:
            return -1
        volume = into
    return volume


def _hardest_first(state, held, joint: int) -> list:
    """The joint's ways as `(slot, a, b, mouth_a)`, its hardest way first, then by falling press, then slot."""
    hardest = int(cell_value(state, held, "squeezed_way", joint))
    ways = [(slot, way_a, way_b, mouth_a) for slot, way_a, way_b, mouth_a, mouth_b in state.ways_of(joint)]
    press = {way[0]: float(cell_value(state, held, "way_press", way[0])) for way in ways}
    return sorted(ways, key=lambda way: (way[0] != hardest, -press[way[0]], way[0]))


def _merging_pair(state, held, joint: int):
    """The two volumes that merge, lower id first, and the joint's code seen from one of them; None for none."""
    for slot, way_a, way_b, mouth_a in _hardest_first(state, held, joint):
        now_a, now_b = _stands_now(state, held, way_a), _stands_now(state, held, way_b)
        if now_a >= 0 and now_b >= 0 and now_a != now_b:
            return min(now_a, now_b), max(now_a, now_b), mouth_a
    return None


def _merge_fluid(state, held, kept: int, gone: int) -> None:
    """Put `gone`'s fluid and heat onto `kept`'s, box onto box, and empty `gone`."""
    hold_cell(state, held, kept, with_block=True)
    hold_cell(state, held, gone, with_block=True)
    summed = held.quanta[kept].astype(np.int64) + held.quanta[gone].astype(np.int64)
    if summed.max() > _MOST_IN_A_BOX:
        raise OverflowError(f"merging {gone} into {kept} puts more than {_MOST_IN_A_BOX} quanta in one box")
    held.phase[kept] = joined_phase_host(held.quanta[kept], held.phase[kept], held.quanta[gone], held.phase[gone])
    held.quanta[kept] = summed.astype(np.int32)
    held.quanta[gone] = np.zeros((BLOCK, BLOCK, BLOCK), dtype=np.int32)
    held.phase[gone] = np.zeros((BLOCK, BLOCK, BLOCK), dtype=np.float64)
    heat = int(cell_value(state, held, "heat_quanta", kept)) + int(cell_value(state, held, "heat_quanta", gone))
    set_value(state, held, "heat_quanta", kept, heat)
    set_value(state, held, "heat_quanta", gone, 0)
    set_value(state, held, "kind", gone, NO_CELL)
    set_value(state, held, "fell_into", gone, kept)


def _hand_relations(state, held, kept: int, gone: int) -> None:
    """Every joint that listed `gone` lists `kept` at the same code, and `kept` mirrors it."""
    listed = set(held.entries[kept])
    for joint, code in held.entries[gone]:
        hold_cell(state, held, joint)
        renamed = [(kept, there) if one == gone else (one, there) for one, there in held.entries[joint]]
        held.entries[joint] = list(dict.fromkeys(renamed))
        if (joint, code) not in listed:
            held.entries[kept].append((joint, code))
            listed.add((joint, code))
    held.entries[gone] = []


def collapse(state, held, joint: int) -> dict:
    """Collapse one joint; returns the event's record."""
    joint = int(joint)
    hold_cell(state, held, joint, with_block=True)
    volumes = _volumes_of(state, held, joint)
    pair = _merging_pair(state, held, joint)
    entries = list(held.entries[joint])
    taken_joint = remove_joint(state, held, joint)
    close_web(state, held, entries)
    record = {"what": "collapse", "joint": joint, "taken_joint": taken_joint,
              "volumes": len({one for one, code in volumes})}
    state.collapses += 1
    if pair is None:
        kept = volumes[0][0] if volumes else None
        record.update(kept=kept, gone=kept, joint_type="closed",
                      taken_volume=take_off(state, held, kept, taken_joint) if volumes else 0)
        return record
    kept, gone, mouth = pair
    _merge_fluid(state, held, kept, gone)
    _hand_relations(state, held, kept, gone)
    record.update(kept=kept, gone=gone, taken_volume=take_off(state, held, kept, taken_joint),
                  joint_type=_JOINT_TYPES[sum(part != 0 for part in step_of_host(mouth))])
    state.merges += 1
    return record
