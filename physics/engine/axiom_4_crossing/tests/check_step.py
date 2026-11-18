"""Check step: axiom 4 read off the state after one step, as plain numbers.

Contract: `check_step(state, settings) -> dict`, read-only, called after every step.
  crossed             plus quanta that crossed a joint this step, both senses counted
  heated              how many of them lost their direction there (friction)
  net_crossed         quanta that really changed volume: half the sum over the volumes
                      of what each took less what it gave, ignoring sign
  heat                quanta held now that have lost their direction
  heat_share          `heat` over all plus quanta
  heat_wrong          cells whose heat is below 0 or above what they hold; must be 0
                      (nothing trims the count: it has to come out right by itself)
  heated_elsewhere    quanta booked as heated on an id that was not made a joint; must
                      be 0 (the only friction is in a joint)
  ways_without_joint  living ways whose joint is not a living joint or whose ends are
                      not two living energy volumes; must be 0 (only through minus)
  crossed_elsewhere   quanta booked as crossed through an id that was not made a joint;
                      must be 0
  sent_by_type        {"face", "edge", "corner"}: quanta sent this step along the ways
                      of each type of mouth, both senses (a measure of the shape a
                      block gives a cell; see "Open" in axiom 3's CLAUDE.md)
"""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_3_structure.definitions.direction_codes import step_of_host
from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT

_TYPE_OF_CODE = np.array([sum(part != 0 for part in step_of_host(code)) for code in range(27)])
_TYPES = ("face", "edge", "corner")


def _off_ways(state, kind: np.ndarray, quanta: np.ndarray, net: np.ndarray) -> int:
    """Quanta that changed volume this step on no way (a settling, axiom 7); -1 in a step with a collapse or tear."""
    if state.changed:
        return -1
    volumes = kind == ENERGY_VOLUME
    change = quanta[volumes].astype(np.int64) - state.host("quanta_before")[volumes] - net[volumes]
    return int(np.abs(change).sum()) // 2


def check_step(state, settings) -> dict:
    """Axiom 4's numbers for the step just taken."""
    kind, quanta = state.host("kind"), state.host("cell_quanta")
    way_a, way_b, way_joint = state.host("way_a"), state.host("way_b"), state.host("way_joint")
    moved = state.host("way_moved").astype(np.int64)
    sent = state.host("way_out_a").astype(np.int64) + state.host("way_out_b")
    living = way_joint >= 0
    net = np.zeros(len(kind), dtype=np.int64)
    np.add.at(net, way_b[living], moved[living])
    np.subtract.at(net, way_a[living], moved[living])
    sound = np.zeros(len(living), dtype=bool)
    sound[living] = (kind[way_joint[living]] == JOINT) & (kind[way_a[living]] == ENERGY_VOLUME) \
        & (kind[way_b[living]] == ENERGY_VOLUME)
    heat, heated = state.host("heat_quanta"), state.host("heated")
    plus = int(quanta[kind == ENERGY_VOLUME].sum())
    types = _TYPE_OF_CODE[state.host("way_mouth_a")[living]]
    return {
        "crossed": int(state.host("crossed").sum()),
        "heated": int(heated.sum()),
        "net_crossed": int(np.abs(net).sum()) // 2,
        "off_ways": _off_ways(state, kind, quanta, net),
        "heat": int(heat.sum()),
        "heat_share": float(heat.sum()) / plus if plus else 0.0,
        "heat_wrong": int(((heat < 0) | (heat > quanta)).sum()),
        "heated_elsewhere": int(heated[state.host("made_kind") != JOINT].sum()),
        "ways_without_joint": int((living & ~sound).sum()),
        "crossed_elsewhere": int(state.host("crossed")[state.host("made_kind") != JOINT].sum()),
        "sent_by_type": {name: int(sent[living][types == number].sum())
                         for number, name in enumerate(_TYPES, start=1)},
    }
