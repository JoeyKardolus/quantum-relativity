"""`cross` on one volume set moving: the flow goes on through its side, friction takes direction, the books hold."""
from __future__ import annotations

import numpy as np

from physics import engine
from physics.engine.axiom_1_energy.logic.count_cells import count_cells
from physics.engine.axiom_4_crossing.logic.cross import cross
from physics.engine.axiom_4_crossing.logic.tests.conftest import PLUS_X, face_way, held_by, ramp, still, universe

EVEN = {"unevenness": 0.0, "inner_unevenness": 0.0}
STEP = 0.5


def _moving(friction: float):
    """An even universe at rest in which one volume's fluid runs toward +x; the volume, and its way through +x."""
    state, settings = universe(friction=friction, **EVEN)
    still(state)
    volume = int(np.flatnonzero(state.host("kind") == engine.ENERGY_VOLUME)[0])
    ramp(state, volume, STEP)
    return state, settings, volume, face_way(state, volume, PLUS_X)


def _out(state, way: dict) -> int:
    """What the way's near end sent through it this step."""
    return int(state.host("way_out_" + way["end"])[way["slot"]])


def _back(state, way: dict) -> int:
    """What came back through the way this step."""
    return int(state.host("way_out_" + ("b" if way["end"] == "a" else "a"))[way["slot"]])


def test_a_volume_whose_fluid_runs_at_a_side_sends_through_it_and_nothing_comes_back():
    state, settings, volume, way = _moving(1e-9)
    before, far_before = held_by(state, volume), held_by(state, way["far"])
    total = int(np.abs(state.quanta.numpy()).sum())
    cross(state, settings)
    count_cells(state)
    assert _out(state, way) > 0 and _back(state, way) == 0
    assert held_by(state, way["far"]) > far_before, "what leaves arrives next door in the same step"
    assert int(state.host("crossed")[way["joint"]]) == _out(state, way), "and is booked on the joint it went through"
    sent = int(state.host("way_out_a").sum() + state.host("way_out_b").sum())
    assert before - held_by(state, volume) <= sent and int(np.abs(state.quanta.numpy()).sum()) == total


def test_what_arrives_is_still_moving_and_the_more_friction_the_less_it_turns_the_fluid_it_joins():
    turns = []
    for friction in (1e-9, 0.5, 1.0):
        state, settings, volume, way = _moving(friction)
        cross(state, settings)
        mouth = state.block("phase", way["far"])[0, 1:5, 1:5]
        turns.append(float(np.abs(mouth).mean()))
    assert turns[0] > turns[1] > turns[2] >= 0.0, turns


def test_friction_takes_the_direction_off_its_share_and_that_is_booked_as_heat_in_the_taking_volume():
    state, settings, volume, way = _moving(0.5)
    cross(state, settings)
    crossed, heated = int(state.host("crossed").sum()), int(state.host("heated").sum())
    assert crossed > 1000 and abs(heated / crossed - 0.5) < 0.02
    assert int(state.host("heat_quanta").sum()) == heated, "all of it was new heat: nothing was heat before"
    lost = int(state.host("way_lost_" + way["end"])[way["slot"]])
    assert lost > 0 and int(state.host("heat_quanta")[way["far"]]) >= lost


def test_nothing_crosses_where_the_joint_between_holds_no_minus_energy():
    state, settings, volume, way = _moving(0.5)
    state.set_block("quanta", way["joint"], np.zeros((engine.BLOCK,) * 3, dtype=np.int32))
    count_cells(state)
    cross(state, settings)
    assert _out(state, way) == 0 and _back(state, way) == 0
