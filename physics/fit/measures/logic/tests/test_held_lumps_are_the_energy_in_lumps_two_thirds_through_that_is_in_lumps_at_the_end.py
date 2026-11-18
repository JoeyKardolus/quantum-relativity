"""Held share: of the plus quanta two thirds through, the share in lump volumes then whose cell, or the cell it fell into, is a lump volume at the end."""
import numpy as np
import pytest

from physics.fit.measures.logic.held_share import held_share
from physics.fit.measures.logic.tests.hand_cells import ring_snapshot

_EVEN = [10.0] * 6 + [1.0] * 6
_NOWHERE = np.full(12, -1)


def test_a_lump_that_stays_holds_one_that_fell_into_a_lump_holds_and_one_that_melted_does_not():
    then = ring_snapshot(_EVEN, lumps=[0, 2, 4], held=[300, 50, 250, 50, 250, 100] + [10] * 6)
    last = ring_snapshot(_EVEN, lumps=[0, 5], held=[550, 50, 0, 50, 50, 300] + [10] * 6)
    fell_into = _NOWHERE.copy()
    fell_into[2] = 0
    held = held_share([then, then, then, last], fell_into)
    assert held == pytest.approx((300 + 250) / 1000), "volume 0 held, 2 fell into 0, 4 is no lump any more"


def test_a_fall_is_followed_to_the_end_of_its_chain():
    then = ring_snapshot(_EVEN, lumps=[1], held=[100, 400, 100, 100, 100, 200] + [10] * 6)
    last = ring_snapshot(_EVEN, lumps=[5], held=[0, 0, 0, 100, 100, 800] + [10] * 6)
    fell_into = _NOWHERE.copy()
    fell_into[1], fell_into[0] = 0, 5
    assert held_share([then, last], fell_into) == pytest.approx(400 / 1000)


def test_the_snapshot_read_is_the_one_two_thirds_through():
    early = ring_snapshot(_EVEN, lumps=[], held=[100] * 6 + [10] * 6)
    then = ring_snapshot(_EVEN, lumps=[3], held=[100, 100, 100, 500, 100, 100] + [10] * 6)
    assert held_share([early, early, then, then], _NOWHERE) == pytest.approx(0.5), "index 2 of 0 .. 3"
    assert held_share([early, then, early, then], _NOWHERE) == 0.0, "no lump two thirds through"


def test_no_lumps_then_or_at_the_end_hold_nothing():
    none = ring_snapshot(_EVEN, lumps=[])
    some = ring_snapshot(_EVEN, lumps=[2], held=[100, 100, 300, 100, 100, 100] + [10] * 6)
    assert held_share([none, none], _NOWHERE) == 0.0 and held_share([some, none], _NOWHERE) == 0.0
