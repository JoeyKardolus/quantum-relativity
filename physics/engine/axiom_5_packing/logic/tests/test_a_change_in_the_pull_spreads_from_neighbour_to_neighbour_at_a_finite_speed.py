"""The pull's own wave on a real universe at rest: one volume made heavier, and when the change arrives where.

Only `spread_pull` runs, on two universes that differ in that one volume; what one hears
before the other is the change. A hop is a neighbour: 155 long here, 620 steps of light.
"""
from __future__ import annotations

import functools

import numpy as np
import pytest
import warp as wp

from physics import engine
from physics.engine.axiom_1_energy.logic.clear_ledgers import clear_ledgers
from physics.engine.axiom_1_energy.logic.count_cells import count_cells
from physics.engine.axiom_3_structure.logic.size_cells import size_cells
from physics.engine.axiom_5_packing.logic.pack_cells import pack_cells
from physics.engine.axiom_5_packing.logic.spread_pull import spread_pull
from physics.engine.axiom_5_packing.logic.start_pull import start_pull

wp.config.quiet = True

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="needs CUDA: 2,600 wave steps of two universes")

SIDE, STEPS, HOPS = 16, 2600, (1, 2, 3, 4)

#: A clause the engine does not keep yet (axiom 5's CLAUDE.md, "Not done": the pull outruns light).
#: Strict: the day it holds, the mark turns the test red and must go.
NOT_KEPT = pytest.mark.xfail(strict=True, reason="not kept: the pull outruns light (axiom 5, Not done)")


def _universe(lump: float, at_rest: bool):
    """An even universe; volume 0 holds `lump` times the others, with the pull as it was or set at rest again."""
    settings = engine.Settings(unevenness=0.0, inner_unevenness=0.0)
    state = engine.new_universe(settings, cells_per_side=SIDE, seed=1, device="cuda")
    if lump != 1.0:
        state.set_block("quanta", 0, np.rint(state.block("quanta", 0) * lump).astype(np.int32))
        count_cells(state)
        size_cells(state, settings)
        pack_cells(state, settings)
        if at_rest:
            start_pull(state, settings)
    return state, settings


def _along(hops: int) -> int:
    """The id of the start cell `hops` neighbours along one line from volume 0 (odd: a joint, even: a volume)."""
    return hops * SIDE * SIDE


@functools.lru_cache(maxsize=None)
def _story(lump: float) -> dict:
    """The change at each hop, step by step, and what it comes to at rest."""
    rest, settings = _universe(lump, True)
    even, settings = _universe(1.0, True)
    far = _along(8)
    target = {hops: float((rest.host("pull") - even.host("pull"))[_along(hops)]
                          - (rest.host("pull") - even.host("pull"))[far]) for hops in HOPS}
    heavy, control = _universe(lump, False)[0], _universe(1.0, False)[0]
    length = heavy.host("length")
    light = float(length[_along(1)] + length[_along(2)]) / 2.0 / (settings.light_speed * settings.step_time)
    change = {hops: [] for hops in HOPS}
    for _ in range(STEPS):
        for state in (heavy, control):
            clear_ledgers(state)
            spread_pull(state, settings)
        differs = heavy.host("pull") - control.host("pull")
        for hops in HOPS:
            change[hops].append(float(differs[_along(hops)]))
    return {"target": target, "change": change, "light": light}


def _first_step(values: list, bar: float) -> int:
    """The first step at which the change is at least `bar` in size; one past the run when it never is."""
    return next((step for step, value in enumerate(values, start=1) if abs(value) >= bar), len(values) + 1)


def test_a_change_reaches_no_cell_sooner_than_one_neighbour_a_step_so_nothing_is_heard_at_once():
    story = _story(1.1)
    for hops in HOPS:
        assert not any(story["change"][hops][:hops - 1]), f"{hops} hops away it changed before step {hops}"
        assert any(story["change"][hops])


def test_a_change_of_a_thousandth_of_a_volume_s_energy_is_heard():
    story = _story(1.001)
    assert abs(story["change"][1][999]) > 1e-4 and abs(story["change"][2][999]) > 1e-5


def test_once_at_rest_the_change_is_weaker_with_every_neighbour_further_away():
    target = _story(1.1)["target"]
    sizes = [abs(target[hops]) for hops in HOPS]
    assert sizes == sorted(sizes, reverse=True) and sizes[-1] > 0.0


@pytest.mark.parametrize("hops", [pytest.param(hops, marks=[NOT_KEPT] if hops in (2, 4) else []) for hops in HOPS])
def test_half_of_a_change_does_not_arrive_before_light_does(hops):
    story = _story(1.1)
    half = _first_step(story["change"][hops], 0.5 * abs(story["target"][hops]))
    assert half >= hops * story["light"], f"half the change was {hops} hops away at step {half}, light at {hops * story['light']:.0f}"


@pytest.mark.parametrize("hops", [pytest.param(hops, marks=[NOT_KEPT]) for hops in HOPS])
def test_a_hundredth_of_a_change_does_not_arrive_before_light_does(hops):
    story = _story(1.1)
    trace = _first_step(story["change"][hops], 0.01 * abs(story["target"][hops]))
    assert trace >= hops * story["light"], f"1% of the change was {hops} hops away at step {trace}, light at {hops * story['light']:.0f}"
