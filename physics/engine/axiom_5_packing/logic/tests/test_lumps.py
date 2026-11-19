"""Staged checks in a grown patch; each gap is a strict xfail with its measured number (seed 1)."""
from dataclasses import fields, replace

import numpy as np
import pytest
import warp as wp

from physics.engine import step
from physics.engine.axiom_2_swirls.logic.read_swirl import flow_along
from physics.engine.axiom_5_packing.logic.lumps import FULLER, fuller, grown, moving, region
from physics.engine.axiom_6_limits.logic.tests.test_grow import _checked_step

patch = pytest.fixture(scope="module")(lambda: grown(4, 6))


def _copy(store):
    """The store again, so each check stages its own."""
    return replace(store, **{one.name: wp.clone(getattr(store, one.name))
                             for one in fields(store) if isinstance(getattr(store, one.name), wp.array)})


def _row(store) -> np.ndarray:
    """The quanta of the volumes on the row through the middle along direction 1, in calm volumes."""
    cells = [region(store, (0, out, 0))[0] for out in range(-4, 5)]
    return store.cell_quanta.numpy()[cells] / 648_000


def _after(patch, stage, steps: int) -> tuple:
    """The rows of the staged patch and of the patch left alone, both `steps` steps on."""
    staged, plain = _copy(patch), _copy(patch)
    stage(staged)
    first = _row(staged) - _row(plain)
    for _ in range(steps):
        step(staged)
        step(plain)
    return first, _row(staged) - _row(plain), staged


def _middle(more: np.ndarray, where=slice(None)) -> float:
    """The middle of the energy a row holds more than the patch left alone, in volumes along it."""
    weight = np.clip(more, 0, None)[where]
    return float((weight * np.arange(-4, 5)[where]).sum() / weight.sum())


def test_staging_makes_a_region_four_times_fuller_and_sets_it_going_with_exact_books(patch):
    store, cells = _copy(patch), region(patch, (0, 0, 0))
    before = store.cell_quanta.numpy()[cells]
    fuller(store, cells)
    moving(store, cells, 1, 0.3)
    assert (store.cell_quanta.numpy()[cells] == FULLER * before).all()
    slot = list(store.own_cell.numpy()).index(cells[0])  # the fluid goes up direction 1 now
    assert flow_along(store.box_quanta.numpy()[slot], store.box_phase.numpy()[slot], 1) > 0
    for _ in range(5):
        _checked_step(store)  # the books stay exact with the pull on


@pytest.mark.xfail(strict=True, reason="(b) a region made 4 times fuller in a fresh patch keeps 2% of "
                   "what it was given after half a turn, as it does with the pull off: through "
                   "thin fresh joints the crossing evens out faster than the pull bends it")
def test_a_fuller_region_gains_from_its_thinner_surroundings(patch):
    cells = region(patch, (0, 0, 0))
    first, more, _staged = _after(patch, lambda store: fuller(store, cells), 23)
    assert more[3:6].sum() > first[3:6].sum()


@pytest.mark.xfail(strict=True, reason="(c) two fuller regions 4.0 volumes apart: their energy's "
                   "middles lie 6.0 apart after half a turn; each spreads out and neither falls")
def test_two_fuller_regions_draw_together(patch):
    stage = lambda store: [fuller(store, region(store, (0, way, 0))) for way in (-2, 2)]  # noqa: E731
    first, more, _staged = _after(patch, stage, 23)
    apart = lambda row: _middle(row, slice(5, 9)) - _middle(row, slice(0, 4))  # noqa: E731
    assert apart(more) < apart(first)


@pytest.mark.xfail(strict=True, reason="(e) a fuller region set going one way: after a turn its energy's "
                   "middle has not gone on (-0.06 of a volume) and 2% of it is still together")
def test_a_fuller_region_given_motion_keeps_going_across_several_cells(patch):
    cells = region(patch, (0, 0, 0))
    first, more, _staged = _after(patch, lambda store: (fuller(store, cells), moving(store, cells, 1, 0.3)), 46)
    assert _middle(more) - _middle(first) > 2 and np.clip(more, 0, None).sum() > 0.5 * first.sum()
