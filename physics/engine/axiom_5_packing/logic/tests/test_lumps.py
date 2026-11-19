"""Staged checks in a grown patch, away from the first volume; each gap is a strict xfail with its measured number."""
from dataclasses import fields, replace

import numpy as np
import pytest
import warp as wp

from physics.engine import step
from physics.engine.axiom_2_swirls.logic.read_swirl import flow_along
from physics.engine.axiom_5_packing.logic.lumps import FULLER, fuller, grown, moving, region
from physics.engine.axiom_6_limits.logic.tests.test_grow import _checked_step

patch = pytest.fixture(scope="module")(lambda: grown(3, 5))  # the first volume still holds 1,044 calm volumes
OUT = np.arange(-3, 4)  # the volumes of the row through the middle

def _copy(store):
    """The store again, so each check stages its own."""
    return replace(store, **{one.name: wp.clone(getattr(store, one.name))
                             for one in fields(store) if isinstance(getattr(store, one.name), wp.array)})

def _stands(store, out: tuple) -> list:
    """Those of the region `out` volumes from the middle that stand already."""
    return [cell for cell in region(store, out) if cell in store.own_cell.numpy()[:store.count]]

def _row(store) -> np.ndarray:
    """The quanta of the volumes on the row through the middle along direction 1, in calm volumes."""
    cells = [region(store, (0, out, 0))[0] for out in OUT]
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
    return float((weight * OUT[where]).sum() / weight.sum())


def test_staging_makes_a_region_four_times_fuller_and_sets_it_going_with_exact_books(patch):
    store, cells = _copy(patch), region(patch, (0, 2, 0))
    before = store.cell_quanta.numpy()[cells]
    fuller(store, cells)
    moving(store, cells, 1, 0.3)
    assert (store.cell_quanta.numpy()[cells] == FULLER * before).all()
    slot = list(store.own_cell.numpy()).index(cells[0])  # the fluid goes up direction 1 now
    assert flow_along(store.box_quanta.numpy()[slot], store.box_phase.numpy()[slot], 1) > 0
    for _ in range(5):
        _checked_step(store)  # the books stay exact with the pull on


@pytest.mark.xfail(strict=True, reason="(b) a region made 4 times fuller beside the first volume keeps "
                   "0.24 of what it was given after half a turn (12 steps): it evens out")
def test_a_fuller_region_gains_from_its_thinner_surroundings(patch):
    cells = region(patch, (0, 2, 0))
    first, more, _staged = _after(patch, lambda store: fuller(store, cells), 12)
    assert more[4:7].sum() > first[4:7].sum()


def test_two_fuller_regions_draw_together(patch):
    stage = lambda store: [fuller(store, _stands(store, (0, way, 0))) for way in (-2, 2)]  # noqa: E731
    first, more, _staged = _after(patch, stage, 12)
    apart = lambda row: _middle(row, slice(4, 7)) - _middle(row, slice(0, 3))  # noqa: E731
    assert apart(more) < apart(first)  # measured: 2.78 volumes apart, then 2.73 (no joint gives way now)


@pytest.mark.xfail(strict=True, reason="(e) a fuller region set going one way: after a turn its energy's "
                   "middle is 0.74 of a volume on, not 2, though 0.80 of it is still in the row")
def test_a_fuller_region_given_motion_keeps_going_across_several_cells(patch):
    cells = _stands(patch, (0, -2, 0))
    first, more, _staged = _after(patch, lambda store: (fuller(store, cells), moving(store, cells, 1, 0.3)), 25)
    assert _middle(more) - _middle(first) > 2 and np.clip(more, 0, None).sum() > 0.5 * first.sum()
