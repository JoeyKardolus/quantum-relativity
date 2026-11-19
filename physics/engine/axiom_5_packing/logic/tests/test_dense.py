"""(4) A dense area in a sheet of calm volumes gathers inward on its middle; (6) a small region set going sideways beside a big one."""
import numpy as np
import pytest

from physics.engine import step
from physics.engine.axiom_5_packing.logic.line import calm_block, dense, volume_at
from physics.engine.axiom_5_packing.logic.lumps import moving

CALM = 648_000


def _sheet(store, wide: int) -> np.ndarray:
    """The sheet's volumes' quanta in calm volumes, (wide, wide)."""
    cells = [volume_at(store, (0, row, column)) for row in range(wide) for column in range(wide)]
    return store.cell_quanta.numpy()[cells].reshape(wide, wide) / CALM


def _middle_and_spread(held: np.ndarray) -> tuple:
    """Where the middle of the energy lies, in volumes from the sheet's middle, and how far it is spread round it."""
    out = np.indices(held.shape) - held.shape[0] // 2
    middle = (held * out).sum(axis=(1, 2)) / held.sum()
    return np.abs(middle).max(), float(np.sqrt((held * ((out - middle[:, None, None]) ** 2).sum(axis=0)).sum() / held.sum()))


def test_a_dense_area_gathers_inward_on_its_middle_and_its_middle_stays_put():
    store = dense()  # the window's scene: `--scene dense`
    seen = [_sheet(store, 7)]
    for _ in range(3):
        for _ in range(25):
            step(store)
        seen.append(_sheet(store, 7))
    area = [float(held[2:5, 2:5].sum()) for held in seen]
    spread = [_middle_and_spread(held)[1] for held in seen]
    print(f"the nine hold {np.round(area, 1)}, their middle one {[round(float(held[3, 3]), 1) for held in seen]}, spread {np.round(spread, 2)}")
    assert (np.diff(area) > 0).all() and area[-1] > 1.4 * area[0]  # 36, 40.5, 47.7, 52.6 of the sheet's 76
    assert (np.diff(spread) < 0).all()  # in volumes: it draws in, it is not thrown apart
    assert seen[-1][3, 3] > 30 and max(_middle_and_spread(held)[0] for held in seen) < 0.02  # 4 to 34; the middle stays
    # not right: three turns on the sheet is no longer even side for side (9.5 above the middle, 3.6 below)


@pytest.mark.xfail(strict=True, reason="(6) no orbit. A volume 4 times fuller, three volumes from one 16 times fuller in "
                   "a sheet of 81 and set going sideways at 1.6 calm lengths a turn (the round-path speed for 16 at 6 "
                   "lengths), is carried one volume sideways in the first turn and thins from 4 to 2.3; then it is no "
                   "body any more, and by 2.6 turns the volume between the two holds 14: it fell in, it did not go round")
def test_a_small_region_set_going_sideways_beside_a_big_one_goes_round_it():
    big, small = (0, 4, 4), (0, 4, 7)
    store = calm_block((1, 9, 9), {big: 16, small: 4})
    moving(store, [volume_at(store, small)], 1, 1.6)
    for _ in range(75):
        step(store)
    held = _sheet(store, 9)
    fullest = np.unravel_index(np.argmax(np.where(np.abs(np.indices(held.shape) - 4).max(axis=0) > 1, held, 0)), held.shape)
    assert held[fullest] > 2 and fullest[0] != 4 and abs(fullest[1] - 4) < 3  # still a body, off the line it began on
