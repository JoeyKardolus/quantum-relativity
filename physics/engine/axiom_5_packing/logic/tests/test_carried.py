"""A line of calm volumes and joints, every rule on: a fuller volume gathers; two draw each other, the heavier moving less; coasting is cut short."""
import numpy as np
import pytest

from physics.engine import step
from physics.engine.axiom_5_packing.logic.line import calm_block, calm_line, line_of

CALM = 648_000


def _line(fuller: tuple = (), going: float = 0.0):
    """The line bench and its cells."""
    store = calm_line(fuller=fuller, going=going)
    return store, line_of(store)


def _held(store, cells) -> np.ndarray:  # the volumes' quanta in calm volumes
    return store.cell_quanta.numpy()[cells[::2]] / CALM


def _at(store, cells, where=slice(None)) -> float:
    """Where the energy above calm stands from the middle of all the line's energy, through the cells' sizes."""
    sizes = CALM / np.abs(store.cell_quanta.numpy()[cells])
    middles, held = (np.cumsum(sizes) - sizes / 2)[::2], _held(store, cells)
    more = np.clip(held - 1, 0, None)[where]
    return float((more * middles[where]).sum() / more.sum() - (held * middles).sum() / held.sum())


def _turns(store, turns: int, act=step) -> None:
    for _ in range(25 * turns):
        act(store)


def _moved(fuller: dict, turns: int = 3) -> list:
    """How far each fuller volume of a line goes toward the line's middle, turn by turn."""
    store = calm_block((1, 11, 1), {(0, place, 0): times for place, times in fuller.items()})
    cells, parts, seen = line_of(store), [slice(0, 5) if place < 5 else slice(6, None) for place in fuller], []
    for _ in range(turns + 1):
        seen.append([abs(_at(store, cells, part)) for part in parts])
        _turns(store, 1)
    return (seen[0] - np.array(seen)).T.tolist()


def test_a_fuller_volume_gains_from_its_thinner_neighbours_and_stays_where_it_is():
    store, cells = _line((5,))
    seen = [_held(store, cells)]
    for _ in range(6):
        _turns(store, 1)
        seen.append(_held(store, cells))
    gained = np.array(seen)[:, 5]
    assert (np.diff(gained[:5]) > 0).all() and gained[4] > 5.8  # 4, 4.49, 5.35, 5.78, 5.91; then 5.90, 5.75
    assert max(seen[4][4], seen[4][6]) < 0.8  # its two neighbours thin to 0.19 and 0.28: they grow, it shrinks
    # not right: it gives a little back at 5 and 6 turns, then gains again (7.93 at 8), and the two sides differ


def test_two_fuller_volumes_draw_each_other_faster_as_they_go_and_further_than_a_lone_one_drifts():
    (left, right), (lone,) = _moved({3: 4, 7: 4}), _moved({7: 4})
    assert np.allclose(left, right, atol=0.06)  # each toward the other nearly alike: 0.27, 0.68, 1.95 and 0.27, 0.67, 1.89
    assert (np.diff(left, 2) > 0).all()  # faster as they go: 0.27, 0.41 and 1.26 a turn
    assert left[3] > 1.8 * lone[3] > 0  # a lone one drifts 0.94 by the line's own pull; the other lump adds as much


def test_of_two_uneven_volumes_the_heavier_moves_less():
    (heavy, light), (heavy_alone,), (light_alone,) = _moved({3: 8, 7: 2}), _moved({3: 8}), _moved({7: 2})
    by_the_other = heavy[3] - heavy_alone[3], light[3] - light_alone[3]  # 0.23 and 1.27 calm lengths
    assert 0 < 3 * by_the_other[0] < by_the_other[1]  # 8 against 2: a quarter by their shares, measured a fifth


@pytest.mark.xfail(strict=True, reason="(5) a fuller volume set going at a calm length a turn goes from volume 1 to "
                   "volume 5 in 6 turns and stops there (volume 4 before): every volume of the line pulls every "
                   "other, the line draws together on its middle and the lump gathers it (its fullest holds 2.93 at least)")
def test_a_fuller_volume_set_going_coasts_across_the_cells_as_a_body():
    store, cells = _line((1,), going=1.0)  # a calm length a turn of a calm swirl: the units' own speed
    peaks = []
    for _ in range(8):
        _turns(store, 1)
        peaks.append(_held(store, cells))
    place = [int(np.argmax(one)) for one in peaks]
    assert place == sorted(place) and place[-1] >= 6  # now 2, 2, 3, 4, 4, 5, 5, 5
    assert min(one.max() for one in peaks) > 2.3  # it stays a body: its fullest volume holds 2.36 at least
