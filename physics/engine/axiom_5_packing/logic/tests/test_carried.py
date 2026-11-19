"""A line of calm volumes and joints, every rule on: a fuller volume gathers, one set going coasts as a body; two do not draw each other."""
import numpy as np
import pytest

from physics.engine import step
from physics.engine.axiom_5_packing.logic.line import calm_line, line_of

CALM = 648_000


def _line(fuller: tuple = (), going: float = 0.0):
    """The line bench and its cells."""
    store = calm_line(fuller=fuller, going=going)
    return store, line_of(store)


def _held(store, cells) -> np.ndarray:  # the volumes' quanta in calm volumes
    return store.cell_quanta.numpy()[cells[::2]] / CALM


def _at(store, cells, where=slice(None)) -> float:
    """Where the energy above a calm volume's stands along the line, in calm lengths through the cells' sizes."""
    sizes = CALM / np.abs(store.cell_quanta.numpy()[cells])
    middles = np.cumsum(sizes) - sizes / 2
    more = np.clip(_held(store, cells) - 1, 0, None)[where]
    return float((more * middles[::2][where]).sum() / more.sum())


def _turns(store, turns: int, act=step) -> None:
    for _ in range(25 * turns):
        act(store)


def test_a_fuller_volume_gains_from_its_thinner_neighbours_and_shrinks():
    store, cells = _line((5,))
    seen = [_held(store, cells)]
    for _ in range(8):
        _turns(store, 1)
        seen.append(_held(store, cells))
    gained = np.array(seen)[:, 5]
    assert (np.diff(gained) > 0).all() and gained[-1] > 6  # 4, 4.2, 4.63, 4.97, 5.18, 5.31, 5.46, 5.75, 6.29
    assert max(seen[4][4], seen[4][6]) < 0.6  # its two neighbours thin to 0.47 and 0.48: they grow, it shrinks


@pytest.mark.xfail(strict=True, reason="(b) two fuller volumes close 4.64 calm lengths in 8 turns, but a lone one "
                   "at the same place goes 3.94 toward the middle by itself: the line's thinning ends push it in, the "
                   "other lump adds nothing (in a line the pull fades to a tenth with every volume)")
def test_two_fuller_volumes_close_faster_than_two_lone_ones_drift():
    (pair, cells), (lone, _cells) = _line((3, 7)), _line((3,))
    apart, alone = _at(pair, cells, slice(6, None)) - _at(pair, cells, slice(0, 5)), _at(lone, cells)
    _turns(pair, 8), _turns(lone, 8)
    closed = apart - (_at(pair, cells, slice(6, None)) - _at(pair, cells, slice(0, 5)))
    assert closed > 2 * (_at(lone, cells) - alone)  # each pulled by the other on top of the line's own pull


def test_a_fuller_volume_set_going_coasts_across_the_cells_as_a_body():
    store, cells = _line((1,), going=1.0)  # a calm length a turn of a calm swirl: the units' own speed
    peaks = []
    for _ in range(8):
        _turns(store, 1)
        peaks.append(_held(store, cells))
    place = [int(np.argmax(one)) for one in peaks]
    assert place == sorted(place) and place[-1] >= 6  # from volume 1 to 7 in 8 turns: 1, 3, 3, 4, 5, 6, 6, 7
    assert min(one.max() for one in peaks) > 2.3  # it stays a body: its fullest volume holds 2.47 to 4.11

