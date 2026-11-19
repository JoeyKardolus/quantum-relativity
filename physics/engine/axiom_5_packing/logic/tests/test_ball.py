"""Gravity in the ball the engine grew (`ball.py`, what the window shows): fall, dense, coast, each against the same ball left alone."""
import numpy as np
import pytest

from physics.engine import step
from physics.engine.axiom_5_packing.logic.ball import OUT, PAIR, staged
from physics.engine.axiom_5_packing.logic.weighed import extra, middle_of, spread_of

TURNS = 3


def _run(*stores) -> list:
    """Each staged ball's extra over the plain one (the first of `stores`), the first volume left out, turn by turn."""
    seen, middle = [], stores[0].side // 4
    for turn in range(TURNS + 1):
        seen.append([extra(one, stores[0]) for one in stores[1:]])
        for more in seen[-1]:
            more[middle, middle, middle] = 0.0
        for one in stores if turn < TURNS else ():
            for _ in range(25):
                step(one)
    return seen


@pytest.fixture(scope="module")
def runs():
    """Three turns of the plain ball beside the pair, one of the pair alone, the dense and the coasting one (20 s)."""
    return _run(staged(), staged(PAIR), staged(PAIR[1:]), staged((OUT,)), staged((OUT,), True))


@pytest.mark.xfail(strict=True, reason="fall: in the grown ball two regions made 4 times fuller, 4 volumes apart, do not "
                   "draw together: their middles stand 3.18, 3.27, 3.18 and 3.27 volumes apart over 3 turns. The first "
                   "volume holds 1,640 of the ball's 1,646 calm volumes and pulls a region 16,000 times as hard as the "
                   "other region does (15.6 against 0.001), so what a region is given falls to the first volume")
def test_two_fuller_regions_of_the_ball_draw_together_more_than_each_alone(runs):
    low, high = np.s_[:, :runs[0][0].shape[0] // 2, :], np.s_[:, runs[0][0].shape[0] // 2 + 1:, :]
    gap = [middle_of(pair, high)[1] - middle_of(pair, low)[1] for pair, *_rest in runs]
    went = [float(middle_of(runs[0][run], high)[1] - middle_of(runs[-1][run], high)[1]) for run in (0, 1)]
    print(f"fall: the two middles {np.round(gap, 2)} volumes apart turn by turn; one went {went[0]:.2f} volumes toward "
          f"the other, and {went[1]:.2f} that way with no other there")
    assert gap[-1] < gap[0] and went[0] > went[1] > -0.1


@pytest.mark.xfail(strict=True, reason="dense: a region of the ball made 4 times fuller does not gather on its own middle: "
                   "its spread goes 0.95, 1.05, 1.23, 1.36 volumes in 3 turns and its own volumes keep 0.67, 0.51, 0.78 "
                   "of what they were given: it holds 1.1 calm volumes beside a first volume of 1,640")
def test_a_fuller_region_of_the_ball_gathers_on_its_own_middle_and_holds(runs):
    spread = [spread_of(dense) for *_rest, dense, _coast in runs]
    kept = [dense[runs[0][2] > 0].sum() / runs[0][2].sum() for *_rest, dense, _coast in runs]
    print(f"dense: spread {np.round(spread, 2)} volumes turn by turn, kept in its volumes {np.round(kept, 2)}")
    assert spread[-1] < spread[0] and kept[-1] > 0.9


@pytest.mark.xfail(strict=True, reason="coast: a fuller region two volumes out set going at a calm length a turn gets 0.01 of a "
                   "volume along its way in 3 turns (0.061 to 0.071; the same region not set going ends at -0.057): the "
                   "ball's outer cells are 30 calm lengths wide, so 3 calm lengths are a tenth of a cell, and the first "
                   "volume's pull is 16,000 times the region's own")
def test_a_fuller_region_set_going_in_the_ball_s_outer_part_crosses_cells(runs):
    along = [middle_of(coast)[1] for *_rest, coast in runs]
    still = [middle_of(dense)[1] for *_rest, dense, _coast in runs]
    print(f"coast: along its way {np.round(along, 3)} volumes turn by turn; not set going {np.round(still, 3)}")
    assert along[-1] - along[0] > 1  # a cell at least
