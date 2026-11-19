"""(3) The pull against distance through the cells' sizes, and when it arrives: the grown patch held still."""
import numpy as np
import pytest

from physics.engine.axiom_3_structure.definitions.places import steps_of
from physics.engine.axiom_5_packing.logic.between import between, held_of
from physics.engine.axiom_5_packing.logic.lumps import fuller
from physics.engine.axiom_5_packing.logic.pairs import pulls
from physics.engine.axiom_5_packing.logic.staging import copy_of
from physics.engine.axiom_6_limits.definitions.collapse_limit import TOP_SPEED

STEPS = 70  # the furthest volume's pull is 54 steps late


@pytest.fixture(scope="module")
def felt(grown_patch):
    """One volume made fuller in a patch where nothing else moves: for every other volume, how far
    off it is, how many steps on its pull has risen (half of the rise), and how much more it is pulled in the end."""
    plain, staged = copy_of(grown_patch), copy_of(grown_patch)
    held = held_of(staged)
    volumes = np.flatnonzero((held > 0).ravel() & (steps_of(np.arange(held.size), staged.side) % 2 == 0).all(axis=1))
    one = int(volumes[np.abs(steps_of(volumes, staged.side) - staged.side // 2).max(axis=1).argmax()])  # the outermost: no row runs on through it
    seen = []
    for stage in (lambda: None, lambda: fuller(staged, [one])):
        stage()
        for _ in range(STEPS):
            cells, _deep, plain_drawn = pulls(plain)
            seen.append(pulls(staged)[2] - plain_drawn)
            plain.step, staged.step = plain.step + 1, staged.step + 1
    more = np.linalg.norm(np.array(seen[STEPS:]), axis=2)  # (steps, volumes) since the staging
    apart = np.linalg.norm(between(held_of(staged), steps_of(cells, staged.side)), axis=2)[cells == one][0]
    others = cells != one
    print(f"{others.sum()} volumes, {apart[others].min():.1f} to {apart[others].max():.0f} calm lengths off")
    print(f"felt at once, through the sizes alone: {(more[0] / more[-1])[others].max():.3f} of the pull at most")
    return apart[others], np.argmax(more > 0.5 * more[-1], axis=0)[others], more[-1][others], staged


def test_the_pull_of_a_volume_fades_as_one_over_the_distance_squared_through_the_cells_sizes(felt):
    apart, _first, more, _store = felt
    assert len(apart) > 60 and apart.max() > 5 * apart.min()  # 45 to 242 calm lengths off
    power = np.polyfit(np.log(apart), np.log(more), 1)[0]
    print(f"the pull fades as the distance to the power {power:.3f}")
    assert abs(power + 2) < 0.02  # -2.015 in the ball: the fuller volume is smaller, and that reaches the others at once


def test_the_pull_arrives_late_in_step_with_the_distance_at_the_one_top_speed(felt):
    apart, first, _more, store = felt
    a_step = TOP_SPEED * store.turn_per_step / (2 * np.pi)  # 4.4 calm lengths a step
    print(f"first felt {first.min()} to {first.max()} steps on")
    assert (first == np.ceil(apart / a_step)).all() and first.max() > 4 * first.min()  # 11 to 55 steps
