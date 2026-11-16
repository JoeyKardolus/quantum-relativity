"""Axiom 08: pull spreads through all space at a finite speed, weaker with distance; a smaller packed volume pulls harder.

Bible rule 7 and rule 2 (owner, 2026-11-16). Measured:
  1. default universe (seed 1, Settings(), axiom_run.py), every 10 steps for 300 steps:
     no box holds a nonzero pull farther beyond the plus boxes' bounding box than the
     steps taken plus how far the plus boxes themselves spread (finite speed: at most
     one box per step on the wave's stencil), and from step 100 on (the wave has
     crossed the filled cube) at least 99% of the occupied boxes hold a nonzero pull
     (it reaches all of space)
  2. staged, pull wave only (pull_wave.py, Settings()): a 64-box cube of joint boxes
     (-1000, they carry the pull but add none) with the same 512,000 plus quanta at its
     centre either packed in 4**3 boxes or spread over 8**3 boxes. Over steps 120..240
     (after the front has reached the walls) the pull's time average along the x axis
     from the centre: after n <= 60 steps no box farther than n boxes from the source
     holds any pull; outside the source the averaged pull gets weaker with distance
     at every box from 8 to 28 boxes out; and at the centre the packed source's
     averaged pull is deeper than the spread one's
"""
import numpy as np

from physics.engine import Settings, State, pull_wave
from physics.engine.tests._gpu import needs_gpu

SIDE = 64
CENTRE = 32


def _source(across: int) -> np.ndarray:
    """Joint boxes everywhere inside a 1-box empty rim, 512,000 plus quanta in a cube `across` wide at the centre."""
    quanta = np.zeros((SIDE,) * 3, dtype=np.int32)
    quanta[1:-1, 1:-1, 1:-1] = -1000
    low = CENTRE - across // 2
    quanta[low:low + across, low:low + across, low:low + across] = 512000 // across ** 3
    return quanta


def _staged(across: int) -> tuple:
    """(farthest pull reach beyond the source at steps 10..60 minus the steps, time-averaged pull along +x)."""
    quanta = _source(across)
    state = State.from_host(quanta, np.zeros(quanta.shape))
    settings = Settings()
    low = CENTRE - across // 2
    overshoot = -10 ** 9
    total = np.zeros(SIDE - CENTRE)
    for count in range(1, 241):
        pull_wave(state, settings)
        if count <= 60 and count % 10 == 0:
            places = np.argwhere(state.pull.numpy() != 0)
            beyond = np.maximum(np.maximum(low - places, places - (low + across - 1)), 0).max()
            overshoot = max(overshoot, int(beyond) - count)
        if count >= 120:
            total += state.pull.numpy()[CENTRE:, CENTRE, CENTRE]
    return overshoot, total / 121.0


@needs_gpu
def test_the_pull_has_a_finite_speed_reaches_all_space_fades_and_is_deeper_from_a_packed_source():
    from physics.engine.tests.axiom_run import axiom_run
    run = axiom_run()
    overshoot = max(check["pull_reach"] - check["step"] - check["plus_spread"] for check in run["checks"][1:])
    cover = min(check["pull_cover"] for check in run["checks"] if check["step"] >= 100)
    packed_reach, packed = _staged(4)
    spread_reach, spread = _staged(8)
    outside = np.abs(packed[8:29])
    fades = bool(np.all(np.diff(outside) < 0))
    print(f"default run: pull reach beyond plus boxes minus (steps + plus spread), worst {overshoot}; least share of "
          f"occupied boxes with pull from step 100: {cover:.4f}")
    print(f"staged: reach past n boxes after n steps, worst {max(packed_reach, spread_reach)}; averaged pull at "
          f"centre packed {packed[0]:.4g}, spread {spread[0]:.4g}; |pull| from 8 to 28 boxes out: "
          + " ".join(f"{value:.3g}" for value in outside) + f"; strictly fading {fades}")
    assert overshoot <= 0 and max(packed_reach, spread_reach) <= 0
    assert cover >= 0.99
    assert fades
    assert packed[0] < spread[0] < 0
